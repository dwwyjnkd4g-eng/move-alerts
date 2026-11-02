"""
US large-move scanner -> news context -> Claude explanation -> Telegram alert.

Flags S&P 500 names with large *idiosyncratic* moves (beta-adjusted vs their
sector ETF, scaled by their own residual vol), pulls headlines + earnings
timing from Finnhub, asks Claude to classify the move as information- or
flow-driven and give a fade verdict, then pushes an alert to Telegram and logs
it to alerts_log.csv for later evaluation (see evaluate.py).

Usage:
    python scanner.py                  # auto: intraday during US session, eod after close
    python scanner.py --mode eod
    python scanner.py --dry-run        # print alerts, don't send or log
    python scanner.py --tickers NKE,INTC --dry-run
    python scanner.py --test-telegram  # send a test message and exit

Env vars (GitHub Actions secrets):
    ANTHROPIC_API_KEY, FINNHUB_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional:
    CLAUDE_MODEL (default claude-sonnet-5-5)
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Config — tune these
# --------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "alerts_log.csv"
WHITELIST_PATH = HERE / "revolut_whitelist.txt"   # optional: one ticker per line

MIN_IDIO_Z = 3.0          # idiosyncratic move, in residual-vol sigmas
MIN_ABS_RET = 0.03        # and at least a 3% raw move
BIG_MOVE = 0.08           # ...or ANY move this large (e.g. -8%) counts on its own, whatever the sigma
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 10e6     # 20d avg $ volume (liquidity floor for the wide universe)
MAX_ALERTS_PER_RUN = 15
BETA_WINDOW = 120         # days for beta to sector ETF
RESID_WINDOW = 60         # days for residual vol
VOL_WINDOW = 20           # days for raw vol / avg volume
NEWS_LOOKBACK_DAYS = 3
MAX_HEADLINES = 15

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}
MARKET_ETF = "SPY"


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
US_LISTS = {   # Wikipedia pages; together ~ every US-listed company above roughly $1bn
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}
UNIVERSE_CACHE = HERE / "cache" / "us_universe.csv"
# Official lists of EVERY US-listed security (Nasdaq, NYSE, NYSE American, Arca, Cboe, IEX)
SYMBOL_DIRS = ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
               "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"]
_NOT_COMMON = re.compile(r"\b(warrants?|units?|rights?|preferred|preference|notes? due|debentures?|"
                         r"perpetual|subordinated|trust preferred|depositary shares,? each representing a 1/)", re.I)
LIQUID_CACHE = HERE / "cache" / "us_liquid.csv"
# Leveraged / inverse ETFs (kept even though other ETFs are excluded), e.g. TQQQ, SOXL, NVDL, SQQQ, SH
LEV_RE = re.compile(r"(\b-?\d(\.\d+)?x\b|\bultra|\bbull\b|\bbear\b|leveraged|inverse|proshares short|"
                    r"daily target|daily .{0,40}\b(long|short)\b)", re.I)
MIN_PRICE_UNIVERSE = 3.0


def all_us_listings() -> pd.DataFrame:
    """Every US-listed common stock / ADR (no ETFs, warrants, units, rights, preferreds, test issues)."""
    rows = []
    for url in SYMBOL_DIRS:
        lines = [l for l in _fetch(url).splitlines() if "|" in l and not l.startswith("File Creation")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)
        sym_col = "Symbol" if "Symbol" in df else "ACT Symbol"
        df = df[df.get("Test Issue", "N") == "N"]
        for _, r in df.iterrows():
            s, name = r[sym_col].strip(), r["Security Name"]
            if not s or re.search(r"[$^+=#*]", s):
                continue
            if r.get("ETF", "N") == "Y":
                if LEV_RE.search(name):
                    rows.append({"symbol": s, "name": name.strip(), "sector": "Leveraged ETF"})
                continue
            if _NOT_COMMON.search(name):
                continue
            rows.append({"symbol": s, "name": name.split(" - ")[0].replace(" Common Stock", "").strip(), "sector": ""})
    return pd.DataFrame(rows).drop_duplicates("symbol")
EXTRA_PATH = HERE / "extra_tickers.txt"   # optional: one US ticker per line, e.g. recent IPOs


def find_table(html_text: str, min_rows: int = 20) -> pd.DataFrame | None:
    """First table on a page that has a ticker/symbol column and enough rows."""
    for t in pd.read_html(io.StringIO(html_text)):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)).strip() for c in t.columns]
        cols = {str(c).lower(): c for c in t.columns}
        sym = next((cols[c] for c in cols if c in ("symbol", "ticker", "ticker symbol", "epic")
                    or c.startswith("ticker") or c.startswith("symbol")), None)
        if sym is not None and len(t) >= min_rows:
            name = next((cols[c] for c in cols if c in ("security", "company", "name", "company name")
                         or c.startswith("company")), None)
            sector = next((cols[c] for c in cols if "sector" in c or "industry" in c), None)
            out = pd.DataFrame({"symbol": t[sym].astype(str).str.strip(),
                                "name": t[name].astype(str) if name is not None else t[sym].astype(str),
                                "sector": t[sector].astype(str) if sector is not None else ""})
            return out[out["symbol"].str.len().between(1, 12)]
    return None


def _fetch(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=30)
    r.raise_for_status()
    return r.text


def load_universe(liquid_only: bool = True) -> pd.DataFrame:
    """The whole US equity market, then a liquidity screen.

    Index lists (S&P 500/400/600, Nasdaq-100) give GICS sectors; the exchange symbol directories
    add every other listed common stock and ADR. With liquid_only, keeps names that traded at least
    MIN_DOLLAR_VOL a day on average and closed above $3 (refreshed daily — see refresh_liquid()).
    Columns: symbol, yf_symbol, name, sector, etf."""
    full = _load_full_universe()
    if not liquid_only:
        return full
    try:
        liq = pd.read_csv(LIQUID_CACHE, keep_default_na=False)
        if datetime.now(timezone.utc).timestamp() - LIQUID_CACHE.stat().st_mtime < 86400:
            keep = set(liq["yf_symbol"])
            return full[full["yf_symbol"].isin(keep)].reset_index(drop=True)
    except Exception:
        pass
    return refresh_liquid(full)


def refresh_liquid(full: pd.DataFrame | None = None) -> pd.DataFrame:
    """Download a year of daily data for every listed stock (slow: ~10-20 min, once a day) and keep the liquid ones."""
    full = full if full is not None else _load_full_universe()
    etfs = sorted(set(full["etf"]) | {MARKET_ETF})
    daily = cached_daily(list(full["yf_symbol"]) + etfs)
    dv = (daily["close"] * daily["volume"]).iloc[-VOL_WINDOW:].mean()
    last = daily["close"].ffill().iloc[-1]
    ok = set(dv[(dv >= MIN_DOLLAR_VOL)].index) & set(last[last >= MIN_PRICE_UNIVERSE].index)
    try:
        import holdings
        ok |= {t.replace(".", "-") for t in holdings.tickers() if not t.endswith(".L")}
    except Exception:
        pass
    liq = full[full["yf_symbol"].isin(ok)].reset_index(drop=True)
    LIQUID_CACHE.parent.mkdir(exist_ok=True)
    liq.to_csv(LIQUID_CACHE, index=False)
    print(f"  liquid US universe: {len(liq)} of {len(full)} listed stocks")
    return liq


def _load_full_universe() -> pd.DataFrame:
    df = None
    try:
        if UNIVERSE_CACHE.exists() and \
                datetime.now(timezone.utc).timestamp() - UNIVERSE_CACHE.stat().st_mtime < 86400:
            df = pd.read_csv(UNIVERSE_CACHE, keep_default_na=False)
    except Exception:
        df = None
    if df is None:
        frames = []
        for label, url in US_LISTS.items():
            try:
                t = find_table(_fetch(url), min_rows=50)
                if t is not None:
                    frames.append(t)
                    print(f"  universe: {label} {len(t)}")
            except Exception as e:
                print(f"  universe: {label} failed ({e})", file=sys.stderr)
        try:
            listed = all_us_listings()
            frames.append(listed)       # after the index lists, so their sectors win on duplicates
            print(f"  universe: all US listings {len(listed)}")
        except Exception as e:
            print(f"  universe: exchange symbol lists failed ({e})", file=sys.stderr)
        if not frames:
            raise RuntimeError("couldn't load any US stock list")
        df = pd.concat(frames)
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df = df.drop_duplicates("symbol")
        UNIVERSE_CACHE.parent.mkdir(exist_ok=True)
        df.to_csv(UNIVERSE_CACHE, index=False)
    extras = set()
    if EXTRA_PATH.exists():
        extras |= {l.strip().upper() for l in EXTRA_PATH.read_text().splitlines() if l.strip() and not l.startswith("#")}
    try:
        import holdings
        extras |= {t for t in holdings.tickers() if "." not in t}
    except Exception:
        pass
    new = sorted(extras - set(df["symbol"]))
    if new:
        df = pd.concat([df, pd.DataFrame({"symbol": new, "name": new, "sector": ""})])
    df = df[["symbol", "name", "sector"]].copy()
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["yf_symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    df["etf"] = df["sector"].map(SECTOR_ETF).fillna(MARKET_ETF)

    if WHITELIST_PATH.exists():
        wl = {
            line.strip().upper()
            for line in WHITELIST_PATH.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        if wl:
            df = df[df["symbol"].str.upper().isin(wl)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
CHUNK = 200


def yf_chunked(syms: list[str], fields: tuple[str, ...], **kw) -> dict[str, pd.DataFrame]:
    """yf.download in chunks (Yahoo throttles very large requests), merged per field."""
    import time as _time
    import yfinance as yf

    parts: dict[str, list[pd.DataFrame]] = {f.lower(): [] for f in fields}
    syms = list(dict.fromkeys(syms))
    for i in range(0, len(syms), CHUNK):
        chunk = syms[i:i + CHUNK]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, group_by="column", threads=True, progress=False, **kw)
                break
            except Exception as e:
                print(f"  download retry ({e})", file=sys.stderr)
                _time.sleep(5 * (attempt + 1))
        else:
            continue
        for f in fields:
            if f not in raw:
                continue
            x = raw[f]
            if isinstance(x, pd.Series):
                x = x.to_frame(chunk[0])
            parts[f.lower()].append(x)
        if len(syms) > CHUNK:
            _time.sleep(1)
    return {k: (pd.concat(v, axis=1).sort_index() if v else pd.DataFrame()) for k, v in parts.items()}


def cached_daily(yf_symbols: list[str], name: str = "us", tz=None) -> dict[str, pd.DataFrame]:
    """1y daily open/close/volume up to YESTERDAY, downloaded once a day and cached on disk."""
    tz = tz or ET
    path = HERE / "cache" / f"daily_{name}.pkl"
    today = datetime.now(tz).date()
    data = None
    try:
        data = pd.read_pickle(path)
        if data["asof"] != today or not set(yf_symbols) <= set(data.get("requested", [])):
            data = None
    except Exception:
        data = None
    if data is None:
        print(f"  refreshing daily history for {len(yf_symbols)} symbols…")
        got = yf_chunked(yf_symbols, ("Open", "Close", "Volume"), period="1y", interval="1d", auto_adjust=True)
        data = {"asof": today, "requested": list(yf_symbols)}
        for k, f in got.items():
            f.index = pd.to_datetime(f.index).tz_localize(None) if getattr(f.index, "tz", None) else pd.to_datetime(f.index)
            data[k] = f[f.index.date < today]
        path.parent.mkdir(exist_ok=True)
        pd.to_pickle(data, path)
    return {k: data[k] for k in ("open", "close", "volume")}


def with_today(daily: dict[str, pd.DataFrame], intra: dict[str, pd.DataFrame], today) -> dict[str, pd.DataFrame]:
    """Append today's regular-session row (open = first bar, close = last bar, volume = sum)."""
    out = {}
    if intra["close"].empty:
        return daily
    ic = intra["close"]
    reg_mask = [(t.date() == today) and _is_regular_bar(t) for t in ic.index]
    if not any(reg_mask):
        return daily
    reg_c, reg_o, reg_v = ic[reg_mask], intra["open"][reg_mask], intra["volume"][reg_mask]
    row_close = reg_c.ffill().iloc[-1]
    row_open = reg_o.bfill().iloc[0]
    row_vol = reg_v.fillna(0).sum()
    ts = pd.Timestamp(today)
    for k, row in (("close", row_close), ("open", row_open), ("volume", row_vol)):
        f = daily[k].copy()
        f.loc[ts] = row.reindex(f.columns)
        out[k] = f.sort_index()
    return out


def download_prices(yf_symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Daily history (cached) + today's live regular-session row."""
    today = datetime.now(ET).date()
    daily = cached_daily(yf_symbols)
    intra = download_intraday(yf_symbols)
    return with_today(daily, intra, today)


def session_fraction(now_et: datetime) -> float:
    """Fraction of the regular session elapsed (for projecting partial-day volume)."""
    open_ = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    frac = (now_et - open_).total_seconds() / (6.5 * 3600)
    return float(min(max(frac, 0.05), 1.0))


# --------------------------------------------------------------------------
# Move metrics (pure function — tested offline)
# --------------------------------------------------------------------------
def compute_metrics(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    stock_to_etf: dict[str, str],
    vol_scale: float = 1.0,
) -> pd.DataFrame:
    """
    For the last row of the price frames, compute per stock:
      ret, z (ret / 20d vol), beta to sector ETF, sector_ret, idio_ret, idio_z,
      gap, rvol, adv_usd, ret_5d_prior, dist_52w_high, dist_52w_low.
    vol_scale projects today's partial volume to a full session (1.0 at EOD).
    """
    rows = []
    last_day = close.index[-1]
    for sym, etf in stock_to_etf.items():
        if sym not in close.columns or etf not in close.columns:
            continue
        # align stock and benchmark on days both traded (exchanges have different holidays)
        pair = pd.concat([close[sym], close[etf]], axis=1, keys=["s", "b"]).dropna()
        if len(pair) < BETA_WINDOW + 8 or pair.index[-1] != last_day:
            continue
        lr = np.log(pair).diff()
        s, b, c = lr["s"], lr["b"], pair["s"]
        volume_s = volume[sym].reindex(pair.index)
        open_s = open_[sym].reindex(pair.index)

        r_today, b_today = s.iloc[-1], b.iloc[-1]
        hist_s = s.iloc[-BETA_WINDOW - 1:-1]
        hist_b = b.iloc[-BETA_WINDOW - 1:-1]
        var_b = hist_b.var()
        beta = hist_s.cov(hist_b) / var_b if var_b > 0 else 1.0

        resid_hist = (hist_s - beta * hist_b).iloc[-RESID_WINDOW:]
        resid_sd = resid_hist.std()
        vol20 = hist_s.iloc[-VOL_WINDOW:].std()
        idio = r_today - beta * b_today

        vol_hist = volume_s.iloc[-VOL_WINDOW - 1:-1]
        avg_vol = vol_hist.mean()
        adv_usd = (c.iloc[-VOL_WINDOW - 1:-1] * vol_hist).mean()
        rvol = (volume_s.iloc[-1] / vol_scale) / avg_vol if avg_vol > 0 else np.nan

        win = c.iloc[-253:-1]
        rows.append({
            "yf_symbol": sym,
            "etf": etf,
            "price": c.iloc[-1],
            "ret": np.expm1(r_today),
            "z": r_today / vol20 if vol20 > 0 else np.nan,
            "beta": beta,
            "sector_ret": np.expm1(b_today),
            "idio_ret": np.expm1(idio),
            "idio_z": idio / resid_sd if resid_sd > 0 else np.nan,
            "gap": open_s.iloc[-1] / c.iloc[-2] - 1,
            "rvol": rvol,
            "adv_usd": adv_usd,
            "ret_5d_prior": c.iloc[-2] / c.iloc[-7] - 1,
            "dist_52w_high": c.iloc[-1] / win.max() - 1,
            "dist_52w_low": c.iloc[-1] / win.min() - 1,
        })
    return pd.DataFrame(rows)


LEV_MIN_Z, LEV_MIN_RET = 3.0, 0.05    # leveraged ETFs: big raw move vs their own vol also counts


def _lev_mask(m: pd.DataFrame) -> pd.Series:
    lev = m["name"].map(is_leveraged) if "name" in m else pd.Series(False, index=m.index)
    return lev & (m["z"].abs() >= LEV_MIN_Z) & (m["ret"].abs() >= LEV_MIN_RET)


def select_movers(m: pd.DataFrame) -> pd.DataFrame:
    big = (((m["idio_z"].abs() >= MIN_IDIO_Z) & (m["ret"].abs() >= MIN_ABS_RET))
           | (m["ret"].abs() >= BIG_MOVE) | _lev_mask(m))
    f = m[
        big
        & (m["price"] >= MIN_PRICE)
        & (m["adv_usd"] >= MIN_DOLLAR_VOL)
    ].copy()
    f["rank"] = f["idio_z"].abs() + 50 * f["ret"].abs()
    return f.sort_values("rank", ascending=False).head(MAX_ALERTS_PER_RUN)


# --------------------------------------------------------------------------
# Extended hours (pre-market / after-hours)
# --------------------------------------------------------------------------
MIN_EXT_DOLLAR_VOL = 1e6     # traded since the close, to ignore stale single prints
EXT_FRESH_MIN = 45
FUTURES = "ES=F"             # S&P 500 futures = market move outside cash hours


def session_now(now_et: datetime) -> str | None:
    """pre 04:00-09:45 | regular 09:45-16:00 | close 16:00-16:30 | post 16:30-20:00 (ET, weekdays)."""
    if now_et.weekday() >= 5:
        return None
    hm = now_et.hour * 100 + now_et.minute
    if 400 <= hm < 945:
        return "pre"
    if 945 <= hm < 1600:
        return "regular"
    if 1600 <= hm < 1630:
        return "close"
    if 1630 <= hm < 2000:
        return "post"
    return None


def download_intraday(yf_symbols: list[str], tz=None, period: str = "5d") -> dict[str, pd.DataFrame]:
    """15-minute bars incl. pre/after-hours, index in `tz` (New York by default)."""
    tz = tz or ET
    got = yf_chunked(yf_symbols, ("Open", "Close", "Volume"), period=period, interval="15m",
                     prepost=True, auto_adjust=False)
    for k, f in got.items():
        if f.empty:
            continue
        idx = f.index if f.index.tz is not None else f.index.tz_localize("UTC")
        f.index = idx.tz_convert(tz)
    return got


def _is_regular_bar(ts: pd.Timestamp) -> bool:
    hm = ts.hour * 100 + ts.minute
    return 930 <= hm < 1600


def compute_extended(
    daily_close: pd.DataFrame,
    intra_close: pd.DataFrame,
    intra_vol: pd.DataFrame,
    symbols: list[str],
    now_et: datetime,
) -> pd.DataFrame:
    """Move since the last regular-session close, adjusted for the S&P futures move."""
    fut = intra_close[FUTURES].dropna() if FUTURES in intra_close else pd.Series(dtype=float)
    spy_d = np.log(daily_close[MARKET_ETF]).diff()
    rows = []
    for sym in symbols:
        if sym not in intra_close or sym not in daily_close:
            continue
        c = intra_close[sym].dropna()
        if c.empty:
            continue
        reg = c[[_is_regular_bar(t) for t in c.index]]
        if reg.empty:
            continue
        ref_ts, ref = reg.index[-1], float(reg.iloc[-1])
        ext = c[c.index > ref_ts]
        if ext.empty or (now_et - ext.index[-1].to_pydatetime()) > timedelta(minutes=EXT_FRESH_MIN):
            continue
        last = float(ext.iloc[-1])
        v = intra_vol[sym].reindex(ext.index).fillna(0)
        ext_dv = float((ext * v).sum())

        # daily history up to the reference session
        d = daily_close[sym]
        d = d[d.index.date <= ref_ts.date()].dropna()
        sd = np.log(d).diff()
        sp = spy_d.reindex(sd.index)
        hs, hb = sd.iloc[-BETA_WINDOW:], sp.iloc[-BETA_WINDOW:]
        if len(hs.dropna()) < 60:
            continue
        beta = hs.cov(hb) / hb.var() if hb.var() > 0 else 1.0
        resid_sd = (hs - beta * hb).iloc[-RESID_WINDOW:].std()
        vol20 = hs.iloc[-VOL_WINDOW:].std()

        mkt = 0.0
        if len(fut):
            f_ref = fut[fut.index <= ref_ts]
            if len(f_ref):
                mkt = float(fut.iloc[-1] / f_ref.iloc[-1] - 1)
        r = math.log(last / ref)
        idio = r - beta * math.log1p(mkt)
        win = d.iloc[-252:]
        rows.append({
            "yf_symbol": sym, "etf": "S&P fut", "price": last, "ret": math.expm1(r),
            "z": r / vol20 if vol20 > 0 else np.nan, "beta": beta, "sector_ret": mkt,
            "idio_ret": math.expm1(idio), "idio_z": idio / resid_sd if resid_sd > 0 else np.nan,
            "gap": np.nan, "rvol": np.nan, "ext_dollar_vol": ext_dv,
            "ret_5d_prior": float(d.iloc[-1] / d.iloc[-6] - 1) if len(d) > 6 else np.nan,
            "dist_52w_high": last / win.max() - 1, "dist_52w_low": last / win.min() - 1,
        })
    return pd.DataFrame(rows)


def select_extended(m: pd.DataFrame, adv: pd.Series) -> pd.DataFrame:
    if m.empty:
        return m
    m = m.copy()
    m["adv_usd"] = m["yf_symbol"].map(adv)
    big = (((m["idio_z"].abs() >= MIN_IDIO_Z) & (m["ret"].abs() >= MIN_ABS_RET))
           | (m["ret"].abs() >= BIG_MOVE) | _lev_mask(m))
    f = m[
        big
        & (m["price"] >= MIN_PRICE)
        & (m["adv_usd"] >= MIN_DOLLAR_VOL)
        & (m["ext_dollar_vol"] >= MIN_EXT_DOLLAR_VOL)
    ].copy()
    f["rank"] = f["idio_z"].abs() + 50 * f["ret"].abs()
    return f.sort_values("rank", ascending=False).head(MAX_ALERTS_PER_RUN)


# --------------------------------------------------------------------------
# Earnings overreaction gauge
# --------------------------------------------------------------------------
def earnings_reactions(close: pd.Series, dates: list[pd.Timestamp]) -> list[float]:
    """Close-to-close reaction for each past earnings date (pre-market -> same day, after close -> next day)."""
    close = close.dropna()
    idx = close.index
    out = []
    for d in dates:
        d_et = d.tz_convert(ET) if d.tzinfo else d.tz_localize(ET)
        day = pd.Timestamp(d_et.date())
        pos = idx.searchsorted(day)
        if pos >= len(idx) or pos == 0:
            continue
        if idx[pos].date() != day.date():
            continue  # not a trading day in our data
        if d_et.hour >= 12:      # after close -> reaction is the next session
            pos += 1
            if pos >= len(idx):
                continue
        out.append(float(close.iloc[pos] / close.iloc[pos - 1] - 1))
    return out


def earnings_gauge(yf_symbol: str, close: pd.Series, move: float) -> dict | None:
    """How big is this move vs the stock's usual earnings reaction? Uses yfinance's earnings dates."""
    try:
        import yfinance as yf
        ed = yf.Ticker(yf_symbol).get_earnings_dates(limit=16)
        if ed is None or ed.empty:
            return None
        now = pd.Timestamp.now(tz=ET)
        past = [t for t in ed.index if t < now - pd.Timedelta(days=3)]
        reac = earnings_reactions(close, past)[:8]
        if len(reac) < 3:
            return None
        avg = float(np.mean(np.abs(reac)))
        return {"avg_abs": avg, "n": len(reac), "ratio": abs(move) / avg if avg > 0 else None}
    except Exception as e:
        print(f"  earnings gauge failed for {yf_symbol}: {e}", file=sys.stderr)
        return None


def earnings_recent(earnings: list[dict], today: date) -> bool:
    for e in earnings:
        try:
            d = date.fromisoformat(e.get("date", ""))
        except ValueError:
            continue
        if today - timedelta(days=2) <= d <= today:
            return True
    return False


# --------------------------------------------------------------------------
# Context: Finnhub news + earnings
# --------------------------------------------------------------------------
FINNHUB = "https://finnhub.io/api/v1"


def _finnhub(path: str, params: dict) -> object:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return None
    for attempt in range(3):
        try:
            r = requests.get(f"{FINNHUB}/{path}", params={**params, "token": key}, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + attempt * 3)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  finnhub {path} error: {e}", file=sys.stderr)
            time.sleep(1)
    return None


def get_news(symbol: str, today: date) -> list[dict]:
    data = _finnhub("company-news", {
        "symbol": symbol,
        "from": (today - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat(),
        "to": today.isoformat(),
    }) or []
    items = sorted(data, key=lambda x: x.get("datetime", 0), reverse=True)
    seen, out = set(), []
    for it in items:
        h = (it.get("headline") or "").strip()
        if not h or h.lower() in seen:
            continue
        seen.add(h.lower())
        out.append({
            "headline": h,
            "summary": (it.get("summary") or "")[:400],
            "source": it.get("source", ""),
            "url": it.get("url", ""),
            "time_et": datetime.fromtimestamp(it.get("datetime", 0), ET).strftime("%a %d %b %H:%M ET"),
        })
        if len(out) >= MAX_HEADLINES:
            break
    return out


def get_earnings(symbol: str, today: date) -> list[dict]:
    data = _finnhub("calendar/earnings", {
        "symbol": symbol,
        "from": (today - timedelta(days=7)).isoformat(),
        "to": (today + timedelta(days=7)).isoformat(),
    }) or {}
    return data.get("earningsCalendar", []) if isinstance(data, dict) else []


# --------------------------------------------------------------------------
# Claude explanation
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an equity analyst supporting a discretionary mean-reversion trader \
who trades large single-stock moves in US equities. For each flagged move you get quantitative \
stats, numbered news items, and earnings-calendar data.

Rules:
- Use ONLY the supplied evidence. Cite news items by number. Never invent figures, quotes or events.
- If nothing supplied plausibly explains the move, say catalyst_type = "no_identifiable_catalyst".
  That is a useful answer, not a failure.
- Be decisive but calibrated; say "unclear" when the evidence genuinely doesn't support a view.

Framework:
- INFORMATION moves tend to persist or drift (post-earnings-announcement drift, guidance resets, \
M&A, regulatory/legal rulings, management change, financing/dilution, major contract wins/losses). \
Fading these is usually wrong unless the reaction is clearly far larger than the news justifies.
- FLOW / low-information moves are better reversion candidates: sector or peer sympathy moves, \
index/ETF/rebalance flows, macro-beta days, analyst notes with no new data, rumours later denied, \
technical breaks, squeezes, moves with no catalyst at all.
- Weigh: idiosyncratic size (idio_z) vs raw move; whether the sector moved; relative volume; \
gap vs intraday trend; prior 5-day move (exhaustion?) and position vs 52-week range; and whether \
earnings were just reported or are imminent (event risk for a fade).
- If earnings_move_vs_typical_x is present: a move well above ~1.5-2x the stock's typical \
earnings reaction is a classic overreaction signal; well below 1x suggests the market has \
barely priced the news.
- Pre-market / after-hours moves: liquidity is thin and prices can overshoot, but these sessions \
are also when earnings and major news land, so an extended-hours move WITH a clear catalyst is \
usually information. Here the benchmark is S&P 500 futures, not a sector ETF.
- Leveraged/inverse ETFs (instrument field present): judge the move by its UNDERLYING (stock, \
index or sector named in the fund name) — the ETF should move ~leverage x the underlying, so the \
catalyst is the underlying's. Note for a fade: daily-reset decay and path dependence make holding \
these for days costly, especially in volatile, choppy markets; size and holding period matter.
"""

EXPLAIN_TOOL = {
    "name": "record_assessment",
    "description": "Record the structured assessment of the stock move.",
    "input_schema": {
        "type": "object",
        "properties": {
            "catalyst_type": {"type": "string", "enum": [
                "earnings", "guidance", "m_and_a", "analyst_action", "regulatory_legal",
                "management", "financing_dilution", "product_contract", "sector_peer_sympathy",
                "macro", "no_identifiable_catalyst", "other"]},
            "catalyst_summary": {"type": "string", "description": "1-2 sentences: what happened."},
            "evidence_ids": {"type": "array", "items": {"type": "integer"},
                             "description": "News item numbers supporting the catalyst."},
            "move_nature": {"type": "string", "enum": ["information", "flow", "mixed", "unclear"]},
            "verdict": {"type": "string", "enum": ["fade_candidate", "do_not_fade", "unclear"]},
            "rationale": {"type": "string", "description": "2-4 sentences on why the move does or doesn't make sense."},
            "what_would_change_view": {"type": "string", "description": "One sentence."},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["catalyst_type", "catalyst_summary", "evidence_ids", "move_nature",
                     "verdict", "rationale", "what_would_change_view", "confidence"],
    },
}


def is_leveraged(name: str) -> bool:
    return bool(LEV_RE.search(str(name))) and bool(re.search(r"(etf|fund|shares|trust)", str(name), re.I))


def build_prompt(row: pd.Series, name: str, news: list[dict], earnings: list[dict], mode: str) -> str:
    def r(x, k=100, d=2):
        return None if pd.isna(x) else round(float(x) * k, d)

    stats = {
        "ticker": row["symbol"], "company": name, "benchmark": row["etf"],
        "data_as_of": AS_OF.get(mode, mode),
        "move_pct": r(row["ret"]),
        "move_in_own_vol_sigmas": r(row["z"], 1),
        "beta_to_benchmark": r(row["beta"], 1),
        "benchmark_move_pct": r(row["sector_ret"]),
        "idiosyncratic_move_pct": r(row["idio_ret"]),
        "idiosyncratic_sigmas": r(row["idio_z"], 1),
        "gap_at_open_pct": r(row.get("gap")),
        "relative_volume_x": r(row.get("rvol"), 1),
        "extended_hours_dollar_volume_m": r(row.get("ext_dollar_vol"), 1e-6, 1),
        "earnings_move_vs_typical_x": r(row.get("earn_ratio"), 1, 1),
        "typical_earnings_reaction_pct": r(row.get("earn_avg")),
        "prior_5d_move_pct": r(row["ret_5d_prior"]),
        "pct_from_52w_high": r(row["dist_52w_high"], 100, 1),
        "pct_from_52w_low": r(row["dist_52w_low"], 100, 1),
    }
    stats = {k: v for k, v in stats.items() if v is not None}
    if is_leveraged(name):
        stats["instrument"] = "leveraged/inverse ETF (see name for leverage and underlying)"
    news_txt = "\n".join(
        f"[{i}] {n['time_et']} | {n['source']} | {n['headline']}\n    {n['summary']}"
        for i, n in enumerate(news, 1)
    ) or "(no news items returned)"
    earn_txt = json.dumps(earnings, default=str) if earnings else "(no earnings within +/-7 days)"
    return (
        f"MOVE STATS\n{json.dumps(stats, indent=1)}\n\n"
        f"NEWS (last {NEWS_LOOKBACK_DAYS} days, newest first)\n{news_txt}\n\n"
        f"EARNINGS CALENDAR\n{earn_txt}\n\n"
        "Assess this move and call record_assessment."
    )


def explain(prompt: str, n_news: int) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        tools=[EXPLAIN_TOOL],
        tool_choice={"type": "tool", "name": "record_assessment"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if block.type == "tool_use":
            return sanitize_assessment(dict(block.input), n_news)
    raise RuntimeError("Claude returned no assessment")


def sanitize_assessment(a: dict, n_news: int) -> dict:
    """Drop citations that point at news items that don't exist."""
    a["evidence_ids"] = [i for i in a.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= n_news]
    return a


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------
VERDICT_LABEL = {"fade_candidate": "FADE CANDIDATE", "do_not_fade": "DON'T FADE", "unclear": "UNCLEAR"}
TAG = {"intraday": "live", "eod": "close", "pre": "pre-market", "post": "after-hours"}
AS_OF = {"intraday": "live intraday", "eod": "official close",
         "pre": "pre-market, vs prior regular close", "post": "after-hours, vs today's regular close"}


def format_alert(row: pd.Series, name: str, a: dict, news: list[dict], mode: str) -> str:
    e = html.escape
    arrow = "🔺" if row["ret"] > 0 else "🔻"
    tag = TAG.get(mode, mode)
    if mode in ("pre", "post"):
        stats_line = (f"S&amp;P fut {row['sector_ret']*100:+.1f}% · β {row['beta']:.2f} · "
                      f"${row['ext_dollar_vol']/1e6:,.1f}m traded since close · "
                      f"prior 5d {row['ret_5d_prior']*100:+.1f}%")
    else:
        stats_line = (f"{e(row['etf'])} {row['sector_ret']*100:+.1f}% · β {row['beta']:.2f} · "
                      f"RVOL {row['rvol']:.1f}x · gap {row['gap']*100:+.1f}% · "
                      f"prior 5d {row['ret_5d_prior']*100:+.1f}%")
    lev = " ⚙️ <i>leveraged ETF</i>" if is_leveraged(name) else ""
    lines = [
        f"{arrow} <b>{e(row['symbol'])}</b> {row['ret']*100:+.1f}%  "
        f"(idio {row['idio_z']:+.1f}σ) · {e(name)} [{tag}]{lev}",
        stats_line,
    ]
    if not pd.isna(row.get("earn_ratio", np.nan)):
        lines.append(f"Earnings: move is <b>{row['earn_ratio']:.1f}×</b> its typical reaction "
                     f"(avg ±{row['earn_avg']*100:.1f}% over last {int(row['earn_n'])})")
    lines += [
        "",
        f"<b>Catalyst:</b> {e(a['catalyst_type'].replace('_', ' '))} — {e(a['catalyst_summary'])}",
        f"<b>Nature:</b> {e(a['move_nature'])} → <b>{VERDICT_LABEL.get(a['verdict'], a['verdict'])}</b> "
        f"({e(a['confidence'])})",
        f"<b>Why:</b> {e(a['rationale'])}",
        f"<b>Changes view:</b> {e(a['what_would_change_view'])}",
    ]
    if a["evidence_ids"]:
        lines.append("")
        for i in a["evidence_ids"][:4]:
            n = news[i - 1]
            src = e(n["source"] or "source")
            if n["url"]:
                lines.append(f'[{i}] <a href="{e(n["url"], quote=True)}">{src}</a>: {e(n["headline"][:140])}')
            else:
                lines.append(f"[{i}] {src}: {e(n['headline'][:140])}")
    text = "\n".join(lines)
    return text


MUTE_PATH = HERE / "mute_state.json"


def is_muted() -> bool:
    try:
        until = datetime.fromisoformat(json.loads(MUTE_PATH.read_text())["until"])
        return datetime.now(timezone.utc) < until
    except Exception:
        return False


def send_telegram(text: str, force: bool = False, chat_id: str | None = None) -> None:
    """Send an HTML message. Skipped while muted (/mute) unless force=True."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not force and is_muted():
        print("Muted — not sent:\n" + text[:200])
        return
    if not token or not chat:
        print("Telegram not configured; printing instead:\n" + text)
        return
    for chunk in split_message(text):
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20,
        )
        if not r.ok:
            print(f"Telegram error {r.status_code}: {r.text}", file=sys.stderr)
            # never lose a message: resend as plain text if the formatting was rejected
            plain = html.unescape(re.sub(r"<[^>]+>", "", chunk))
            r2 = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                               json={"chat_id": chat, "text": plain[:4096], "disable_web_page_preview": True},
                               timeout=20)
            if not r2.ok:
                print(f"Telegram plain-text retry failed {r2.status_code}: {r2.text}", file=sys.stderr)


_TAG_RE = re.compile(r"<(/?)(b|i|u|a|pre|code)\b[^>]*>")


def _balanced(chunk: str) -> bool:
    depth = 0
    for m in _TAG_RE.finditer(chunk):
        depth += -1 if m.group(1) else 1
    return depth == 0


def split_message(text: str, limit: int = 3800) -> list[str]:
    """Split a long HTML message into Telegram-sized parts at blank lines / line breaks,
    never inside a tag pair."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for block in text.split("\n"):
        cand = (cur + "\n" + block) if cur else block
        if len(cand) > limit and cur and _balanced(cur):
            parts.append(cur)
            cur = block
        else:
            cur = cand
    if cur:
        parts.append(cur)
    out = []
    for p in parts:   # last resort for a single enormous line
        out += [p[i:i + 4000] for i in range(0, len(p), 4000)] if len(p) > 4096 else [p]
    return out


# --------------------------------------------------------------------------
# Log / dedupe
# --------------------------------------------------------------------------
LOG_COLS = ["run_ts_utc", "date", "mode", "symbol", "price", "ret", "z", "idio_z", "beta",
            "etf", "sector_ret", "rvol", "gap", "ret_5d_prior", "catalyst_type", "move_nature",
            "verdict", "confidence", "catalyst_summary"]


SESSION_GROUP = {"intraday": "regular", "eod": "regular", "pre": "pre", "post": "post"}
EQUITY_STATE_PATH = HERE / "equity_state.json"


def load_equity_state() -> dict:
    try:
        return json.loads(EQUITY_STATE_PATH.read_text())
    except Exception:
        return {}


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLS)


def append_log(records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)[LOG_COLS]
    df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "intraday", "eod", "pre", "post"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tickers", help="comma-separated subset (S&P 500 symbols)")
    ap.add_argument("--test-telegram", action="store_true")
    ap.add_argument("--latest", action="store_true",
                    help="scan the most recent trading day even on weekends/holidays (manual runs)")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the whole-market universe, liquidity screen and daily history cache (slow)")
    args = ap.parse_args()

    if args.test_telegram:
        send_telegram("✅ move-alerts: Telegram is wired up.")
        return 0

    lock = HERE / "cache" / "refresh.lock"
    if args.refresh:
        lock.parent.mkdir(exist_ok=True)
        lock.write_text(datetime.now(timezone.utc).isoformat())
        try:
            for p in (UNIVERSE_CACHE, LIQUID_CACHE):
                p.unlink(missing_ok=True)
            liq = refresh_liquid()
            cached_daily(list(liq["yf_symbol"]) + sorted(set(liq["etf"]) | {MARKET_ETF}))
        finally:
            lock.unlink(missing_ok=True)
        return 0
    if lock.exists() and datetime.now(timezone.utc).timestamp() - lock.stat().st_mtime < 3600:
        print("Daily universe refresh in progress — skipping this scan.")
        return 0

    now_et = datetime.now(ET)
    today = now_et.date()
    mode = args.mode
    if args.latest:
        mode = "eod"
    elif mode == "auto":
        session = session_now(now_et)
        if session is None:
            print(f"Outside US trading sessions ({now_et:%a %H:%M} ET) — nothing to do.")
            return 0
        mode = {"pre": "pre", "regular": "intraday", "close": "eod", "post": "post"}[session]

    uni = load_universe()
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        uni = uni[uni["symbol"].isin(wanted)]
    print(f"Universe: {len(uni)} names, mode={mode}")
    yf_syms = list(uni["yf_symbol"])

    if mode in ("pre", "post"):
        daily = cached_daily(yf_syms + [MARKET_ETF] + sorted(set(uni["etf"])))
        dv = (daily["close"] * daily["volume"]).iloc[-VOL_WINDOW:].mean()
        intra = download_intraday(yf_syms + [FUTURES])
        m = compute_extended(daily["close"], intra["close"], intra["volume"], yf_syms, now_et)
        if m.empty:
            print("No extended-hours trading data yet.")
            return 0
        m = m.merge(uni[["yf_symbol", "symbol", "name"]], on="yf_symbol")
        movers = select_extended(m, dv)
        closes = daily["close"]
    else:
        etfs = sorted(set(uni["etf"]) | {MARKET_ETF})
        px = download_prices(yf_syms + etfs)
        last_bar = px["close"].index[-1].date()
        if last_bar != today:
            if args.latest:
                print(f"Using latest trading day {last_bar}.")
                today = last_bar
            else:
                print(f"No bar for {today} (last {last_bar}) — market holiday or data delay.")
                return 0
        vol_scale = session_fraction(now_et) if mode == "intraday" else 1.0
        m = compute_metrics(px["close"], px["open"], px["volume"],
                            dict(zip(uni["yf_symbol"], uni["etf"])), vol_scale)
        m = m.merge(uni[["yf_symbol", "symbol", "name"]], on="yf_symbol")
        movers = select_movers(m)
        closes = px["close"]

    log = load_log()
    group = SESSION_GROUP.get(mode, mode)
    if len(log):
        same = (log["date"] == today.isoformat()) & (log["mode"].map(lambda x: SESSION_GROUP.get(x, x)) == group)
        already = set(log.loc[same, "symbol"])
    else:
        already = set()
    new_movers = movers[~movers["symbol"].isin(already)]
    print(f"Flagged {len(movers)} ({len(new_movers)} new).")

    records = []
    for _, row in new_movers.iterrows():
        sym = row["symbol"]
        news = get_news(sym, today)
        earnings = get_earnings(sym, today)
        if earnings_recent(earnings, today) and row["yf_symbol"] in closes:
            g = earnings_gauge(row["yf_symbol"], closes[row["yf_symbol"]], row["ret"])
            if g and g["ratio"] is not None:
                row["earn_ratio"], row["earn_avg"], row["earn_n"] = g["ratio"], g["avg_abs"], g["n"]
        try:
            a = explain(build_prompt(row, row["name"], news, earnings, mode), len(news))
        except Exception as ex:  # keep going; still alert the raw move
            print(f"  explain failed for {sym}: {ex}", file=sys.stderr)
            a = {"catalyst_type": "other", "catalyst_summary": "Explanation unavailable.",
                 "evidence_ids": [], "move_nature": "unclear", "verdict": "unclear",
                 "rationale": f"Model call failed ({type(ex).__name__}: {str(ex)[:160]}); check headlines manually.",
                 "what_would_change_view": "-", "confidence": "low"}
        text = format_alert(row, row["name"], a, news, mode)
        if args.dry_run:
            print("\n" + text + "\n" + "-" * 60)
        else:
            send_telegram(text)
        records.append({
            "run_ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "date": today.isoformat(), "mode": mode, "symbol": sym,
            **{k: row[k] for k in ["price", "ret", "z", "idio_z", "beta", "etf",
                                   "sector_ret", "rvol", "gap", "ret_5d_prior"]},
            **{k: a[k] for k in ["catalyst_type", "move_nature", "verdict",
                                 "confidence", "catalyst_summary"]},
        })

    state = load_equity_state()
    if mode == "eod" and len(movers) and not args.dry_run and state.get("eod_sent") != today.isoformat():
        summary = ["<b>US close — large idiosyncratic moves</b>"]
        for _, r in movers.iterrows():
            summary.append(f"{html.escape(r['symbol'])} {r['ret']*100:+.1f}% "
                           f"(idio {r['idio_z']:+.1f}σ, {html.escape(r['etf'])} {r['sector_ret']*100:+.1f}%)")
        send_telegram("\n".join(summary))
        state["eod_sent"] = today.isoformat()
        EQUITY_STATE_PATH.write_text(json.dumps(state))

    if not args.dry_run:
        append_log(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())

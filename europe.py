"""
UK & European stock alerts (London hours, every 15 minutes on weekdays).

Universe: FTSE 100 + FTSE 250, DAX, MDAX, CAC 40, AEX, IBEX 35, FTSE MIB, SMI, OMX Stockholm 30,
Euro Stoxx 50 (Wikipedia lists, cached daily). Each stock is measured against its home index
(beta-adjusted), scaled by its own residual volatility, with a liquidity floor. Flagged moves get
company headlines (Google News) + macro headlines and a Claude catalyst / fade verdict.

  python europe.py [--dry-run] [--mode auto|live|close]
"""

from __future__ import annotations

import argparse
import html
import io
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import scanner as S

HERE = Path(__file__).resolve().parent
LONDON = ZoneInfo("Europe/London")
UNIVERSE_CACHE = HERE / "cache" / "eu_universe.csv"

# page, default Yahoo suffix, home index
EU_LISTS = [
    ("https://en.wikipedia.org/wiki/FTSE_100_Index", ".L", "^FTSE"),
    ("https://en.wikipedia.org/wiki/FTSE_250_Index", ".L", "^FTMC"),
    ("https://en.wikipedia.org/wiki/DAX", ".DE", "^GDAXI"),
    ("https://en.wikipedia.org/wiki/MDAX", ".DE", "^MDAXI"),
    ("https://en.wikipedia.org/wiki/CAC_40", ".PA", "^FCHI"),
    ("https://en.wikipedia.org/wiki/AEX_index", ".AS", "^AEX"),
    ("https://en.wikipedia.org/wiki/IBEX_35", ".MC", "^IBEX"),
    ("https://en.wikipedia.org/wiki/FTSE_MIB", ".MI", "FTSEMIB.MI"),
    ("https://en.wikipedia.org/wiki/Swiss_Market_Index", ".SW", "^SSMI"),
    ("https://en.wikipedia.org/wiki/OMX_Stockholm_30", ".ST", "^OMX"),
    ("https://en.wikipedia.org/wiki/Euro_Stoxx_50", ".DE", "^STOXX50E"),
]
SUFFIX_INDEX = {".L": "^FTSE", ".DE": "^GDAXI", ".PA": "^FCHI", ".AS": "^AEX", ".MC": "^IBEX",
                ".MI": "FTSEMIB.MI", ".SW": "^SSMI", ".ST": "^OMX", ".BR": "^BFX", ".HE": "^OMXH25",
                ".CO": "^OMXC25", ".IR": "^STOXX50E", ".LS": "^STOXX50E", ".VI": "^ATX", ".OL": "^STOXX50E"}
# rough FX to USD for the liquidity floor only (GBp = pence)
TO_USD = {".L": 1.30 / 100, ".DE": 1.08, ".PA": 1.08, ".AS": 1.08, ".MC": 1.08, ".MI": 1.08,
          ".BR": 1.08, ".HE": 1.08, ".IR": 1.08, ".LS": 1.08, ".VI": 1.08, ".SW": 1.13, ".ST": 0.095, ".CO": 0.145,
          ".OL": 0.093}
MIN_DOLLAR_VOL = 5e6
SESSION = ((8, 15), (16, 30))     # UK time, live scanning
CLOSE_WINDOW = ((16, 30), (17, 0))


def to_yahoo(sym: str, suffix: str) -> str:
    sym = sym.strip().upper().replace(" ", "")
    if re.search(r"\.[A-Z]{1,2}$", sym) and not (suffix == ".L" and re.search(r"\.[A-Z]$", sym)):
        return sym                      # already has an exchange suffix, e.g. ADS.DE
    if suffix == ".L":
        sym = sym.rstrip(".").replace(".", "-")   # BT.A -> BT-A.L
    return sym + suffix


# Every listed company on Europe's main exchanges (FinanceDatabase, free, per-exchange files)
FDB = "https://raw.githubusercontent.com/JerBouma/FinanceDatabase/main/database/equities/{x}.csv"
FDB_EXCHANGES = {"LSE": ".L", "PAR": ".PA", "GER": ".DE", "AMS": ".AS", "BRU": ".BR", "MIL": ".MI",
                 "MCE": ".MC", "EBS": ".SW", "STO": ".ST", "CPH": ".CO", "HEL": ".HE", "OSL": ".OL",
                 "LIS": ".LS", "VIE": ".VI", "ISE": ".IR"}
FDB_CAPS = {"Mega Cap", "Large Cap", "Mid Cap", "Small Cap"}
EUROPE_COUNTRIES = {"United Kingdom", "France", "Germany", "Netherlands", "Belgium", "Italy", "Spain",
                    "Switzerland", "Sweden", "Denmark", "Finland", "Norway", "Portugal", "Austria", "Ireland",
                    "Luxembourg", "Jersey", "Guernsey", "Isle of Man", "Bermuda", "Greece", "Poland", "Iceland",
                    "Czech Republic", "Hungary", "Malta", "Cyprus", "Monaco", "Faroe Islands", "Gibraltar"}
LIQUID_CACHE = HERE / "cache" / "eu_liquid.csv"


def all_european_listings() -> pd.DataFrame:
    rows = []
    for x, suffix in FDB_EXCHANGES.items():
        try:
            df = pd.read_csv(io.StringIO(S._fetch(FDB.format(x=x))), dtype=str, keep_default_na=False,
                             usecols=lambda c: c in ("symbol", "name", "market_cap", "delisted", "country"))
            df = df[df["symbol"].str.endswith(suffix) & df["market_cap"].isin(FDB_CAPS)]
            if "country" in df:   # drop cross-listings of US/Asian companies (e.g. META.VI, 0XXX.L)
                df = df[df["country"].isin(EUROPE_COUNTRIES)]
            df = df[~df["symbol"].str.match(r"^\d")]
            if "delisted" in df:
                df = df[df["delisted"].str.lower() != "true"]
            for _, r in df.iterrows():
                rows.append({"yf_symbol": r["symbol"].strip(), "name": r["name"].strip(),
                             "bench": SUFFIX_INDEX.get(suffix, "^STOXX50E")})
            print(f"  {x}: {len(df)}")
        except Exception as e:
            print(f"  {x} listings failed: {e}", file=sys.stderr)
    return pd.DataFrame(rows)


def load_universe(liquid_only: bool = True) -> pd.DataFrame:
    """Every listed small/mid/large cap on the main European exchanges + index lists, then
    (by default) a daily liquidity screen."""
    if liquid_only:
        try:
            if LIQUID_CACHE.exists() and datetime.now().timestamp() - LIQUID_CACHE.stat().st_mtime < 86400:
                return pd.read_csv(LIQUID_CACHE, keep_default_na=False)
        except Exception:
            pass
        return refresh_liquid()
    try:
        if UNIVERSE_CACHE.exists() and datetime.now().timestamp() - UNIVERSE_CACHE.stat().st_mtime < 86400:
            return pd.read_csv(UNIVERSE_CACHE, keep_default_na=False)
    except Exception:
        pass
    rows = []
    for url, suffix, idx in EU_LISTS:
        try:
            t = S.find_table(S._fetch(url), min_rows=20)
            if t is None:
                print(f"  {url.rsplit('/', 1)[-1]}: no table", file=sys.stderr)
                continue
            for _, r in t.iterrows():
                y = to_yahoo(r["symbol"], suffix)
                suf = "." + y.rsplit(".", 1)[-1]
                rows.append({"yf_symbol": y, "name": r["name"], "bench": idx if suf == suffix else SUFFIX_INDEX.get(suf, "^STOXX50E")})
            print(f"  {url.rsplit('/', 1)[-1]}: {len(t)}")
        except Exception as e:
            print(f"  {url.rsplit('/', 1)[-1]} failed: {e}", file=sys.stderr)
    frames = [pd.DataFrame(rows)] if rows else []
    try:
        listed = all_european_listings()
        if len(listed):
            frames.append(listed)       # index lists first, so their home-index benchmark wins
    except Exception as e:
        print(f"  European listings failed: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError("no European lists loaded")
    df = pd.concat(frames).drop_duplicates("yf_symbol")
    UNIVERSE_CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(UNIVERSE_CACHE, index=False)
    return df


def refresh_liquid() -> pd.DataFrame:
    """Download a year of daily data for every listed European stock (slow, once a day) and keep liquid ones."""
    full = load_universe(liquid_only=False)
    benches = sorted(set(full["bench"]))
    daily = S.cached_daily(list(full["yf_symbol"]) + benches, name="eu", tz=LONDON)
    dv = (daily["close"] * daily["volume"]).iloc[-S.VOL_WINDOW:].mean()
    fx = pd.Series({s: TO_USD.get("." + s.rsplit(".", 1)[-1], 1.0) for s in dv.index})
    dv_usd = dv * fx.reindex(dv.index).fillna(1.0)
    ok = set(dv_usd[dv_usd >= MIN_DOLLAR_VOL].index)
    liq = full[full["yf_symbol"].isin(ok)].reset_index(drop=True)
    LIQUID_CACHE.parent.mkdir(exist_ok=True)
    liq.to_csv(LIQUID_CACHE, index=False)
    print(f"  liquid European universe: {len(liq)} of {len(full)} listed stocks")
    return liq


def eu_base_rates() -> dict:
    import quant

    def loader():
        uni = load_universe()
        benches = sorted(set(uni["bench"]))
        daily = S.cached_daily(list(uni["yf_symbol"]) + benches, name="eu", tz=LONDON)
        # liquidity is already screened; GBp prices make a $ floor misleading here
        return daily["close"], daily["open"], daily["volume"], dict(zip(uni["yf_symbol"], uni["bench"])), 0.0
    return quant.base_rates_cached("eu", loader)


def session(now: datetime) -> str | None:
    if now.weekday() >= 5:
        return None
    hm = (now.hour, now.minute)
    if SESSION[0] <= hm < SESSION[1]:
        return "live"
    if CLOSE_WINDOW[0] <= hm < CLOSE_WINDOW[1]:
        return "close"
    return None


def session_fraction(now: datetime) -> float:
    start = now.replace(hour=8, minute=0, second=0, microsecond=0)
    return float(min(max((now - start).total_seconds() / (8.5 * 3600), 0.05), 1.0))


def is_regular_eu(t: pd.Timestamp) -> bool:
    hm = t.hour * 100 + t.minute
    return 800 <= hm < 1630


def build_frames(syms: list[str]) -> dict[str, pd.DataFrame]:
    today = datetime.now(LONDON).date()
    daily = S.cached_daily(syms, name="eu", tz=LONDON)
    intra = S.download_intraday(syms, tz=LONDON, period="2d")
    if intra["close"].empty:
        return daily
    ic = intra["close"]
    mask = [(t.date() == today) and is_regular_eu(t) for t in ic.index]
    if not any(mask):
        return daily
    out = {}
    rows = {"close": ic[mask].ffill().iloc[-1], "open": intra["open"][mask].bfill().iloc[0],
            "volume": intra["volume"][mask].fillna(0).sum()}
    for k, row in rows.items():
        f = daily[k].copy()
        f.loc[pd.Timestamp(today)] = row.reindex(f.columns)
        out[k] = f.sort_index()
    return out


def company_news(name: str, limit: int = 10) -> list[dict]:
    """Recent headlines about a company via Google News RSS (works for any market)."""
    from news import parse_feed
    url = f"https://news.google.com/rss/search?q={quote_plus(chr(34) + name + chr(34) + ' when:3d')}&hl=en-GB&gl=GB&ceid=GB:en"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=15)
        r.raise_for_status()
        items = sorted(parse_feed(r.content, "Google News"), key=lambda x: x["ts"], reverse=True)[:limit]
    except Exception as e:
        print(f"  company news failed for {name}: {e}", file=sys.stderr)
        items = []
    return [{"headline": it["headline"], "summary": "", "source": it["source"], "url": it["url"],
             "time_et": it["ts"].astimezone(LONDON).strftime("%a %d %b %H:%M UK")} for it in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "live", "close"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="rebuild universe + daily history cache only")
    args = ap.parse_args()
    lock = HERE / "cache" / "eu_refresh.lock"
    if args.refresh:
        lock.parent.mkdir(exist_ok=True)
        lock.write_text("x")
        try:
            UNIVERSE_CACHE.unlink(missing_ok=True)
            LIQUID_CACHE.unlink(missing_ok=True)
            uni = load_universe()
            S.cached_daily(list(uni["yf_symbol"]) + sorted(set(uni["bench"])), name="eu", tz=LONDON)
            eu_base_rates()
            print(f"European universe: {len(uni)} stocks")
        finally:
            lock.unlink(missing_ok=True)
        return 0
    if lock.exists() and datetime.now().timestamp() - lock.stat().st_mtime < 2400:
        print("European refresh in progress — skipping this scan.")
        return 0

    now = datetime.now(LONDON)
    mode = args.mode if args.mode != "auto" else session(now)
    if mode is None:
        print(f"Outside European session ({now:%a %H:%M} UK).")
        return 0
    today = now.date()
    uni = load_universe()
    benches = sorted(set(uni["bench"]))
    syms = list(uni["yf_symbol"])
    px = build_frames(syms + benches)
    if px["close"].index[-1].date() != today:
        print("No bar for today yet (holiday or data delay).")
        return 0

    vol_scale = session_fraction(now) if mode == "live" else 1.0
    have = set(c for c in px["close"].columns if px["close"][c].notna().any())
    fallback = "^STOXX50E" if "^STOXX50E" in have else None
    bench = {s: (b if b in have else fallback) for s, b in zip(uni["yf_symbol"], uni["bench"])}
    bench = {s: b for s, b in bench.items() if b}
    m = S.compute_metrics(px["close"], px["open"], px["volume"], bench, vol_scale)
    if m.empty:
        return 0
    m = m.merge(uni[["yf_symbol", "name"]], on="yf_symbol")
    m["adv_usd"] = m.apply(lambda r: r["adv_usd"] * TO_USD.get("." + r["yf_symbol"].rsplit(".", 1)[-1], 1.0), axis=1)
    m["symbol"] = m["yf_symbol"]
    saved = S.MIN_DOLLAR_VOL
    S.MIN_DOLLAR_VOL = MIN_DOLLAR_VOL
    movers = S.select_movers(m)
    S.MIN_DOLLAR_VOL = saved

    log = S.load_log()
    if len(log):
        same = (log["date"] == today.isoformat()) & (log["mode"] == "eu")
        already = set(log.loc[same, "symbol"])
    else:
        already = set()
    new = movers[~movers["symbol"].isin(already)]
    print(f"Flagged {len(movers)} ({len(new)} new).")

    from macro import get_headlines
    macro_heads = None
    records = []
    base = eu_base_rates() if len(new) else {}
    for _, row in new.iterrows():
        news = company_news(str(row["name"]))
        plan = None
        try:
            import quant
            plan = quant.equity_plan(row, px["close"], base)
        except Exception as ex:
            print(f"  quant plan failed for {row['symbol']}: {ex}", file=sys.stderr)
        if macro_heads is None:
            macro_heads = [{"headline": h["headline"], "summary": "", "source": h["source"], "url": h["url"],
                            "time_et": h["ts"].astimezone(LONDON).strftime("%a %d %b %H:%M UK")}
                           for h in get_headlines(hours=8, limit=15)]
        allnews = news + macro_heads
        try:
            a = S.explain(S.build_prompt(row, row["name"], allnews, [], "intraday" if mode == "live" else "eod", plan), len(allnews))
        except Exception as ex:
            print(f"  explain failed: {ex}", file=sys.stderr)
            a = {"catalyst_type": "other", "catalyst_summary": "Explanation unavailable.", "evidence_ids": [],
                 "move_nature": "unclear", "verdict": "unclear", "rationale": f"Model call failed ({type(ex).__name__}: {str(ex)[:160]}).",
                 "what_would_change_view": "-", "confidence": "low"}
        text = S.format_alert(row, row["name"], a, allnews, "intraday" if mode == "live" else "eod", plan)
        text = text.replace(" [live]", " [Europe · live]").replace(" [close]", " [Europe · close]")
        print(text) if args.dry_run else S.send_telegram(text)
        records.append({
            "run_ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "date": today.isoformat(),
            "mode": "eu", "symbol": row["symbol"],
            **{k: row[k] for k in ["price", "ret", "z", "idio_z", "beta", "etf", "sector_ret", "rvol", "gap", "ret_5d_prior"]},
            **{k: a[k] for k in ["catalyst_type", "move_nature", "verdict", "confidence", "catalyst_summary"]},
            **S.plan_fields(plan),
        })
    if not args.dry_run:
        S.append_log(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import os
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
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 20e6     # 20d avg $ volume
MAX_ALERTS_PER_RUN = 10
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
def load_universe() -> pd.DataFrame:
    """S&P 500 constituents with GICS sector. Columns: symbol, yf_symbol, name, sector, etf."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=30)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    df = df.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"})
    df = df[["symbol", "name", "sector"]].copy()
    df["symbol"] = df["symbol"].str.strip()
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
def download_prices(yf_symbols: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        yf_symbols, period="1y", interval="1d", auto_adjust=True,
        group_by="column", threads=True, progress=False,
    )
    out = {}
    for field in ("Open", "Close", "Volume"):
        f = raw[field]
        if isinstance(f, pd.Series):
            f = f.to_frame(yf_symbols[0])
        out[field.lower()] = f.sort_index()
    return out


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
    logr = np.log(close).diff()
    rows = []
    for sym, etf in stock_to_etf.items():
        if sym not in close.columns or etf not in close.columns:
            continue
        s = logr[sym]
        b = logr[etf]
        c = close[sym]
        if c.iloc[-BETA_WINDOW - 2:].isna().any():
            continue

        r_today, b_today = s.iloc[-1], b.iloc[-1]
        hist_s = s.iloc[-BETA_WINDOW - 1:-1]
        hist_b = b.iloc[-BETA_WINDOW - 1:-1]
        var_b = hist_b.var()
        beta = hist_s.cov(hist_b) / var_b if var_b > 0 else 1.0

        resid_hist = (hist_s - beta * hist_b).iloc[-RESID_WINDOW:]
        resid_sd = resid_hist.std()
        vol20 = hist_s.iloc[-VOL_WINDOW:].std()
        idio = r_today - beta * b_today

        vol_hist = volume[sym].iloc[-VOL_WINDOW - 1:-1]
        avg_vol = vol_hist.mean()
        adv_usd = (close[sym].iloc[-VOL_WINDOW - 1:-1] * vol_hist).mean()
        rvol = (volume[sym].iloc[-1] / vol_scale) / avg_vol if avg_vol > 0 else np.nan

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
            "gap": open_[sym].iloc[-1] / c.iloc[-2] - 1,
            "rvol": rvol,
            "adv_usd": adv_usd,
            "ret_5d_prior": c.iloc[-2] / c.iloc[-7] - 1,
            "dist_52w_high": c.iloc[-1] / win.max() - 1,
            "dist_52w_low": c.iloc[-1] / win.min() - 1,
        })
    return pd.DataFrame(rows)


def select_movers(m: pd.DataFrame) -> pd.DataFrame:
    f = m[
        (m["idio_z"].abs() >= MIN_IDIO_Z)
        & (m["ret"].abs() >= MIN_ABS_RET)
        & (m["price"] >= MIN_PRICE)
        & (m["adv_usd"] >= MIN_DOLLAR_VOL)
    ].copy()
    f["abs_idio_z"] = f["idio_z"].abs()
    return f.sort_values("abs_idio_z", ascending=False).head(MAX_ALERTS_PER_RUN)


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


def build_prompt(row: pd.Series, name: str, news: list[dict], earnings: list[dict], mode: str) -> str:
    stats = {
        "ticker": row["symbol"], "company": name, "sector_etf": row["etf"],
        "data_as_of": "live intraday" if mode == "intraday" else "official close",
        "move_pct": round(row["ret"] * 100, 2),
        "move_in_own_vol_sigmas": round(row["z"], 2),
        "beta_to_sector": round(row["beta"], 2),
        "sector_etf_move_pct": round(row["sector_ret"] * 100, 2),
        "idiosyncratic_move_pct": round(row["idio_ret"] * 100, 2),
        "idiosyncratic_sigmas": round(row["idio_z"], 2),
        "gap_at_open_pct": round(row["gap"] * 100, 2),
        "relative_volume_x": round(row["rvol"], 2),
        "prior_5d_move_pct": round(row["ret_5d_prior"] * 100, 2),
        "pct_from_52w_high": round(row["dist_52w_high"] * 100, 1),
        "pct_from_52w_low": round(row["dist_52w_low"] * 100, 1),
    }
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


def format_alert(row: pd.Series, name: str, a: dict, news: list[dict], mode: str) -> str:
    e = html.escape
    arrow = "🔺" if row["ret"] > 0 else "🔻"
    tag = "live" if mode == "intraday" else "close"
    lines = [
        f"{arrow} <b>{e(row['symbol'])}</b> {row['ret']*100:+.1f}%  "
        f"(idio {row['idio_z']:+.1f}σ) · {e(name)} [{tag}]",
        f"{e(row['etf'])} {row['sector_ret']*100:+.1f}% · β {row['beta']:.2f} · RVOL {row['rvol']:.1f}x · "
        f"gap {row['gap']*100:+.1f}% · prior 5d {row['ret_5d_prior']*100:+.1f}%",
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
    return text[:4000]


def send_telegram(text: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured; printing instead:\n" + text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=20,
    )
    if not r.ok:
        print(f"Telegram error {r.status_code}: {r.text}", file=sys.stderr)


# --------------------------------------------------------------------------
# Log / dedupe
# --------------------------------------------------------------------------
LOG_COLS = ["run_ts_utc", "date", "mode", "symbol", "price", "ret", "z", "idio_z", "beta",
            "etf", "sector_ret", "rvol", "gap", "ret_5d_prior", "catalyst_type", "move_nature",
            "verdict", "confidence", "catalyst_summary"]


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
    ap.add_argument("--mode", choices=["auto", "intraday", "eod"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tickers", help="comma-separated subset (S&P 500 symbols)")
    ap.add_argument("--test-telegram", action="store_true")
    args = ap.parse_args()

    if args.test_telegram:
        send_telegram("✅ move-alerts: Telegram is wired up.")
        return 0

    now_et = datetime.now(ET)
    today = now_et.date()
    if now_et.weekday() >= 5:
        print("Weekend — nothing to do.")
        return 0
    mode = args.mode
    if mode == "auto":
        mode = "eod" if now_et.hour >= 16 else "intraday"
    if mode == "intraday" and (now_et.hour, now_et.minute) < (9, 45):
        print("Too early in the session.")
        return 0

    uni = load_universe()
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        uni = uni[uni["symbol"].isin(wanted)]
    print(f"Universe: {len(uni)} names, mode={mode}")

    etfs = sorted(set(uni["etf"]) | {MARKET_ETF})
    px = download_prices(list(uni["yf_symbol"]) + etfs)
    last_bar = px["close"].index[-1].date()
    if last_bar != today:
        print(f"No bar for {today} (last {last_bar}) — market holiday or data delay.")
        return 0

    vol_scale = session_fraction(now_et) if mode == "intraday" else 1.0
    m = compute_metrics(px["close"], px["open"], px["volume"],
                        dict(zip(uni["yf_symbol"], uni["etf"])), vol_scale)
    m = m.merge(uni[["yf_symbol", "symbol", "name"]], on="yf_symbol")
    movers = select_movers(m)

    log = load_log()
    already = set(log.loc[log["date"] == today.isoformat(), "symbol"]) if len(log) else set()
    new_movers = movers[~movers["symbol"].isin(already)]
    print(f"Flagged {len(movers)} ({len(new_movers)} new).")

    records = []
    for _, row in new_movers.iterrows():
        sym = row["symbol"]
        news = get_news(sym, today)
        earnings = get_earnings(sym, today)
        try:
            a = explain(build_prompt(row, row["name"], news, earnings, mode), len(news))
        except Exception as ex:  # keep going; still alert the raw move
            print(f"  explain failed for {sym}: {ex}", file=sys.stderr)
            a = {"catalyst_type": "other", "catalyst_summary": "Explanation unavailable.",
                 "evidence_ids": [], "move_nature": "unclear", "verdict": "unclear",
                 "rationale": "Model call failed; check headlines manually.",
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

    if mode == "eod" and len(movers) and not args.dry_run:
        summary = ["<b>US close — large idiosyncratic moves</b>"]
        for _, r in movers.iterrows():
            summary.append(f"{html.escape(r['symbol'])} {r['ret']*100:+.1f}% "
                           f"(idio {r['idio_z']:+.1f}σ, {html.escape(r['etf'])} {r['sector_ret']*100:+.1f}%)")
        send_telegram("\n".join(summary))

    if not args.dry_run:
        append_log(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())

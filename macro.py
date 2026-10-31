"""
24/7 global macro move alerts: USD/FX, precious metals, bitcoin, US index
futures (covering pre-market/after-hours/overnight), global indices, rates,
energy and copper.

Every run (hourly by default) checks 15-minute bars for big moves over the
last 1h and 4h, scaled by each instrument's own recent volatility. A flagged
move gets recent headlines (FinancialJuice + Finnhub), a Claude explanation,
and a Telegram alert. Weekends: FX and CME metals are shut, so it watches
BTC and PAXG (a gold-backed token that trades 24/7 and tracks spot gold).

Usage:
    python macro.py              # normal run
    python macro.py --dry-run    # print, don't send or save state
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import xml.etree.ElementTree as ET_xml
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scanner import CLAUDE_MODEL, _finnhub, send_telegram

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "macro_state.json"
LONDON = __import__("zoneinfo").ZoneInfo("Europe/London")

# name, source, symbol, minimum % move to alert (as fraction)
INSTRUMENTS = [
    # Crypto (24/7)
    ("Bitcoin", "kraken", "XBTUSD", 0.025),
    # Precious metals (CME, ~23/5)
    ("Gold", "yf", "GC=F", 0.008),
    ("Silver", "yf", "SI=F", 0.015),
    ("Platinum", "yf", "PL=F", 0.015),
    ("Palladium", "yf", "PA=F", 0.020),
    # USD / FX (24/5)
    ("DXY (USD index)", "yf", "DX-Y.NYB", 0.003),
    ("EURUSD", "yf", "EURUSD=X", 0.003),
    ("GBPUSD", "yf", "GBPUSD=X", 0.003),
    ("USDJPY", "yf", "JPY=X", 0.004),
    ("USDCHF", "yf", "CHF=X", 0.003),
    ("AUDUSD", "yf", "AUDUSD=X", 0.004),
    ("USDCAD", "yf", "CAD=X", 0.003),
    ("USDCNH", "yf", "CNH=X", 0.002),
    ("USDMXN", "yf", "MXN=X", 0.006),
    # US equity index futures (~23/5) — cover pre-market, after-hours and overnight
    ("S&P 500 futures", "yf", "ES=F", 0.006),
    ("Nasdaq 100 futures", "yf", "NQ=F", 0.008),
    ("Russell 2000 futures", "yf", "RTY=F", 0.010),
    ("Dow futures", "yf", "YM=F", 0.006),
    ("Nikkei futures (CME)", "yf", "NIY=F", 0.010),
    # Global cash indices (their own local hours)
    ("FTSE 100", "yf", "^FTSE", 0.007),
    ("DAX", "yf", "^GDAXI", 0.008),
    ("Euro Stoxx 50", "yf", "^STOXX50E", 0.008),
    ("Hang Seng", "yf", "^HSI", 0.012),
    ("Shanghai Composite", "yf", "000001.SS", 0.010),
    ("VIX", "yf", "^VIX", 0.080),
    # Rates (futures ~23/5; yield index = cash hours)
    ("US 10y note futures", "yf", "ZN=F", 0.003),
    ("US 10y yield", "yf", "^TNX", 0.015),
    # Energy / industrial metals
    ("WTI crude", "yf", "CL=F", 0.015),
    ("Brent crude", "yf", "BZ=F", 0.015),
    ("Natural gas", "yf", "NG=F", 0.030),
    ("Copper", "yf", "HG=F", 0.010),
]
# Weekend / out-of-hours gold proxy: only used when gold futures are closed.
PAXG = ("Gold (PAXG token, 24/7 proxy)", "kraken", "PAXGUSD", 0.008)

Z_THRESHOLD = 3.0          # move vs own vol, in sigmas
WINDOWS = {"1h": 4, "4h": 16}  # in 15-minute bars
STALE_MINUTES = 45         # no fresh bar for this long => market closed
COOLDOWN_HOURS = 6         # don't repeat an alert unless the move extends
HEADLINE_HOURS = 6

FJ_RSS_URL = os.environ.get("FJ_RSS_URL", "https://www.financialjuice.com/feed.ashx?xy=rss")


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def kraken_closes(pair: str, interval: int = 15) -> pd.Series:
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": pair, "interval": interval}, timeout=20)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(js["error"])
    key = next(k for k in js["result"] if k != "last")
    rows = js["result"][key]
    idx = pd.to_datetime([int(x[0]) for x in rows], unit="s", utc=True)
    return pd.Series([float(x[4]) for x in rows], index=idx).sort_index()


def yf_closes(symbols: list[str]) -> dict[str, pd.Series]:
    import yfinance as yf

    raw = yf.download(symbols, period="5d", interval="15m", progress=False, auto_adjust=False)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    out = {}
    for s in symbols:
        if s in close.columns:
            ser = close[s].dropna()
            if ser.index.tz is None:
                ser.index = ser.index.tz_localize("UTC")
            out[s] = ser.tz_convert("UTC")
    return out


def load_all_series() -> dict[str, pd.Series]:
    yf_syms = [i[2] for i in INSTRUMENTS if i[1] == "yf"]
    series: dict[str, pd.Series] = {}
    try:
        series.update(yf_closes(yf_syms))
    except Exception as e:
        print(f"yfinance failed: {e}", file=sys.stderr)
    for name, src, sym, _ in INSTRUMENTS + [PAXG]:
        if src == "kraken":
            try:
                series[sym] = kraken_closes(sym)
            except Exception as e:
                print(f"kraken {sym} failed: {e}", file=sys.stderr)
    return series


# --------------------------------------------------------------------------
# Signal (pure — tested offline)
# --------------------------------------------------------------------------
def is_fresh(s: pd.Series, now: datetime) -> bool:
    return len(s) > 0 and (now - s.index[-1].to_pydatetime()) <= timedelta(minutes=STALE_MINUTES)


def move_stats(s: pd.Series) -> dict | None:
    """Largest vol-scaled move over the 1h/4h windows ending at the last bar."""
    if len(s) < 120:
        return None
    lr = np.log(s).diff().dropna()
    hist = lr.iloc[-400:-max(WINDOWS.values())]
    hist = hist[hist != 0]
    vol = hist.std()
    if not vol or math.isnan(vol):
        return None
    best = None
    for label, n in WINDOWS.items():
        ret = math.log(s.iloc[-1] / s.iloc[-1 - n])
        z = ret / (vol * math.sqrt(n))
        if best is None or abs(z) > abs(best["z"]):
            best = {"window": label, "ret": math.expm1(ret), "z": z}
    day_ago = s.index[-1] - pd.Timedelta(hours=24)
    prior = s[s.index <= day_ago]
    best["chg_24h"] = s.iloc[-1] / prior.iloc[-1] - 1 if len(prior) else float("nan")
    best["price"] = float(s.iloc[-1])
    best["vol_15m"] = float(vol)
    return best


def should_alert(name: str, st: dict, min_abs: float, state: dict, now: datetime) -> bool:
    if abs(st["z"]) < Z_THRESHOLD or abs(st["ret"]) < min_abs:
        return False
    prev = state.get(name)
    if not prev:
        return True
    age_h = (now - datetime.fromisoformat(prev["ts"])).total_seconds() / 3600
    if age_h >= COOLDOWN_HOURS:
        return True
    # within cooldown: only re-alert if the move has extended by another min_abs in the same direction
    same_dir = math.copysign(1, st["ret"]) == prev["dir"]
    extension = (st["price"] / prev["price"] - 1) * prev["dir"]
    return same_dir and extension >= min_abs


# --------------------------------------------------------------------------
# Headlines
# --------------------------------------------------------------------------
def financialjuice_headlines(since: datetime) -> list[dict]:
    try:
        r = requests.get(FJ_RSS_URL, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=20)
        r.raise_for_status()
        root = ET_xml.fromstring(r.content)
    except Exception as e:
        print(f"FinancialJuice RSS failed: {e}", file=sys.stderr)
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        title = title.removeprefix("FinancialJuice:").strip()
        try:
            ts = parsedate_to_datetime(item.findtext("pubDate") or "")
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if title and ts >= since:
            out.append({"ts": ts, "source": "FinancialJuice", "headline": title,
                        "url": (item.findtext("link") or "").strip()})
    return out


def finnhub_headlines(since: datetime, categories=("general", "forex", "crypto")) -> list[dict]:
    out = []
    for cat in categories:
        for it in _finnhub("news", {"category": cat}) or []:
            ts = datetime.fromtimestamp(it.get("datetime", 0), timezone.utc)
            if ts >= since and it.get("headline"):
                out.append({"ts": ts, "source": it.get("source", "Finnhub"),
                            "headline": it["headline"].strip(), "url": it.get("url", "")})
    return out


def get_headlines(hours: float, limit: int = 60) -> list[dict]:
    """Recent headlines from all news sources (see news.py FEEDS), deduped, newest first."""
    from news import fetch_all

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        items = fetch_all(since)
    except Exception as e:
        print(f"news fetch failed: {e}", file=sys.stderr)
        items = financialjuice_headlines(since) + finnhub_headlines(since)
    out = []
    for it in items[:limit]:
        srcs = it.get("sources") or [it["source"]]
        out.append({**it, "source": " / ".join(srcs[:3])})
    return out


def headlines_block(items: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {it['ts'].astimezone(LONDON):%a %H:%M} UK | {it['source']} | {it['headline']}"
        for i, it in enumerate(items, 1)
    ) or "(no headlines returned)"


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a macro strategist supporting a discretionary trader who fades \
large moves across global macro: USD and FX, precious metals, bitcoin, US and global equity \
indices and futures, rates, energy and copper. You get move statistics, a snapshot of \
related markets, and numbered headlines.

Rules:
- Use ONLY the supplied evidence. Cite headlines by number. Never invent data or events.
- If nothing explains the move, say catalyst_type = "no_identifiable_catalyst".
- Cross-check: does the move fit what related markets did (e.g. gold vs USD and yields, \
silver vs gold, BTC vs risk)? A move that is NOT confirmed elsewhere is more likely flow.
- INFORMATION moves (data surprises, central-bank shifts, policy/tariff/sanction news, \
geopolitical escalation) tend to persist; FLOW moves (stop runs, thin weekend/Asia liquidity, \
options/expiry, liquidations, no news) are better fade candidates.
- Be decisive but calibrated; "unclear" is allowed.
"""

TOOL = {
    "name": "record_assessment",
    "description": "Record the assessment of the move.",
    "input_schema": {
        "type": "object",
        "properties": {
            "catalyst_type": {"type": "string", "enum": [
                "economic_data", "central_bank", "geopolitics", "policy_trade_sanctions",
                "flows_positioning", "liquidations", "crypto_specific", "cross_asset_spillover",
                "no_identifiable_catalyst", "other"]},
            "catalyst_summary": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "integer"}},
            "cross_asset_confirmation": {"type": "string", "description": "One sentence."},
            "move_nature": {"type": "string", "enum": ["information", "flow", "mixed", "unclear"]},
            "verdict": {"type": "string", "enum": ["fade_candidate", "do_not_fade", "unclear"]},
            "rationale": {"type": "string"},
            "what_would_change_view": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["catalyst_type", "catalyst_summary", "evidence_ids", "cross_asset_confirmation",
                     "move_nature", "verdict", "rationale", "what_would_change_view", "confidence"],
    },
}


def explain(name: str, st: dict, snapshot: dict, heads: list[dict]) -> dict:
    import anthropic

    prompt = (
        f"FLAGGED MOVE\n{json.dumps({'instrument': name, 'window': st['window'], 'move_pct': round(st['ret']*100, 2), 'sigmas': round(st['z'], 1), 'change_24h_pct': round(st['chg_24h']*100, 2), 'price': st['price']}, indent=1)}\n\n"
        f"RELATED MARKETS (24h % change, latest)\n{json.dumps(snapshot, indent=1)}\n\n"
        f"HEADLINES (last {HEADLINE_HOURS}h, newest first)\n{headlines_block(heads)}\n\n"
        "Assess this move and call record_assessment."
    )
    msg = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL, max_tokens=1200, system=SYSTEM_PROMPT, tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_assessment"},
        messages=[{"role": "user", "content": prompt}],
    )
    for b in msg.content:
        if b.type == "tool_use":
            a = dict(b.input)
            a["evidence_ids"] = [i for i in a.get("evidence_ids", [])
                                 if isinstance(i, int) and 1 <= i <= len(heads)]
            return a
    raise RuntimeError("no assessment")


def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.0f}"
    if p >= 10:
        return f"{p:,.2f}"
    return f"{p:.4f}"


VERDICT = {"fade_candidate": "FADE CANDIDATE", "do_not_fade": "DON'T FADE", "unclear": "UNCLEAR"}


def format_alert(name: str, st: dict, a: dict, heads: list[dict], weekend: bool) -> str:
    e = html.escape
    arrow = "🔺" if st["ret"] > 0 else "🔻"
    lines = [
        f"{arrow} <b>{e(name)}</b> {st['ret']*100:+.2f}% in {st['window']} ({st['z']:+.1f}σ) · "
        f"24h {st['chg_24h']*100:+.2f}% · last {fmt_price(st['price'])}" + (" · weekend" if weekend else ""),
        "",
        f"<b>Catalyst:</b> {e(a['catalyst_type'].replace('_', ' '))} — {e(a['catalyst_summary'])}",
        f"<b>Cross-asset:</b> {e(a['cross_asset_confirmation'])}",
        f"<b>Nature:</b> {e(a['move_nature'])} → <b>{VERDICT.get(a['verdict'], a['verdict'])}</b> ({e(a['confidence'])})",
        f"<b>Why:</b> {e(a['rationale'])}",
        f"<b>Changes view:</b> {e(a['what_would_change_view'])}",
    ]
    if a["evidence_ids"]:
        lines.append("")
        for i in a["evidence_ids"][:4]:
            h = heads[i - 1]
            label = e(h["source"])
            if h["url"]:
                label = f'<a href="{e(h["url"], quote=True)}">{label}</a>'
            lines.append(f"[{i}] {label}: {e(h['headline'][:160])}")
    return "\n".join(lines)[:4000]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    weekend = now.astimezone(LONDON).weekday() >= 5
    series = load_all_series()

    instruments = list(INSTRUMENTS)
    gold = series.get("GC=F")
    if gold is None or not is_fresh(gold, now):
        instruments.append(PAXG)   # gold futures shut -> watch the 24/7 proxy

    stats = {}
    for name, _, sym, min_abs in instruments:
        s = series.get(sym)
        if s is None or not is_fresh(s, now):
            continue
        st = move_stats(s)
        if st:
            stats[name] = (st, min_abs)
    print(f"Live instruments: {', '.join(stats) or 'none'}")

    snapshot = {n: {"chg_24h_pct": round(st["chg_24h"] * 100, 2), "last": round(st["price"], 4)}
                for n, (st, _) in stats.items()}
    state = load_state()
    heads = None
    sent = False
    for name, (st, min_abs) in stats.items():
        if not should_alert(name, st, min_abs, state, now):
            continue
        if heads is None:
            heads = get_headlines(HEADLINE_HOURS)
        try:
            a = explain(name, st, snapshot, heads)
        except Exception as ex:
            print(f"explain failed: {ex}", file=sys.stderr)
            a = {"catalyst_type": "other", "catalyst_summary": "Explanation unavailable.",
                 "evidence_ids": [], "cross_asset_confirmation": "-", "move_nature": "unclear",
                 "verdict": "unclear", "rationale": "Model call failed; check headlines.",
                 "what_would_change_view": "-", "confidence": "low"}
        text = format_alert(name, st, a, heads, weekend)
        if args.dry_run:
            print(text + "\n" + "-" * 60)
        else:
            send_telegram(text)
            state[name] = {"ts": now.isoformat(timespec="seconds"), "price": st["price"],
                           "dir": math.copysign(1, st["ret"])}
            sent = True

    if sent:
        STATE_PATH.write_text(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

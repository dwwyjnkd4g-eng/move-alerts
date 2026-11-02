"""
Your portfolio: positions, P&L, PA holding-period dates, and tighter move alerts.

Data lives in holdings.json (kept OFF GitHub — see .gitignore). Manage it from
Telegram with /add, /remove, /holdings, /holdperiod (see bot.py), or here:

    python holdings.py --seed "AAPL,MSFT,NVDA"   # add tickers with no qty/cost yet
    python holdings.py --watch                    # check holdings for big moves (run every 15 min)
    python holdings.py --show                     # print positions, P&L, earliest sale dates
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HOLDINGS_PATH = HERE / "holdings.json"
WATCH_STATE = HERE / "holdings_state.json"
LONDON = ZoneInfo("Europe/London")
ET = ZoneInfo("America/New_York")

DEFAULT_HOLD_DAYS = 30
STOCK_MIN_MOVE, ETF_MIN_MOVE = 0.025, 0.015   # tighter than the market-wide scanner
MIN_Z = 2.0


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
def load() -> dict:
    try:
        d = json.loads(HOLDINGS_PATH.read_text())
    except Exception:
        d = {}
    d.setdefault("positions", [])
    d.setdefault("hold_days", DEFAULT_HOLD_DAYS)
    return d


def save(d: dict) -> None:
    HOLDINGS_PATH.write_text(json.dumps(d, indent=1))


def find(d: dict, ticker: str) -> dict | None:
    t = ticker.upper()
    return next((p for p in d["positions"] if p["ticker"].upper() == t), None)


def add(ticker: str, qty: float | None = None, price: float | None = None,
        buy_date: str | None = None, name: str | None = None) -> str:
    """Add or top up a position. Topping up averages the cost and resets the PA clock."""
    d = load()
    t = ticker.upper()
    p = find(d, t)
    bd = buy_date or date.today().isoformat()
    if p is None:
        d["positions"].append({"ticker": t, "name": name or t, "qty": qty, "cost": price, "buy_date": bd})
        msg = f"Added {t}"
    else:
        if qty and price and p.get("qty") and p.get("cost"):
            new_qty = p["qty"] + qty
            p["cost"] = (p["qty"] * p["cost"] + qty * price) / new_qty
            p["qty"] = new_qty
        elif qty is not None:
            p["qty"], p["cost"] = qty, price
        p["buy_date"] = bd
        msg = f"Updated {t}"
    save(d)
    return msg


def remove(ticker: str) -> str:
    d = load()
    before = len(d["positions"])
    d["positions"] = [p for p in d["positions"] if p["ticker"].upper() != ticker.upper()]
    save(d)
    return f"Removed {ticker.upper()}" if len(d["positions"]) < before else f"{ticker.upper()} not found"


def set_hold_days(days: int) -> str:
    d = load()
    d["hold_days"] = int(days)
    save(d)
    return f"PA holding period set to {days} days"


def tickers() -> list[str]:
    return [p["ticker"] for p in load()["positions"]]


def is_etf(ticker: str) -> bool:
    return ticker.endswith(".L")   # your ETFs are London-listed; US tickers are single stocks


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def _currency(ticker: str) -> str:
    try:
        import yfinance as yf
        return yf.Ticker(ticker).fast_info.get("currency") or "USD"
    except Exception:
        return "USD"


def snapshot(tks: list[str]) -> dict[str, dict]:
    """Latest price (incl. pre/after-hours for US), previous session close, daily vol, currency."""
    import yfinance as yf

    if not tks:
        return {}
    daily = yf.download(tks, period="1y", interval="1d", auto_adjust=False, progress=False)["Close"]
    intra = yf.download(tks, period="5d", interval="15m", prepost=True, auto_adjust=False, progress=False)["Close"]
    if isinstance(daily, pd.Series):
        daily = daily.to_frame(tks[0])
    if isinstance(intra, pd.Series):
        intra = intra.to_frame(tks[0])
    out = {}
    for t in tks:
        try:
            d = daily[t].dropna()
            i = intra[t].dropna() if t in intra else pd.Series(dtype=float)
            last_ts = i.index[-1] if len(i) else d.index[-1]
            last = float(i.iloc[-1]) if len(i) else float(d.iloc[-1])
            tz = LONDON if t.endswith(".L") else ET
            last_ts = last_ts.tz_convert(tz) if last_ts.tzinfo else last_ts.tz_localize(tz)
            last_day = last_ts.date()
            prev = d[d.index.date < last_day]
            prev_close = float(prev.iloc[-1]) if len(prev) else float(d.iloc[-2])
            vol = float(np.log(d).diff().iloc[-21:-1].std())
            cur = _currency(t)
            scale = 0.01 if cur == "GBp" else 1.0
            out[t] = {
                "last": last * scale, "prev_close": prev_close * scale, "vol": vol,
                "currency": "GBP" if cur in ("GBp", "GBX") else cur,
                "last_ts": last_ts, "move": last / prev_close - 1,
                "week_ago": float(d.iloc[-6]) * scale if len(d) > 6 else None,
            }
        except Exception as e:
            print(f"  price failed for {t}: {e}", file=sys.stderr)
    return out


def gbp_per(currency: str) -> float:
    if currency == "GBP":
        return 1.0
    try:
        import yfinance as yf
        fx = yf.download(f"GBP{currency}=X", period="5d", interval="1d", progress=False)["Close"].dropna()
        return 1.0 / float(fx.iloc[-1].squeeze())
    except Exception:
        return 1.0 / 1.30 if currency == "USD" else 1.0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def earliest_sale(p: dict, hold_days: int) -> date | None:
    try:
        return date.fromisoformat(p["buy_date"]) + timedelta(days=hold_days)
    except Exception:
        return None


def summary_lines(snap: dict | None = None, today: date | None = None, weekly: bool = False) -> list[str]:
    d = load()
    if not d["positions"]:
        return ["No holdings saved. Add with /add TICKER QTY @ PRICE"]
    today = today or datetime.now(LONDON).date()
    snap = snap if snap is not None else snapshot([p["ticker"] for p in d["positions"]])
    fx_cache: dict[str, float] = {}
    rows, tot_val, tot_pnl, tot_day = [], 0.0, 0.0, 0.0
    for p in d["positions"]:
        s = snap.get(p["ticker"])
        if not s:
            rows.append(f"{p['ticker']:<8} n/a")
            continue
        ref = s["week_ago"] if weekly and s.get("week_ago") else s["prev_close"]
        chg = s["last"] / ref - 1
        line = f"{p['ticker']:<8}{s['last']:>10,.2f} {chg*100:+6.2f}%"
        if p.get("qty"):
            fx = fx_cache.setdefault(s["currency"], gbp_per(s["currency"]))
            val = p["qty"] * s["last"] * fx
            tot_val += val
            tot_day += p["qty"] * (s["last"] - ref) * fx
            if p.get("cost"):
                pnl = p["qty"] * (s["last"] - p["cost"]) * fx
                tot_pnl += pnl
                line += f"  £{val:,.0f} ({(s['last']/p['cost']-1)*100:+.1f}%)"
            else:
                line += f"  £{val:,.0f}"
        rows.append(line)
    label = "week" if weekly else "day"
    if tot_val:
        rows.append(f"{'Total':<8}£{tot_val:,.0f} · {label} {'+' if tot_day >= 0 else '-'}£{abs(tot_day):,.0f}"
                    + (f" · open P&L {'+' if tot_pnl >= 0 else '-'}£{abs(tot_pnl):,.0f}" if tot_pnl else ""))
    return rows


def pa_lines(today: date | None = None) -> list[str]:
    d = load()
    today = today or datetime.now(LONDON).date()
    out = []
    for p in d["positions"]:
        es = earliest_sale(p, d["hold_days"])
        if es and es > today:
            out.append(f"{p['ticker']}: earliest sale {es:%d %b} ({(es - today).days}d)")
    return out


def upcoming_earnings(days: int = 7) -> list[str]:
    from scanner import _finnhub
    today = datetime.now(LONDON).date()
    out = []
    for t in tickers():
        if t.endswith(".L"):
            continue
        data = _finnhub("calendar/earnings", {"symbol": t, "from": today.isoformat(),
                                              "to": (today + timedelta(days=days)).isoformat()}) or {}
        for e in (data.get("earningsCalendar", []) if isinstance(data, dict) else []):
            when = {"bmo": "pre-mkt", "amc": "after close"}.get(e.get("hour"), "")
            out.append(f"{t}: {e.get('date')} {when}".strip())
    return out


# --------------------------------------------------------------------------
# Watcher: tighter move alerts for things you own
# --------------------------------------------------------------------------
HOLDING_PROMPT = """A trader holds this position and it is moving sharply. Using ONLY the data \
and numbered headlines supplied, explain the likely catalyst, whether it is stock-specific or \
market/sector-driven, and whether the move looks like information (likely to persist) or flow \
(likely to revert). Cite headlines by number. Be concise and calibrated; do not invent facts. \
Do not give personal financial advice; frame it as an analysis of the move."""

HOLDING_TOOL = {
    "name": "record",
    "description": "Record the assessment.",
    "input_schema": {"type": "object", "properties": {
        "catalyst": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        "driver": {"type": "string", "enum": ["stock_specific", "sector", "market", "unclear"]},
        "move_nature": {"type": "string", "enum": ["information", "flow", "mixed", "unclear"]},
        "verdict": {"type": "string", "enum": ["fade_candidate", "do_not_fade", "unclear"]},
        "rationale": {"type": "string"},
    }, "required": ["catalyst", "evidence_ids", "driver", "move_nature", "verdict", "rationale"]},
}


def should_alert(t: str, s: dict, state: dict, today: str) -> bool:
    min_move = ETF_MIN_MOVE if is_etf(t) else STOCK_MIN_MOVE
    z = math.log1p(s["move"]) / s["vol"] if s["vol"] else 0
    if abs(s["move"]) < min_move or abs(z) < MIN_Z:
        return False
    prev = state.get(t)
    if not prev or prev.get("date") != today:
        return True
    # same day: only if the move has extended by another threshold in the same direction
    return (s["move"] - prev["move"]) * math.copysign(1, s["move"]) >= min_move


def explain_holding(t: str, name: str, s: dict, heads: list[dict]) -> dict | None:
    import anthropic
    from macro import headlines_block
    from scanner import CLAUDE_MODEL
    prompt = (f"POSITION: {name} ({t})\nMOVE: {s['move']*100:+.2f}% vs previous close, "
              f"{math.log1p(s['move'])/s['vol']:+.1f} sigma of daily vol\n\n"
              f"HEADLINES\n{headlines_block(heads)}\n\nCall record.")
    try:
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=800, system=HOLDING_PROMPT, tools=[HOLDING_TOOL],
            tool_choice={"type": "tool", "name": "record"}, messages=[{"role": "user", "content": prompt}])
        for b in msg.content:
            if b.type == "tool_use":
                a = dict(b.input)
                a["evidence_ids"] = [i for i in a.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= len(heads)]
                return a
    except Exception as e:
        print(f"  holding explain failed: {e}", file=sys.stderr)
    return None


def holding_headlines(t: str, name: str) -> list[dict]:
    from macro import get_headlines
    heads = []
    if not t.endswith(".L"):
        from scanner import get_news
        for n in get_news(t, datetime.now(ET).date())[:10]:
            heads.append({"ts": datetime.now(timezone.utc), "source": n["source"],
                          "headline": n["headline"], "url": n["url"]})
    heads += get_headlines(hours=8, limit=30)
    return heads[:40]


def watch(dry: bool = False) -> None:
    from scanner import send_telegram
    d = load()
    if not d["positions"]:
        return
    snap = snapshot([p["ticker"] for p in d["positions"]])
    try:
        state = json.loads(WATCH_STATE.read_text())
    except Exception:
        state = {}
    today = datetime.now(LONDON).date().isoformat()
    fx_cache: dict[str, float] = {}
    for p in d["positions"]:
        t, s = p["ticker"], snap.get(p["ticker"])
        if not s or (datetime.now(timezone.utc) - s["last_ts"].to_pydatetime()) > timedelta(hours=1):
            continue  # no fresh price (market closed)
        if not should_alert(t, s, state, today):
            continue
        name = p.get("name") or t
        heads = holding_headlines(t, name)
        a = explain_holding(t, name, s, heads)
        e = html.escape
        arrow = "🔺" if s["move"] > 0 else "🔻"
        lines = [f"📌 <b>YOUR HOLDING</b> {arrow} <b>{e(t)}</b> {s['move']*100:+.1f}% "
                 f"({math.log1p(s['move'])/s['vol']:+.1f}σ) · last {s['last']:,.2f} {e(s['currency'])}"]
        if p.get("qty"):
            fx = fx_cache.setdefault(s["currency"], gbp_per(s["currency"]))
            lines.append(f"Position today: {'+' if s['move'] >= 0 else '-'}£"
                         f"{abs(p['qty'] * (s['last'] - s['prev_close']) * fx):,.0f}")
        if a:
            lines += ["", f"<b>Catalyst:</b> {e(a['catalyst'])}",
                      f"<b>Driver:</b> {e(a['driver'].replace('_', ' '))} · <b>Nature:</b> {e(a['move_nature'])} "
                      f"→ <b>{e(a['verdict'].replace('_', ' ').upper())}</b>",
                      f"<b>Why:</b> {e(a['rationale'])}"]
            for i in a["evidence_ids"][:3]:
                h = heads[i - 1]
                src = f'<a href="{e(h["url"], quote=True)}">{e(h["source"])}</a>' if h.get("url") else e(h["source"])
                lines.append(f"[{i}] {src}: {e(h['headline'][:150])}")
        es = earliest_sale(p, d["hold_days"])
        if es and es > datetime.now(LONDON).date():
            lines.append(f"<i>PA: earliest sale {es:%d %b}</i>")
        text = "\n".join(lines)
        if dry:
            print(text)
        else:
            send_telegram(text)
            state[t] = {"date": today, "move": s["move"]}
    if not dry:
        WATCH_STATE.write_text(json.dumps(state))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", help="comma-separated tickers to add with no qty/cost")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.seed:   # "AAPL:Apple,MSFT:Microsoft,CSPX.L"  (names are optional, used to spot news)
        for item in args.seed.split(","):
            t, _, name = item.strip().partition(":")
            if t and not find(load(), t):
                add(t, buy_date="2000-01-01", name=name or None)   # existing holdings: no PA clock
        print("Holdings:", ", ".join(tickers()))
    if args.watch:
        watch(dry=args.dry_run)
    if args.show:
        print("\n".join(summary_lines()))
        print("\n".join(pa_lines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

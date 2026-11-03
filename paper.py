"""
Paper-trading ledger: every alert whose quant plan has an edge (score >= MIN_PLAN_SCORE) is
"executed" on paper, then marked against real 15-minute bars until it hits its target, its stop,
or its planned exit time. Nothing is traded for real.

  python paper.py --mark     mark open positions, close any that hit target/stop/time (every 15 min)
  python paper.py --report   send the paper-book report to Telegram (daily, after the US close)
  python paper.py --dry-run  with either flag: print instead of sending

Fills are deliberately conservative:
  - entry = OPEN of the first 15-min bar that starts after the alert (you can't fill at the alert print)
  - if a bar touches both target and stop, the stop is assumed to have hit first
  - a gap through a level fills at the bar's open, not at the level
  - the plan's round-trip cost estimate (spread, impact, stamp duty / FTT) is deducted

State lives in paper_book.json on the server (never in the repo).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BOOK = HERE / "paper_book.json"
CAPITAL = float(os.environ.get("PAPER_CAPITAL", "100000"))
LONDON = ZoneInfo("Europe/London")
CLOSE_LOCAL = {"us": ("America/New_York", 16, 0), "uk": ("Europe/London", 16, 30), "eu": ("Europe/Berlin", 17, 30)}


# --------------------------------------------------------------------------
# Book storage
# --------------------------------------------------------------------------
def load() -> list[dict]:
    try:
        return json.loads(BOOK.read_text())
    except Exception:
        return []


def save(book: list[dict]) -> None:
    tmp = BOOK.with_suffix(".tmp")
    tmp.write_text(json.dumps(book, indent=1, default=str))
    tmp.replace(BOOK)


def book_trade(market: str, source: str, symbol: str, label: str, plan: dict | None,
               verdict: str | None = None, alert_price: float | None = None) -> bool:
    """Called by the scanners when an alert goes out. Books only plans with an edge."""
    if not plan or not plan.get("edge") or plan.get("side") not in ("long", "short"):
        return False
    try:
        now = datetime.now(timezone.utc)
        book = load()
        if any(p["symbol"] == symbol and p["status"] == "open" for p in book):
            return False                          # one position per instrument
        x = plan.get("exit_by")
        if isinstance(x, datetime):
            exit_by = x.astimezone(timezone.utc).isoformat(timespec="minutes")
        elif isinstance(x, date):
            exit_by = x.isoformat()
        else:
            exit_by = None
        book.append({
            "id": f"{market}:{symbol}:{now:%Y%m%d%H%M}", "market": market, "source": source,
            "symbol": symbol, "label": label, "side": plan["side"],
            "opened": now.isoformat(timespec="seconds"), "alert_price": alert_price or plan.get("entry"),
            "tgt_pct": float(plan["tgt_pct"]), "stop_pct": float(plan["stop_pct"]),
            "hold": plan.get("hold"), "unit": plan.get("unit", "d"), "exit_by": exit_by,
            "score": int(plan.get("score", 0)), "cost": float(plan.get("cost") or 0.0),
            "size": round(CAPITAL * float(plan.get("risk_size") or 0.05), 2),
            "verdict": verdict, "status": "open", "entry": None,
        })
        save(book)
        return True
    except Exception as e:
        print(f"  paper booking failed for {symbol}: {e}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def bars_yf(symbol: str, since: datetime) -> pd.DataFrame:
    import yfinance as yf
    days = (datetime.now(timezone.utc) - since).days
    period = "5d" if days < 4 else ("1mo" if days < 25 else "60d")
    d = yf.download(symbol, period=period, interval="15m", progress=False, auto_adjust=False, prepost=False)
    if d is None or d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def bars_kraken(pair: str, since: datetime) -> pd.DataFrame:
    import requests
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": pair, "interval": 15, "since": int(since.timestamp()) - 900}, timeout=20)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(js["error"])
    key = next(k for k in js["result"] if k != "last")
    rows = js["result"][key]
    d = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "vwap", "vol", "n"])
    d.index = pd.to_datetime(d["t"], unit="s", utc=True)
    return d[["Open", "High", "Low", "Close"]].astype(float)


def bars(p: dict) -> pd.DataFrame:
    since = datetime.fromisoformat(p["opened"])
    if p["source"] == "kraken":
        return bars_kraken(p["symbol"], since)
    return bars_yf(p["symbol"], since)


# --------------------------------------------------------------------------
# Marking
# --------------------------------------------------------------------------
def exit_deadline(p: dict) -> datetime | None:
    x = p.get("exit_by")
    if not x:
        return None
    if "T" in x:
        return datetime.fromisoformat(x)
    tz, hh, mm = CLOSE_LOCAL.get(p["market"], CLOSE_LOCAL["us"])
    d = date.fromisoformat(x)
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def mark_one(p: dict, b: pd.DataFrame, now: datetime) -> dict:
    """Update one open position from its bars. Returns the (possibly closed) position."""
    opened = datetime.fromisoformat(p["opened"])
    after = b[b.index >= opened]
    if after.empty:
        return p
    sgn = 1 if p["side"] == "long" else -1
    if p.get("entry") is None:
        p["entry"] = float(after["Open"].iloc[0])
        p["entry_ts"] = after.index[0].isoformat()
    e = p["entry"]
    tgt = e * (1 + sgn * p["tgt_pct"])
    stp = e * (1 - sgn * p["stop_pct"])
    deadline = exit_deadline(p)
    walk = b[b.index >= pd.Timestamp(p["entry_ts"])]
    exit_px, how, ts = None, None, None
    for t, r in walk.iterrows():
        o, h, l = r["Open"], r["High"], r["Low"]
        if deadline and t >= deadline:
            break
        hit_stop = (l <= stp) if sgn > 0 else (h >= stp)
        hit_tgt = (h >= tgt) if sgn > 0 else (l <= tgt)
        if hit_stop:                                   # conservative: stop first if both touched
            gapped = (o <= stp) if sgn > 0 else (o >= stp)
            exit_px, how, ts = (o if gapped else stp), "stop", t
            break
        if hit_tgt:
            gapped = (o >= tgt) if sgn > 0 else (o <= tgt)
            exit_px, how, ts = (o if gapped else tgt), "target", t
            break
    if exit_px is None and deadline and now >= deadline:
        upto = walk[walk.index < deadline]
        if len(upto):
            exit_px, how, ts = float(upto["Close"].iloc[-1]), "time", upto.index[-1]
    last = float(walk["Close"].iloc[-1]) if len(walk) else e
    p["last"] = last
    p["unreal"] = sgn * (last / e - 1) - p["cost"]
    if exit_px is not None:
        pnl = sgn * (exit_px / e - 1) - p["cost"]
        p.update({"status": how, "exit": float(exit_px), "closed": pd.Timestamp(ts).isoformat(),
                  "pnl": pnl, "pnl_gbp": round(p["size"] * pnl, 2)})
    return p


def mark(dry: bool = False) -> list[dict]:
    book = load()
    now = datetime.now(timezone.utc)
    closed = []
    for i, p in enumerate(book):
        if p["status"] != "open":
            continue
        try:
            b = bars(p)
            if b.empty:
                continue
            q = mark_one(dict(p), b, now)
            book[i] = q
            if q["status"] != "open":
                closed.append(q)
        except Exception as e:
            print(f"  mark {p['symbol']} failed: {e}", file=sys.stderr)
    if not dry:
        save(book)
    if closed:
        icon = {"target": "✅", "stop": "❌", "time": "⏱"}
        lines = [f"{icon[q['status']]} {html.escape(q['label'])} {q['side']} (edge {q['score']}) "
                 f"{ {'target': 'hit target', 'stop': 'stopped', 'time': 'time exit'}[q['status']] } "
                 f"· {q['pnl']*100:+.2f}% net · £{q['pnl_gbp']:+,.0f}" for q in closed]
        text = "📒 <b>Paper trade closed</b>\n" + "\n".join(lines)
        if dry:
            print(text)
        else:
            from scanner import send_telegram
            send_telegram(text)
    print(f"Marked {sum(p['status'] == 'open' for p in book)} open, closed {len(closed)}.")
    return closed


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _stats(df: pd.DataFrame) -> str:
    n = len(df)
    if not n:
        return "none yet"
    x = df["pnl"]
    t = x.mean() / x.std() * math.sqrt(n) if n > 2 and x.std() > 0 else float("nan")
    return (f"{n} trades · win {(x > 0).mean()*100:.0f}% · avg {x.mean()*100:+.2f}% · "
            f"£{df['pnl_gbp'].sum():+,.0f}" + (f" · t={t:.1f}" if not math.isnan(t) else ""))


def report_text() -> str:
    book = load()
    if not book:
        return ("📒 <b>Paper book</b>\nNo paper trades yet. Every alert whose quant plan clears the edge "
                "threshold is booked here automatically.")
    df = pd.DataFrame(book)
    closed = df[df["status"] != "open"].copy()
    open_ = df[df["status"] == "open"]
    start = pd.to_datetime(df["opened"]).min()
    lines = [f"📒 <b>Paper book</b> since {start:%d %b %Y} · notional £{CAPITAL:,.0f} · net of costs"]
    lines.append(f"<b>All closed:</b> {_stats(closed)}")
    if len(closed):
        eq = closed.sort_values("closed")["pnl_gbp"].cumsum()
        dd = float((eq - eq.cummax()).min())
        lines.append(f"Return {closed['pnl_gbp'].sum()/CAPITAL*100:+.2f}% of capital · worst drawdown £{dd:,.0f}")
        for lab, g in closed.groupby("market"):
            lines.append(f"  {lab.upper()}: {_stats(g)}")
        agree = closed[closed["verdict"] == "fade_candidate"]
        dis = closed[closed["verdict"] != "fade_candidate"]
        if len(agree) and len(dis):
            lines.append(f"  AI agreed (FADE): {_stats(agree)}")
            lines.append(f"  AI didn't: {_stats(dis)}")
        exits = closed["status"].value_counts()
        lines.append("  Exits: " + " · ".join(f"{k} {v}" for k, v in exits.items()))
        today = closed[pd.to_datetime(closed["closed"]).dt.tz_convert(LONDON).dt.date == datetime.now(LONDON).date()]
        if len(today):
            lines.append("<b>Closed today:</b>")
            for _, q in today.iterrows():
                lines.append(f"  {html.escape(q['label'])} {q['side']} {q['status']} {q['pnl']*100:+.2f}% (£{q['pnl_gbp']:+,.0f})")
    if len(open_):
        lines.append(f"<b>Open ({len(open_)}):</b>")
        for _, p in open_.iterrows():
            u = p.get("unreal")
            u_txt = f"{u*100:+.2f}%" if u is not None and not pd.isna(u) else "awaiting fill"
            lines.append(f"  {html.escape(p['label'])} {p['side']} · edge {p['score']} · {u_txt} · "
                         f"tgt {p['tgt_pct']*100:+.1f}% / stop −{p['stop_pct']*100:.1f}% · exit {str(p['exit_by'])[:16]}")
    lines.append("<i>Fills: next 15-min bar open after the alert; stop assumed first if a bar hits both.</i>")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.mark:
        mark(dry=a.dry_run)
    if a.report:
        t = report_text()
        if a.dry_run:
            print(t)
        else:
            from scanner import send_telegram
            send_telegram(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())

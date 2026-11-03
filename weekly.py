"""
Sunday-evening week-ahead (18:00 UK):
  - key themes (Claude, from the weekend's headlines + the calendar)
  - the week's high-impact calendar, by day (UK time)
  - earnings: your holdings first, then S&P 500 names, by day
  - last week's biggest moves across macro and the S&P 500
  - your holdings over the week
  - the fade scorecard

  python weekly.py [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

LONDON = ZoneInfo("Europe/London")
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def calendar_by_day() -> dict[str, list[str]]:
    try:
        r = requests.get(FF_URL, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=20)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        return {"": [f"(calendar unavailable: {e.__class__.__name__})"]}
    out: dict[str, list[str]] = defaultdict(list)
    for ev in sorted(events, key=lambda x: x.get("date", "")):
        imp, ccy = ev.get("impact"), ev.get("country")
        if not (imp == "High" or (imp == "Medium" and ccy == "USD")):
            continue
        try:
            ts = datetime.fromisoformat(ev["date"]).astimezone(LONDON)
        except Exception:
            continue
        dot = "🔴" if imp == "High" else "🟠"
        f = f" (f {ev['forecast']})" if ev.get("forecast") else ""
        out[f"{ts:%a %d %b}"].append(f"{ts:%H:%M} {dot} {ccy} {ev.get('title')}{f}")
    return out


def earnings_by_day(start, end) -> dict[str, list[str]]:
    from scanner import _finnhub, load_universe
    import holdings
    data = _finnhub("calendar/earnings", {"from": start.isoformat(), "to": end.isoformat()}) or {}
    cal = data.get("earningsCalendar", []) if isinstance(data, dict) else []
    try:
        spx = set(load_universe()["symbol"])
    except Exception:
        spx = set()
    mine = set(holdings.tickers())
    when = {"bmo": "pre", "amc": "post"}
    out: dict[str, list[str]] = defaultdict(list)
    for c in sorted(cal, key=lambda x: (x.get("date", ""), x.get("symbol", ""))):
        s = c.get("symbol")
        if s in mine or s in spx:
            d = datetime.fromisoformat(c["date"]).strftime("%a %d %b")
            tag = f"📌{s}" if s in mine else s
            out[d].append(f"{tag}({when.get(c.get('hour'), '?')})")
    return out


def macro_week_moves(top: int = 10) -> list[str]:
    import yfinance as yf
    from macro import INSTRUMENTS, kraken_closes
    from macro import OTC_FUT
    yf_syms = {sym: name for name, src, sym, _ in INSTRUMENTS if src in ("yf", "yf_prepost")}
    yf_syms.update({OTC_FUT[sym]: name for name, src, sym, _ in INSTRUMENTS if src == "otc"})
    px = yf.download(list(yf_syms), period="3mo", interval="1d", auto_adjust=False, progress=False)["Close"]
    rows = []
    for sym, name in yf_syms.items():
        if sym not in px:
            continue
        s = px[sym].dropna()
        if len(s) < 30:
            continue
        wk = s.iloc[-1] / s.iloc[-6] - 1
        vol = np.log(s).diff().iloc[-60:].std() * math.sqrt(5)
        rows.append((abs(math.log1p(wk)) / vol if vol else 0, name, wk))
    try:
        b = kraken_closes("XBTUSD", interval=1440)
        wk = b.iloc[-1] / b.iloc[-8] - 1
        vol = np.log(b).diff().iloc[-60:].std() * math.sqrt(7)
        rows.append((abs(math.log1p(wk)) / vol, "Bitcoin", wk))
    except Exception:
        pass
    rows.sort(reverse=True)
    return [f"{name:<22}{wk*100:+6.2f}%  ({z:.1f}σ)" for z, name, wk in rows[:top]]


def spx_week_moves(n: int = 5) -> list[str]:
    import yfinance as yf
    from scanner import load_universe
    uni = load_universe()
    from scanner import yf_chunked
    px = yf_chunked(list(uni["yf_symbol"]), ("Close",), period="1mo", interval="1d", auto_adjust=True)["close"]
    wk = (px.iloc[-1] / px.iloc[-6] - 1).dropna().sort_values()
    name = dict(zip(uni["yf_symbol"], uni["symbol"]))
    dn = [f"{name.get(s, s)} {v*100:+.1f}%" for s, v in wk.head(n).items()]
    up = [f"{name.get(s, s)} {v*100:+.1f}%" for s, v in wk.tail(n)[::-1].items()]
    return ["Up: " + ", ".join(up), "Down: " + ", ".join(dn)]


def themes(cal: dict[str, list[str]]) -> list[str]:
    try:
        import anthropic
        from macro import get_headlines, headlines_block
        from scanner import CLAUDE_MODEL
        heads = get_headlines(hours=48, limit=60)
        cal_txt = "\n".join(f"{d}: " + "; ".join(v) for d, v in cal.items())
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=600,
            messages=[{"role": "user", "content":
                       "You brief a UK macro trader (USD/FX, rates, precious metals, bitcoin, US equities) "
                       "for the week ahead. Using ONLY the calendar and headlines below, give 4-5 short "
                       "bullets: the key themes and events to watch and why. No invented facts. Output just "
                       "the bullets, each starting with '• '.\n\nCALENDAR\n" + cal_txt +
                       f"\n\nHEADLINES\n{headlines_block(heads)}"}])
        return [l.strip() for l in "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").splitlines() if l.strip().startswith("•")][:6]
    except Exception as e:
        print(f"themes failed: {e}", file=sys.stderr)
        return []


def build_messages() -> list[str]:
    import followups
    import holdings
    e = html.escape
    now = datetime.now(LONDON)
    mon = (now + timedelta(days=(7 - now.weekday()) % 7 or 7)).date() if now.weekday() >= 5 else now.date()
    fri = mon + timedelta(days=4 - mon.weekday())

    cal = calendar_by_day()
    m1 = [f"🗓 <b>Week ahead</b> · {mon:%d %b} – {fri:%d %b}"]
    th = themes(cal)
    if th:
        m1 += ["", "<b>Themes</b>"] + [e(t) for t in th]
    m1 += ["", "<b>Calendar (UK time)</b>"]
    for d, evs in cal.items():
        m1 += [f"<u>{e(d)}</u>"] + [e(x) for x in evs]

    m2 = []
    try:
        earn = earnings_by_day(mon, fri)
        if earn:
            m2 += ["<b>Earnings</b> (US >$1bn; 📌 = yours)"] + [f"<u>{e(d)}</u> {e(', '.join(v[:30]))}" for d, v in earn.items()]
    except Exception as ex:
        m2.append(f"(earnings unavailable: {e(ex.__class__.__name__)})")
    try:
        m2 += ["", "<b>Last week's biggest macro moves</b> (vs normal weekly range)",
               "<pre>" + e("\n".join(macro_week_moves())) + "</pre>"]
    except Exception as ex:
        m2.append(f"(macro moves unavailable: {e(ex.__class__.__name__)})")
    try:
        m2 += ["<b>US stocks last week</b>"] + [e(x) for x in spx_week_moves()]
    except Exception as ex:
        m2.append(f"(S&P moves unavailable: {e(ex.__class__.__name__)})")
    try:
        if holdings.tickers():
            m2 += ["", "<b>Your holdings this week</b>",
                   "<pre>" + e("\n".join(holdings.summary_lines(weekly=True))) + "</pre>"]
    except Exception as ex:
        m2.append(f"(holdings unavailable: {e(ex.__class__.__name__)})")

    try:
        m3 = followups.scorecard_text()
    except Exception as ex:
        m3 = f"📊 Scorecard unavailable ({e(ex.__class__.__name__)})"
    return ["\n".join(m1), "\n".join(m2), m3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from scanner import send_telegram
    for msg in build_messages():
        print(msg) if args.dry_run else send_telegram(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

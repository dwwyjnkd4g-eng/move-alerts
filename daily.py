"""
Daily 07:00 UK brief, every day including weekends:
  - hello + health check of the other alert jobs (via the GitHub API)
  - market levels: USD, metals, bitcoin, equity futures, yields, oil
  - today's economic calendar (ForexFactory) and S&P 500 earnings (Finnhub)
  - overnight headlines (FinancialJuice + Finnhub), summarised by Claude

Usage:
    python daily.py            # only sends if it's 07:xx in London (DST-safe)
    python daily.py --force    # send now
    python daily.py --dry-run  # print instead of sending
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from macro import LONDON, get_headlines, headlines_block, kraken_closes
from scanner import CLAUDE_MODEL, _finnhub, load_universe, send_telegram

NAME = "there"
SEND_HOUR_LONDON = 7

LEVELS = [  # label, source, symbol, decimals
    ("DXY", "yf", "DX-Y.NYB", 2),
    ("EURUSD", "yf", "EURUSD=X", 4),
    ("USDJPY", "yf", "JPY=X", 2),
    ("Gold", "yf", "GC=F", 1),
    ("Silver", "yf", "SI=F", 2),
    ("Platinum", "yf", "PL=F", 1),
    ("Palladium", "yf", "PA=F", 1),
    ("Bitcoin", "kraken", "XBTUSD", 0),
    ("PAXG (24/7 gold)", "kraken", "PAXGUSD", 1),
    ("S&P fut", "yf", "ES=F", 1),
    ("Nasdaq fut", "yf", "NQ=F", 1),
    ("FTSE 100", "yf", "^FTSE", 1),
    ("DAX", "yf", "^GDAXI", 1),
    ("Nikkei 225", "yf", "^N225", 0),
    ("Hang Seng", "yf", "^HSI", 0),
    ("VIX", "yf", "^VIX", 2),
    ("US 10y %", "yf", "^TNX", 3),
    ("Brent", "yf", "BZ=F", 2),
    ("Copper", "yf", "HG=F", 3),
]
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
WORKFLOWS = {"alerts.yml": "Equity alerts", "macro.yml": "Macro alerts"}


# --------------------------------------------------------------------------
def health_lines(now: datetime) -> list[str]:
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return ["(health check runs on GitHub only)"]
    out = []
    for wf, label in WORKFLOWS.items():
        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/runs",
                params={"per_page": 20, "event": "schedule"},
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                timeout=20,
            )
            r.raise_for_status()
            runs = r.json().get("workflow_runs", [])
        except Exception as e:
            out.append(f"⚠️ {label}: couldn't check ({e.__class__.__name__})")
            continue
        if not runs:
            out.append(f"⚪ {label}: no scheduled runs yet")
            continue
        last = runs[0]
        when = datetime.fromisoformat(last["created_at"].replace("Z", "+00:00"))
        fails = sum(1 for x in runs if x.get("conclusion") == "failure")
        ago = (now - when).total_seconds() / 3600
        icon = "✅" if last.get("conclusion") in ("success", None) and fails == 0 else "⚠️"
        out.append(f"{icon} {label}: last run {ago:.0f}h ago · {fails} failed of last {len(runs)}")
    return out


def levels(now: datetime) -> list[str]:
    import yfinance as yf

    lines = []
    yf_syms = [s for _, src, s, _ in LEVELS if src == "yf"]
    try:
        d = yf.download(yf_syms, period="7d", interval="1d", progress=False, auto_adjust=False)["Close"]
    except Exception as e:
        print(f"yfinance levels failed: {e}", file=sys.stderr)
        d = pd.DataFrame()
    for label, src, sym, dp in LEVELS:
        try:
            if src == "yf":
                s = d[sym].dropna()
                last, prev = s.iloc[-1], s.iloc[-2]
                tag = f"{s.index[-1]:%a}"
            else:
                s = kraken_closes(sym, interval=60)
                last = s.iloc[-1]
                prev = s[s.index <= s.index[-1] - pd.Timedelta(hours=24)].iloc[-1]
                tag = "24h"
            lines.append(f"{label:<17}{last:>12,.{dp}f}  {(last / prev - 1) * 100:+6.2f}% ({tag})")
        except Exception:
            lines.append(f"{label:<17}{'n/a':>12}")
    return lines


def calendar_today(today_london) -> list[str]:
    try:
        r = requests.get(FF_URL, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=20)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        return [f"(calendar unavailable: {e.__class__.__name__})"]
    out = []
    for ev in events:
        try:
            ts = datetime.fromisoformat(ev["date"]).astimezone(LONDON)
        except Exception:
            continue
        impact, ccy = ev.get("impact", ""), ev.get("country", "")
        if ts.date() != today_london:
            continue
        if not (impact == "High" or (impact == "Medium" and ccy == "USD")):
            continue
        dot = "🔴" if impact == "High" else "🟠"
        extra = ", ".join(x for x in [f"f {ev['forecast']}" if ev.get("forecast") else "",
                                      f"p {ev['previous']}" if ev.get("previous") else ""] if x)
        out.append(f"{ts:%H:%M} {dot} {ccy} {ev.get('title', '')}" + (f" ({extra})" if extra else ""))
    return sorted(out) or ["No high-impact releases today."]


def earnings_today(today) -> list[str]:
    data = _finnhub("calendar/earnings", {"from": today.isoformat(), "to": today.isoformat()}) or {}
    cal = data.get("earningsCalendar", []) if isinstance(data, dict) else []
    if not cal:
        return []
    try:
        spx = set(load_universe()["symbol"])
    except Exception:
        spx = set()
    when = {"bmo": "pre-mkt", "amc": "after close", "dmh": "intraday"}
    rows = [f"{c['symbol']} ({when.get(c.get('hour'), '?')})" for c in cal
            if not spx or c.get("symbol") in spx]
    return rows[:25]


BRIEF_TOOL = {
    "name": "record_brief",
    "description": "Record the morning brief.",
    "input_schema": {
        "type": "object",
        "properties": {
            "overnight": {"type": "array", "items": {"type": "string"},
                          "description": "3-5 short bullets: what moved overnight and why."},
            "top_headlines": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "integer"}, "why_it_matters": {"type": "string"}},
                "required": ["id", "why_it_matters"]},
                "description": "The 6-8 most market-relevant headlines, most important first."},
            "watch_today": {"type": "array", "items": {"type": "string"},
                            "description": "2-4 bullets: what matters today for USD, metals, BTC, US equities."},
        },
        "required": ["overnight", "top_headlines", "watch_today"],
    },
}


def claude_brief(levels_txt: str, cal: list[str], heads: list[dict]) -> dict | None:
    if not heads or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    prompt = (
        "You write a concise morning brief for a UK-based trader focused on USD, precious metals, "
        "bitcoin and large single-stock moves in US equities. Use ONLY the data below; cite "
        "headlines by number; no invented facts.\n\n"
        f"LEVELS\n{levels_txt}\n\nTODAY'S CALENDAR (UK time)\n" + "\n".join(cal) +
        f"\n\nHEADLINES (newest first)\n{headlines_block(heads)}\n\nCall record_brief."
    )
    try:
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=1500, tools=[BRIEF_TOOL],
            tool_choice={"type": "tool", "name": "record_brief"},
            messages=[{"role": "user", "content": prompt}],
        )
        for b in msg.content:
            if b.type == "tool_use":
                return dict(b.input)
    except Exception as e:
        print(f"brief failed: {e}", file=sys.stderr)
    return None


def build_messages(now: datetime) -> list[str]:
    e = html.escape
    local = now.astimezone(LONDON)
    weekend = local.weekday() >= 5

    hello = [f"👋 Morning {NAME} — bot is alive. {local:%A %d %b %Y}", ""]
    hello += [e(x) for x in health_lines(now)]
    try:
        from news import source_health
        ok, total, failing = source_health()
        hello.append(f"{'✅' if not failing else '⚠️'} News sources: {ok}/{total} responding"
                     + (f" (not: {e(', '.join(failing))})" if failing else ""))
    except Exception as ex:
        hello.append(f"⚠️ News sources: check failed ({ex.__class__.__name__})")

    lv = levels(now)
    cal = [] if weekend else calendar_today(local.date())
    earn = [] if weekend else earnings_today(local.date())
    heads = get_headlines(hours=14 if not weekend else 24)
    brief = claude_brief("\n".join(lv), cal, heads)

    m1 = hello + ["", "<b>Levels</b>", "<pre>" + e("\n".join(lv)) + "</pre>"]
    if weekend:
        m1.append("Weekend: FX and CME metals shut; BTC and PAXG trade live.")
    if cal:
        m1 += ["", "<b>Calendar (UK time)</b>"] + [e(x) for x in cal]
    if earn:
        m1 += ["", "<b>S&amp;P 500 earnings today</b>", e(", ".join(earn))]

    m2 = []
    if brief:
        m2 += ["<b>Overnight</b>"] + [f"• {e(x)}" for x in brief["overnight"]]
        m2 += ["", "<b>Top headlines</b>"]
        for h in brief["top_headlines"][:8]:
            i = h.get("id")
            if not isinstance(i, int) or not 1 <= i <= len(heads):
                continue
            it = heads[i - 1]
            src = e(it["source"])
            if it["url"]:
                src = f'<a href="{e(it["url"], quote=True)}">{src}</a>'
            m2.append(f"• {e(it['headline'][:180])} — {src}\n   <i>{e(h['why_it_matters'])}</i>")
        m2 += ["", "<b>Watch today</b>"] + [f"• {e(x)}" for x in brief["watch_today"]]
    elif heads:
        m2 += ["<b>Latest headlines</b>"] + [
            f"• {it['ts'].astimezone(LONDON):%H:%M} {e(it['headline'][:180])} ({e(it['source'])})"
            for it in heads[:10]]
    return ["\n".join(m1)[:4000]] + (["\n".join(m2)[:4000]] if m2 else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    if not args.force and now.astimezone(LONDON).hour != SEND_HOUR_LONDON:
        print("Not 07:xx in London — skipping (the other cron slot covers the clock change).")
        return 0
    for msg in build_messages(now):
        if args.dry_run:
            print(msg + "\n" + "=" * 60)
        else:
            send_telegram(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

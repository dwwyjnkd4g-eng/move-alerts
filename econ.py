"""
Data-release alerts.

  ⏰ ~10 minutes before each high-impact release (all currencies) and medium-impact USD release
  📊 5-25 minutes after: the actual vs forecast (from squawk headlines) and the market reaction

Run every 1-2 minutes on the server (news/listener feed the 'actual' lines), or with the
GitHub macro job (less timely). Calendar: ForexFactory weekly JSON (no actuals in it).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
STATE = HERE / "econ_state.json"
SQUAWK_LOG = HERE / "squawk_recent.jsonl"   # written by listener.py
LONDON = ZoneInfo("Europe/London")
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

HEADSUP_MIN = (7, 13)      # send the heads-up when the event is this many minutes away
RESULT_MIN = (5, 25)       # send the result in this window after the release
REACTION = [("DXY", "DX-Y.NYB"), ("EURUSD", "EURUSD=X"), ("USDJPY", "JPY=X"), ("Gold", "GC=F"),
            ("Silver", "SI=F"), ("S&P fut", "ES=F"), ("US 10y note fut", "ZN=F"), ("Bitcoin", "BTC-USD")]
COUNTRY_WORDS = {"USD": ["us", "u.s.", "united states", "fed", "fomc"], "EUR": ["eurozone", "euro", "ecb", "german", "germany"],
                 "GBP": ["uk", "u.k.", "british", "boe"], "JPY": ["japan", "japanese", "boj"],
                 "CNY": ["china", "chinese", "pboc"], "CAD": ["canada", "canadian", "boc"],
                 "AUD": ["australia", "australian", "rba"], "NZD": ["new zealand", "rbnz"], "CHF": ["swiss", "snb"]}


CAL_CACHE = HERE / "econ_calendar.json"


def _calendar_json() -> list[dict]:
    """ForexFactory's weekly JSON, cached for an hour (this runs every couple of minutes)."""
    try:
        c = json.loads(CAL_CACHE.read_text())
        if datetime.now(timezone.utc) - datetime.fromisoformat(c["fetched"]) < timedelta(minutes=60):
            return c["events"]
    except Exception:
        pass
    r = requests.get(FF_URL, headers={"User-Agent": "Mozilla/5.0 (move-alerts)"}, timeout=20)
    r.raise_for_status()
    events = r.json()
    CAL_CACHE.write_text(json.dumps({"fetched": datetime.now(timezone.utc).isoformat(), "events": events}))
    return events


def load_calendar() -> list[dict]:
    out = []
    for ev in _calendar_json():
        try:
            ev["ts"] = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except Exception:
            continue
        if ev.get("impact") == "High" or (ev.get("impact") == "Medium" and ev.get("country") == "USD"):
            out.append(ev)
    return out


def ev_key(ev: dict) -> str:
    return f"{ev['ts'].isoformat()}|{ev.get('country')}|{ev.get('title')}"


def ev_line(ev: dict) -> str:
    dot = "🔴" if ev.get("impact") == "High" else "🟠"
    extra = ", ".join(x for x in [f"f {ev['forecast']}" if ev.get("forecast") else "",
                                  f"p {ev['previous']}" if ev.get("previous") else ""] if x)
    return f"{dot} {ev.get('country')} {ev.get('title')}" + (f" ({extra})" if extra else "")


# --------------------------------------------------------------------------
# Actuals from squawk + market reaction
# --------------------------------------------------------------------------
_STOP = {"m/m", "y/y", "q/q", "mom", "yoy", "qoq", "prelim", "final", "flash", "rate", "s", "the", "of", "and"}


def _keywords(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def recent_squawk(since: datetime) -> list[dict]:
    items = []
    if SQUAWK_LOG.exists():
        for line in SQUAWK_LOG.read_text().splitlines()[-3000:]:
            try:
                it = json.loads(line)
                it["ts"] = datetime.fromisoformat(it["ts"])
                if it["ts"] >= since:
                    items.append(it)
            except Exception:
                continue
    if not items:  # no listener (e.g. on GitHub) — fall back to the polled feeds
        try:
            from news import fetch_all
            items = [{"ts": x["ts"], "source": x["source"], "text": x["headline"]} for x in fetch_all(since)]
        except Exception as e:
            print(f"fallback news failed: {e}", file=sys.stderr)
    return items


def match_actuals(ev: dict, items: list[dict], limit: int = 3) -> list[dict]:
    kw = _keywords(ev.get("title", ""))
    cw = COUNTRY_WORDS.get(ev.get("country", ""), [])
    scored = []
    for it in items:
        t = it["text"].lower()
        if not (ev["ts"] - timedelta(minutes=1) <= it["ts"] <= ev["ts"] + timedelta(minutes=20)):
            continue
        hits = sum(1 for k in kw if re.search(rf"\b{re.escape(k)}\b", t))
        if not hits:
            continue
        score = hits + (2 if "actual" in t else 0) + (1 if any(c in t for c in cw) else 0) \
            + (1 if re.search(r"\d", t) else 0)
        scored.append((score, it))
    scored.sort(key=lambda x: (-x[0], x[1]["ts"]))
    return [it for s, it in scored if s >= 3][:limit]


def reaction(ts: datetime) -> list[str]:
    import yfinance as yf
    syms = [s for _, s in REACTION]
    try:
        px = yf.download(syms, period="1d", interval="1m", progress=False, auto_adjust=False)["Close"]
    except Exception as e:
        return [f"(reaction unavailable: {e.__class__.__name__})"]
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    out = []
    for label, s in REACTION:
        if s not in px:
            continue
        ser = px[s].dropna()
        before = ser[ser.index < ts]
        after = ser[ser.index >= ts]
        if before.empty or after.empty:
            continue
        chg = float(after.iloc[-1] / before.iloc[-1] - 1)
        mins = int((after.index[-1] - ts).total_seconds() // 60)
        out.append(f"{label} {chg*100:+.2f}%")
    return out + ([f"(over ~{mins} min)"] if out else [])


def classify(ev: dict, lines: list[str]) -> str | None:
    """One-line hot/cool read from Claude Haiku, using only the squawk text."""
    try:
        import anthropic
        from news import NEWS_MODEL
        msg = anthropic.Anthropic().messages.create(
            model=NEWS_MODEL, max_tokens=120,
            messages=[{"role": "user", "content":
                       f"Release: {ev.get('country')} {ev.get('title')}, forecast {ev.get('forecast') or 'n/a'}, "
                       f"previous {ev.get('previous') or 'n/a'}.\nSquawk lines:\n" + "\n".join(lines) +
                       "\n\nIn ONE short line (max 15 words): actual vs forecast, and whether that is "
                       "hot/strong, soft/weak or in line for the currency. If the lines don't contain "
                       "the actual, reply exactly: unknown"}])
        txt = msg.content[0].text.strip()
        return None if txt.lower().startswith("unknown") else txt
    except Exception as e:
        print(f"classify failed: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
def run(now: datetime | None = None, dry: bool = False) -> None:
    from scanner import send_telegram
    now = now or datetime.now(timezone.utc)
    try:
        cal = load_calendar()
    except Exception as e:
        print(f"calendar failed: {e}", file=sys.stderr)
        return
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}

    # heads-ups, grouped by release time
    due: dict[datetime, list[dict]] = {}
    for ev in cal:
        mins = (ev["ts"] - now).total_seconds() / 60
        if HEADSUP_MIN[0] <= mins <= HEADSUP_MIN[1] and not state.get("h|" + ev_key(ev)):
            due.setdefault(ev["ts"], []).append(ev)
    for ts, evs in due.items():
        text = (f"⏰ <b>In {int((ts - now).total_seconds() // 60)} min</b> "
                f"({ts.astimezone(LONDON):%H:%M} UK)\n" + "\n".join(html.escape(ev_line(e)) for e in evs))
        print(text) if dry else send_telegram(text)
        for e in evs:
            state["h|" + ev_key(e)] = True

    # results, grouped by release time
    done: dict[datetime, list[dict]] = {}
    for ev in cal:
        mins = (now - ev["ts"]).total_seconds() / 60
        if RESULT_MIN[0] <= mins <= RESULT_MIN[1] and not state.get("r|" + ev_key(ev)):
            done.setdefault(ev["ts"], []).append(ev)
    for ts, evs in done.items():
        items = recent_squawk(ts - timedelta(minutes=2))
        parts = [f"📊 <b>Data</b> · {ts.astimezone(LONDON):%H:%M} UK"]
        for e in evs:
            parts.append(f"\n<b>{html.escape(ev_line(e))}</b>")
            hits = match_actuals(e, items)
            if hits:
                read = classify(e, [h["text"] for h in hits])
                if read:
                    parts.append(f"→ {html.escape(read)}")
                for h in hits:
                    parts.append(f"   • {html.escape(h['text'][:200])} <i>({html.escape(h['source'])})</i>")
            else:
                parts.append("   (actual not seen on squawk yet)")
        r = reaction(ts)
        if r:
            parts.append("\n<b>Reaction:</b> " + html.escape(", ".join(r)))
        text = "\n".join(parts)
        print(text) if dry else send_telegram(text)
        for e in evs:
            state["r|" + ev_key(e)] = True

    # prune old keys
    cutoff = (now - timedelta(days=8)).isoformat()
    state = {k: v for k, v in state.items() if k.split("|")[1] >= cutoff}
    if not dry:
        STATE.write_text(json.dumps(state))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

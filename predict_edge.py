"""
Prediction-market edge scanner: flags odds that look rich or cheap against an independent fair value.

Five checks, run every 15 minutes on the snapshot predict.py saves (no extra load on the venues):

  1. FED vs FUTURES   Polymarket/Kalshi Fed-decision odds vs what 30-day fed funds futures imply for the
                      same meeting (the CME FedWatch method). Compared in expected bp of change, so
                      "61% chance of 25bp" = -15bp. Flags gaps >= 2.5bp (= 10 pts of a 25bp move).
  2. CROSS-VENUE      The same question on Polymarket and Kalshi (pairs matched by Claude, re-checked
                      every 6h; only identical outcome + resolution date). Flags gaps >= 8 pts.
  3. OVER/UNDERROUND  Mutually exclusive outcomes of one event should sum to ~100%. Flags >= 7 pts off
                      (underround only when every outcome is liquid and included).
  4. LADDERS          "Above 3.0%" can't be cheaper than "above 3.1%". Flags inversions >= 5 pts.
  5. DEADLINES        "By March" can't be dearer than "by June". Flags inversions >= 5 pts.

A signal must show on two consecutive scans (stale quotes don't count), then re-alerts only after 6h or
if the gap widens by 5+ pts. /edges shows what's live now.

These are SIGNALS, not advice: check fees, depth, settlement wording, venue access (Kalshi is US-only)
and your employer's PA policy before trading anything.
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "predict_edge_state.json"
NEWS_MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
FED_GAP_BP = 2.5
XV_GAP = 0.08
ROUND_GAP = 0.07
INV_GAP = 0.05
REALERT_H = 6
WIDEN = 0.05
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | \
         {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
FUT_CODES = "FGHJKMNQUVXZ"


def _load(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _save(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)


def pct(p: float) -> str:
    return f"{p*100:.0f}%"


# --------------------------------------------------------------------------
# 1. Fed odds vs fed funds futures
# --------------------------------------------------------------------------
def ff_prices(months: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Last settle of the 30-day fed funds future for each (year, month) via Yahoo (ZQ<code><yy>.CBT)."""
    import yfinance as yf
    syms = {f"ZQ{FUT_CODES[m - 1]}{y % 100:02d}.CBT": (y, m) for y, m in months}
    out = {}
    try:
        d = yf.download(list(syms), period="5d", interval="1d", progress=False, auto_adjust=False)["Close"]
        if not hasattr(d, "columns"):
            d = d.to_frame(list(syms)[0])
        for s, ym in syms.items():
            if s in d and d[s].notna().any():
                out[ym] = float(d[s].dropna().iloc[-1])
    except Exception as e:
        print(f"  fed funds futures: {e}", file=sys.stderr)
    return out


def _add_month(y: int, m: int, k: int) -> tuple[int, int]:
    i = y * 12 + (m - 1) + k
    return i // 12, i % 12 + 1


def futures_implied(meeting: date, fomc: set[date], px: dict) -> dict | None:
    """Expected change in bp at `meeting`, FedWatch-style: the month's average rate is a day-weighted
    mix of the pre- and post-meeting rates; anchor the pre (or post) rate on a meeting-free month."""
    y, m = meeting.year, meeting.month
    n = calendar.monthrange(y, m)[1]
    d = meeting.day                     # decision at 2pm on day d, new rate effective d+1
    if (y, m) not in px:
        return None
    avg = 100 - px[(y, m)]
    meet_months = {(x.year, x.month) for x in fomc}
    prev, nxt = _add_month(y, m, -1), _add_month(y, m, 1)
    if prev not in meet_months and prev in px:
        r0 = 100 - px[prev]
        r1 = (avg * n - r0 * d) / (n - d)
    elif nxt not in meet_months and nxt in px:
        r1 = 100 - px[nxt]
        r0 = (avg * n - r1 * (n - d)) / d
    else:
        return None
    bp = (r1 - r0) * 100
    return {"bp": bp, "r0": r0, "r1": r1, "p_cut25": max(0.0, min(1.0, -bp / 25)), "p_hike25": max(0.0, min(1.0, bp / 25))}


OUT_RE = re.compile(r"(no change|hold|unchanged|maintain|pause)|(\d+)\s*\+?\s*(?:bps?|basis points?)?\s*"
                    r"(decrease|cut|lower|reduction|increase|hike|raise)|(cuts?|decreases?|hikes?|increases?|raises?)\s*(?:rates?\s*)?(?:by\s*)?(\d+)", re.I)


def outcome_bp(text: str) -> int | None:
    t = text.lower()
    m = OUT_RE.search(t)
    if not m:
        return None
    if m.group(1):
        return 0
    if m.group(2):
        n, word = int(m.group(2)), m.group(3)
    else:
        word, n = m.group(4), int(m.group(5))
    return -n if word.rstrip("s") in ("decrease", "cut", "lower", "reduction") else n


def meeting_for(title: str, fomc: set[date], today: date) -> date | None:
    t = title.lower()
    upcoming = sorted(x for x in fomc if x >= today)
    for mtg in upcoming[:6]:
        names = (calendar.month_name[mtg.month].lower(), calendar.month_abbr[mtg.month].lower())
        if any(re.search(rf"\b{n}\b", t) for n in names):
            yr = re.search(r"\b(20\d\d)\b", t)
            if yr and int(yr.group(1)) != mtg.year:
                continue
            return mtg
    if re.search(r"\bnext (fed|fomc) (meeting|decision)\b", t) and upcoming:
        return upcoming[0]
    return None


def fed_signals(markets: list[dict]) -> list[dict]:
    from quant import FOMC
    today = datetime.now(timezone.utc).date()
    groups: dict[str, list[dict]] = {}
    for m in markets:
        if m.get("cat") != "fed_rates" and not re.search(r"\b(fed|fomc)\b", m["event"], re.I):
            continue
        if not re.search(r"decision|rates?|cut|hike|bps", m["event"], re.I):
            continue
        groups.setdefault(m["ekey"], []).append(m)
    cand = []
    for k, ms in groups.items():
        mtg = meeting_for(ms[0]["event"], FOMC, today)
        if not mtg:
            continue
        outs = []
        for m in ms:
            bp = outcome_bp(m["label"] or m["question"])
            if bp is not None:
                outs.append((bp, m["p"], m))
        if len(outs) < 2 or not any(b == 0 for b, _, _ in outs):
            continue
        tot = sum(p for _, p, _ in outs)
        if not 0.85 <= tot <= 1.15:
            continue
        crowd = sum(b * p for b, p, _ in outs) / tot
        cand.append((mtg, k, crowd, outs, ms[0]))
    if not cand:
        return []
    need = set()
    for mtg, *_ in cand:
        for k in (-1, 0, 1):
            need.add(_add_month(mtg.year, mtg.month, k))
    px = ff_prices(sorted(need))
    out = []
    for mtg, k, crowd, outs, m0 in cand:
        fut = futures_implied(mtg, FOMC, px)
        if not fut:
            continue
        gap = crowd - fut["bp"]
        p_cut_crowd = sum(p for b, p, _ in outs if b < 0) / sum(p for _, p, _ in outs)
        side = "more dovish" if gap < 0 else "more hawkish"
        out.append({
            "key": f"fed:{k}", "type": "Fed vs futures", "gap": abs(gap) / 25, "show": abs(gap) >= FED_GAP_BP,
            "venue": m0["venue"], "url": m0.get("url", ""),
            "title": f"FOMC {mtg:%d %b %Y}: {m0['venue']} vs fed funds futures",
            "lines": [f"Crowd: {crowd:+.1f}bp expected ({pct(p_cut_crowd)} chance of a cut) · "
                      + " / ".join(f"{('no change' if b == 0 else f'{b:+d}bp')} {pct(p)}" for b, p, _ in sorted(outs, key=lambda x: x[0])),
                      f"Futures: {fut['bp']:+.1f}bp expected (≈{pct(fut['p_cut25'])} of a 25bp cut) · "
                      f"pre {fut['r0']:.3f}% → post {fut['r1']:.3f}%",
                      f"Gap {gap:+.1f}bp ≈ {abs(gap)/25*100:.0f} pts of a 25bp move — the crowd is {side} than rates."],
            "trade": ("crowd odds of a cut look rich vs futures (sell 'cut' / buy 'no change'), or futures are the "
                      "cheaper way to own the dovish view") if gap < 0 else
                     ("crowd odds of a cut look cheap vs futures (buy 'cut'), or futures are the cheaper way to "
                      "own the hawkish view"),
        })
    return out


# --------------------------------------------------------------------------
# 2. Cross-venue pairs (Claude-matched, cached)
# --------------------------------------------------------------------------
PAIR_TOOL = {
    "name": "match_markets",
    "description": "Pair markets that are the SAME question on the two venues.",
    "input_schema": {"type": "object", "properties": {"pairs": {"type": "array", "items": {
        "type": "object", "properties": {
            "pm": {"type": "integer"}, "ks": {"type": "integer"},
            "invert": {"type": "boolean", "description": "true if Kalshi YES equals Polymarket NO"},
        }, "required": ["pm", "ks", "invert"]}}}, "required": ["pairs"]},
}


def match_pairs(markets: list[dict], st: dict) -> list[dict]:
    now = time.time()
    if now - st.get("pairs_ts", 0) < 6 * 3600 and st.get("pairs") is not None:
        return st["pairs"]
    pm = sorted([m for m in markets if m["venue"] == "Polymarket" and m.get("imp", 1) >= 2], key=lambda m: -m["vol24"])[:150]
    ks = sorted([m for m in markets if m["venue"] == "Kalshi" and m.get("imp", 1) >= 2], key=lambda m: -m["vol24"])[:150]
    if not pm or not ks:
        st["pairs"], st["pairs_ts"] = [], now
        return []

    def line(i, m):
        lab = f" — {m['label']}" if m["label"] else ""
        return f"{i}. {m['event'][:110]}{lab} (resolves {str(m.get('end') or '')[:10]})"

    prompt = ("POLYMARKET\n" + "\n".join(line(i, m) for i, m in enumerate(pm)) + "\n\nKALSHI\n"
              + "\n".join(line(i, m) for i, m in enumerate(ks))
              + "\n\nPair only markets asking the SAME question with the SAME outcome and the same deadline/date "
                "(e.g. same Fed meeting and same move size; same race and same party). Different thresholds, dates "
                "or wording that could resolve differently = NOT a pair. It's fine to return few or no pairs.")
    try:
        from llm import tool_call
        res = tool_call(NEWS_MODEL, PAIR_TOOL, [{"role": "user", "content": prompt}], max_tokens=3000)
        pairs = []
        for x in res.get("pairs", []):
            i, j = x.get("pm"), x.get("ks")
            if isinstance(i, int) and isinstance(j, int) and 0 <= i < len(pm) and 0 <= j < len(ks):
                pairs.append({"pm": pm[i]["mid"], "ks": ks[j]["mid"], "invert": bool(x.get("invert"))})
    except Exception as e:
        print(f"  pair matching failed: {e}", file=sys.stderr)
        pairs = st.get("pairs") or []
    st["pairs"], st["pairs_ts"] = pairs, now
    return pairs


def cross_venue_signals(markets: list[dict], st: dict) -> list[dict]:
    by = {m["mid"]: m for m in markets}
    out = []
    for pr in match_pairs(markets, st):
        a, b = by.get(pr["pm"]), by.get(pr["ks"])
        if not a or not b:
            continue
        pb = 1 - b["p"] if pr["invert"] else b["p"]
        gap = a["p"] - pb
        tight = all((x.get("spread") is None or x["spread"] <= 0.05) for x in (a, b))
        rich, cheap = ("Polymarket", "Kalshi") if gap > 0 else ("Kalshi", "Polymarket")
        out.append({
            "key": f"xv:{a['mid']}:{b['mid']}", "type": "Cross-venue", "gap": abs(gap), "show": abs(gap) >= XV_GAP and tight,
            "venue": "Polymarket/Kalshi", "url": a.get("url", ""),
            "title": a["event"][:120] + (f" — {a['label']}" if a["label"] else ""),
            "lines": [f"Polymarket {pct(a['p'])} vs Kalshi {pct(pb)}" + (" (Kalshi NO side)" if pr["invert"] else "")
                      + f" · gap {abs(gap)*100:.0f} pts"],
            "trade": f"{rich} looks rich vs {cheap}: sell {rich} / buy {cheap} (check fees and settlement wording)",
        })
    return out


# --------------------------------------------------------------------------
# 3-5. Internal consistency
# --------------------------------------------------------------------------
def round_signals(markets: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for m in markets:
        if m.get("mutex"):
            groups.setdefault(m["ekey"], []).append(m)
    out = []
    for k, ms in groups.items():
        if len(ms) < 3:
            continue
        s = sum(m["p"] for m in ms)
        complete = len(ms) >= (ms[0].get("n_out") or 999)
        over, under = s - 1 >= ROUND_GAP, complete and 1 - s >= ROUND_GAP
        if not (over or under):
            continue
        out.append({
            "key": f"rd:{k}", "type": "Overround" if over else "Underround", "gap": abs(s - 1), "show": True,
            "venue": ms[0]["venue"], "url": ms[0].get("url", ""), "title": ms[0]["event"][:120],
            "lines": [f"{len(ms)} mutually exclusive outcomes sum to {s*100:.0f}%"
                      + ("" if complete else f" ({len(ms)} of {ms[0].get('n_out')} outcomes are liquid enough to count)"),
                      "Top: " + ", ".join(f"{m['label'] or m['question'][:30]} {pct(m['p'])}" for m in sorted(ms, key=lambda x: -x["p"])[:4])],
            "trade": (f"the field is overpriced by ~{(s-1)*100:.0f} pts: buying NO across outcomes locks that in before fees"
                      if over else f"the field is underpriced by ~{(1-s)*100:.0f} pts: buying YES on every outcome locks that in"),
        })
    return out


LADDER_RE = re.compile(r"(above|over|greater than|more than|at least|higher than|>=|≥|>|below|under|less than|"
                       r"lower than|at most|<=|≤|<)\s*\$?\s*(-?[\d,]*\.?\d+)\s*(%|bps|bp|k|m|bn|b|t)?", re.I)
UP_WORDS = ("above", "over", "greater than", "more than", "at least", "higher than", ">=", "≥", ">")


def ladder_signals(markets: list[dict]) -> list[dict]:
    groups: dict[tuple, list[tuple]] = {}
    for m in markets:
        txt = m["label"] or m["question"]
        x = LADDER_RE.search(txt)
        if not x:
            continue
        up = x.group(1).lower() in UP_WORDS
        try:
            v = float(x.group(2).replace(",", ""))
        except ValueError:
            continue
        groups.setdefault((m["ekey"], up, (x.group(3) or "").lower()), []).append((v, m))
    out = []
    for (k, up, unit), rows in groups.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r[0])
        worst = None
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                lo, hi = rows[i], rows[j]
                # "above": P(above hi) must be <= P(above lo); "below": P(below lo) must be <= P(below hi)
                bad = (hi[1]["p"] - lo[1]["p"]) if up else (lo[1]["p"] - hi[1]["p"])
                if bad >= INV_GAP and (worst is None or bad > worst[0]):
                    worst = (bad, lo, hi)
        if worst:
            bad, lo, hi = worst
            a, b = (lo, hi) if up else (hi, lo)          # a should be >= b
            out.append({
                "key": f"ld:{k}:{a[1]['mid']}:{b[1]['mid']}", "type": "Ladder inversion", "gap": bad, "show": True,
                "venue": a[1]["venue"], "url": a[1].get("url", ""), "title": a[1]["event"][:120],
                "lines": [f"'{a[1]['label'] or a[1]['question'][:50]}' {pct(a[1]['p'])} < "
                          f"'{b[1]['label'] or b[1]['question'][:50]}' {pct(b[1]['p'])} — the easier outcome is cheaper by {bad*100:.0f} pts"],
                "trade": f"buy '{a[1]['label'] or 'the easier strike'}' / sell '{b[1]['label'] or 'the harder strike'}' "
                         f"(the spread can't lose if both settle on the same number)",
            })
    return out


def _label_date(label: str, today: date) -> date | None:
    t = label.strip().lower().replace(",", "")
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(20\d\d))?\b", t)
    if m and m.group(1) in MONTHS:
        mo, dd = MONTHS[m.group(1)], int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else today.year
        try:
            d = date(yr, mo, dd)
        except ValueError:
            return None
        if not m.group(3) and d < today - timedelta(days=7):
            d = date(yr + 1, mo, dd)
        return d
    m = re.search(r"\b(?:in\s+)?([a-z]{3,9})\s+(20\d\d)\b", t)
    if m and m.group(1) in MONTHS:
        y, mo = int(m.group(2)), MONTHS[m.group(1)]
        return date(y, mo, calendar.monthrange(y, mo)[1])
    m = re.search(r"\b(20\d\d)\b", t)
    if m and len(t) <= 12:
        return date(int(m.group(1)), 12, 31)
    return None


def deadline_signals(markets: list[dict]) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    groups: dict[str, list[tuple]] = {}
    for m in markets:
        if not re.search(r"\b(by|before)\b", m["event"], re.I) or m.get("mutex"):
            continue
        d = _label_date(m["label"], today) if m["label"] else None
        if d:
            groups.setdefault(m["ekey"], []).append((d, m))
    out = []
    for k, rows in groups.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r[0])
        worst = None
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                bad = rows[i][1]["p"] - rows[j][1]["p"]        # earlier deadline priced higher
                if bad >= INV_GAP and (worst is None or bad > worst[0]):
                    worst = (bad, rows[i], rows[j])
        if worst:
            bad, e, l = worst
            out.append({
                "key": f"dl:{k}:{e[1]['mid']}:{l[1]['mid']}", "type": "Deadline inversion", "gap": bad, "show": True,
                "venue": e[1]["venue"], "url": e[1].get("url", ""), "title": e[1]["event"][:120],
                "lines": [f"by {e[0]:%d %b %Y}: {pct(e[1]['p'])} > by {l[0]:%d %b %Y}: {pct(l[1]['p'])} "
                          f"— an earlier deadline can't be more likely ({bad*100:.0f} pts inverted)"],
                "trade": f"buy 'by {l[0]:%b %Y}' / sell 'by {e[0]:%b %Y}'",
            })
    return out


# --------------------------------------------------------------------------
def scan(dry: bool = False) -> int:
    from predict import load_snapshot
    markets = load_snapshot(max_age_min=20)
    if not markets:
        print("No fresh prediction-market snapshot (predict.py runs every 5 min).")
        return 0
    st = _load(STATE, {"seen": {}, "alerted": {}, "current": []})
    sigs = []
    for fn in (fed_signals, round_signals, ladder_signals, deadline_signals):
        try:
            sigs += fn(markets)
        except Exception as e:
            print(f"  {fn.__name__} failed: {e}", file=sys.stderr)
    try:
        sigs += cross_venue_signals(markets, st)
    except Exception as e:
        print(f"  cross-venue failed: {e}", file=sys.stderr)
    now = time.time()
    live = [s for s in sigs if s["show"]]
    fire = []
    for s in live:
        prev = st["seen"].get(s["key"])
        confirmed = prev is not None and now - prev <= 40 * 60
        last = st["alerted"].get(s["key"])
        fresh = last is None or now - last["ts"] >= REALERT_H * 3600 or s["gap"] - last["gap"] >= WIDEN
        if confirmed and fresh:
            fire.append(s)
    st["seen"] = {s["key"]: now for s in live}
    st["current"] = [{k: s[k] for k in ("type", "title", "lines", "trade", "gap", "url")} for s in
                     sorted(live, key=lambda x: -x["gap"])][:15]
    # Fed vs futures is also useful context when NOT flagged: keep the latest reading
    st["fed"] = [{"title": s["title"], "lines": s["lines"]} for s in sigs if s["type"] == "Fed vs futures"]
    if fire:
        fire.sort(key=lambda x: -x["gap"])
        text = format_signals(fire[:6], "⚖️ <b>Prediction-market edge</b>")
        if dry:
            print(text)
        else:
            from scanner import send_telegram
            send_telegram(text)
            for s in fire:
                st["alerted"][s["key"]] = {"ts": now, "gap": s["gap"]}
    st["alerted"] = {k: v for k, v in st["alerted"].items() if now - v["ts"] < 7 * 86400}
    if not dry:
        _save(STATE, st)
    print(f"{len(sigs)} checks, {len(live)} live signals, {len(fire)} alerted.")
    return 0


def format_signals(sigs: list[dict], header: str) -> str:
    e = html.escape
    out = [header]
    for s in sigs:
        out.append(f"\n<b>{e(s['type'])}</b> · {e(s['title'])}")
        out += [f"   {e(x)}" for x in s["lines"]]
        out.append(f"   → <i>{e(s['trade'])}</i>")
        if s.get("url"):
            out.append(f'   <a href="{e(s["url"], quote=True)}">open market</a>')
    out.append("\n<i>Signals, not advice: check fees, depth, settlement wording, venue access (Kalshi is US-only) "
               "and your PA policy.</i>")
    return "\n".join(out)


def edges_text() -> str:
    st = _load(STATE, {})
    cur = st.get("current") or []
    parts = []
    if cur:
        parts.append(format_signals(cur[:8], "⚖️ <b>Prediction-market edges live now</b>"))
    else:
        parts.append("⚖️ No prediction-market mispricings above threshold right now.")
    if st.get("fed"):
        parts.append("\n<b>Fed: crowd vs futures</b>\n" + "\n".join(
            html.escape(f['title']) + "\n   " + "\n   ".join(html.escape(x) for x in f["lines"]) for f in st["fed"][:3]))
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    return scan(dry=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())

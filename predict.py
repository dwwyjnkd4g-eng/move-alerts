"""
Prediction-market feed: alerts when the crowd reprices a macro or geopolitical event.

Every 5 minutes it reads the most-traded open markets on Polymarket and Kalshi, keeps the ones that
matter for macro/FX/metals/equities (Claude Haiku classifies each event once, so sports, celebrity,
"will X tweet" and short-dated price-bin markets are dropped), and alerts when a probability moves
sharply AND the move holds:

  - over 1h  >= 10 pts, over 4h >= 15 pts, since yesterday >= 20 pts
  - or a big move in odds terms from a low/high base (e.g. 3% -> 12%): >= 1.2 logit and >= 6 pts
  - the move must still be there on the next 5-min check, unless it's 1.5x the threshold
  - markets must have real money in them (24h volume / liquidity floors); wide-spread prints ignored
  - low-importance events need 1.5x bigger moves
  - an event re-alerts only when a market has moved a further threshold since its LAST alert

Each alert groups the moved outcomes of one event, says whether a headline explains it (from the
squawk feed and news already sent) or flags "no headline yet - odds moved first", and names the
markets it matters for. Recent moves are also fed to the macro analyst, the morning brief has a
"what prediction markets are pricing" block, and /odds shows the board on demand.

  python predict.py              poll, alert
  python predict.py --dry-run    poll, print instead of sending
  python predict.py --board      print the board

Odds history for relevant markets is logged every 30 min to cache/predict/YYYY-MM.csv, so they can
be backtested as model features later.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).resolve().parent
STATE = HERE / "predict_state.json"
META = HERE / "predict_meta.json"
LOGDIR = HERE / "cache" / "predict"
LONDON = ZoneInfo("Europe/London")
UA = {"User-Agent": "move-alerts/1.0 (market-data reader)", "Accept": "application/json"}

MIN_VOL24 = float(os.environ.get("PRED_MIN_VOL24", "15000"))     # $ traded in 24h (or contracts on Kalshi)
MIN_LIQ = float(os.environ.get("PRED_MIN_LIQ", "50000"))         # $ resting liquidity / open interest
THRESH = {"1h": 0.10, "4h": 0.15, "24h": 0.20}
WINDOWS = {"1h": 3600, "4h": 4 * 3600, "24h": 24 * 3600}
WLABEL = {"1h": "in the last hour", "4h": "in 4h", "24h": "since yesterday"}
LOGIT_MIN, LOGIT_PTS = 1.2, 0.06
MAX_SPREAD = 0.10
EVENT_GAP_MIN = 30          # minutes between alerts on the same event
MAX_EVENTS_MSG = 8
NEWS_MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")

CATEGORIES = ["fed_rates", "central_banks", "inflation_data", "growth_recession", "us_politics",
              "elections_global", "trade_tariffs", "geopolitics_conflict", "energy", "fiscal_debt",
              "crypto_policy", "companies_tech", "commodities_weather", "other"]
CAT_LABEL = {"fed_rates": "Fed", "central_banks": "Central banks", "inflation_data": "Data",
             "growth_recession": "Growth", "us_politics": "US politics", "elections_global": "Elections",
             "trade_tariffs": "Trade/tariffs", "geopolitics_conflict": "Geopolitics", "energy": "Energy",
             "fiscal_debt": "Fiscal", "crypto_policy": "Crypto policy", "companies_tech": "Companies",
             "commodities_weather": "Commodities/weather", "other": "Other"}
SKIP_RE = re.compile(r"\b(nba|nfl|nhl|mlb|ufc|premier league|champions league|world cup|super bowl|grammy|oscar|"
                     r"emmy|box office|album|song|tweet|tweets|posts? on x|mrbeast|taylor swift|kardashian|"
                     r"up or down|highest temperature|temperature in|bachelor|survivor|f1|formula 1|tennis|golf|"
                     r"wimbledon|eurovision|youtube|tiktok|spotify|netflix top)\b", re.I)
KEEP_RE = re.compile(r"\b(fed|fomc|rate (cut|hike)|interest rate|cpi|inflation|recession|gdp|unemployment|payroll|"
                     r"tariff|trade deal|sanction|war|ceasefire|invade|invasion|strike on|missile|nato|china|taiwan|"
                     r"russia|ukraine|iran|israel|gaza|opec|oil|shutdown|debt ceiling|election|president|prime minister|"
                     r"senate|house|congress|ecb|boe|bank of england|boj|treasury|default|gold|bitcoin etf|sec)\b", re.I)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def _f(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _jl(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x or "[]")
    except Exception:
        return []


def fetch_polymarket(pages: int = 5) -> list[dict]:
    out = []
    for k in range(pages):
        try:
            r = requests.get("https://gamma-api.polymarket.com/events", headers=UA, timeout=25,
                             params={"active": "true", "closed": "false", "order": "volume24hr",
                                     "ascending": "false", "limit": 100, "offset": 100 * k})
            r.raise_for_status()
            evs = r.json()
        except Exception as e:
            print(f"  polymarket page {k}: {e}", file=sys.stderr)
            break
        if not evs:
            break
        for ev in evs:
            tags = [t.get("label", "") for t in ev.get("tags") or [] if isinstance(t, dict)]
            for m in ev.get("markets") or []:
                if m.get("closed") or m.get("active") is False:
                    continue
                outcomes = _jl(m.get("outcomes"))
                prices = _jl(m.get("outcomePrices"))
                bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
                if bid and ask and 0 < ask - bid <= MAX_SPREAD:
                    p = (bid + ask) / 2
                else:
                    p = _f(m.get("lastTradePrice")) if not prices else _f(prices[0])
                if p is None:
                    continue
                spread = (ask - bid) if bid and ask else None
                label = m.get("groupItemTitle") or ""
                if outcomes and str(outcomes[0]).lower() != "yes":
                    label = (label + " " if label else "") + f"({outcomes[0]})"
                out.append({
                    "venue": "Polymarket", "ekey": f"pm:{ev.get('id')}", "event": ev.get("title") or m.get("question"),
                    "mid": f"pm:{m.get('id')}", "question": m.get("question") or "", "label": label.strip(),
                    "p": p, "spread": spread, "vol24": _f(m.get("volume24hr"), 0.0) or 0.0,
                    "ev_vol24": _f(ev.get("volume24hr"), 0.0) or 0.0,
                    "liq": _f(m.get("liquidityNum"), None) or _f(m.get("liquidity"), 0.0) or 0.0,
                    "d1h": _f(m.get("oneHourPriceChange")), "d24h": _f(m.get("oneDayPriceChange")),
                    "end": m.get("endDate") or ev.get("endDate"), "tags": tags,
                    "url": f"https://polymarket.com/event/{ev.get('slug')}" if ev.get("slug") else "",
                })
        time.sleep(0.3)
    return out


KALSHI_BASES = ["https://api.elections.kalshi.com/trade-api/v2", "https://external-api.kalshi.com/trade-api/v2"]
KALSHI_SKIP_CATS = {"sports", "entertainment", "social", "transportation", "mentions"}


def _kpx(m: dict, *keys):
    for k in keys:
        v = _f(m.get(k))
        if v is not None:
            return v / 100 if v > 1.0 and not k.endswith("_dollars") else v
    return None


def fetch_kalshi(pages: int = 6) -> list[dict]:
    out = []
    for base in KALSHI_BASES:
        cursor, ok = None, False
        for _ in range(pages):
            try:
                params = {"status": "open", "with_nested_markets": "true", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                r = requests.get(f"{base}/events", headers=UA, params=params, timeout=25)
                r.raise_for_status()
                js = r.json()
            except Exception as e:
                print(f"  kalshi {base.split('/')[2]}: {e}", file=sys.stderr)
                break
            ok = True
            for ev in js.get("events") or []:
                cat = str(ev.get("category") or "")
                if cat.lower() in KALSHI_SKIP_CATS:
                    continue
                for m in ev.get("markets") or []:
                    if str(m.get("status", "active")).lower() not in ("active", "open", "initialized"):
                        continue
                    bid, ask = _kpx(m, "yes_bid_dollars", "yes_bid"), _kpx(m, "yes_ask_dollars", "yes_ask")
                    last = _kpx(m, "last_price_dollars", "last_price")
                    if bid and ask and 0 < ask - bid <= MAX_SPREAD:
                        p = (bid + ask) / 2
                    else:
                        p = last
                    if p is None or p <= 0:
                        continue
                    prev = _kpx(m, "previous_price_dollars", "previous_price")
                    series = ev.get("series_ticker") or ""
                    out.append({
                        "venue": "Kalshi", "ekey": f"ks:{ev.get('event_ticker')}", "event": ev.get("title") or m.get("title"),
                        "mid": f"ks:{m.get('ticker')}", "question": m.get("title") or "",
                        "label": (m.get("yes_sub_title") or m.get("subtitle") or "").strip(),
                        "p": p, "spread": (ask - bid) if bid and ask else None,
                        "vol24": _f(m.get("volume_24h_fp"), None) or _f(m.get("volume_24h"), 0.0) or 0.0, "ev_vol24": 0.0,
                        "liq": (_f(m.get("liquidity_dollars"), None) or (_f(m.get("liquidity"), 0.0) or 0.0) / 100
                                or _f(m.get("open_interest_fp"), None) or _f(m.get("open_interest"), 0.0) or 0.0),
                        "d1h": None, "d24h": (p - prev) if prev else None,
                        "end": m.get("close_time") or m.get("expiration_time"), "tags": [cat] if cat else [],
                        "url": f"https://kalshi.com/markets/{series.lower()}" if series else "https://kalshi.com",
                    })
            cursor = js.get("cursor")
            if not cursor:
                break
            time.sleep(0.3)
        if ok:
            break
    return out


def liquid(m: dict) -> bool:
    if m["spread"] is not None and m["spread"] > MAX_SPREAD:
        return False
    return m["vol24"] >= MIN_VOL24 or m["liq"] >= MIN_LIQ


# --------------------------------------------------------------------------
# Relevance (Claude Haiku, once per event; keyword fallback)
# --------------------------------------------------------------------------
CLASSIFY_TOOL = {
    "name": "classify_events",
    "description": "Classify prediction-market events by relevance to a macro trader.",
    "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {
            "i": {"type": "integer"},
            "relevant": {"type": "boolean"},
            "category": {"type": "string", "enum": CATEGORIES},
            "importance": {"type": "integer", "minimum": 1, "maximum": 3},
            "assets": {"type": "string", "description": "markets most affected, <= 6 words, e.g. 'USD, gold, front-end rates'"},
        }, "required": ["i", "relevant", "category", "importance", "assets"]}}}, "required": ["items"]},
}
CLASSIFY_SYSTEM = """You screen prediction-market events for a professional macro trader (precious metals, \
FX, rates, commodities, global equities). Mark relevant=true only if the outcome could move those markets: \
central banks and rate decisions, inflation/jobs/GDP prints, recession, tariffs and trade deals, sanctions, \
wars, ceasefires and military strikes, elections and government formation that matter for markets, \
shutdowns and debt ceilings, OPEC and energy supply, crypto regulation/ETF approvals, major company events \
(mega-cap earnings beats, big M&A, antitrust). Mark relevant=false for sports, entertainment, celebrities, \
awards, social-media posting counts, city temperature bins, and short-dated price-bin markets that just \
mirror an asset's price ('Bitcoin above $X on date', 'S&P up or down today', 'gold price at end of week'). \
importance: 3 = moves global markets (Fed decision, US election control, major war escalation, big tariffs); \
2 = moves a sector, country or asset; 1 = marginal."""


def classify(new_events: dict[str, dict], meta: dict, max_calls: int = 4) -> None:
    """new_events: ekey -> {"event", "labels", "tags"}. Writes into meta."""
    keys = list(new_events)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    calls = 0
    while keys and calls < max_calls:
        batch, keys = keys[:50], keys[50:]
        lines = []
        for i, k in enumerate(batch):
            ev = new_events[k]
            extra = "; ".join(x for x in ev["labels"][:4] if x)
            tag = ", ".join(ev["tags"][:3])
            lines.append(f"{i}. {ev['event']}" + (f" | outcomes: {extra}" if extra else "") + (f" | tags: {tag}" if tag else ""))
        try:
            from llm import tool_call
            res = tool_call(NEWS_MODEL, CLASSIFY_TOOL, [{"role": "user", "content": "\n".join(lines)}],
                            system=CLASSIFY_SYSTEM, max_tokens=4000)
            got = {int(x["i"]): x for x in res.get("items", []) if isinstance(x, dict) and "i" in x}
        except Exception as e:
            print(f"  classify failed ({e}); keyword fallback", file=sys.stderr)
            got = {}
        calls += 1
        for i, k in enumerate(batch):
            x = got.get(i)
            if x is None:
                txt = new_events[k]["event"] + " " + " ".join(new_events[k]["labels"][:4])
                rel = bool(KEEP_RE.search(txt)) and not SKIP_RE.search(txt)
                x = {"relevant": rel, "category": "other", "importance": 2 if rel else 1, "assets": "", "fallback": True}
            meta[k] = {"relevant": bool(x.get("relevant")), "category": x.get("category", "other"),
                       "importance": int(x.get("importance", 1) or 1), "assets": str(x.get("assets", ""))[:60],
                       "event": new_events[k]["event"][:160], "ts": now, "fallback": x.get("fallback", False)}
        # keyword-fallback verdicts are retried with the model on a later run


# --------------------------------------------------------------------------
# Move detection
# --------------------------------------------------------------------------
def _logit(p: float) -> float:
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))


def base_at(hist: list, secs: int, now: float) -> float | None:
    """Probability ~secs ago from our own samples (needs a sample at least 75% that old)."""
    target = now - secs
    old = [h for h in hist if h[0] <= now - 0.75 * secs]
    if not old:
        return None
    return min(old, key=lambda h: abs(h[0] - target))[1]


def detect(m: dict, hist: list, last_alert_p: float | None, importance: int, now: float) -> dict | None:
    """Return the strongest qualifying move for one market, or None."""
    p = m["p"]
    mult = 1.5 if importance <= 1 else 1.0
    best = None
    for w, secs in WINDOWS.items():
        base = base_at(hist, secs, now)
        boot = False
        if base is None:                       # new market: bootstrap the 1h window from the venue's own field
            d = m.get("d1h") if w == "1h" else None
            if d is None:
                continue
            base, boot = p - d, True
        d = p - base
        thr = THRESH[w] * mult
        big_pts = abs(d) >= thr
        big_odds = abs(_logit(p) - _logit(base)) >= LOGIT_MIN * mult and abs(d) >= LOGIT_PTS * mult
        if not (big_pts or big_odds):
            continue
        # must hold on the previous check too, unless very large (or no history to check against)
        strong = abs(d) >= 1.5 * thr
        if not strong:
            if boot or len(hist) < 1:
                continue
            prev_p = hist[-1][1]
            if abs(prev_p - base) < 0.7 * min(abs(d), thr) or (prev_p - base) * d <= 0:
                continue
        # only new information since the last alert on this market
        if last_alert_p is not None and abs(p - last_alert_p) < min(THRESH["1h"], max(abs(d) * 0.5, LOGIT_PTS)):
            continue
        sc = abs(d) / thr
        if best is None or sc > best["sc"]:
            best = {"w": w, "base": base, "d": d, "sc": sc, "boot": boot}
    return best


# --------------------------------------------------------------------------
# Drivers: recent headlines we already have locally (squawk + news sent)
# --------------------------------------------------------------------------
def local_headlines(hours: float = 3, limit: int = 60) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    for fname, tk, hk, sk in (("squawk_recent.jsonl", "ts", "text", "source"), ("sent_headlines.jsonl", "ts", "h", None)):
        p = HERE / fname
        if not p.exists():
            continue
        for line in p.read_text().splitlines()[-3000:]:
            try:
                r = json.loads(line)
                ts = datetime.fromisoformat(r[tk])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    out.append({"ts": ts, "headline": r[hk], "source": r.get(sk, "news") if sk else "news"})
            except Exception:
                continue
    seen, uniq = set(), []
    for h in sorted(out, key=lambda x: x["ts"], reverse=True):
        k = re.sub(r"\W+", " ", h["headline"].lower())[:80]
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    return uniq[:limit]


DRIVER_TOOL = {
    "name": "explain_odds_moves",
    "description": "Link each prediction-market move to supporting headlines and its market impact.",
    "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {
            "e": {"type": "integer"},
            "headline_ids": {"type": "array", "items": {"type": "integer"}},
            "explained": {"type": "boolean", "description": "true only if a listed headline plausibly caused the move"},
            "impact": {"type": "string", "description": "which markets and which way, <= 10 words, e.g. 'USD weaker, gold up, 2y yields down'"},
        }, "required": ["e", "headline_ids", "explained", "impact"]}}}, "required": ["items"]},
}


def drivers(events: list[dict], heads: list[dict]) -> dict[int, dict]:
    if not events:
        return {}
    ev_txt = []
    for i, e in enumerate(events):
        mv = "; ".join(f"{x['label'] or 'Yes'}: {x['base']*100:.0f}% -> {x['p']*100:.0f}%" for x in e["moves"][:4])
        ev_txt.append(f"{i}. {e['event']} ({e['venue']}) — {mv} {WLABEL[e['moves'][0]['w']]}")
    hd = "\n".join(f"[{j}] {h['ts'].astimezone(LONDON):%H:%M} {h['source']}: {h['headline'][:200]}"
                   for j, h in enumerate(heads, 1)) or "(none)"
    prompt = (f"PREDICTION-MARKET MOVES\n" + "\n".join(ev_txt) + f"\n\nRECENT HEADLINES (last 3h)\n{hd}\n\n"
              "For each move: cite headline ids that plausibly drove it (or none), and say in <=10 words what it "
              "means for markets (USD, gold, rates, oil, equities, sectors). Use only the headlines given.")
    try:
        from llm import tool_call
        res = tool_call(NEWS_MODEL, DRIVER_TOOL, [{"role": "user", "content": prompt}], max_tokens=1500)
        return {int(x["e"]): x for x in res.get("items", []) if isinstance(x, dict) and "e" in x}
    except Exception as e:
        print(f"  driver lookup failed: {e}", file=sys.stderr)
        return {}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def _load(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _save(p: Path, obj) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), default=str))
    tmp.replace(p)


def _pct(p: float) -> str:
    return f"{p*100:.1f}%" if p < 0.1 or p > 0.9 else f"{p*100:.0f}%"


def _money(x: float, venue: str) -> str:
    unit = "" if venue == "Kalshi" else "$"
    if x >= 1e6:
        return f"{unit}{x/1e6:.1f}m"
    if x >= 1e3:
        return f"{unit}{x/1e3:.0f}k"
    return f"{unit}{x:.0f}"


def format_message(events: list[dict], drv: dict[int, dict], heads: list[dict], extra: list[dict]) -> str:
    e = html.escape
    lines = ["🎲 <b>Prediction markets</b>"]
    for i, ev in enumerate(events):
        top = ev["moves"][0]
        arrow = "🔺" if top["d"] > 0 else "🔻"
        lines.append(f"\n{arrow} <b>{e(ev['event'][:140])}</b> — {ev['venue']}"
                     + (f" · {CAT_LABEL.get(ev['category'], '')}" if ev.get("category") else ""))
        for x in ev["moves"][:4]:
            q = x["question"].strip()
            lab = x["label"] or (q[:80] if q and q != ev["event"] else "Yes")
            lines.append(f"   {e(lab)}: {_pct(x['base'])} → <b>{_pct(x['p'])}</b> ({x['d']*100:+.0f} pts {WLABEL[x['w']]})"
                         f" · {_money(x['vol24'], ev['venue'])} 24h")
        d = drv.get(i) or {}
        ids = [j for j in d.get("headline_ids", []) if isinstance(j, int) and 1 <= j <= len(heads)]
        if d.get("explained") and ids:
            h = heads[ids[0] - 1]
            lines.append(f"   📰 {e(h['headline'][:160])} <i>({e(h['source'])} {h['ts'].astimezone(LONDON):%H:%M})</i>")
        elif drv:
            lines.append("   ⚡ <b>No headline yet</b> — odds moved before the news")
        impact = d.get("impact") or ev.get("assets")
        if impact:
            lines.append(f"   Matters for: {e(impact)}")
        if ev.get("url"):
            lines.append(f'   <a href="{e(ev["url"], quote=True)}">open market</a>')
    if extra:
        lines.append("\nAlso moving: " + "; ".join(
            f"{e(x['event'][:60])} {x['moves'][0]['d']*100:+.0f} pts" for x in extra[:6]))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main poll
# --------------------------------------------------------------------------
def poll(dry: bool = False) -> int:
    now_dt = datetime.now(timezone.utc)
    now = now_dt.timestamp()
    markets = [m for m in fetch_polymarket() + fetch_kalshi() if liquid(m)]
    print(f"{len(markets)} liquid markets fetched.")
    if not markets:
        return 0
    meta = _load(META, {})
    st = _load(STATE, {"hist": {}, "alerted": {}, "ev_alert": {}, "recent": [], "board": []})
    warmup = not st["hist"]            # first run: record a baseline, don't alert on stale moves

    # classify events we haven't seen (or only keyword-guessed) yet
    new = {}
    for m in markets:
        k = m["ekey"]
        if k not in meta or meta[k].get("fallback"):
            ne = new.setdefault(k, {"event": m["event"], "labels": [], "tags": m["tags"]})
            if m["label"] and len(ne["labels"]) < 6:
                ne["labels"].append(m["label"])
    if new:
        # pre-filter obvious junk without spending a model call
        junk = {k for k, v in new.items() if SKIP_RE.search(v["event"]) and not KEEP_RE.search(v["event"])}
        for k in junk:
            meta[k] = {"relevant": False, "category": "other", "importance": 1, "assets": "",
                       "event": new[k]["event"][:160], "ts": now_dt.isoformat(timespec="seconds")}
        todo = {k: v for k, v in new.items() if k not in junk}
        if todo:
            classify(dict(sorted(todo.items(), key=lambda kv: -max(
                (m["vol24"] for m in markets if m["ekey"] == kv[0]), default=0))), meta)
        _save(META, meta)

    rel = [m for m in markets if meta.get(m["ekey"], {}).get("relevant")]
    print(f"{len(rel)} relevant markets in {len({m['ekey'] for m in rel})} events.")

    # detect
    by_event: dict[str, dict] = {}
    for m in rel:
        info = meta[m["ekey"]]
        hist = st["hist"].get(m["mid"], [])
        mv = detect(m, hist, st["alerted"].get(m["mid"]), info.get("importance", 2), now)
        if mv:
            ev = by_event.setdefault(m["ekey"], {"ekey": m["ekey"], "event": m["event"], "venue": m["venue"],
                                                 "url": m["url"], "category": info.get("category"),
                                                 "assets": info.get("assets"), "importance": info.get("importance", 2),
                                                 "moves": []})
            ev["moves"].append({**mv, "mid": m["mid"], "p": m["p"], "label": m["label"], "question": m["question"],
                                "vol24": m["vol24"]})

    # event cooldown, ranking
    events = []
    for ev in by_event.values():
        last = st["ev_alert"].get(ev["ekey"])
        if last and now - last < EVENT_GAP_MIN * 60:
            continue
        ev["moves"].sort(key=lambda x: -abs(x["d"]))
        vol = sum(x["vol24"] for x in ev["moves"])
        ev["score"] = max(abs(x["d"]) for x in ev["moves"]) * (1 + ev["importance"]) * math.log10(vol + 10)
        events.append(ev)
    events.sort(key=lambda x: -x["score"])
    if warmup:
        print(f"Warm-up run: baseline recorded ({len(events)} moves ignored).")
        events = []
    show, extra = events[:MAX_EVENTS_MSG], events[MAX_EVENTS_MSG:]

    if show:
        heads = local_headlines(3)
        drv = drivers(show, heads)
        text = format_message(show, drv, heads, extra)
        if dry:
            print(text)
        else:
            from scanner import send_telegram
            send_telegram(text)
        for ev in events:
            st["ev_alert"][ev["ekey"]] = now
            for x in ev["moves"]:
                st["alerted"][x["mid"]] = x["p"]
            top = ev["moves"][0]
            st["recent"].append({"ts": now, "text": f"{ev['event'][:90]}"
                                 + (f" [{top['label']}]" if top["label"] else "")
                                 + f": {_pct(top['base'])} -> {_pct(top['p'])} {WLABEL[top['w']]} ({ev['venue']})"})

    # history (relevant markets only), pruned to ~26h
    keep_after = now - 26 * 3600
    for m in rel:
        h = [x for x in st["hist"].get(m["mid"], []) if x[0] >= keep_after]
        h.append([round(now), round(m["p"], 4)])
        st["hist"][m["mid"]] = h
    live = {m["mid"] for m in rel}
    st["hist"] = {k: v for k, v in st["hist"].items() if k in live or (v and v[-1][0] >= keep_after)}
    st["alerted"] = {k: v for k, v in st["alerted"].items() if k in st["hist"]}
    st["ev_alert"] = {k: v for k, v in st["ev_alert"].items() if now - v < 3 * 86400}
    st["recent"] = [r for r in st["recent"] if now - r["ts"] < 48 * 3600][-200:]
    st["board"] = build_board(rel, meta, st["hist"], now)
    st["updated"] = now_dt.isoformat(timespec="seconds")
    if not dry:
        log_history(rel, now_dt, st)
        _save(STATE, st)
    print(f"Alerted {len(events)} events." if events else "No qualifying moves.")
    return 0


def build_board(rel: list[dict], meta: dict, hist: dict, now: float) -> list[dict]:
    """Top markets by category for the brief and /odds (leading outcome per event)."""
    best: dict[str, dict] = {}
    for m in rel:
        cur = best.get(m["ekey"])
        if cur is None or m["vol24"] + m.get("ev_vol24", 0) > cur["vol24"] + cur.get("ev_vol24", 0):
            if cur is None or m["p"] >= 0.05:
                best[m["ekey"]] = m
    rows = []
    for k, m in best.items():
        info = meta.get(k, {})
        b24 = base_at(hist.get(m["mid"], []), 24 * 3600, now)
        d24 = (m["p"] - b24) if b24 is not None else m.get("d24h")
        rows.append({"cat": info.get("category", "other"), "imp": info.get("importance", 1), "event": m["event"][:110],
                     "label": m["label"][:50], "p": m["p"], "d24": d24, "vol": m["vol24"] + m.get("ev_vol24", 0),
                     "venue": m["venue"]})
    rows.sort(key=lambda r: (-r["imp"], -r["vol"]))
    out, per = [], {}
    for r in rows:
        if per.get(r["cat"], 0) >= 2:
            continue
        per[r["cat"]] = per.get(r["cat"], 0) + 1
        out.append(r)
        if len(out) >= 14:
            break
    return out


def log_history(rel: list[dict], now_dt: datetime, st: dict) -> None:
    slot = now_dt.strftime("%Y-%m-%dT%H:") + ("00" if now_dt.minute < 30 else "30")
    if st.get("log_slot") == slot:
        return
    st["log_slot"] = slot
    try:
        LOGDIR.mkdir(parents=True, exist_ok=True)
        f = LOGDIR / f"{now_dt:%Y-%m}.csv"
        new = not f.exists()
        with f.open("a") as fh:
            if new:
                fh.write("ts,venue,market,event,label,p,vol24\n")
            for m in rel:
                ev = m["event"].replace('"', "'")
                lab = m["label"].replace('"', "'")
                fh.write(f'{slot},{m["venue"]},{m["mid"]},"{ev[:120]}","{lab[:60]}",{m["p"]:.4f},{m["vol24"]:.0f}\n')
    except Exception as e:
        print(f"  odds log failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# Used by other modules
# --------------------------------------------------------------------------
def context_lines(hours: float = 6) -> list[str]:
    """Recent prediction-market repricings, for the macro analyst's prompt."""
    st = _load(STATE, {})
    now = time.time()
    return [datetime.fromtimestamp(r["ts"], LONDON).strftime("%a %H:%M UK ") + r["text"]
            for r in st.get("recent", []) if now - r["ts"] <= hours * 3600][-15:]


def board_lines(limit: int = 12) -> list[str]:
    st = _load(STATE, {})
    out = []
    for r in st.get("board", [])[:limit]:
        d = r.get("d24")
        out.append(f"{CAT_LABEL.get(r['cat'], r['cat'])}: {r['event']}" + (f" — {r['label']}" if r["label"] else "")
                   + f": {_pct(r['p'])}" + (f" ({d*100:+.0f} pts 24h)" if d is not None and abs(d) >= 0.01 else ""))
    return out


def board_text() -> str:
    st = _load(STATE, {})
    lines = board_lines(14)
    if not lines:
        return "🎲 No prediction-market data yet (it updates every 5 minutes)."
    upd = st.get("updated", "")[:16].replace("T", " ")
    return ("🎲 <b>What prediction markets are pricing</b> (Polymarket/Kalshi, " + upd + " UTC)\n"
            + "\n".join("• " + html.escape(x) for x in lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--board", action="store_true")
    a = ap.parse_args()
    if a.board:
        print(board_text())
        return 0
    return poll(dry=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())

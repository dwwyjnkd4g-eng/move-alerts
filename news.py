"""
Breaking-news feed: every 30 minutes, pull headlines from many free sources,
have Claude keep only the market-moving ones (merging the same story across
outlets), and push one Telegram digest.

Usage:
    python news.py              # normal run
    python news.py --dry-run    # print, don't send or save state
    python news.py --sources    # just test which feeds respond
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET_xml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import requests

from scanner import _finnhub, send_telegram

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "news_state.json"
LONDON = ZoneInfo("Europe/London")

# Cheaper model for headline triage (runs 48x a day). Override with NEWS_MODEL.
NEWS_MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
# 3 = only top-tier market movers; 2 = also clearly relevant news. Override with NEWS_MIN_IMPORTANCE.
MIN_IMPORTANCE = int(os.environ.get("NEWS_MIN_IMPORTANCE", "2"))
LOOKBACK_MIN = 75          # look back a bit more than the 30-min cadence (cron can be late)
MAX_CANDIDATES = 90
MAX_PER_DIGEST = 10


def gnews(q: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-GB&gl=GB&ceid=GB:en"


# name -> RSS/Atom URL. Feeds that fail are skipped; check with `python news.py --sources`.
FEEDS = {
    "FinancialJuice": os.environ.get("FJ_RSS_URL", "https://www.financialjuice.com/feed.ashx?xy=rss"),
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "CNBC Markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "MarketWatch Pulse": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Mining.com": "https://www.mining.com/feed/",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "Bank of England": "https://www.bankofengland.co.uk/rss/news",
    # Google News searches surface Reuters / Bloomberg / FT headlines.
    "Google News: macro": gnews('(Fed OR ECB OR "Bank of Japan" OR Treasury OR inflation OR tariffs OR sanctions) when:2h'),
    "Google News: metals/USD/BTC": gnews('(gold OR silver OR platinum OR "US dollar" OR yuan OR bitcoin) when:2h'),
    "Google News: Reuters/Bloomberg": gnews('(site:reuters.com OR site:bloomberg.com) markets when:2h'),
}
FINNHUB_CATEGORIES = ("general", "forex", "crypto", "merger")
# Public Telegram channels read via their web preview (t.me/s/<name>).
TELEGRAM_CHANNELS = {"FinancialJuice": "financialjuice"}
UA = {"User-Agent": "Mozilla/5.0 (move-alerts news reader)"}


# --------------------------------------------------------------------------
# Feed parsing (RSS 2.0 and Atom)
# --------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(el, *names):
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        ts = parsedate_to_datetime(s)
    except Exception:
        try:
            ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _clean_title(t: str, source: str) -> tuple[str, str]:
    t = html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()
    t = re.sub(r"^FinancialJuice:\s*", "", t)
    if source.startswith("Google News"):
        m = re.match(r"^(.*)\s+-\s+([^-]{2,40})$", t)   # "Headline - Reuters"
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return t, "Google News"
    return t, source


def parse_feed(content: bytes, source: str) -> list[dict]:
    root = ET_xml.fromstring(content)
    out = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        t = _child(el, "title")
        title, src = _clean_title(t.text if t is not None else "", source)
        link_el = _child(el, "link")
        link = ""
        if link_el is not None:
            link = (link_el.text or link_el.get("href") or "").strip()
        src_el = _child(el, "source")
        if src_el is not None and src_el.text and source.startswith("Google News"):
            src = src_el.text.strip()
        d = _child(el, "pubdate", "published", "updated", "date")
        ts = _parse_ts(d.text if d is not None else None)
        if title and ts:
            out.append({"ts": ts, "source": src, "headline": title, "url": link, "feed": source})
    return out


def fetch_feed(name: str, url: str) -> tuple[str, list[dict], str | None]:
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        return name, parse_feed(r.content, name), None
    except Exception as e:
        return name, [], f"{e.__class__.__name__}: {str(e)[:80]}"


def parse_tg_preview(page: str, label: str, channel: str) -> list[dict]:
    out = []
    for block in re.split(r'<div class="tgme_widget_message_wrap', page)[1:]:
        post = re.search(r'data-post="([^"]+)"', block)
        txt = re.search(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.S)
        ts = re.search(r'<time[^>]*datetime="([^"]+)"', block)
        if not (txt and ts):
            continue
        text = html.unescape(re.sub(r"<br\s*/?>", " ", txt.group(1)))
        text = re.sub(r"<[^>]+>", "", text).strip()
        t = _parse_ts(ts.group(1))
        if text and t:
            url = f"https://t.me/{post.group(1)}" if post else f"https://t.me/s/{channel}"
            out.append({"ts": t, "source": label, "headline": text[:300], "url": url, "feed": f"{label} (Telegram)"})
    return out


def fetch_tg(label: str, channel: str) -> tuple[str, list[dict], str | None]:
    name = f"{label} (Telegram)"
    try:
        r = requests.get(f"https://t.me/s/{channel}", headers=UA, timeout=15)
        r.raise_for_status()
        return name, parse_tg_preview(r.text, label, channel), None
    except Exception as e:
        return name, [], f"{e.__class__.__name__}: {str(e)[:80]}"


def fetch_finnhub() -> list[dict]:
    out = []
    for cat in FINNHUB_CATEGORIES:
        for it in _finnhub("news", {"category": cat}) or []:
            if it.get("headline"):
                out.append({"ts": datetime.fromtimestamp(it.get("datetime", 0), timezone.utc),
                            "source": it.get("source") or "Finnhub", "headline": it["headline"].strip(),
                            "url": it.get("url", ""), "feed": "Finnhub"})
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def _similar(a: str, b: str) -> bool:
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.6


def dedupe(items: list[dict]) -> list[dict]:
    """Collapse near-identical headlines; keep the earliest, remember other sources."""
    items = sorted(items, key=lambda x: x["ts"])
    out: list[dict] = []
    for it in items:
        for o in out[-200:]:
            if _norm(o["headline"]) == _norm(it["headline"]) or _similar(o["headline"], it["headline"]):
                if it["source"] not in o["sources"]:
                    o["sources"].append(it["source"])
                break
        else:
            out.append({**it, "sources": [it["source"]]})
    return sorted(out, key=lambda x: x["ts"], reverse=True)


def fetch_all(since: datetime, verbose: bool = False) -> list[dict]:
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(lambda kv: fetch_feed(*kv), FEEDS.items()))
        results += list(ex.map(lambda kv: fetch_tg(*kv), TELEGRAM_CHANNELS.items()))
    items = []
    for name, got, err in results:
        fresh = [x for x in got if x["ts"] >= since]
        if verbose or err:
            print(f"  {name:<32} {'ERROR ' + err if err else f'{len(got)} items, {len(fresh)} recent'}")
        items += fresh
    fh = [x for x in fetch_finnhub() if x["ts"] >= since]
    if verbose:
        print(f"  {'Finnhub':<32} {len(fh)} recent")
    return dedupe(items + fh)


def source_health() -> tuple[int, int, list[str]]:
    """(responding, total, failing names) — for the daily brief."""
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(lambda kv: fetch_feed(*kv), FEEDS.items()))
        results += list(ex.map(lambda kv: fetch_tg(*kv), TELEGRAM_CHANNELS.items()))
    failing =[name for name, got, err in results if err or not got]
    return len(results) - len(failing), len(results), failing


def item_id(it: dict) -> str:
    return hashlib.sha1(_norm(it["headline"]).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Claude triage
# --------------------------------------------------------------------------
TRIAGE_PROMPT = """You filter a live news wire for a UK-based trader who trades global macro \
(USD and FX, rates, equity indices and futures, oil, copper), precious metals (gold, silver, \
PGMs), bitcoin, and large single-stock moves in US equities including pre-market and after-hours."""
TRIAGE_PROMPT += """

From the numbered headlines, select only those likely to move those markets or that a macro \
trader would want to know about now. Score importance:
 3 = top-tier: central bank decisions/surprise comments, major data surprises, tariffs/sanctions, \
     geopolitical escalation, big market moves/halts, major M&A or earnings shocks in mega-caps, \
     large crypto/metals-specific events.
 2 = clearly relevant: scheduled data in line, notable official comments, sector/company news \
     with plausible market impact, metals supply/demand news.
Ignore: opinion pieces, explainers, listicles, personal finance, "stocks to buy", stale recaps, \
promotional content, minor company news.

Group headlines about the same story into one entry (list all their ids). Write a crisp \
one-line headline in wire style (no source names), a short why-it-matters, and affected assets. \
Use only the supplied headlines; do not add facts. Return an empty list if nothing qualifies."""

TRIAGE_TOOL = {
    "name": "record_selection",
    "description": "Record the selected market-moving stories.",
    "input_schema": {
        "type": "object",
        "properties": {
            "stories": {"type": "array", "items": {"type": "object", "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
                "headline": {"type": "string"},
                "importance": {"type": "integer", "enum": [2, 3]},
                "why": {"type": "string", "description": "Max ~20 words."},
                "assets": {"type": "array", "items": {"type": "string"},
                           "description": "e.g. USD, Gold, Silver, BTC, US equities, a ticker"},
            }, "required": ["ids", "headline", "importance", "why", "assets"]}},
        },
        "required": ["stories"],
    },
}


def triage(cands: list[dict]) -> list[dict]:
    import anthropic

    listing = "\n".join(
        f"[{i}] {c['ts'].astimezone(LONDON):%H:%M} | {', '.join(c['sources'][:3])} | {c['headline']}"
        for i, c in enumerate(cands, 1)
    )
    msg = anthropic.Anthropic().messages.create(
        model=NEWS_MODEL, max_tokens=2000, system=TRIAGE_PROMPT, tools=[TRIAGE_TOOL],
        tool_choice={"type": "tool", "name": "record_selection"},
        messages=[{"role": "user", "content": f"HEADLINES\n{listing}\n\nCall record_selection."}],
    )
    for b in msg.content:
        if b.type == "tool_use":
            stories = b.input.get("stories", [])
            for s in stories:
                s["ids"] = [i for i in s.get("ids", []) if isinstance(i, int) and 1 <= i <= len(cands)]
            return [s for s in stories if s["ids"]]
    return []


def format_digest(stories: list[dict], cands: list[dict], now: datetime) -> str:
    e = html.escape
    lines = [f"📰 <b>Breaking</b> · {now.astimezone(LONDON):%H:%M} UK"]
    for s in stories:
        first = cands[s["ids"][0] - 1]
        srcs = []
        for i in s["ids"]:
            c = cands[i - 1]
            for src in c["sources"]:
                if src not in [x[0] for x in srcs]:
                    srcs.append((src, c["url"]))
        src_txt = ", ".join(
            f'<a href="{e(u, quote=True)}">{e(n)}</a>' if u and k == 0 else e(n)
            for k, (n, u) in enumerate(srcs[:4])
        )
        dot = "🔴" if s["importance"] == 3 else "🟠"
        lines.append("")
        lines.append(f"{dot} <b>{e(s['headline'])}</b> ({first['ts'].astimezone(LONDON):%H:%M})")
        lines.append(f"   {e(s['why'])} <i>[{e(', '.join(s['assets'][:4]))}]</i>")
        lines.append(f"   {src_txt}")
    return "\n".join(lines)[:4000]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"seen": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sources", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    if args.sources:
        fetch_all(now - timedelta(hours=24), verbose=True)
        return 0

    state = load_state()
    seen = set(state.get("seen", []))
    items = fetch_all(now - timedelta(minutes=LOOKBACK_MIN), verbose=True)
    cands = [it for it in items if item_id(it) not in seen][:MAX_CANDIDATES]
    print(f"{len(items)} recent, {len(cands)} new")
    if not cands:
        return 0

    try:
        stories = triage(cands)
    except Exception as ex:
        print(f"triage failed: {ex}", file=sys.stderr)
        return 1   # don't mark as seen; retry next run
    stories = [s for s in stories if s["importance"] >= MIN_IMPORTANCE]
    stories.sort(key=lambda s: (-s["importance"], s["ids"][0]))
    stories = stories[:MAX_PER_DIGEST]

    if stories:
        text = format_digest(stories, cands, now)
        print(text) if args.dry_run else send_telegram(text)
    if not args.dry_run:
        state["seen"] = (state.get("seen", []) + [item_id(c) for c in cands])[-3000:]
        STATE_PATH.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Live squawk listener (runs on the always-on server).

Connects to Telegram as YOU (via Telegram's official API, using your api_id/api_hash from
my.telegram.org), joins the squawk channels, and sees every new post within seconds:

  - logs every line to squawk_recent.jsonl (used by econ.py for data actuals)
  - batches posts for a few seconds, removes duplicates across channels and vs what's
    already been sent, and asks Claude which ones matter
  - 🔴 top-tier: pushed to you immediately
  - 🟠 relevant: sent as soon as scored (SQUAWK_INSTANT=1, default) or as a 5-minute digest (=0)
  - 📌 anything mentioning one of your holdings is always included

First run (interactive, once):  python listener.py --login
Then it runs as a service:      python listener.py
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SQUAWK_LOG = HERE / "squawk_recent.jsonl"
LONDON = ZoneInfo("Europe/London")

CHANNELS = [c.strip() for c in os.environ.get(
    "SQUAWK_CHANNELS", "financialjuice,WalterBloomberg,TreeNewsFeed").split(",") if c.strip()]
BATCH_SECONDS = 5
ORANGE_FLUSH_MIN = 5
# 1 = send relevant (🟠) stories as soon as they're scored too; 0 = 5-minute digest
INSTANT = os.environ.get("SQUAWK_INSTANT", "1") == "1"

queue: list[dict] = []
orange: list[dict] = []


# --------------------------------------------------------------------------
def holding_matchers() -> list[tuple[str, list[re.Pattern]]]:
    """Patterns per holding: $TICKER always; bare TICKER (case-sensitive) if 3+ letters;
    plus any names saved with the holding (case-insensitive), e.g. 'Boeing|BA'."""
    try:
        import holdings
        out = []
        for p in holdings.load()["positions"]:
            t = p["ticker"].split(".")[0].upper()
            pats = [re.compile(rf"\${re.escape(t)}\b")]
            if len(t) >= 3:
                pats.append(re.compile(rf"\b{re.escape(t)}\b"))
            for n in str(p.get("name") or "").split("|"):
                n = n.strip()
                if n and n.upper() != t and len(n) >= 3:
                    pats.append(re.compile(rf"\b{re.escape(n)}\b", re.I))
            out.append((p["ticker"], pats))
        return out
    except Exception:
        return []


def mentions_holding(text: str, matchers) -> str | None:
    for t, pats in matchers:
        if any(p.search(text) for p in pats):
            return t
    return None


def fmt_story(s: dict) -> str:
    e = html.escape
    dot = "🔴" if s["importance"] == 3 else "🟠"
    pin = f"📌{e(s['holding'])} " if s.get("holding") else ""
    src = e(s["source"])
    if s.get("url"):
        src = f'<a href="{e(s["url"], quote=True)}">{src}</a>'
    assets = f" <i>[{e(', '.join(s['assets'][:4]))}]</i>" if s.get("assets") else ""
    return (f"{dot} {pin}<b>{e(s['headline'])}</b> ({s['ts'].astimezone(LONDON):%H:%M})\n"
            f"   {e(s['why'])}{assets}\n   {src}")


def log_squawk(item: dict) -> None:
    with SQUAWK_LOG.open("a") as f:
        f.write(json.dumps({"ts": item["ts"].isoformat(), "source": item["source"],
                            "text": item["headline"], "url": item["url"]}) + "\n")


def prune_log() -> None:
    if not SQUAWK_LOG.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    keep = []
    for line in SQUAWK_LOG.read_text().splitlines():
        try:
            if datetime.fromisoformat(json.loads(line)["ts"]) >= cutoff:
                keep.append(line)
        except Exception:
            continue
    SQUAWK_LOG.write_text("\n".join(keep) + ("\n" if keep else ""))


# --------------------------------------------------------------------------
def process_batch(batch: list[dict]) -> None:
    """Dedupe -> Claude triage -> send reds now, park oranges. Runs in a worker thread."""
    import news
    import sent_store
    from scanner import send_telegram

    cands = [c for c in news.dedupe(batch) if not sent_store.is_dup(c["headline"])]
    if not cands:
        return
    matchers = holding_matchers()
    try:
        stories = news.triage(cands)
    except Exception as e:
        print(f"triage failed: {e}", file=sys.stderr, flush=True)
        stories = []
    picked = set()
    out = []
    for s in stories:
        first = cands[s["ids"][0] - 1]
        picked.update(s["ids"])
        out.append({**s, "ts": first["ts"], "source": ", ".join(first["sources"][:3]), "url": first["url"],
                    "holding": mentions_holding(" ".join(cands[i - 1]["headline"] for i in s["ids"]), matchers),
                    "originals": [cands[i - 1]["headline"] for i in s["ids"]]})
    # anything about your holdings that Claude skipped still gets through as orange
    for i, c in enumerate(cands, 1):
        h = mentions_holding(c["headline"], matchers)
        if h and i not in picked:
            out.append({"headline": c["headline"], "importance": 2, "why": "Mentions your holding.",
                        "assets": [h], "ts": c["ts"], "source": c["source"], "url": c["url"],
                        "holding": h, "originals": [c["headline"]]})
    for s in out:
        if s["importance"] >= 3:
            send_telegram(fmt_story(s))
            sent_store.add([s["headline"]] + s["originals"])
        elif s["importance"] >= news.MIN_IMPORTANCE or s.get("holding"):
            if INSTANT:
                send_telegram(fmt_story(s))
                sent_store.add([s["headline"]] + s["originals"])
            else:
                orange.append(s)
    # everything we looked at counts as seen, so the polled feed doesn't re-send it
    sent_store.add([c["headline"] for i, c in enumerate(cands, 1) if i not in picked])


def flush_orange() -> None:
    import sent_store
    from scanner import send_telegram
    if not orange:
        return
    items = sorted(orange, key=lambda s: s["ts"])
    orange.clear()
    text = f"📰 <b>Squawk</b> · {datetime.now(LONDON):%H:%M} UK\n\n" + "\n\n".join(fmt_story(s) for s in items[:15])
    send_telegram(text)
    sent_store.add([h for s in items for h in [s["headline"]] + s["originals"]])


async def batch_loop() -> None:
    while True:
        await asyncio.sleep(BATCH_SECONDS)
        if queue:
            batch = queue[:]
            queue.clear()
            try:
                await asyncio.to_thread(process_batch, batch)
            except Exception as e:
                print(f"batch error: {e}", file=sys.stderr, flush=True)


async def flush_loop() -> None:
    n = 0
    while True:
        await asyncio.sleep(ORANGE_FLUSH_MIN * 60)
        try:
            await asyncio.to_thread(flush_orange)
        except Exception as e:
            print(f"flush error: {e}", file=sys.stderr, flush=True)
        n += 1
        if n % 12 == 0:
            await asyncio.to_thread(prune_log)


# --------------------------------------------------------------------------
async def run(login_only: bool) -> None:
    from telethon import TelegramClient, events
    from telethon.tl.functions.channels import JoinChannelRequest

    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        sys.exit("Set TG_API_ID and TG_API_HASH in .env (from my.telegram.org).")
    client = TelegramClient(str(HERE / "squawk"), int(api_id), api_hash)
    if not login_only:
        # As a service there is no keyboard: if the saved login is gone, say so ONCE and wait,
        # instead of crash-looping on the phone-number prompt.
        await client.connect()
        if not await client.is_user_authorized():
            await client.disconnect()
            flag = HERE / ".listener_logged_out"
            if not flag.exists():
                flag.write_text(datetime.now(timezone.utc).isoformat())
                try:
                    from scanner import send_telegram
                    send_telegram("⚠️ <b>Live squawk listener is logged out of Telegram</b> — instant headlines are "
                                  "paused (the 5-min news feed still runs). To fix, in the server console run:\n"
                                  "<code>sudo systemctl stop move-alerts-listener</code>\n"
                                  "<code>cd ~/move-alerts &amp;&amp; set -a &amp;&amp; . ./.env &amp;&amp; set +a &amp;&amp; "
                                  ".venv/bin/python listener.py --login</code>\n"
                                  "<code>sudo systemctl start move-alerts-listener</code>", force=True)
                except Exception:
                    pass
            print("Not logged in — run: python listener.py --login. Waiting 1h.", file=sys.stderr, flush=True)
            await asyncio.sleep(3600)
            sys.exit(1)
        (HERE / ".listener_logged_out").unlink(missing_ok=True)
    await client.start()   # first time: asks for your phone number and the login code
    me = await client.get_me()
    print(f"Logged in as {me.first_name}.", flush=True)
    if login_only:
        await client.disconnect()
        return

    chats = []
    for ch in CHANNELS:
        try:
            ent = await client.get_entity(ch)
            await client(JoinChannelRequest(ent))
            chats.append(ent)
            print(f"Listening to {getattr(ent, 'title', ch)}", flush=True)
        except Exception as e:
            print(f"Couldn't join {ch}: {e}", file=sys.stderr, flush=True)

    if not chats:
        sys.exit("Couldn't join any squawk channel — check SQUAWK_CHANNELS in .env.")

    @client.on(events.NewMessage(chats=chats))
    async def on_message(ev):
        text = (ev.raw_text or "").strip()
        if not text or len(text) < 8:
            return
        chat = await ev.get_chat()
        username = getattr(chat, "username", None)
        item = {"ts": ev.date.astimezone(timezone.utc), "source": getattr(chat, "title", username or "Squawk"),
                "headline": re.sub(r"\s+", " ", text)[:300],
                "url": f"https://t.me/{username}/{ev.id}" if username else "", "feed": "listener"}
        log_squawk(item)
        queue.append(item)

    asyncio.create_task(batch_loop())
    asyncio.create_task(flush_loop())
    print("Live.", flush=True)
    await client.run_until_disconnected()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true", help="log in once, interactively, then exit")
    args = ap.parse_args()
    asyncio.run(run(args.login))


if __name__ == "__main__":
    main()

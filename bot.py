"""
Telegram commands for your bot (runs on the always-on server).

Only answers messages from your own chat (TELEGRAM_CHAT_ID, or TELEGRAM_OWNER_ID
if your alerts go to a channel). Commands:

  /help                         this list
  /why TICKER                   explain a stock's (or ETF's) move right now
  /brief                        the morning brief, now
  /levels                       current market levels
  /snapshot                     30-min style update across all 48 macro markets, now
  /holdings                     your positions, P&L and PA earliest-sale dates
  /add TICKER QTY @ PRICE [YYYY-MM-DD]   add or top up a position
  /remove TICKER                remove a position
  /holdperiod DAYS              set your PA minimum holding period
  /scorecard                    how the fade calls have done
  /cot                          latest CFTC positioning
  /week                         week-ahead preview
  /mute 2h | 30m                pause alerts       /unmute   resume
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from scanner import MUTE_PATH, send_telegram

HERE = Path(__file__).resolve().parent
STATE = HERE / "bot_state.json"


def owner_id() -> str:
    return os.environ.get("TELEGRAM_OWNER_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")


def reply(chat_id, text: str) -> None:
    send_telegram(text, force=True, chat_id=str(chat_id))


# --------------------------------------------------------------------------
# Command handlers (each returns HTML text)
# --------------------------------------------------------------------------
def cmd_help(_args: str) -> str:
    return "<pre>" + html.escape(__doc__.split("Commands:")[1].strip()) + "</pre>"


def cmd_mute(args: str) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*([mh]?)\s*", args or "1h")
    if not m:
        return "Usage: /mute 2h or /mute 30m"
    n, unit = int(m.group(1)), m.group(2) or "h"
    until = datetime.now(timezone.utc) + (timedelta(hours=n) if unit == "h" else timedelta(minutes=n))
    MUTE_PATH.write_text(json.dumps({"until": until.isoformat()}))
    from zoneinfo import ZoneInfo
    return f"🔕 Muted until {until.astimezone(ZoneInfo('Europe/London')):%H:%M} UK. /unmute to resume."


def cmd_unmute(_args: str) -> str:
    MUTE_PATH.unlink(missing_ok=True)
    return "🔔 Alerts back on."


def cmd_add(args: str) -> str:
    m = re.fullmatch(r"\s*([A-Za-z0-9.\-=]+)\s+([\d.]+)\s*@\s*([\d.]+)\s*(\d{4}-\d{2}-\d{2})?\s*", args or "")
    if not m:
        return "Usage: /add TICKER QTY @ PRICE [YYYY-MM-DD]\ne.g. /add AAPL 10 @ 225"
    import holdings
    return html.escape(holdings.add(m.group(1), float(m.group(2)), float(m.group(3)), m.group(4)))


def cmd_remove(args: str) -> str:
    if not args.strip():
        return "Usage: /remove TICKER"
    import holdings
    return html.escape(holdings.remove(args.strip()))


def cmd_holdperiod(args: str) -> str:
    if not args.strip().isdigit():
        return "Usage: /holdperiod 30"
    import holdings
    return html.escape(holdings.set_hold_days(int(args.strip())))


def cmd_holdings(_args: str) -> str:
    import holdings
    lines = holdings.summary_lines()
    pa = holdings.pa_lines()
    earn = holdings.upcoming_earnings()
    out = "<b>Holdings</b>\n<pre>" + html.escape("\n".join(lines)) + "</pre>"
    if pa:
        out += "\n<b>PA holding period</b>\n" + html.escape("\n".join(pa))
    if earn:
        out += "\n<b>Earnings next 7 days</b>\n" + html.escape("\n".join(earn))
    return out


def cmd_why(args: str) -> str:
    t = args.strip().upper()
    if not t:
        return "Usage: /why TICKER (US ticker, or .L for London e.g. VUSA.L)"
    import holdings
    snap = holdings.snapshot([t]).get(t)
    if not snap:
        return f"Couldn't get a price for {html.escape(t)}."
    heads = holdings.holding_headlines(t, t)
    a = holdings.explain_holding(t, t, snap, heads)
    e = html.escape
    lines = [f"<b>{e(t)}</b> {snap['move']*100:+.2f}% vs prev close · last {snap['last']:,.2f} {e(snap['currency'])}"]
    if a:
        lines += [f"<b>Catalyst:</b> {e(a['catalyst'])}",
                  f"<b>Driver:</b> {e(a['driver'].replace('_', ' '))} · <b>Nature:</b> {e(a['move_nature'])} "
                  f"→ <b>{e(a['verdict'].replace('_', ' ').upper())}</b>",
                  f"<b>Why:</b> {e(a['rationale'])}"]
        for i in a["evidence_ids"][:4]:
            h = heads[i - 1]
            src = f'<a href="{e(h["url"], quote=True)}">{e(h["source"])}</a>' if h.get("url") else e(h["source"])
            lines.append(f"[{i}] {src}: {e(h['headline'][:150])}")
    return "\n".join(lines)


def cmd_brief(_args: str) -> list[str]:
    import daily
    return daily.build_messages(datetime.now(timezone.utc))


def cmd_levels(_args: str) -> str:
    import daily
    return "<pre>" + html.escape("\n".join(daily.levels(datetime.now(timezone.utc)))) + "</pre>"


def cmd_scorecard(_args: str) -> str:
    import followups
    return followups.scorecard_text()


def cmd_cot(_args: str) -> str:
    import cot
    return cot.build_message()


def cmd_snapshot(_args: str) -> str:
    import snapshot
    return snapshot.build_message(force=True) or "No market data right now."


def cmd_week(_args: str) -> list[str]:
    import weekly
    return weekly.build_messages()


COMMANDS = {
    "help": cmd_help, "start": cmd_help, "mute": cmd_mute, "unmute": cmd_unmute,
    "add": cmd_add, "remove": cmd_remove, "holdperiod": cmd_holdperiod, "holdings": cmd_holdings,
    "why": cmd_why, "brief": cmd_brief, "levels": cmd_levels, "scorecard": cmd_scorecard,
    "cot": cmd_cot, "week": cmd_week, "snapshot": cmd_snapshot, "markets": cmd_snapshot,
}
SLOW = {"why", "brief", "holdings", "scorecard", "cot", "week", "levels", "snapshot", "markets"}


def handle(text: str) -> list[str]:
    m = re.match(r"^/(\w+)(?:@\w+)?\s*(.*)$", text.strip(), re.S)
    if not m:
        return ["Unknown command. /help for the list."]
    fn = COMMANDS.get(m.group(1).lower())
    if not fn:
        return ["Unknown command. /help for the list."]
    out = fn(m.group(2))
    return out if isinstance(out, list) else [out]


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    api = f"https://api.telegram.org/bot{token}"
    try:
        offset = json.loads(STATE.read_text())["offset"]
    except Exception:
        offset = 0
    print("bot.py listening for commands…", flush=True)
    while True:
        try:
            r = requests.get(f"{api}/getUpdates", params={"timeout": 50, "offset": offset}, timeout=60)
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                STATE.write_text(json.dumps({"offset": offset}))
                msg = upd.get("message") or {}
                chat, sender = msg.get("chat", {}), msg.get("from", {})
                text = msg.get("text", "")
                if not text.startswith("/"):
                    continue
                if str(chat.get("id")) != owner_id() and str(sender.get("id")) != owner_id():
                    continue  # ignore strangers
                cmd = text.split()[0].lstrip("/").split("@")[0].lower()
                if cmd in SLOW:
                    reply(chat["id"], "⏳ On it…")
                try:
                    for part in handle(text):
                        reply(chat["id"], part)
                except Exception as e:
                    traceback.print_exc()
                    reply(chat["id"], f"⚠️ {html.escape(e.__class__.__name__)}: {html.escape(str(e)[:200])}")
        except Exception as e:
            print(f"poll error: {e}", file=sys.stderr, flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()

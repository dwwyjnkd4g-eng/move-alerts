"""
Always-on scheduler for the server. Runs every job on exact timings, each in its own
thread so a slow job never delays the others, and records results in runner_status.json
(shown in the morning brief's health check).

  every 2 min       econ.py (data heads-ups / results)
  every 5 min       news.py (polled web sources; squawk channels are live via listener.py),
                    predict.py (Polymarket/Kalshi odds repricings on macro & geopolitical events)
  every 15 min      macro.py, scanner.py (it picks pre-market / live / close / after-hours),
                    europe.py (UK & European stocks, London hours),
                    holdings.py --watch, paper.py --mark (paper-trading ledger),
                    predict_edge.py (prediction-market mispricings vs futures / other venue / themselves),
                    watchdog (listener/bot services, disk space)
  every 30 min      divergence.py, snapshot.py (market update across the 48 macro markets)
  07:00 UK daily    daily.py (morning brief)
  07:30 UK daily    scanner.py --refresh (whole US market list + liquidity screen + history)
  21:20 UK Mon-Fri  followups.py (fade follow-ups)
  21:30 UK daily    paper.py --report (paper book)
  03:30 UK daily    housekeeping (trim logs and old cache files)
  Sun 02:00 / 04:00 backtest.py US / Europe (refreshes the backtest prior the analyst sees)
  any job failing 3 times in a row -> one Telegram warning, and a note when it recovers
  Fri 21:00 - Sat   cot.py hourly (sends once per new CFTC report)
  Sun 18:00 UK      weekly.py (week-ahead + scorecard)

The live squawk listener (listener.py) and Telegram commands (bot.py) run as separate services.
"""

import html
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
LONDON = ZoneInfo("Europe/London")
STATUS = HERE / "runner_status.json"
PY = sys.executable
TIMEOUTS = {"predict.py": 240, "predict_edge.py": 300, "backtest-us": 7200, "backtest-eu": 7200, "paper.py": 600, "scanner.py": 1800, "scanner-refresh": 3600, "europe.py": 1800, "europe-refresh": 2400,
            "weekly.py": 1800, "cot.py": 600, "macro.py": 600, "snapshot.py": 600, "divergence.py": 600,
            "holdings.py": 600, "daily.py": 900}

_running: set[str] = set()
_lock = threading.Lock()


def tg(text: str) -> None:
    """Minimal Telegram send (keeps pandas out of the long-running scheduler)."""
    import os
    import urllib.request
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        body = json.dumps({"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print(f"telegram warning failed: {e}", flush=True)


FAIL_ALERT_AFTER = 3


def record(name: str, ok: bool, detail: str = "") -> None:
    with _lock:
        try:
            st = json.loads(STATUS.read_text())
        except Exception:
            st = {}
        now = datetime.now(timezone.utc)
        info = st.get(name, {"fails": []})
        info["last"] = now.isoformat(timespec="seconds")
        fails = [f for f in info.get("fails", []) if datetime.fromisoformat(f) > now - timedelta(hours=24)]
        if not ok:
            fails.append(now.isoformat(timespec="seconds"))
        info["fails"], info["fails_24h"] = fails, len(fails)
        streak = 0 if ok else info.get("streak", 0) + 1
        if streak == FAIL_ALERT_AFTER:
            tg(f"⚠️ <b>{name}</b> has failed {streak} times in a row.\n<code>{html.escape(detail[-300:])}</code>\n"
               "The rest of the bot keeps running. Check with: journalctl -u move-alerts-runner -n 50")
        if ok and info.get("streak", 0) >= FAIL_ALERT_AFTER:
            tg(f"✅ <b>{name}</b> is working again.")
        info["streak"] = streak
        st[name] = info
        STATUS.write_text(json.dumps(st, indent=1))


def _run(name: str, args: list[str]) -> None:
    t0 = time.time()
    script = args[0]
    try:
        p = subprocess.run([PY, str(HERE / script), *args[1:]], cwd=HERE, capture_output=True, text=True,
                           timeout=TIMEOUTS.get(name, TIMEOUTS.get(script, 300)))
        ok = p.returncode == 0
        noise = ("Warning", "warn(", "See https://pandas", "inplace=True", "FutureWarning", "DeprecationWarning")
        lines = [l for l in (p.stdout + p.stderr).strip().splitlines() if l.strip() and not any(n in l for n in noise)]
        key = [l for l in lines if any(k in l for k in ("Flagged", "failed", "Error", "error", "Traceback", "liquid US", "universe:"))]
        tail = (key[-3:] if key else lines[-2:])
        print(f"{datetime.now(LONDON):%d %b %H:%M:%S} {name} {'ok' if ok else 'FAIL'} "
              f"{time.time()-t0:.0f}s | " + " / ".join(tail), flush=True)
        detail = " / ".join(tail)
    except subprocess.TimeoutExpired:
        ok = False
        detail = "timed out"
        print(f"{datetime.now(LONDON):%H:%M} {name} TIMED OUT", flush=True)
    record(name, ok, detail)
    with _lock:
        _running.discard(name)


def launch(name: str, *args: str) -> None:
    """Start a job in the background unless the previous run is still going."""
    with _lock:
        if name in _running:
            return
        _running.add(name)
    threading.Thread(target=_run, args=(name, list(args) or [name]), daemon=True).start()


_down: set[str] = set()


def watchdog() -> None:
    """Warn once if the listener or bot service stops, or the disk is nearly full."""
    import shutil
    for svc in ("move-alerts-listener", "move-alerts-bot"):
        try:
            state = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            continue
        if state != "active" and svc not in _down:
            _down.add(svc)
            tg(f"⚠️ Service <b>{svc}</b> is {state}. systemd will try to restart it; if this repeats run:\n"
               f"<code>sudo systemctl restart {svc}</code>")
        elif state == "active" and svc in _down:
            _down.discard(svc)
            tg(f"✅ Service <b>{svc}</b> is running again.")
    du = shutil.disk_usage("/")
    if du.used / du.total > 0.88 and "disk" not in _down:
        _down.add("disk")
        tg(f"⚠️ Server disk {du.used/du.total*100:.0f}% full. Housekeeping runs at 03:30; "
           "if it stays high, run: journalctl --vacuum-size=100M")
    elif du.used / du.total < 0.8:
        _down.discard("disk")


def housekeeping() -> None:
    """Keep the 1GB server tidy: cap the system journal, trim growing files, drop stale cache."""
    try:
        subprocess.run(["journalctl", "--vacuum-size=200M"], capture_output=True, timeout=120)
    except Exception:
        pass
    for f, keep in (("squawk_recent.jsonl", 5000), ("sent_headlines.jsonl", 20000)):
        p = HERE / f
        try:
            if p.exists():
                lines = p.read_text().splitlines()
                if len(lines) > keep:
                    p.write_text("\n".join(lines[-keep:]) + "\n")
        except Exception:
            pass
    cutoff = time.time() - 30 * 86400
    for p in (HERE / "cache").glob("*"):
        try:
            age = time.time() - p.stat().st_mtime
            if p.is_file() and ((p.suffix == ".log" and p.stat().st_mtime < cutoff)
                                or (p.name.startswith("bt_") and p.suffix == ".pkl" and age > 2 * 86400)):
                p.unlink()
        except Exception:
            pass
    print(f"{datetime.now(LONDON):%d %b %H:%M} housekeeping done", flush=True)


def main() -> None:
    last: dict[str, object] = {}
    print("run_forever started", flush=True)
    while True:
        now = datetime.now(LONDON)
        slot2 = (now.date(), now.hour, now.minute // 2)
        slot5 = (now.date(), now.hour, now.minute // 5)
        slot15 = (now.date(), now.hour, now.minute // 15)
        slot30 = (now.date(), now.hour, now.minute // 30)
        if last.get("2") != slot2:
            last["2"] = slot2
            launch("econ.py")
        if last.get("5") != slot5:          # squawk channels are live via listener.py; this polls the rest
            last["5"] = slot5
            launch("news.py")
            launch("predict.py")
        if last.get("15") != slot15:
            last["15"] = slot15
            launch("macro.py")
            launch("scanner.py")
            launch("europe.py")
            launch("holdings.py", "holdings.py", "--watch")
            launch("paper.py", "paper.py", "--mark")
            launch("predict_edge.py")
            threading.Thread(target=watchdog, daemon=True).start()
        if last.get("30") != slot30:
            last["30"] = slot30
            launch("divergence.py")
            launch("snapshot.py")
        if (now.hour, now.minute) >= (7, 30) and last.get("refresh") != now.date():
            last["refresh"] = now.date()      # whole-US-market universe + history, before US pre-market
            launch("scanner-refresh", "scanner.py", "--refresh")
        if (now.hour, now.minute) >= (6, 45) and last.get("eu-refresh") != now.date():
            last["eu-refresh"] = now.date()   # UK/Europe universe + history, before London opens
            launch("europe-refresh", "europe.py", "--refresh")
        if now.hour == 7 and last.get("daily") != now.date():
            last["daily"] = now.date()
            launch("daily.py", "daily.py", "--force")
        if now.weekday() < 5 and (now.hour, now.minute) >= (21, 20) and last.get("follow") != now.date():
            last["follow"] = now.date()
            launch("followups.py")
        cot_window = (now.weekday() == 4 and now.hour >= 21) or now.weekday() == 5
        if cot_window and last.get("cot") != (now.date(), now.hour):
            last["cot"] = (now.date(), now.hour)
            launch("cot.py")
        if (now.hour, now.minute) >= (21, 30) and last.get("paper") != now.date():
            last["paper"] = now.date()
            launch("paper-report", "paper.py", "--report")
        if now.hour == 3 and now.minute >= 30 and last.get("house") != now.date():
            last["house"] = now.date()
            threading.Thread(target=housekeeping, daemon=True).start()
        if now.weekday() == 6 and now.hour == 2 and last.get("bt-us") != now.date():
            last["bt-us"] = now.date()
            launch("backtest-us", "backtest.py", "--market", "us", "--years", "3", "--telegram")
        if now.weekday() == 6 and now.hour == 4 and last.get("bt-eu") != now.date():
            last["bt-eu"] = now.date()
            launch("backtest-eu", "backtest.py", "--market", "eu", "--years", "3", "--telegram")
        if now.weekday() == 6 and now.hour == 18 and last.get("weekly") != now.date():
            last["weekly"] = now.date()
            launch("weekly.py")
        time.sleep(15)


if __name__ == "__main__":
    main()

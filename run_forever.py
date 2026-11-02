"""
Always-on scheduler for the server. Runs every job on exact timings, each in its own
thread so a slow job never delays the others, and records results in runner_status.json
(shown in the morning brief's health check).

  every 2 min       econ.py (data heads-ups / results)
  every 5 min       news.py (polled web sources; squawk channels are live via listener.py)
  every 15 min      macro.py, scanner.py (it picks pre-market / live / close / after-hours),
                    europe.py (UK & European stocks, London hours),
                    holdings.py --watch
  every 30 min      divergence.py, snapshot.py (market update across the 48 macro markets)
  07:00 UK daily    daily.py (morning brief)
  07:30 UK daily    scanner.py --refresh (whole US market list + liquidity screen + history)
  21:20 UK Mon-Fri  followups.py (fade follow-ups)
  Fri 21:00 - Sat   cot.py hourly (sends once per new CFTC report)
  Sun 18:00 UK      weekly.py (week-ahead + scorecard)

The live squawk listener (listener.py) and Telegram commands (bot.py) run as separate services.
"""

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
TIMEOUTS = {"scanner.py": 1800, "scanner-refresh": 3600, "europe.py": 1800, "europe-refresh": 2400,
            "weekly.py": 1800, "cot.py": 600, "macro.py": 600, "snapshot.py": 600, "divergence.py": 600,
            "holdings.py": 600, "daily.py": 900}

_running: set[str] = set()
_lock = threading.Lock()


def record(name: str, ok: bool) -> None:
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
    except subprocess.TimeoutExpired:
        ok = False
        print(f"{datetime.now(LONDON):%H:%M} {name} TIMED OUT", flush=True)
    record(name, ok)
    with _lock:
        _running.discard(name)


def launch(name: str, *args: str) -> None:
    """Start a job in the background unless the previous run is still going."""
    with _lock:
        if name in _running:
            return
        _running.add(name)
    threading.Thread(target=_run, args=(name, list(args) or [name]), daemon=True).start()


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
        if last.get("15") != slot15:
            last["15"] = slot15
            launch("macro.py")
            launch("scanner.py")
            launch("europe.py")
            launch("holdings.py", "holdings.py", "--watch")
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
        if now.weekday() == 6 and now.hour == 18 and last.get("weekly") != now.date():
            last["weekly"] = now.date()
            launch("weekly.py")
        time.sleep(15)


if __name__ == "__main__":
    main()

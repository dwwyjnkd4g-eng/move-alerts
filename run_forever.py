"""
Always-on runner for a small server (Option 2). Replaces the GitHub schedules
with exact timing:

  news.py     every 2 minutes
  macro.py    every 15 minutes (on :00, :15, :30, :45)
  scanner.py  every 15 minutes (it decides pre-market / live / close / after-hours itself)
  daily.py    once a day at 07:00 UK

Run with:  python run_forever.py
(Keys go in a .env file or the environment, same names as the GitHub secrets.)
If you use this, disable the three GitHub workflows so you don't get duplicates.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
LONDON = ZoneInfo("Europe/London")
PY = sys.executable


def load_env() -> None:
    import os
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def run(script: str, *args: str, timeout: int = 600) -> None:
    t0 = time.time()
    try:
        p = subprocess.run([PY, str(HERE / script), *args], cwd=HERE, timeout=timeout,
                           capture_output=True, text=True)
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        print(f"{datetime.now(LONDON):%d %b %H:%M:%S} {script} exit={p.returncode} "
              f"{time.time()-t0:.0f}s | " + " / ".join(tail), flush=True)
    except subprocess.TimeoutExpired:
        print(f"{datetime.now(LONDON):%H:%M} {script} TIMED OUT", flush=True)


def main() -> None:
    load_env()
    last_news = 0.0
    last_slot = None
    last_daily = None
    while True:
        now = datetime.now(LONDON)
        if time.time() - last_news >= 120:
            last_news = time.time()
            run("news.py")
        slot = (now.date(), now.hour, now.minute // 15)
        if slot != last_slot:
            last_slot = slot
            run("macro.py")
            run("scanner.py")
        if now.hour == 7 and last_daily != now.date():
            last_daily = now.date()
            run("daily.py", "--force")
        time.sleep(20)


if __name__ == "__main__":
    main()

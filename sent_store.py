"""Shared memory of headlines already sent, so the live listener and the polled news feed
don't send you the same story twice."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "sent_headlines.jsonl"


def _recent(hours: float) -> list[str]:
    if not PATH.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    for line in PATH.read_text().splitlines()[-2000:]:
        try:
            r = json.loads(line)
            if datetime.fromisoformat(r["ts"]) >= cutoff:
                out.append(r["h"])
        except Exception:
            continue
    return out


def is_dup(headline: str, hours: float = 6) -> bool:
    from news import _norm, _similar
    n = _norm(headline)
    return any(n == _norm(h) or _similar(h, headline) for h in _recent(hours))


def add(headlines: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with PATH.open("a") as f:
        for h in headlines:
            f.write(json.dumps({"ts": now, "h": h}) + "\n")
    lines = PATH.read_text().splitlines()
    if len(lines) > 5000:
        PATH.write_text("\n".join(lines[-3000:]) + "\n")

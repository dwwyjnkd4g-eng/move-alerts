"""
30-minute market snapshot across the 48 macro markets: last price, 30-minute and 24-hour
change for every market that's trading, grouped by asset class, plus a short Claude summary
of what's moving and why (from the latest headlines).

Skips when fewer than MIN_LIVE markets are trading (i.e. most of the weekend) and during
quiet hours (SNAPSHOT_QUIET, default 23-06 UK; set SNAPSHOT_QUIET= to disable).

  python snapshot.py [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import math
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

LONDON = ZoneInfo("Europe/London")
MIN_LIVE = 5

GROUPS = [
    ("USD & FX", ["DXY (USD index)", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "USDCNH", "USDMXN"]),
    ("Precious metals (OTC spot)", ["Gold (XAU/USD)", "Gold (PAXG token, 24/7 proxy)", "Silver (XAG/USD)",
                                    "Platinum (XPT/USD)", "Palladium (XPD/USD)"]),
    ("Crypto", ["Bitcoin"]),
    ("Index futures", ["S&P 500 futures", "Nasdaq 100 futures", "Russell 2000 futures", "Dow futures", "Nikkei futures (CME)"]),
    ("Global indices & vol", ["FTSE 100", "DAX", "Euro Stoxx 50", "Hang Seng", "Shanghai Composite", "VIX"]),
    ("Rates", ["US 10y note futures", "US 10y yield"]),
    ("Energy & metals", ["WTI crude", "Brent crude", "Natural gas", "Heating oil", "RBOB gasoline", "Dutch TTF gas", "Copper", "Aluminium"]),
    ("Ags & livestock", ["Wheat", "Corn", "Soybeans", "Coffee", "Cocoa", "Sugar", "Cotton", "Live cattle", "Lean hogs"]),
    ("Freight, uranium, lithium", ["Dry bulk freight (BDRY)", "Dry bulk shippers (Oslo)", "Tanker freight (BWET)",
                                   "Tanker shippers (Oslo)", "Uranium (URA)", "Uranium (Yellow Cake, London)",
                                   "Lithium & battery (LIT)", "Lithium miners (ASX)"]),
]
SHORT = {"DXY (USD index)": "DXY", "Gold (XAU/USD)": "XAU", "Silver (XAG/USD)": "XAG", "Platinum (XPT/USD)": "XPT",
         "Palladium (XPD/USD)": "XPD", "Gold (PAXG token, 24/7 proxy)": "Gold (PAXG)", "S&P 500 futures": "S&P fut",
         "Nasdaq 100 futures": "Nasdaq fut", "Russell 2000 futures": "Russell fut", "Dow futures": "Dow fut",
         "Nikkei futures (CME)": "Nikkei fut", "Shanghai Composite": "Shanghai", "US 10y note futures": "10y note fut",
         "US 10y yield": "10y yield %", "Dry bulk freight (BDRY)": "BDRY dry bulk", "Tanker freight (BWET)": "BWET tankers",
         "Uranium (URA)": "URA uranium", "Lithium & battery (LIT)": "LIT lithium",
         "Dry bulk shippers (Oslo)": "Bulk shippers", "Tanker shippers (Oslo)": "Tanker shpprs",
         "Uranium (Yellow Cake, London)": "Yellow Cake", "Lithium miners (ASX)": "Li miners ASX", "Euro Stoxx 50": "Stoxx 50",
         "Live cattle": "Cattle", "Lean hogs": "Hogs", "RBOB gasoline": "Gasoline", "Dutch TTF gas": "TTF gas",
         "Natural gas": "Nat gas", "Heating oil": "Heating oil"}


def quiet_now(now: datetime) -> bool:
    spec = os.environ.get("SNAPSHOT_QUIET", "23-06")
    if not spec:
        return False
    try:
        a, b = (int(x) for x in spec.split("-"))
    except ValueError:
        return False
    h = now.hour
    return (a <= h or h < b) if a > b else (a <= h < b)


def fmt_px(p: float) -> str:
    if p >= 10000:
        return f"{p:,.0f}"
    if p >= 100:
        return f"{p:,.1f}"
    if p >= 10:
        return f"{p:.2f}"
    return f"{p:.4f}"


def collect() -> dict[str, dict]:
    """name -> {last, chg30, chg24, z30, live}"""
    from macro import INSTRUMENTS, PAXG, is_fresh, load_all_series
    now = datetime.now(timezone.utc)
    series = load_all_series()
    out = {}
    for name, _, sym, _ in INSTRUMENTS + [PAXG]:
        s = series.get(sym)
        if s is None or len(s) < 30:
            continue
        live = is_fresh(s, now)
        last = float(s.iloc[-1])
        chg30 = float(s.iloc[-1] / s.iloc[-3] - 1) if live else 0.0
        prior = s[s.index <= s.index[-1] - np.timedelta64(24, "h")]
        chg24 = float(last / prior.iloc[-1] - 1) if len(prior) else float("nan")
        lr = np.log(s).diff().dropna()
        lr = lr[lr != 0].iloc[-400:]
        vol30 = float(lr.std() * math.sqrt(2)) if len(lr) > 50 else float("nan")
        out[name] = {"last": last, "chg30": chg30, "chg24": chg24, "live": live,
                     "z30": chg30 / vol30 if vol30 and not math.isnan(vol30) and live else 0.0}
    # PAXG only matters when the OTC gold market is shut (weekends)
    if out.get("Gold (XAU/USD)", {}).get("live"):
        out.pop("Gold (PAXG token, 24/7 proxy)", None)
    return out


def table(data: dict[str, dict]) -> list[str]:
    e = html.escape
    lines = []
    for title, names in GROUPS:
        rows = []
        for n in names:
            d = data.get(n)
            if not d:
                continue
            label = SHORT.get(n, n)[:13]
            if d["live"]:
                flag = "⚡" if abs(d["z30"]) >= 2 else "  "
                rows.append(f"{label:<13}{fmt_px(d['last']):>10} {d['chg30']*100:+6.2f}% {d['chg24']*100:+6.2f}%{flag}")
            else:
                rows.append(f"{label:<13}{fmt_px(d['last']):>10}  closed {d['chg24']*100:+6.2f}%")
        if rows:
            lines += [f"<b>{e(title)}</b>", "<pre>" + e("\n".join(rows)) + "</pre>"]
    return lines


def summary(data: dict[str, dict]) -> list[str]:
    try:
        import anthropic
        from macro import get_headlines, headlines_block
        from news import NEWS_MODEL
        live = {n: d for n, d in data.items() if d["live"]}
        movers = sorted(live.items(), key=lambda kv: -abs(kv[1]["z30"]))[:12]
        mv = "\n".join(f"{n}: 30m {d['chg30']*100:+.2f}% ({d['z30']:+.1f}σ), 24h {d['chg24']*100:+.2f}%" for n, d in movers)
        heads = get_headlines(hours=1, limit=40)
        msg = anthropic.Anthropic().messages.create(
            model=NEWS_MODEL, max_tokens=400,
            messages=[{"role": "user", "content":
                       "You write a 30-minute market update for a UK macro trader. Using ONLY the data and "
                       "headlines below, write 2-4 short bullets (each starting '• '): the notable moves in the "
                       "last 30 minutes / 24 hours and the likely reason if a headline supports it (say 'no clear "
                       "catalyst' otherwise). Mention cross-asset links where the numbers show them. If it's quiet, "
                       "one bullet saying so is fine. No invented facts.\n\nBIGGEST MOVERS (by 30-min sigma)\n"
                       f"{mv}\n\nHEADLINES (last hour)\n{headlines_block(heads)}"}])
        return [l.strip() for l in msg.content[0].text.splitlines() if l.strip().startswith("•")][:4]
    except Exception as ex:
        print(f"summary failed: {ex}", file=sys.stderr)
        return []


def build_message(force: bool = False) -> str | None:
    now = datetime.now(LONDON)
    data = collect()
    n_live = sum(1 for d in data.values() if d["live"])
    if not force and n_live < MIN_LIVE:
        print(f"Only {n_live} markets live — skipping.")
        return None
    e = html.escape
    lines = [f"📊 <b>Markets</b> · {now:%a %H:%M} UK · {n_live} live",
             "<i>last · 30m · 24h   ⚡ = unusual 30-min move · metals = OTC spot</i>", ""]
    bullets = summary(data)
    if bullets:
        lines += [e(b) for b in bullets] + [""]
    lines += table(data)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="send even in quiet hours / when few markets are live")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.force and quiet_now(datetime.now(LONDON)):
        print("Quiet hours — skipping.")
        return 0
    text = build_message(force=args.force)
    if text:
        from scanner import send_telegram
        print(text) if args.dry_run else send_telegram(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

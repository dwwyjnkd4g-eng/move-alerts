"""
Weekly CFTC Commitments of Traders summary (released Fridays 15:30 New York time).

Speculative positioning (managed money for commodities, leveraged funds for financials),
the weekly change, and how extreme it is vs the last 3 years. Crowded positioning + a
sharp move is a classic fade setup.

  python cot.py            send if a new report is out and not yet sent
  python cot.py --force    send the latest now
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
STATE = HERE / "cot_state.json"
API = "https://publicreporting.cftc.gov/resource/{ds}.json"
DISAGG, TFF = "72hh-3qpy", "gpe5-46if"

# label, dataset, CFTC contract code, trader group
MARKETS = [
    ("Gold", DISAGG, "088691", "mm"), ("Silver", DISAGG, "084691", "mm"),
    ("Copper", DISAGG, "085692", "mm"), ("Platinum", DISAGG, "076651", "mm"),
    ("Palladium", DISAGG, "075651", "mm"), ("WTI crude", DISAGG, "067651", "mm"),
    ("Nat gas", DISAGG, "023651", "mm"),
    ("USD index", TFF, "098662", "lev"), ("EUR", TFF, "099741", "lev"), ("JPY", TFF, "097741", "lev"),
    ("GBP", TFF, "096742", "lev"), ("AUD", TFF, "232741", "lev"), ("CAD", TFF, "090741", "lev"),
    ("CHF", TFF, "092741", "lev"), ("MXN", TFF, "095741", "lev"),
    ("S&P 500 e-mini", TFF, "13874A", "lev"), ("Nasdaq e-mini", TFF, "209742", "lev"),
    ("US 10y note", TFF, "043602", "lev"), ("Bitcoin (CME)", TFF, "133741", "lev"),
]
COLS = {"mm": ("m_money_positions_long_all", "m_money_positions_short_all"),
        "lev": ("lev_money_positions_long", "lev_money_positions_short")}


def fetch(ds: str, code: str, weeks: int = 160) -> pd.DataFrame:
    r = requests.get(API.format(ds=ds), params={
        "$where": f"cftc_contract_market_code='{code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": weeks}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def summarise(df: pd.DataFrame, group: str) -> dict | None:
    lc, sc = COLS[group]
    if df.empty or lc not in df or sc not in df:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df = df.sort_values("date")
    net = pd.to_numeric(df[lc], errors="coerce") - pd.to_numeric(df[sc], errors="coerce")
    oi = pd.to_numeric(df.get("open_interest_all"), errors="coerce")
    net = net.dropna()
    if len(net) < 10:
        return None
    pct = float((net < net.iloc[-1]).mean())
    return {"date": df["date"].iloc[-1].date(), "net": float(net.iloc[-1]),
            "chg": float(net.iloc[-1] - net.iloc[-2]), "pctile": pct,
            "net_oi": float(net.iloc[-1] / oi.iloc[-1]) if oi.notna().iloc[-1] and oi.iloc[-1] else None}


def collect() -> list[tuple[str, dict]]:
    out = []
    for label, ds, code, grp in MARKETS:
        try:
            s = summarise(fetch(ds, code), grp)
            if s:
                out.append((label, s))
        except Exception as e:
            print(f"COT {label} failed: {e}", file=sys.stderr)
    return out


def build_message(rows: list[tuple[str, dict]] | None = None) -> str:
    rows = rows if rows is not None else collect()
    if not rows:
        return "COT data unavailable right now."
    date = max(s["date"] for _, s in rows)
    tbl = [f"{'':<15}{'net':>9}{'w/w':>9}{'3y %ile':>8}"]
    flags = []
    for label, s in rows:
        mark = "🔥" if s["pctile"] >= 0.9 else ("🧊" if s["pctile"] <= 0.1 else "  ")
        tbl.append(f"{label:<15}{s['net']:>9,.0f}{s['chg']:>+9,.0f}{s['pctile']*100:>7.0f}% {mark}")
        if s["pctile"] >= 0.9:
            flags.append(f"🔥 {label}: specs near 3-year max long (3y percentile {s['pctile']*100:.0f})")
        elif s["pctile"] <= 0.1:
            flags.append(f"🧊 {label}: specs near 3-year max short (3y percentile {s['pctile']*100:.0f})")
    lines = [f"📋 <b>CFTC positioning</b> · week to {date:%d %b}",
             "<i>Managed money (commodities) / leveraged funds (financials), contracts</i>",
             "<pre>" + html.escape("\n".join(tbl)) + "</pre>"]
    if flags:
        lines += ["<b>Crowded</b>"] + [html.escape(f) for f in flags]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from scanner import send_telegram
    rows = collect()
    if not rows:
        return 1
    latest = max(s["date"] for _, s in rows).isoformat()
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}
    if not args.force and state.get("sent") == latest:
        print(f"COT {latest} already sent.")
        return 0
    text = build_message(rows)
    if args.dry_run:
        print(text)
    else:
        send_telegram(text)
        STATE.write_text(json.dumps({"sent": latest}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Cross-asset divergence alerts: when a market moves very differently from what its usual
driver implies (e.g. gold up hard while the dollar is also up). Each pair's relationship is
estimated from the last 120 trading days; today's residual is scored in sigmas.

Also tracks the gold/silver ratio vs its 3-year range.

  python divergence.py [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATE = HERE / "divergence_state.json"

PAIRS = [  # label, asset, driver
    ("Gold vs USD (DXY)", "GC=F", "DX-Y.NYB"),
    ("Gold vs US 10y note futures", "GC=F", "ZN=F"),
    ("Silver vs Gold", "SI=F", "GC=F"),
    ("Platinum vs Gold", "PL=F", "GC=F"),
    ("Bitcoin vs Nasdaq futures", "BTC-USD", "NQ=F"),
    ("USDJPY vs US 10y note futures", "JPY=X", "ZN=F"),
    ("AUDUSD vs Copper", "AUDUSD=X", "HG=F"),
    ("USDCAD vs WTI crude", "CAD=X", "CL=F"),
    ("Nasdaq vs S&P futures", "NQ=F", "ES=F"),
]
NAMES = {"GC=F": "Gold", "SI=F": "Silver", "PL=F": "Platinum", "DX-Y.NYB": "DXY", "ZN=F": "10y note fut",
         "BTC-USD": "Bitcoin", "NQ=F": "Nasdaq fut", "ES=F": "S&P fut", "JPY=X": "USDJPY",
         "AUDUSD=X": "AUDUSD", "HG=F": "Copper", "CAD=X": "USDCAD", "CL=F": "WTI"}
Z_MIN, CORR_MIN, MIN_RESID = 2.5, 0.3, 0.004
WINDOW = 120


def pair_stats(a: pd.Series, b: pd.Series) -> dict | None:
    """a, b: daily closes whose LAST value is today's live price."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < WINDOW + 5:
        return None
    r = np.log(df).diff().dropna()
    hist, today = r.iloc[-WINDOW - 1:-1], r.iloc[-1]
    beta = hist["a"].cov(hist["b"]) / hist["b"].var()
    corr = hist["a"].corr(hist["b"])
    resid_sd = (hist["a"] - beta * hist["b"]).std()
    resid = today["a"] - beta * today["b"]
    return {"beta": beta, "corr": corr, "z": resid / resid_sd if resid_sd else 0.0,
            "resid": math.expm1(resid), "ret_a": math.expm1(today["a"]), "ret_b": math.expm1(today["b"]),
            "implied_a": math.expm1(beta * today["b"])}


def gold_silver_ratio(gold: pd.Series, silver: pd.Series) -> dict | None:
    df = pd.concat([gold, silver], axis=1, keys=["g", "s"]).dropna()
    if len(df) < 250:
        return None
    ratio = df["g"] / df["s"]
    return {"ratio": float(ratio.iloc[-1]), "pctile": float((ratio < ratio.iloc[-1]).mean())}


def load_prices() -> pd.DataFrame:
    import yfinance as yf
    syms = sorted({s for _, a, b in PAIRS for s in (a, b)})
    px = yf.download(syms, period="3y", interval="1d", auto_adjust=False, progress=False)["Close"]
    px.index = pd.to_datetime(px.index.date)
    return px


def run(dry: bool = False) -> None:
    from scanner import send_telegram
    px = load_prices()
    today = px.index[-1]
    if today.date() < (datetime.now(timezone.utc) - timedelta(days=1)).date():
        print("Stale data (markets closed).")
        return
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}
    day = today.date().isoformat()
    for label, a, b in PAIRS:
        if a not in px or b not in px:
            continue
        # both legs must have a value for today
        if pd.isna(px[a].iloc[-1]) or pd.isna(px[b].iloc[-1]):
            continue
        st = pair_stats(px[a], px[b])
        if not st or abs(st["z"]) < Z_MIN or abs(st["corr"]) < CORR_MIN or abs(st["resid"]) < MIN_RESID:
            continue
        key = f"{label}|{day}"
        if state.get(key):
            continue
        na, nb = NAMES.get(a, a), NAMES.get(b, b)
        e = html.escape
        text = (f"↔️ <b>Divergence: {e(label)}</b>\n"
                f"{e(na)} {st['ret_a']*100:+.2f}% while {e(nb)} {st['ret_b']*100:+.2f}%. "
                f"Usual relationship (β {st['beta']:+.2f}, corr {st['corr']:+.2f}) implies "
                f"{e(na)} {st['implied_a']*100:+.2f}% → gap {st['resid']*100:+.2f}% "
                f"(<b>{st['z']:+.1f}σ</b>).\n"
                f"<i>Relative-value fade idea: {e(na)} {'rich' if st['resid'] > 0 else 'cheap'} vs {e(nb)}. "
                f"Check headlines for a reason the relationship should break.</i>")
        print(text) if dry else send_telegram(text)
        state[key] = True

    gsr = gold_silver_ratio(px.get("GC=F", pd.Series(dtype=float)), px.get("SI=F", pd.Series(dtype=float)))
    if gsr and (gsr["pctile"] >= 0.95 or gsr["pctile"] <= 0.05):
        wk = f"GSR|{today.isocalendar().year}-{today.isocalendar().week}"
        if not state.get(wk):
            side = "high (silver cheap vs gold)" if gsr["pctile"] >= 0.95 else "low (silver rich vs gold)"
            text = (f"⚖️ <b>Gold/silver ratio {gsr['ratio']:.1f}</b> — {side}, "
                    f"3-year percentile {gsr['pctile']*100:.0f}.")
            print(text) if dry else send_telegram(text)
            state[wk] = True
    cutoff = (today - pd.Timedelta(days=10)).date().isoformat()
    state = {k: v for k, v in state.items() if k.startswith("GSR") or k.split("|")[1] >= cutoff}
    if not dry:
        STATE.write_text(json.dumps(state))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    run(dry=ap.parse_args().dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

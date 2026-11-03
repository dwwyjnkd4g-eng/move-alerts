"""
Event-implied pricing: what prediction-market odds mean for asset prices.

For the biggest live macro / political / geopolitical events on Polymarket it measures, from hourly
history, how much each linked asset moves per change in the odds (the event's "beta"), then:

  SCENARIO MAP   For each outcome, the move it would cause from here = beta x (outcome odds -> 100%).
                 Fed decisions use one driver: expected bp of change (sum of outcome bp x odds), so
                 "hold" when the crowd prices 61% cut = +15bp of hawkish surprise x gold's beta per bp.
                 Shows which outcome is the bigger surprise and which asset pays most for a view.
  LAG ALERTS     Odds jump but an asset hasn't moved by beta x the odds change -> "asset lagging the
                 odds" alert (checked every 15 min, 6h cooldown per pair).
  /scenarios     The map on demand; the morning brief carries the top events.

Betas: OLS of the asset's move over that hour and the next on the hour's odds change, using only hours
where the odds moved >= 2 pts (winsorised), over the last ~120 days. A beta is only USED when n >= 20
and |t| >= 2.5; otherwise the asset shows as "no reliable link". Re-fitted daily at 06:30 UK.

  python scenarios.py --fit       fit betas (daily)
  python scenarios.py --lag       check for lagging assets (every 15 min)
  python scenarios.py --show      print the scenario map
  add --dry-run to print alerts instead of sending
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
BETAS = CACHE / "scenario_betas.json"
LAGSTATE = HERE / "scenario_lag_state.json"

# asset label -> (yahoo symbol, unit) ; unit "pct" = % price change, "bp" = yield change in bp
ASSETS = {
    "Gold": ("GC=F", "pct"), "Silver": ("SI=F", "pct"), "DXY": ("DX-Y.NYB", "pct"), "EURUSD": ("EURUSD=X", "pct"),
    "USDJPY": ("JPY=X", "pct"), "US 10y": ("^TNX", "bp"), "US 2y": ("2YY=F", "bp"), "S&P fut": ("ES=F", "pct"),
    "Nasdaq fut": ("NQ=F", "pct"), "WTI": ("CL=F", "pct"), "Bitcoin": ("BTC-USD", "pct"),
    "Defence (ITA)": ("ITA", "pct"), "Semis (SMH)": ("SMH", "pct"), "Reg. banks (KRE)": ("KRE", "pct"),
    "Energy (XLE)": ("XLE", "pct"), "Homebuilders (XHB)": ("XHB", "pct"),
}
CAT_ASSETS = {
    "fed_rates": ["Gold", "Silver", "DXY", "USDJPY", "US 2y", "US 10y", "S&P fut", "Nasdaq fut", "Reg. banks (KRE)",
                  "Homebuilders (XHB)", "Bitcoin"],
    "inflation_data": ["Gold", "DXY", "US 2y", "US 10y", "S&P fut", "Homebuilders (XHB)"],
    "growth_recession": ["Gold", "DXY", "US 10y", "S&P fut", "WTI", "Reg. banks (KRE)"],
    "central_banks": ["EURUSD", "USDJPY", "DXY", "Gold"],
    "us_politics": ["DXY", "S&P fut", "Gold", "Bitcoin", "US 10y", "Defence (ITA)", "Energy (XLE)", "Reg. banks (KRE)"],
    "elections_global": ["EURUSD", "DXY", "Gold", "S&P fut"],
    "trade_tariffs": ["S&P fut", "Nasdaq fut", "Semis (SMH)", "DXY", "USDJPY", "EURUSD", "Gold"],
    "geopolitics_conflict": ["WTI", "Gold", "Defence (ITA)", "Energy (XLE)", "S&P fut", "USDJPY"],
    "fiscal_debt": ["Gold", "US 10y", "DXY", "S&P fut"],
    "energy": ["WTI", "Energy (XLE)"],
    "crypto_policy": ["Bitcoin"],
}
MIN_N, MIN_T = 20, 2.5
MIN_DP = 0.02
LAG_DP = 0.08          # odds move (2h) that triggers a lag check
LAG_RATIO = 0.35       # asset has done < 35% of its implied move
LAG_COOLDOWN_H = 6


def _load(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)


# --------------------------------------------------------------------------
# Drivers: what single number describes each event's odds
# --------------------------------------------------------------------------
def build_drivers(snapshot: list[dict], max_events: int = 25) -> list[dict]:
    """Pick the events to model and define each one's driver.
    fed: expected bp change = sum(outcome bp x p) / sum(p)  -> beta per bp
    binary: the leading outcome's probability                -> beta per 100 pts"""
    from predict_edge import outcome_bp
    ev: dict[str, list[dict]] = {}
    for m in snapshot:
        if m.get("venue") != "Polymarket" or not m.get("token") or m.get("cat") not in CAT_ASSETS:
            continue
        if (m.get("imp") or 1) < 2:
            continue
        ev.setdefault(m["ekey"], []).append(m)
    ranked = sorted(ev.items(), key=lambda kv: (-max(x.get("imp") or 1 for x in kv[1]), -sum(x["vol24"] for x in kv[1])))
    out = []
    for k, ms in ranked[:max_events]:
        cat = ms[0]["cat"]
        outs = [(outcome_bp(m["label"] or m["question"]), m) for m in ms]
        outs = [(b, m) for b, m in outs if b is not None]
        if cat == "fed_rates" and len(outs) >= 2 and any(b == 0 for b, _ in outs):
            out.append({"ekey": k, "kind": "fed", "event": ms[0]["event"], "cat": cat, "end": ms[0].get("end"),
                        "url": ms[0].get("url"),
                        "legs": [{"token": m["token"], "label": m["label"] or m["question"][:40], "bp": b, "p": m["p"]}
                                 for b, m in outs]})
        else:
            lead = max(ms, key=lambda m: m["vol24"])
            out.append({"ekey": k, "kind": "binary", "event": ms[0]["event"], "cat": cat, "end": lead.get("end"),
                        "url": lead.get("url"),
                        "legs": [{"token": lead["token"], "label": lead["label"] or "Yes", "p": lead["p"]}],
                        "others": [{"label": m["label"], "p": m["p"]} for m in sorted(ms, key=lambda m: -m["p"])[:4]
                                   if m is not lead and m["label"]]})
    return out


def driver_series(d: dict, hist: dict[str, pd.Series]) -> pd.Series:
    if d["kind"] == "fed":
        cols = {i: hist.get(leg["token"]) for i, leg in enumerate(d["legs"])}
        cols = {i: s for i, s in cols.items() if s is not None and len(s)}
        if len(cols) < 2:
            return pd.Series(dtype=float)
        P = pd.DataFrame(cols).ffill(limit=6).dropna()
        bp = np.array([d["legs"][i]["bp"] for i in P.columns], dtype=float)
        w = P.values
        return pd.Series((w * bp).sum(axis=1) / w.sum(axis=1).clip(min=1e-6), index=P.index)
    s = hist.get(d["legs"][0]["token"])
    return s if s is not None else pd.Series(dtype=float)


def driver_now(d: dict) -> float:
    if d["kind"] == "fed":
        tot = sum(l["p"] for l in d["legs"]) or 1
        return sum(l["bp"] * l["p"] for l in d["legs"]) / tot
    return d["legs"][0]["p"]


# --------------------------------------------------------------------------
# Betas
# --------------------------------------------------------------------------
def asset_hourly(labels: list[str], days: int = 120) -> pd.DataFrame:
    import yfinance as yf
    syms = {ASSETS[a][0]: a for a in labels if a in ASSETS}
    d = yf.download(list(syms), period=f"{days}d", interval="60m", progress=False, auto_adjust=False)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(list(syms)[0])
    d.index = pd.to_datetime(d.index, utc=True).floor("h")
    d = d[~d.index.duplicated(keep="last")].rename(columns=syms)
    for a in d.columns:
        if ASSETS[a][1] == "bp" and d[a].median() > 20:      # some yield indices are quoted x10
            d[a] = d[a] / 10
    return d


def asset_moves(px: pd.Series, unit: str) -> pd.Series:
    """Per-bar move: % (log) for prices, bp for yields."""
    return (px.diff() * 100) if unit == "bp" else np.log(px).diff()


def fit_beta(drv: pd.Series, px: pd.Series, unit: str, fed: bool) -> dict | None:
    px = px.dropna()
    if len(px) < 100 or drv.empty:
        return None
    x = drv.reindex(px.index, method="ffill")           # driver value at each asset bar
    dx = x.diff()
    r = asset_moves(px, unit)
    y = r + r.shift(-1)                                  # same bar + the next: allows a short reaction lag
    gap = pd.Series(px.index, index=px.index).diff()
    nxt_gap = gap.shift(-1)
    j = pd.concat([dx, y, gap, nxt_gap], axis=1, keys=["dx", "y", "gap", "ng"]).dropna()
    # only continuous trading: skip bars spanning a close/weekend (the odds change there isn't matched by one bar)
    j = j[(j["gap"] <= pd.Timedelta(hours=1)) & (j["ng"] <= pd.Timedelta(hours=1))]
    thr = 0.5 if fed else MIN_DP                         # fed driver is in bp
    cap = 12 if fed else 0.3
    j = j[(j["dx"].abs() >= thr) & (j["dx"].abs() <= cap)]    # drop resolution-sized jumps
    if len(j) < MIN_N:
        return None
    lo, hi = j["y"].quantile([0.01, 0.99])
    j["y"] = j["y"].clip(lo, hi)
    b = float((j["dx"] * j["y"]).sum() / (j["dx"] ** 2).sum())
    resid = j["y"] - b * j["dx"]
    se = float(math.sqrt((resid ** 2).sum() / max(len(j) - 1, 1) / (j["dx"] ** 2).sum()))
    t = b / se if se > 0 else 0.0
    sd2 = float(r.rolling(2).sum().dropna().std())       # typical 2-bar move, for the lag test
    return {"beta": b, "t": t, "n": int(len(j)), "unit": unit, "sd2": sd2}


def fit(dry: bool = False) -> int:
    from predict import load_snapshot
    from predict_backtest import odds_history
    snap = load_snapshot(max_age_min=90)
    if not snap:
        print("No prediction-market snapshot yet (predict.py runs every 5 min).")
        return 0
    drivers = build_drivers(snap)
    start = int((datetime.now(timezone.utc) - timedelta(days=120)).timestamp())
    tokens = {leg["token"] for d in drivers for leg in d["legs"]}
    print(f"{len(drivers)} events, {len(tokens)} odds histories to fetch…", flush=True)
    hist = {}
    for tok in tokens:
        try:
            s = odds_history(tok, start)
            if len(s) >= 48:
                hist[tok] = s
        except Exception as e:
            print(f"  odds history failed: {e}", file=sys.stderr)
        time.sleep(0.1)
    need = sorted({a for d in drivers for a in CAT_ASSETS.get(d["cat"], [])})
    px = asset_hourly(need)
    out = []
    for d in drivers:
        s = driver_series(d, hist)
        if s.empty:
            continue
        betas = {}
        for a in CAT_ASSETS.get(d["cat"], []):
            if a not in px:
                continue
            b = fit_beta(s, px[a], ASSETS[a][1], d["kind"] == "fed")
            if b:
                betas[a] = b
        out.append({**d, "betas": betas})
    _save(BETAS, {"fitted": datetime.now(timezone.utc).isoformat(timespec="seconds"), "events": out})
    good = sum(1 for d in out for b in d["betas"].values() if b["n"] >= MIN_N and abs(b["t"]) >= MIN_T)
    print(f"Fitted {len(out)} events, {good} reliable event→asset links.")
    return 0


def reliable(b: dict) -> bool:
    return b.get("n", 0) >= MIN_N and abs(b.get("t", 0)) >= MIN_T


# --------------------------------------------------------------------------
# Scenario map
# --------------------------------------------------------------------------
def _fmt_move(v: float, unit: str) -> str:
    return f"{v:+.0f}bp" if unit == "bp" else f"{v*100:+.1f}%"


def _live(d: dict, snap_by_token: dict) -> dict:
    """Refresh the event's odds from the latest snapshot."""
    d = json.loads(json.dumps(d))
    for leg in d["legs"]:
        m = snap_by_token.get(leg["token"])
        if m:
            leg["p"] = m["p"]
    return d


def scenarios(d: dict) -> list[dict]:
    """[{outcome, p, moves:{asset: (move, unit)}}] using reliable betas only."""
    now = driver_now(d)
    rel = {a: b for a, b in d["betas"].items() if reliable(b)}
    outs = []
    if d["kind"] == "fed":
        for leg in sorted(d["legs"], key=lambda l: -l["p"]):
            if leg["p"] < 0.03:
                continue
            dx = leg["bp"] - now
            outs.append({"outcome": leg["label"], "p": leg["p"],
                         "moves": {a: (b["beta"] * dx, b["unit"]) for a, b in rel.items()}})
    else:
        p = d["legs"][0]["p"]
        lab = d["legs"][0]["label"]
        outs.append({"outcome": lab if lab != "Yes" else "Yes", "p": p,
                     "moves": {a: (b["beta"] * (1 - p), b["unit"]) for a, b in rel.items()}})
        outs.append({"outcome": "No" if lab == "Yes" else f"not {lab}", "p": 1 - p,
                     "moves": {a: (b["beta"] * (0 - p), b["unit"]) for a, b in rel.items()}})
    return outs


def show_text(max_events: int = 6, compact: bool = False) -> str:
    data = _load(BETAS, {})
    evs = data.get("events") or []
    if not evs:
        return "📊 Scenario map not built yet — it fits each morning (needs a day of prediction-market data)."
    try:
        from predict import load_snapshot
        snap = {m.get("token"): m for m in load_snapshot(max_age_min=60) if m.get("token")}
    except Exception:
        snap = {}
    e = html.escape
    lines = ["📊 <b>What the odds imply</b> — move from here if each outcome happens (event betas, last 120d)"]
    shown = 0
    for d in evs:
        d = _live(d, snap)
        sc = scenarios(d)
        if not sc or not sc[0]["moves"]:
            continue
        shown += 1
        lines.append(f"\n<b>{e(d['event'][:110])}</b>")
        assets = sorted(sc[0]["moves"], key=lambda a: -abs(d["betas"][a]["t"]))[:3 if compact else 5]
        for o in sc[:3]:
            mv = " · ".join(f"{e(a)} {_fmt_move(*o['moves'][a])}" for a in assets)
            lines.append(f"   {e(str(o['outcome'])[:28])} {o['p']*100:.0f}% → {mv}")
        if len(sc) >= 2:
            big = max(sc[:3], key=lambda o: max(abs(v if u == 'bp' else v * 100) for v, u in o["moves"].values()))
            lines.append(f"   <i>Bigger surprise: {e(str(big['outcome'])[:30])} ({big['p']*100:.0f}% priced)</i>")
        if d["kind"] == "fed":
            fed = _load(HERE / "predict_edge_state.json", {}).get("fed") or []
            for f in fed[:1]:
                fut = [x for x in f["lines"] if x.startswith("Futures")]
                if fut:
                    lines.append(f"   {e(fut[0][:110])}")
        if shown >= max_events:
            break
    if shown == 0:
        return ("📊 No reliable event→asset links yet (each link needs ≥20 informative hours and |t| ≥ 2.5). "
                "They build up as odds move.")
    lines.append(f"\n<i>Betas fitted {e(str(data.get('fitted', ''))[:16])} UTC · only links with |t| ≥ {MIN_T} shown · "
                 "estimates, not forecasts.</i>")
    return "\n".join(lines)


def brief_lines(n: int = 2) -> list[str]:
    """Plain-text lines for the morning brief."""
    t = show_text(max_events=n, compact=True)
    if t.startswith("📊 Scenario map not built") or t.startswith("📊 No reliable"):
        return []
    return [re.sub(r"<[^>]+>", "", html.unescape(x)) for x in t.splitlines()[1:] if x.strip() and not x.startswith("<i>Betas")]


# --------------------------------------------------------------------------
# Lag alerts
# --------------------------------------------------------------------------
def recent_assets(labels: list[str]) -> pd.DataFrame:
    import yfinance as yf
    syms = {ASSETS[a][0]: a for a in labels if a in ASSETS}
    d = yf.download(list(syms), period="2d", interval="15m", progress=False, auto_adjust=False, prepost=True)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(list(syms)[0])
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.rename(columns=syms)
    for a in d.columns:
        if ASSETS[a][1] == "bp" and d[a].median() > 20:
            d[a] = d[a] / 10
    return d


def lag_check(dry: bool = False) -> int:
    data = _load(BETAS, {})
    evs = data.get("events") or []
    if not evs:
        print("No betas yet.")
        return 0
    from predict import STATE as PSTATE, base_at
    pst = _load(PSTATE, {})
    hist = pst.get("hist", {})
    snap_tokens = {}
    try:
        from predict import load_snapshot
        for m in load_snapshot(max_age_min=20):
            if m.get("token"):
                snap_tokens[m["token"]] = m
    except Exception:
        pass
    now = time.time()
    cands = []
    for d in evs:
        live = _live(d, snap_tokens)
        # driver 2h ago from predict.py's own 5-minute samples
        past = json.loads(json.dumps(live))
        ok = True
        for leg in past["legs"]:
            mid = next((m["mid"] for m in snap_tokens.values() if m.get("token") == leg["token"]), None)
            b = base_at(hist.get(mid, []), 2 * 3600, now) if mid else None
            if b is None:
                ok = False
                break
            leg["p"] = b
        if not ok:
            continue
        dx = driver_now(live) - driver_now(past)
        trig = abs(dx) >= (4.0 if d["kind"] == "fed" else LAG_DP)
        if not trig:
            continue
        for a, b in d["betas"].items():
            if reliable(b):
                cands.append((live, a, b, dx))
    if not cands:
        print("No odds moves big enough for a lag check.")
        return 0
    px = recent_assets(sorted({a for _, a, _, _ in cands}))
    st = _load(LAGSTATE, {})
    fire = []
    for d, a, b, dx in cands:
        if a not in px:
            continue
        s = px[a].dropna()
        s2 = s[s.index >= pd.Timestamp(now - 2 * 3600, unit="s", tz="UTC")]
        if len(s2) < 2:
            continue                                         # asset not trading now
        actual = float(asset_moves(pd.Series([s2.iloc[0], s2.iloc[-1]]), b["unit"]).iloc[-1])
        implied = b["beta"] * dx
        big = abs(implied) >= max(1.0 * b["sd2"], 0.002 if b["unit"] == "pct" else 1.5)
        lagging = big and (np.sign(actual) != np.sign(implied) or abs(actual) < LAG_RATIO * abs(implied))
        key = f"{d['ekey']}:{a}"
        if lagging and now - st.get(key, 0) >= LAG_COOLDOWN_H * 3600:
            fire.append({"d": d, "a": a, "b": b, "dx": dx, "implied": implied, "actual": actual})
            st[key] = now
    if fire:
        bt = _load(CACHE / "predict_backtest_summary.json", {})
        tl = bt.get("follow_odds_24h_t")
        e = html.escape
        lines = ["🧭 <b>Assets lagging the odds</b>"]
        for f in fire[:6]:
            d, b = f["d"], f["b"]
            what = (f"expected change {f['dx']:+.1f}bp" if d["kind"] == "fed" else f"odds {f['dx']*100:+.0f} pts")
            lines.append(f"\n<b>{e(d['event'][:100])}</b>: {what} in 2h")
            lines.append(f"   {e(f['a'])}: moved {_fmt_move(f['actual'], b['unit'])} vs "
                         f"<b>{_fmt_move(f['implied'], b['unit'])} implied</b> (beta t={b['t']:.1f}, n={b['n']})")
        lines.append("\n<i>" + ("Lead-lag is supported by the odds backtest" if tl is not None and tl >= 2 else
                                "Lead-lag not yet proven in the odds backtest — treat as a heads-up") +
                     (f" (t={tl:.1f})" if tl is not None else "") + ".</i>")
        text = "\n".join(lines)
        if dry:
            print(text)
        else:
            from scanner import send_telegram
            send_telegram(text)
            _save(LAGSTATE, {k: v for k, v in st.items() if now - v < 3 * 86400})
    print(f"{len(cands)} event→asset pairs checked, {len(fire)} lagging.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--lag", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit(a.dry_run)
    if a.lag:
        lag_check(a.dry_run)
    if a.show or not (a.fit or a.lag):
        print(re.sub(r"<[^>]+>", "", show_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

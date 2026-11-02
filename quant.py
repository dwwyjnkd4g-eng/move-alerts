"""
Quant layer: turns a flagged move into a statistically grounded trade plan.

1. Universe base rates (event study). For every liquid stock over the past year, find days with
   a large idiosyncratic move (|idio z| >= 3 and |move| >= 3%), and measure the forward idio
   return over 1/2/3/5/10 days. Events are bucketed by direction, size (sigma), relative volume
   and gap-vs-intraday, with a hierarchical fallback when a bucket is thin. Cached daily.
2. Stock analogs. The same study on the flagged stock's own 5-year history (|z| >= 2.5, same
   direction), shrunk towards the universe base rate (James-Stein style, k = 20 events).
3. Setup features: RSI(2)/RSI(14), distance from 20/50/200-day averages in sigma, 20-day
   trend, 52-week position, close location in today's range, move since the open.
4. Plan: pick the hold horizon with the best per-trade risk-adjusted fade return
   (mean / std, positive mean only); target = expected fade, stretch = 75th pct, stop = the
   adverse 15th-percentile outcome; win rate, reward/risk, EV, and a 0-100 edge score from
   the t-statistic. If no horizon has a positive expectancy, the plan is "no edge"
   (history favours continuation).

Macro markets use the same logic on the instrument's own recent intraday history.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
HORIZONS = [1, 2, 3, 5, 10]
BETA_W, SD_W = 60, 40            # shorter than the scanner's windows so a 1y cache yields ~150 usable days
EV_Z, EV_RET = 3.0, 0.03
ANALOG_Z = 2.5
SHRINK_K = 20
MIN_BUCKET = 30


# --------------------------------------------------------------------------
# Event study (vectorised)
# --------------------------------------------------------------------------
def idio_panel(close: pd.DataFrame, bench_of: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling-beta idiosyncratic log returns, their z-scores and raw log returns, per stock."""
    r = np.log(close).diff()
    resid_parts, z_parts = [], []
    groups: dict[str, list[str]] = {}
    for s, b in bench_of.items():
        if s in r and b in r:
            groups.setdefault(b, []).append(s)
    for b, syms in groups.items():
        X, y = r[syms], r[b]
        mx, my = X.rolling(BETA_W).mean(), y.rolling(BETA_W).mean()
        cov = X.mul(y, axis=0).rolling(BETA_W).mean() - mx.mul(my, axis=0)
        var = y.rolling(BETA_W).var(ddof=0)
        beta = cov.div(var, axis=0).shift(1).clip(-3, 5)
        resid = X - beta.mul(y, axis=0)
        sd = resid.rolling(SD_W).std().shift(1)
        resid_parts.append(resid)
        z_parts.append(resid / sd)
    if not resid_parts:
        return pd.DataFrame(), pd.DataFrame(), r
    return pd.concat(resid_parts, axis=1), pd.concat(z_parts, axis=1), r


def events_table(close: pd.DataFrame, open_: pd.DataFrame | None, volume: pd.DataFrame | None,
                 bench_of: dict[str, str], z_min: float = EV_Z, ret_min: float = EV_RET,
                 min_dv: float = 0.0, direction: int | None = None) -> pd.DataFrame:
    """One row per large idiosyncratic move, with forward fade returns (positive = reverted)."""
    resid, z, r = idio_panel(close, bench_of)
    if z.empty:
        return pd.DataFrame()
    syms = list(z.columns)
    raw = np.expm1(r[syms])
    mask = (z.abs() >= z_min) & (raw.abs() >= ret_min)
    if direction is not None:
        mask &= np.sign(z) == direction
    if volume is not None and not volume.empty and min_dv:
        dv = (close[syms] * volume.reindex(columns=syms)).rolling(20).mean().shift(1)
        mask &= dv >= min_dv
    fwd = {h: resid.rolling(h).sum().shift(-h) for h in HORIZONS}
    rvol = None
    if volume is not None and not volume.empty:
        v = volume.reindex(columns=syms)
        rvol = v / v.rolling(20).mean().shift(1)
    gapf = None
    if open_ is not None and not open_.empty:
        o = open_.reindex(columns=syms)
        gap = np.log(o / close[syms].shift(1))
        gapf = (gap / r[syms]).clip(-1, 2)
    st = mask.stack()
    st = st[st]
    if st.empty:
        return pd.DataFrame()
    idx = st.index
    zz = z.stack().reindex(idx)
    out = pd.DataFrame({"z": zz.values, "ret": raw.stack().reindex(idx).values}, index=idx)
    out["dir"] = np.sign(out["z"])
    out["rvol"] = rvol.stack().reindex(idx).values if rvol is not None else np.nan
    out["gapf"] = gapf.stack().reindex(idx).values if gapf is not None else np.nan
    for h, f in fwd.items():
        out[f"fade_{h}"] = -out["dir"] * f.stack().reindex(idx).values
    return out.reset_index(names=["date", "symbol"]) if hasattr(out, "reset_index") else out


def _bucket_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zb"] = pd.cut(df["z"].abs(), [0, 4, 6, np.inf], labels=["3-4σ", "4-6σ", "6σ+"], right=False).astype(str)
    df["rvb"] = pd.cut(df["rvol"].fillna(1.0), [-np.inf, 2, 5, np.inf], labels=["rvol<2", "rvol2-5", "rvol5+"]).astype(str)
    df["gb"] = np.where(df["gapf"].fillna(0) >= 0.6, "gap", "intraday")
    return df


def _stats(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 3:
        return {"n": len(x)}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()),
            "hit": float((x > 0).mean()), "p15": float(x.quantile(0.15)), "p75": float(x.quantile(0.75))}


def build_base_rates(ev: pd.DataFrame) -> dict:
    """Stats per bucket level: (dir,zb,rvb,gb) -> (dir,zb,rvb) -> (dir,zb) -> (dir,)."""
    if ev.empty:
        return {"levels": {}, "n_events": 0, "asof": datetime.now(timezone.utc).isoformat()}
    ev = _bucket_cols(ev)
    levels = {}
    for keys in (("dir", "zb", "rvb", "gb"), ("dir", "zb", "rvb"), ("dir", "zb"), ("dir",)):
        tab = {}
        for k, g in ev.groupby(list(keys)):
            k = k if isinstance(k, tuple) else (k,)
            tab[tuple(str(int(v)) if isinstance(v, (float, np.floating)) else str(v) for v in k)] = \
                {h: _stats(g[f"fade_{h}"]) for h in HORIZONS}
        levels["|".join(keys)] = tab
    return {"levels": levels, "n_events": int(len(ev)), "asof": datetime.now(timezone.utc).isoformat()}


def lookup(br: dict, direction: int, z: float, rvol: float | None, gapf: float | None) -> tuple[dict, str]:
    """Most specific bucket with enough events."""
    one = pd.DataFrame([{"z": z, "rvol": rvol if rvol is not None else np.nan,
                         "gapf": gapf if gapf is not None else np.nan}])
    b = _bucket_cols(one).iloc[0]
    d = str(int(direction))
    tries = [("dir|zb|rvb|gb", (d, b["zb"], b["rvb"], b["gb"]), f"{b['zb']}, {b['rvb']}, {b['gb']}"),
             ("dir|zb|rvb", (d, b["zb"], b["rvb"]), f"{b['zb']}, {b['rvb']}"),
             ("dir|zb", (d, b["zb"]), b["zb"]),
             ("dir", (d,), "all sizes")]
    for level, key, label in tries:
        s = br.get("levels", {}).get(level, {}).get(key)
        if s and s.get(5, {}).get("n", 0) >= MIN_BUCKET:
            return s, label
    return {}, ""


def base_rates_cached(name: str, loader) -> dict:
    """Load today's base rates for a market ('us'/'eu'); build them with loader() if missing."""
    path = CACHE / f"base_rates_{name}.pkl"
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        br = pd.read_pickle(path)
        if br.get("day") == today:
            return br
    except Exception:
        pass
    try:
        close, open_, volume, bench_of, min_dv = loader()
        ev = events_table(close, open_, volume, bench_of, min_dv=min_dv)
        br = build_base_rates(ev)
        br["day"] = today
        CACHE.mkdir(exist_ok=True)
        pd.to_pickle(br, path)
        print(f"  base rates ({name}): {br['n_events']} events")
        return br
    except Exception as e:
        print(f"  base rates ({name}) failed: {e}", file=sys.stderr)
        return {"levels": {}, "n_events": 0}


# --------------------------------------------------------------------------
# Stock's own history
# --------------------------------------------------------------------------
def stock_analogs(yf_symbol: str, bench: str, direction: int) -> dict:
    """Same-direction big idio moves in the stock's own 5-year history."""
    path = CACHE / "analogs" / f"{yf_symbol}_{bench}_{direction}.pkl".replace("^", "_")
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        a = pd.read_pickle(path)
        if a.get("day") == today:
            return a
    except Exception:
        pass
    out = {"day": today}
    try:
        import yfinance as yf
        px = yf.download([yf_symbol, bench], period="5y", interval="1d", auto_adjust=True, progress=False)
        close = px["Close"].dropna(how="all")
        ev = events_table(close, None, None, {yf_symbol: bench}, z_min=ANALOG_Z, ret_min=0.02, direction=direction)
        # drop the most recent few days (that's the move we're analysing)
        if not ev.empty:
            ev = ev[pd.to_datetime(ev["date"]) < close.index[-1] - pd.Timedelta(days=3)]
        out.update({h: _stats(ev[f"fade_{h}"]) if not ev.empty else {"n": 0} for h in HORIZONS})
    except Exception as e:
        print(f"  analogs failed for {yf_symbol}: {e}", file=sys.stderr)
        out.update({h: {"n": 0} for h in HORIZONS})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def rsi(c: pd.Series, n: int) -> float:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    v = 100 - 100 / (1 + rs)
    return float(v.iloc[-1]) if len(v) and not np.isnan(v.iloc[-1]) else float("nan")


def setup_features(close: pd.Series, intraday: pd.Series | None = None) -> dict:
    c = close.dropna()
    if len(c) < 30:
        return {}
    lr = np.log(c).diff()
    sd = float(lr.iloc[-61:-1].std())
    f = {"rsi2": rsi(c, 2), "rsi14": rsi(c, 14), "ret20": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else np.nan}
    for n in (20, 50, 200):
        if len(c) > n:
            ma = c.iloc[-n:].mean()
            f[f"ma{n}_sig"] = float(math.log(c.iloc[-1] / ma) / (sd * math.sqrt(n / 4))) if sd else np.nan
    w = c.iloc[-252:]
    f["pos52"] = float((c.iloc[-1] - w.min()) / (w.max() - w.min())) if w.max() > w.min() else np.nan
    f["vol_d"] = sd
    if intraday is not None and len(intraday) >= 2:
        hi, lo = intraday.max(), intraday.min()
        f["clv"] = float((intraday.iloc[-1] - lo) / (hi - lo)) if hi > lo else np.nan     # 0 = at the low, 1 = at the high
        f["since_open"] = float(intraday.iloc[-1] / intraday.iloc[0] - 1)
    return f


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def combine(base: dict, own: dict) -> dict:
    """Shrink the stock's own stats towards the universe base rate, per horizon."""
    out = {}
    for h in HORIZONS:
        b, o = base.get(h, {}), own.get(h, {}) if own else {}
        if not b.get("n"):
            out[h] = o if o.get("n", 0) >= 10 else {}
            continue
        no = o.get("n", 0) if o.get("mean") is not None else 0
        w = no / (no + SHRINK_K)
        c = dict(b)
        for k in ("mean", "hit", "median"):
            if no and o.get(k) is not None:
                c[k] = (1 - w) * b[k] + w * o[k]
        c["n_own"], c["w_own"] = no, w
        out[h] = c
    return out


def make_plan(price: float, move: float, stats: dict, unit: str = "d") -> dict:
    """Choose horizon, target, stop, EV, edge score. move = today's (signed) move as a fraction."""
    best = None
    for h, s in stats.items():
        if not s or s.get("mean") is None or s.get("std") in (None, 0):
            continue
        sharpe = s["mean"] / s["std"]
        if s["mean"] > 0 and (best is None or sharpe > best[1]):
            best = (h, sharpe, s)
    if best is None:
        s5 = stats.get(5) or stats.get(min(stats, default=1), {})
        return {"edge": False, "mean5": s5.get("mean"), "hit5": s5.get("hit"), "n": s5.get("n", 0),
                "note": "history favours continuation, not reversal"}
    h, sharpe, s = best
    side = -1 if move > 0 else 1            # fade: short after up-moves, long after down-moves
    tgt_pct, stretch_pct = s["mean"], max(s.get("p75", s["mean"]), s["mean"])
    stop_pct = max(-s.get("p15", -s["std"]), 0.5 * s["std"], 0.005)
    n_eff = s["n"] + s.get("n_own", 0)
    t = s["mean"] / (s["std"] / math.sqrt(max(n_eff, 1)))
    return {
        "edge": True, "side": "long" if side > 0 else "short", "hold": h, "unit": unit,
        "entry": price, "target": price * (1 + side * tgt_pct), "stretch": price * (1 + side * stretch_pct),
        "stop": price * (1 - side * stop_pct), "tgt_pct": tgt_pct, "stop_pct": stop_pct,
        "hit": s.get("hit"), "rr": tgt_pct / stop_pct if stop_pct else None,
        "ev": s.get("hit", 0.5) * tgt_pct - (1 - s.get("hit", 0.5)) * stop_pct,
        "n": s["n"], "n_own": s.get("n_own", 0), "t": t, "sharpe": sharpe,
        # edge score: per-trade quality (mean/std) scaled by statistical confidence (t-stat, saturating at 3)
        "score": int(round(50 + 50 * math.tanh(1.5 * sharpe) * min(1.0, max(t, 0) / 3))),
    }


def equity_plan(row, closes: pd.DataFrame, base: dict, intraday_close: pd.Series | None = None) -> dict:
    """Full quant view for a flagged stock (row from the scanner's metrics)."""
    sym, bench = row["yf_symbol"], row["etf"] if row.get("etf") in closes else None
    direction = int(np.sign(row["idio_z"])) or 1
    gapf = None
    if row.get("gap") is not None and not pd.isna(row.get("gap")) and row["ret"]:
        gapf = math.log1p(row["gap"]) / math.log1p(row["ret"])
    b, label = lookup(base, direction, float(row["idio_z"]), row.get("rvol"), gapf)
    own = stock_analogs(sym, bench or "SPY", direction)
    stats = combine(b, own)
    plan = make_plan(float(row["price"]), float(row["ret"]), stats)
    plan["bucket"] = label
    plan["own"] = {h: own.get(h, {}) for h in HORIZONS}
    plan["features"] = setup_features(closes[sym], intraday_close) if sym in closes else {}
    g = row.get("gap")
    if "since_open" not in plan["features"] and g is not None and not pd.isna(g):
        plan["features"]["since_open"] = float((1 + row["ret"]) / (1 + g) - 1)
    return plan


def macro_plan(src: str, sym: str, st: dict) -> dict:
    """Analogs from the instrument's own recent intraday history: moves of the same size (sigma)
    and direction over the same window, then forward fade over 1h / 4h / 1d / 3d."""
    try:
        if src == "kraken":
            from macro import kraken_closes
            s = kraken_closes(sym, interval=60)
            per_h = 1
        else:
            import yfinance as yf
            from macro import OTC_FUT
            ysym = OTC_FUT.get(sym, sym)
            d = yf.download(ysym, period="60d", interval="60m", progress=False, auto_adjust=False)["Close"]
            s = (d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d).dropna()
            per_h = 1
        if len(s) < 200:
            return {"edge": False, "n": 0, "note": "not enough history"}
        lr = np.log(s).diff()
        win = 1 if st["window"] == "1h" else 4
        mv = lr.rolling(win).sum()
        sd = lr.rolling(240).std().shift(win) * math.sqrt(win)
        z = mv / sd
        direction = 1 if st["ret"] > 0 else -1
        ev_idx = z.index[(np.sign(z) == direction) & (z.abs() >= 0.75 * abs(st["z"]))][:-1]
        # de-cluster: at most one event per 6 hours
        kept, last = [], None
        for t in ev_idx:
            if last is None or (t - last) >= pd.Timedelta(hours=6):
                kept.append(t)
                last = t
        hz = {1: 1 * per_h, 4: 4 * per_h, 24: 24 * per_h, 72: 72 * per_h}
        stats = {}
        for label, n in hz.items():
            vals = []
            for t in kept:
                i = s.index.get_loc(t)
                if i + n < len(s):
                    vals.append(-direction * (s.iloc[i + n] / s.iloc[i] - 1))
            stats[label] = _stats(pd.Series(vals))
        plan = make_plan(float(st["price"]), float(st["ret"]), stats, unit="h")
        plan["n_events"] = len(kept)
        return plan
    except Exception as e:
        print(f"  macro plan failed for {sym}: {e}", file=sys.stderr)
        return {"edge": False, "n": 0, "note": "unavailable"}


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
def _p(x, d=1, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x*100:+.{d}f}%" if pct else f"{x:.{d}f}"


def fmt_px(p: float) -> str:
    return f"{p:,.0f}" if p >= 1000 else (f"{p:,.2f}" if p >= 10 else f"{p:.4f}")


def hold_label(plan: dict) -> str:
    h, u = plan.get("hold"), plan.get("unit", "d")
    if u == "h":
        return f"{h}h" if h < 24 else f"{h // 24}d"
    return f"{h} day" + ("s" if h != 1 else "")


def plan_lines(plan: dict, ccy_note: str = "") -> list[str]:
    """HTML lines for the alert."""
    import html as _h
    e = _h.escape
    out = []
    if plan.get("edge"):
        out.append(
            f"📐 <b>Quant: {plan['side'].upper()} fade</b> · edge {plan['score']}/100 · "
            f"hold ~{hold_label(plan)}")
        out.append(
            f"   Entry {fmt_px(plan['entry'])} → <b>target {fmt_px(plan['target'])}</b> ({_p(plan['tgt_pct'])}), "
            f"stretch {fmt_px(plan['stretch'])}, <b>stop {fmt_px(plan['stop'])}</b> (−{plan['stop_pct']*100:.1f}%){ccy_note}")
        out.append(
            f"   Win {plan['hit']*100:.0f}% · R/R {plan['rr']:.2f} · EV {_p(plan['ev'], 2)} · "
            f"n={plan['n']}" + (f" (+{plan['n_own']} own)" if plan.get("n_own") else "") +
            f" · t={plan['t']:.1f}" + (f" · {e(plan['bucket'])}" if plan.get("bucket") else ""))
    else:
        n = plan.get("n", 0)
        mean5 = plan.get("mean5")
        out.append(f"📐 <b>Quant: no fade edge</b> — {e(plan.get('note', ''))}"
                   + (f" (5d avg {_p(mean5)}, n={n})" if mean5 is not None else ""))
    f = plan.get("features") or {}
    if f:
        bits = []
        if not np.isnan(f.get("rsi2", np.nan)):
            bits.append(f"RSI2 {f['rsi2']:.0f}")
        if not np.isnan(f.get("rsi14", np.nan)):
            bits.append(f"RSI14 {f['rsi14']:.0f}")
        for n in (20, 50, 200):
            v = f.get(f"ma{n}_sig")
            if v is not None and not np.isnan(v):
                bits.append(f"{v:+.1f}σ vs {n}DMA")
        if f.get("ret20") is not None and not np.isnan(f["ret20"]):
            bits.append(f"20d {_p(f['ret20'], 0)}")
        if f.get("pos52") is not None and not np.isnan(f["pos52"]):
            bits.append(f"52w pos {f['pos52']*100:.0f}%")
        if f.get("clv") is not None and not np.isnan(f["clv"]):
            bits.append(f"at {f['clv']*100:.0f}% of day range")
        if f.get("since_open") is not None and not np.isnan(f["since_open"]):
            bits.append(f"since open {_p(f['since_open'])}")
        out.append("   Setup: " + " · ".join(bits))
    return out


def plan_for_prompt(plan: dict) -> dict:
    """Compact, model-friendly version of the plan."""
    keep = {k: plan.get(k) for k in ("edge", "side", "hold", "unit", "tgt_pct", "stop_pct", "hit", "rr", "ev",
                                     "n", "n_own", "t", "score", "bucket", "note", "mean5")}
    keep["features"] = {k: (round(v, 3) if isinstance(v, float) and not np.isnan(v) else None)
                        for k, v in (plan.get("features") or {}).items()}
    own = plan.get("own") or {}
    keep["own_history_5d"] = own.get(5)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in keep.items()}

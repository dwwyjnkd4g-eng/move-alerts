"""
Quant engine v2 — turns a flagged move into a statistically grounded, event-aware trade plan.

MODEL
  1. Event panel. Every large idiosyncratic move (|idio z| >= 3, |move| >= 3%) across the liquid
     universe over the past year, with forward idio fade returns at 1/2/3/5/10 trading days and
     features: move size (|z|), log relative volume, gap fraction, liquidity (log 20d $ volume —
     the market-cap proxy), vol regime (20d / 1y vol), prior 5d/20d momentum in the move's
     direction, RSI(2) extremeness, distance from the 20-day mean (sigma), market trend (index
     20d), breadth (% of universe above its 50-day mean), VIX level and VIX/VIX3M term structure.
  2. Ridge regression per horizon (penalty chosen by 5-fold CV) -> conditional expected fade.
  3. k-nearest-neighbour analogs (similar size, volume, gap, liquidity, vol regime, direction,
     VIX) -> empirical outcome distribution: hit rate, 15th / 50th / 75th percentiles, dispersion.
  4. Ensemble with the bucket base rate and the stock's own 5-year history (precision-weighted).
  5. Ornstein-Uhlenbeck fit of the expected path m(h) = A(1 - e^(-kh)) -> half-life ln2/k and
     total expected retrace A.
  6. Volatility: post-shock EWMA of the stock's idio variance decaying to its long-run level
     (GARCH-style persistence 0.94) -> sigma over the hold.
  7. Hold = argmax over 1..10 trading days of (m(h) - costs) / sigma(h), then capped so the
     position is flat before the stock's next earnings.
  8. Target = m(h*); stretch = neighbours' 75th pct; stop = max(k * sigma(h*) * event factor,
     neighbours' adverse 15th pct). Costs by liquidity tier (+ UK stamp duty / French FTT on
     buys). Half-Kelly and fixed-risk position size.
EVENTS in the hold window: earnings, FOMC, US elections, high-impact data (ForexFactory), market
holidays and early closes, monthly options expiry / quarterly triple-witching & index rebalances.
REGIME: VIX level and term structure, index trend, breadth -> risk-on / neutral / risk-off.

Macro markets use the instrument's own last ~60 days of hourly data with the same OU / vol / event logic.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
HORIZONS = [1, 2, 3, 5, 10]
BETA_W, SD_W = 60, 40
EV_Z, EV_RET = 3.0, 0.03
ANALOG_Z = 2.5
SHRINK_K = 20
MIN_BUCKET = 30
EWMA_PHI = 0.94
STOP_K = 1.0
MIN_RR = float(os.environ.get("MIN_RR", "0.33"))   # below this reward/risk the plan is "no edge"
# Backtest (US, Jun 2024 - Oct 2026): only plans scoring 75+ made money net of costs; below that they lost.
MIN_PLAN_SCORE = int(os.environ.get("MIN_PLAN_SCORE", "75"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.005"))   # 0.5% of capital at risk per trade
FX_COST = float(os.environ.get("FX_COST", "0"))                     # e.g. 0.005 for 0.25% each way

FEATURES = ["dir", "absz", "lrvol", "gapf", "lv", "volreg", "mom5d", "mom20d", "rsi2x", "dist20d",
            "mkt20d", "breadth", "vix", "vixts"]
KNN_FEATURES = {"dir": 3.0, "absz": 1.5, "lrvol": 1.0, "gapf": 1.0, "lv": 1.0, "volreg": 0.8, "vix": 0.8,
                "mom20d": 0.5}

# --------------------------------------------------------------------------
# Calendars (NYSE official 2026-2028; FOMC published schedule; LSE/Euronext core holidays)
# --------------------------------------------------------------------------
US_HOLIDAYS = {date(*d) for d in [
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19), (2026, 7, 3),
    (2026, 9, 7), (2026, 11, 26), (2026, 12, 25),
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26), (2027, 5, 31), (2027, 6, 18), (2027, 7, 5),
    (2027, 9, 6), (2027, 11, 25), (2027, 12, 24),
    (2028, 1, 17), (2028, 2, 21), (2028, 4, 14), (2028, 5, 29), (2028, 6, 19), (2028, 7, 4), (2028, 9, 4),
    (2028, 11, 23), (2028, 12, 25)]}
US_EARLY_CLOSE = {date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26), date(2028, 7, 3), date(2028, 11, 24)}
UK_HOLIDAYS = {date(*d) for d in [
    (2026, 1, 1), (2026, 4, 3), (2026, 4, 6), (2026, 5, 4), (2026, 5, 25), (2026, 8, 31), (2026, 12, 25),
    (2026, 12, 28), (2027, 1, 1), (2027, 3, 26), (2027, 3, 29), (2027, 5, 3), (2027, 5, 31), (2027, 8, 30),
    (2027, 12, 27), (2027, 12, 28)]}
EU_HOLIDAYS = {date(*d) for d in [
    (2026, 1, 1), (2026, 4, 3), (2026, 4, 6), (2026, 5, 1), (2026, 12, 24), (2026, 12, 25), (2026, 12, 31),
    (2027, 1, 1), (2027, 3, 26), (2027, 3, 29), (2027, 12, 24), (2027, 12, 31)]}
FOMC = {date(*d) for d in [
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17), (2026, 7, 29), (2026, 9, 16), (2026, 10, 28),
    (2026, 12, 9), (2027, 1, 27), (2027, 3, 17), (2027, 4, 28), (2027, 6, 9), (2027, 7, 28), (2027, 9, 15),
    (2027, 10, 27), (2027, 12, 8)]}
ELECTIONS = {date(2026, 11, 3): "US midterm elections", date(2028, 11, 7): "US presidential election"}
EXTRA_EVENTS = HERE / "events.json"    # optional: {"2026-12-01": "OPEC+ meeting", ...}


def market_of(sym: str) -> str:
    if sym.endswith(".L"):
        return "uk"
    if "." in sym[-4:] and not sym.endswith(".US"):
        return "eu"
    return "us"


def holidays_for(mkt: str) -> set[date]:
    return {"us": US_HOLIDAYS, "uk": UK_HOLIDAYS, "eu": EU_HOLIDAYS}.get(mkt, US_HOLIDAYS)


def add_trading_days(start: date, n: int, mkt: str) -> date:
    d, hol, k = start, holidays_for(mkt), 0
    while k < n:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            k += 1
    return d


def trading_days_between(a: date, b: date, mkt: str) -> int:
    """Trading days strictly after a, up to and including b."""
    hol, n, d = holidays_for(mkt), 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            n += 1
    return n


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def calendar_events(start: date, end: date, mkt: str, ff_events: list[dict] | None = None) -> list[dict]:
    """Market-moving / liquidity events between start (exclusive of past) and end (inclusive)."""
    out = []
    d = start
    extra = {}
    try:
        extra = {date.fromisoformat(k): v for k, v in json.loads(EXTRA_EVENTS.read_text()).items()}
    except Exception:
        pass
    while d <= end:
        if d in FOMC:
            out.append({"date": d, "label": "FOMC decision", "kind": "macro", "severity": 2})
        if d in ELECTIONS:
            out.append({"date": d, "label": ELECTIONS[d], "kind": "macro", "severity": 2})
        if d in extra:
            out.append({"date": d, "label": extra[d], "kind": "macro", "severity": 1})
        if d in holidays_for(mkt) and d.weekday() < 5:
            out.append({"date": d, "label": "market holiday", "kind": "liquidity", "severity": 1})
        if mkt == "us" and d in US_EARLY_CLOSE:
            out.append({"date": d, "label": "early close (thin liquidity)", "kind": "liquidity", "severity": 1})
        if d == third_friday(d.year, d.month):
            q = d.month in (3, 6, 9, 12)
            out.append({"date": d, "label": "quarterly triple witching / index rebalance" if q else "monthly options expiry",
                        "kind": "liquidity", "severity": 2 if q else 1})
        d += timedelta(days=1)
    for ev in ff_events or []:
        if start <= ev["date"] <= end:
            out.append(ev)
    return sorted(out, key=lambda e: e["date"])


def ff_high_impact() -> list[dict]:
    """This week's high-impact releases from the ForexFactory feed (cached by econ.py)."""
    try:
        from econ import load_calendar
        out = []
        for ev in load_calendar():
            if ev.get("impact") == "High":
                out.append({"date": ev["ts"].date(), "label": f"{ev.get('country')} {ev.get('title')}",
                            "kind": "macro", "severity": 2 if ev.get("country") in ("USD",) else 1})
        return out
    except Exception:
        return []


def next_earnings(sym: str, days: int = 25) -> date | None:
    if market_of(sym) != "us":
        return None
    try:
        from scanner import _finnhub
        t = date.today()
        data = _finnhub("calendar/earnings", {"symbol": sym.replace("-", "."), "from": t.isoformat(),
                                              "to": (t + timedelta(days=days)).isoformat()}) or {}
        ds = sorted(date.fromisoformat(e["date"]) for e in data.get("earningsCalendar", []) if e.get("date"))
        return ds[0] if ds else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Regime
# --------------------------------------------------------------------------
def vix_history() -> pd.DataFrame:
    path = CACHE / "vix.pkl"
    try:
        v = pd.read_pickle(path)
        if datetime.now(timezone.utc).timestamp() - path.stat().st_mtime < 3600:
            return v
    except Exception:
        pass
    try:
        import yfinance as yf
        px = yf.download(["^VIX", "^VIX3M"], period="2y", interval="1d", progress=False, auto_adjust=False)["Close"]
        px.index = pd.to_datetime(px.index.date)
        v = pd.DataFrame({"vix": px["^VIX"], "vixts": px["^VIX"] / px["^VIX3M"]}).ffill()
        CACHE.mkdir(exist_ok=True)
        pd.to_pickle(v, path)
        return v
    except Exception as e:
        print(f"  VIX history failed: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["vix", "vixts"])


def regime_now(base: dict) -> dict:
    v = vix_history()
    out = {"vix": float(v["vix"].iloc[-1]) if len(v) else None,
           "vixts": float(v["vixts"].iloc[-1]) if len(v) else None,
           "mkt20d": base.get("mkt20_last"), "breadth": base.get("breadth_last")}
    score = 0
    if out["vix"] is not None:
        score += (out["vix"] > 22) + (out["vix"] > 30) - (out["vix"] < 15)
    if out["vixts"] is not None:
        score += out["vixts"] > 1.0
    if out["mkt20d"] is not None:
        score += out["mkt20d"] < -0.03
        score -= out["mkt20d"] > 0.03
    if out["breadth"] is not None:
        score += out["breadth"] < 0.4
        score -= out["breadth"] > 0.65
    out["label"] = "risk-off" if score >= 2 else ("risk-on" if score <= -1 else "neutral")
    return out


# --------------------------------------------------------------------------
# Event panel (vectorised)
# --------------------------------------------------------------------------
def idio_panel(close: pd.DataFrame, bench_of: dict[str, str]):
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


def _rsi_panel(close: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def events_table(close: pd.DataFrame, open_: pd.DataFrame | None, volume: pd.DataFrame | None,
                 bench_of: dict[str, str], z_min: float = EV_Z, ret_min: float = EV_RET,
                 min_dv: float = 0.0, direction: int | None = None, market: str | None = None,
                 vix: pd.DataFrame | None = None, with_features: bool = True) -> pd.DataFrame:
    resid, z, r = idio_panel(close, bench_of)
    if z.empty:
        return pd.DataFrame()
    syms = list(z.columns)
    c = close[syms]
    raw = np.expm1(r[syms])
    mask = (z.abs() >= z_min) & (raw.abs() >= ret_min)
    if direction is not None:
        mask &= np.sign(z) == direction
    v = volume.reindex(columns=syms) if volume is not None and not volume.empty else None
    dv = (c * v).rolling(20).mean().shift(1) if v is not None else None
    if dv is not None and min_dv:
        mask &= dv >= min_dv
    st = mask.stack()
    st = st[st]
    if st.empty:
        return pd.DataFrame()
    idx = st.index

    def at(df):
        return df.stack().reindex(idx).values if df is not None else np.nan

    out = pd.DataFrame({"z": at(z), "ret": at(raw)}, index=idx)
    out["z"] = pd.to_numeric(out["z"], errors="coerce").astype("float64")
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce").astype("float64")
    out["dir"] = np.sign(out["z"])
    for h in HORIZONS:
        out[f"fade_{h}"] = -out["dir"] * at(resid.rolling(h).sum().shift(-h))
    if not with_features:
        return out.reset_index(names=["date", "symbol"])
    vol20 = r[syms].rolling(20).std().shift(1)
    vol250 = r[syms].rolling(250, min_periods=120).std().shift(1)
    out["absz"] = out["z"].abs()
    out["lrvol"] = np.log(np.clip(at(v / v.rolling(20).mean().shift(1)) if v is not None else np.nan, 0.05, 50))
    if open_ is not None and not open_.empty:
        o = open_.reindex(columns=syms)
        out["gapf"] = np.clip(at(np.log(o / c.shift(1)) / r[syms]), -1, 2)
    else:
        out["gapf"] = np.nan
    out["lv"] = np.log10(np.clip(at(dv), 1e5, None)) if dv is not None else np.nan
    out["volreg"] = np.clip(at(vol20 / vol250), 0.2, 5)
    out["mom5d"] = out["dir"] * at(c.shift(1) / c.shift(6) - 1)
    out["mom20d"] = out["dir"] * at(c.shift(1) / c.shift(21) - 1)
    rsi2 = at(_rsi_panel(c, 2))
    out["rsi2x"] = np.where(out["dir"] < 0, 100 - rsi2, rsi2)        # high = more stretched in the move's direction
    out["dist20d"] = out["dir"] * at(np.log(c / c.rolling(20).mean()) / (vol20 * math.sqrt(5)))
    dates = idx.get_level_values(0)
    if market and market in close:
        m20 = (close[market] / close[market].shift(20) - 1)
        out["mkt20d"] = out["dir"] * m20.reindex(dates).values
    else:
        out["mkt20d"] = np.nan
    breadth = (c > c.rolling(50).mean()).where(c.notna()).mean(axis=1)
    out["breadth"] = breadth.reindex(dates).values
    if vix is not None and len(vix):
        vv = vix.reindex(pd.to_datetime(dates), method="ffill")
        out["vix"], out["vixts"] = vv["vix"].values, vv["vixts"].values
    else:
        out["vix"], out["vixts"] = np.nan, np.nan
    return _numeric(out.reset_index(names=["date", "symbol"]))


def _numeric(ev: pd.DataFrame) -> pd.DataFrame:
    """Force feature/outcome columns to float64 (pandas 3 can hand back object dtype from stack/reindex)."""
    if ev is None or ev.empty:
        return ev
    num = set(FEATURES) | set(KNN_FEATURES) | {"z", "ret", "vol250", "price"} | {f"fade_{h}" for h in HORIZONS}
    bad = [c for c in ev.columns if c in num and ev[c].dtype != np.float64]
    if not bad:                       # already clean: no copy (called per event in the backtest)
        return ev
    ev = ev.copy()
    for c in bad:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("float64")
    return ev


# --------------------------------------------------------------------------
# Buckets (kept as a robust prior)
# --------------------------------------------------------------------------
def _bucket_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rv = np.exp(df["lrvol"]) if "lrvol" in df else df.get("rvol", pd.Series(1.0, index=df.index))
    df["zb"] = pd.cut(df["z"].abs(), [0, 4, 6, np.inf], labels=["3-4σ", "4-6σ", "6σ+"], right=False).astype(str)
    df["rvb"] = pd.cut(pd.Series(rv, index=df.index).fillna(1.0), [-np.inf, 2, 5, np.inf],
                       labels=["rvol<2", "rvol2-5", "rvol5+"]).astype(str)
    df["gb"] = np.where(df["gapf"].fillna(0) >= 0.6, "gap", "intraday")
    return df


def _stats(x: pd.Series, w: np.ndarray | None = None) -> dict:
    x = pd.Series(x).astype(float)
    ok = x.notna().values
    x = x[ok]
    if len(x) < 3:
        return {"n": int(len(x))}
    if w is None:
        return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()),
                "hit": float((x > 0).mean()), "p15": float(x.quantile(0.15)), "p75": float(x.quantile(0.75))}
    w = np.asarray(w)[ok]
    w = w / w.sum()
    xs = x.values
    order = np.argsort(xs)
    cw = np.cumsum(w[order])
    q = lambda p: float(xs[order][np.searchsorted(cw, p)])
    mean = float(np.sum(w * xs))
    return {"n": int(len(xs)), "mean": mean, "median": q(0.5),
            "std": float(math.sqrt(max(np.sum(w * (xs - mean) ** 2), 1e-12))),
            "hit": float(np.sum(w * (xs > 0))), "p15": q(0.15), "p75": q(0.75)}


def build_base_rates(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {"levels": {}, "n_events": 0}
    ev = _bucket_cols(ev)
    levels = {}
    for keys in (("dir", "zb", "rvb", "gb"), ("dir", "zb", "rvb"), ("dir", "zb"), ("dir",)):
        tab = {}
        for k, g in ev.groupby(list(keys)):
            k = k if isinstance(k, tuple) else (k,)
            tab[tuple(str(int(v)) if isinstance(v, (float, np.floating)) else str(v) for v in k)] = \
                {h: _stats(g[f"fade_{h}"]) for h in HORIZONS}
        levels["|".join(keys)] = tab
    return {"levels": levels, "n_events": int(len(ev))}


def lookup(br: dict, direction: int, z: float, rvol: float | None, gapf: float | None) -> tuple[dict, str]:
    one = pd.DataFrame([{"z": z, "lrvol": math.log(max(rvol, 0.05)) if rvol and not pd.isna(rvol) else np.nan,
                         "gapf": gapf if gapf is not None else np.nan}])
    b = _bucket_cols(one).iloc[0]
    d = str(int(direction))
    for level, key, label in [("dir|zb|rvb|gb", (d, b["zb"], b["rvb"], b["gb"]), f"{b['zb']}, {b['rvb']}, {b['gb']}"),
                              ("dir|zb|rvb", (d, b["zb"], b["rvb"]), f"{b['zb']}, {b['rvb']}"),
                              ("dir|zb", (d, b["zb"]), b["zb"]), ("dir", (d,), "all sizes")]:
        s = br.get("levels", {}).get(level, {}).get(key)
        if s and s.get(5, {}).get("n", 0) >= MIN_BUCKET:
            return s, label
    return {}, ""


# --------------------------------------------------------------------------
# Ridge + kNN
# --------------------------------------------------------------------------
def _standardise(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    mu = {c: float(df[c].mean()) if df[c].notna().any() else 0.0 for c in cols}
    sd = {c: float(df[c].std()) if df[c].notna().sum() > 2 and df[c].std() > 0 else 1.0 for c in cols}
    X = np.column_stack([((df[c] - mu[c]) / sd[c]).fillna(0.0).values for c in cols]).astype("float64")
    return X, {"mu": mu, "sd": sd}


def _apply_std(x: dict, cols: list[str], scaler: dict) -> np.ndarray:
    out = []
    for c in cols:
        try:
            v = float(x.get(c))
        except (TypeError, ValueError):
            v = float("nan")
        out.append(0.0 if math.isnan(v) else (v - scaler["mu"][c]) / scaler["sd"][c])
    return np.array(out, dtype="float64")


def fit_ridge(ev: pd.DataFrame) -> dict:
    ev = _numeric(ev)
    cols = [c for c in FEATURES if c in ev and ev[c].notna().mean() > 0.5]
    if len(ev) < 80 or not cols:
        return {}
    X, scaler = _standardise(ev, cols)
    rng = np.random.default_rng(0)
    folds = rng.integers(0, 5, len(ev))
    models = {}
    for h in HORIZONS:
        y = ev[f"fade_{h}"].to_numpy(dtype="float64")
        ok = ~np.isnan(y)
        Xh, yh, fh = X[ok], y[ok], folds[ok]
        best = None
        for lam in (1.0, 10.0, 100.0, 1000.0):
            err = 0.0
            for k in range(5):
                tr, te = fh != k, fh == k
                if te.sum() < 5 or tr.sum() < 20:
                    continue
                mu_y = yh[tr].mean()
                A = Xh[tr].T @ Xh[tr] + lam * np.eye(len(cols))
                b = np.linalg.solve(A, Xh[tr].T @ (yh[tr] - mu_y))
                err += float(np.sum((yh[te] - mu_y - Xh[te] @ b) ** 2))
            if best is None or err < best[0]:
                best = (err, lam)
        lam = best[1]
        mu_y = yh.mean()
        b = np.linalg.solve(Xh.T @ Xh + lam * np.eye(len(cols)), Xh.T @ (yh - mu_y))
        resid = yh - mu_y - Xh @ b
        models[h] = {"b": b.tolist(), "mu_y": float(mu_y), "lam": lam, "rmse": float(resid.std()), "n": int(ok.sum()),
                     "lo": float(np.quantile(yh, 0.10)), "hi": float(np.quantile(yh, 0.90))}
    return {"cols": cols, "scaler": scaler, "models": models}


USE_GBM = os.environ.get("QUANT_GBM", "0") == "1"   # turn on after the backtest supports it


def fit_gbm(train: pd.DataFrame):
    """Shallow gradient-boosted trees per horizon (captures interactions ridge can't)."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:
        return None
    train = _numeric(train)
    cols = [c for c in FEATURES if c in train]
    models = {}
    for h in HORIZONS:
        y = train[f"fade_{h}"]
        ok = y.notna()
        if ok.sum() < 200:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                          n_iter_no_change=20, random_state=0)
        m.fit(train.loc[ok, cols].to_numpy(dtype="float64"), y[ok].to_numpy(dtype="float64"))
        models[h] = m
    return {"cols": cols, "models": models} if models else None


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def gbm_predict_one(g: dict | None, x: dict) -> dict:
    if not g:
        return {}
    row = np.array([[_f(x.get(c)) for c in g["cols"]]], dtype=float)
    return {h: float(m.predict(row)[0]) for h, m in g["models"].items()}


def ridge_predict(model: dict, x: dict) -> dict:
    if not model:
        return {}
    xs = np.clip(_apply_std(x, model["cols"], model["scaler"]), -3, 3)     # no extrapolation beyond ±3 sd
    return {h: float(np.clip(m["mu_y"] + float(np.dot(xs, m["b"])), m.get("lo", -1), m.get("hi", 1)))
            for h, m in model["models"].items()}


def knn_stats(ev: pd.DataFrame, x: dict, k: int | None = None) -> dict:
    ev = _numeric(ev)
    cols = [c for c in KNN_FEATURES if c in ev and ev[c].notna().mean() > 0.5]
    if len(ev) < 40 or not cols:
        return {}
    X, scaler = _standardise(ev, cols)
    w = np.array([KNN_FEATURES[c] for c in cols])
    q = _apply_std(x, cols, scaler)
    d = np.sqrt((((X - q) * w) ** 2).sum(axis=1))
    k = k or int(np.clip(len(ev) // 8, 25, 80))
    nn = np.argsort(d)[:k]
    wts = 1.0 / (d[nn] + 0.25)
    out = {h: _stats(ev[f"fade_{h}"].values[nn], wts) for h in HORIZONS}
    out["k"] = int(k)
    out["dist"] = float(np.median(d[nn]))
    return out


# --------------------------------------------------------------------------
# Base-rate cache (panel + bucket stats + ridge model)
# --------------------------------------------------------------------------
def base_rates_cached(name: str, loader, market: str | None = None) -> dict:
    path = CACHE / f"base_v2_{name}.pkl"
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        br = pd.read_pickle(path)
        if br.get("day") == today:
            return br
    except Exception:
        pass
    try:
        close, open_, volume, bench_of, min_dv = loader()
        mkt = market if market in close else None
        ev = events_table(close, open_, volume, bench_of, min_dv=min_dv, market=mkt, vix=vix_history())
        br = build_base_rates(ev)
        br["ev"] = ev
        br["ridge"] = fit_ridge(ev)
        br["gbm"] = fit_gbm(ev) if USE_GBM else None
        c = close[[s for s in bench_of if s in close]]
        br["breadth_last"] = float((c.iloc[-1] > c.iloc[-50:].mean()).mean()) if len(c) > 50 else None
        br["mkt20_last"] = float(close[mkt].iloc[-1] / close[mkt].iloc[-21] - 1) if mkt else None
        br["market"] = mkt
        br["day"] = today
        CACHE.mkdir(exist_ok=True)
        pd.to_pickle(br, path)
        print(f"  base rates v2 ({name}): {br['n_events']} events, ridge on {len(br['ridge'].get('cols', []))} features")
        return br
    except Exception as e:
        print(f"  base rates ({name}) failed: {e}", file=sys.stderr)
        return {"levels": {}, "n_events": 0}


# --------------------------------------------------------------------------
# Stock's own history
# --------------------------------------------------------------------------
def stock_analogs(yf_symbol: str, bench: str, direction: int) -> dict:
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
        ev = events_table(close, None, None, {yf_symbol: bench}, z_min=ANALOG_Z, ret_min=0.02,
                          direction=direction, with_features=False)
        if not ev.empty:
            ev = ev[pd.to_datetime(ev["date"]) < close.index[-1] - pd.Timedelta(days=3)]
        out.update({h: _stats(ev[f"fade_{h}"]) if not ev.empty else {"n": 0} for h in HORIZONS})
        # idio vol for the EWMA forecast
        resid, _, _ = idio_panel(close, {yf_symbol: bench})
        if not resid.empty:
            rs = resid[yf_symbol].dropna()
            out["sig_lr"] = float(rs.iloc[-250:].std())
    except Exception as e:
        print(f"  analogs failed for {yf_symbol}: {e}", file=sys.stderr)
        out.update({h: {"n": 0} for h in HORIZONS})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


# --------------------------------------------------------------------------
# Features for the current move
# --------------------------------------------------------------------------
def rsi(c: pd.Series, n: int) -> float:
    v = _rsi_panel(c.to_frame("x"), n)["x"]
    return float(v.iloc[-1]) if len(v) and not np.isnan(v.iloc[-1]) else float("nan")


def setup_features(close: pd.Series, intraday: pd.Series | None = None) -> dict:
    c = close.dropna()
    if len(c) < 30:
        return {}
    lr = np.log(c).diff()
    sd = float(lr.iloc[-61:-1].std())
    f = {"rsi2": rsi(c, 2), "rsi14": rsi(c, 14), "ret20": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else np.nan,
         "ret5p": float(c.iloc[-2] / c.iloc[-7] - 1) if len(c) > 7 else np.nan,
         "ret20p": float(c.iloc[-2] / c.iloc[-22] - 1) if len(c) > 22 else np.nan}
    for n in (20, 50, 200):
        if len(c) > n:
            ma = c.iloc[-n:].mean()
            f[f"ma{n}_sig"] = float(math.log(c.iloc[-1] / ma) / (sd * math.sqrt(n / 4))) if sd else np.nan
    w = c.iloc[-252:]
    f["pos52"] = float((c.iloc[-1] - w.min()) / (w.max() - w.min())) if w.max() > w.min() else np.nan
    f["vol_d"] = sd
    f["vol20"] = float(lr.iloc[-21:-1].std())
    f["vol250"] = float(lr.iloc[-251:-1].std())
    if intraday is not None and len(intraday) >= 2:
        hi, lo = intraday.max(), intraday.min()
        f["clv"] = float((intraday.iloc[-1] - lo) / (hi - lo)) if hi > lo else np.nan
        f["since_open"] = float(intraday.iloc[-1] / intraday.iloc[0] - 1)
    return f


def model_x(row, feats: dict, regime: dict, dv: float | None) -> dict:
    direction = 1.0 if row["idio_z"] > 0 else -1.0
    rv = row.get("rvol")
    g = row.get("gap")
    gapf = math.log1p(g) / math.log1p(row["ret"]) if g is not None and not pd.isna(g) and row["ret"] else None
    rsi2 = feats.get("rsi2", np.nan)
    return {
        "dir": direction, "absz": abs(float(row["idio_z"])),
        "lrvol": math.log(max(float(rv), 0.05)) if rv is not None and not pd.isna(rv) else None,
        "gapf": max(-1.0, min(2.0, gapf)) if gapf is not None else None,
        "lv": math.log10(max(dv, 1e5)) if dv else None,
        "volreg": (feats["vol20"] / feats["vol250"]) if feats.get("vol250") else None,
        "mom5d": direction * feats["ret5p"] if not pd.isna(feats.get("ret5p", np.nan)) else None,
        "mom20d": direction * feats["ret20p"] if not pd.isna(feats.get("ret20p", np.nan)) else None,
        "rsi2x": (100 - rsi2 if direction < 0 else rsi2) if not pd.isna(rsi2) else None,
        "dist20d": direction * feats["ma20_sig"] if not pd.isna(feats.get("ma20_sig", np.nan)) else None,
        "mkt20d": direction * regime["mkt20d"] if regime.get("mkt20d") is not None else None,
        "breadth": regime.get("breadth"), "vix": regime.get("vix"), "vixts": regime.get("vixts"),
    }


# --------------------------------------------------------------------------
# Maths: ensemble, OU fit, vol forecast, costs, sizing
# --------------------------------------------------------------------------
def ensemble(ridge: dict, knn: dict, bucket: dict, own: dict) -> dict:
    out = {}
    for h in HORIZONS:
        parts = []   # (estimate, weight)
        if h in ridge:
            parts.append((ridge[h], 1.5))
        if knn.get(h, {}).get("mean") is not None:
            parts.append((knn[h]["mean"], 1.5))
        if bucket.get(h, {}).get("mean") is not None:
            parts.append((bucket[h]["mean"], 0.75))
        no = own.get(h, {}).get("n", 0) if own else 0
        if no and own[h].get("mean") is not None:
            parts.append((own[h]["mean"], 1.5 * no / (no + SHRINK_K)))
        if not parts:
            continue
        wsum = sum(w for _, w in parts)
        disp = knn.get(h) or bucket.get(h) or {}
        out[h] = {"mean": sum(e * w for e, w in parts) / wsum, "std": disp.get("std"), "hit": disp.get("hit"),
                  "p15": disp.get("p15"), "p75": disp.get("p75"), "n": disp.get("n", 0)}
    return out


def fit_ou(curve: dict[int, float]) -> tuple[float, float, float] | None:
    """m(h) = A(1 - exp(-k h)). Returns (A, k, sse) for the best k with A > 0, else None."""
    hs = np.array(sorted(curve))
    m = np.array([curve[h] for h in hs])
    best = None
    for k in np.geomspace(0.03, 3.0, 80):
        g = 1 - np.exp(-k * hs)
        A = float(g @ m / (g @ g))
        if A <= 0:
            continue
        sse = float(np.sum((m - A * g) ** 2))
        if best is None or sse < best[2]:
            best = (A, float(k), sse)
    return best


def sigma_path(sig0: float, sig_lr: float, h: int, phi: float = EWMA_PHI) -> float:
    """Std of the cumulative idio return over h days with variance decaying from sig0^2 to sig_lr^2."""
    var = sum(sig_lr ** 2 + (phi ** i) * (sig0 ** 2 - sig_lr ** 2) for i in range(1, h + 1))
    return math.sqrt(max(var, 1e-12))


def cost_estimate(sym: str, dv: float | None, side: str) -> float:
    """Round-trip cost as a fraction: spread + slippage by liquidity tier, plus transaction taxes on buys."""
    dv = dv or 1e7
    spread = 2e-4 if dv >= 1e9 else 5e-4 if dv >= 2e8 else 1e-3 if dv >= 5e7 else 2e-3 if dv >= 2e7 else 3.5e-3
    cost = 2 * spread + FX_COST
    if side == "long":
        if sym.endswith(".L"):
            cost += 0.005      # UK stamp duty on purchases (AIM shares exempt)
        elif sym.endswith(".PA"):
            cost += 0.003      # French FTT on large caps
        elif sym.endswith(".MI"):
            cost += 0.001      # Italian FTT
    return cost


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def make_plan_v2(sym: str, price: float, move: float, stats: dict, sig0: float, sig_lr: float,
                 dv: float | None, mkt: str, events: list[dict], earnings: date | None,
                 today: date | None = None) -> dict:
    today = today or date.today()
    side = "long" if move < 0 else "short"
    sgn = 1 if side == "long" else -1
    curve = {h: s["mean"] for h, s in stats.items() if s.get("mean") is not None}
    cost = cost_estimate(sym, dv, side)
    if not curve or max(curve.values()) <= cost:
        m5 = curve.get(5)
        return {"edge": False, "note": "history and conditions favour continuation (no fade edge after costs)"
                if curve else "not enough comparable history", "mean5": m5, "n": stats.get(5, {}).get("n", 0),
                "cost": cost, "side": side}
    ou = fit_ou(curve)
    A, k = (ou[0], ou[1]) if ou else (max(curve.values()), 1.0)
    half_life = math.log(2) / k
    # hold optimisation on the OU path, net of costs, vs forecast dispersion
    best = None
    for h in range(1, 11):
        m_h = A * (1 - math.exp(-k * h))
        s_model = sigma_path(sig0, sig_lr, h)
        s_emp = stats.get(min(HORIZONS, key=lambda x: abs(x - h)), {}).get("std")
        s_h = 0.5 * s_model + 0.5 * s_emp if s_emp else s_model
        score = (m_h - cost) / s_h if s_h else -1
        if best is None or score > best[1]:
            best = (h, score, m_h, s_h)
    h, sharpe, m_h, s_h = best
    notes = []
    # earnings: be flat before the report
    if earnings:
        td = trading_days_between(today, earnings, mkt)
        if td <= h:
            cap = max(td - 1, 0)
            if cap < 1:
                return {"edge": False, "side": side, "note": f"earnings on {earnings:%a %d %b} — binary event inside any hold",
                        "mean5": curve.get(5), "n": stats.get(5, {}).get("n", 0), "cost": cost}
            notes.append(f"hold cut from {h}d to {cap}d to exit before earnings {earnings:%a %d %b}")
            h = cap
            m_h = A * (1 - math.exp(-k * h))
            s_h = sigma_path(sig0, sig_lr, h)
            sharpe = (m_h - cost) / s_h if s_h else 0
    exit_day = add_trading_days(today, h, mkt)
    window = [e for e in events if today <= e["date"] <= exit_day]
    sev = sum(e.get("severity", 1) for e in window if e["kind"] == "macro")
    liq = sum(1 for e in window if e["kind"] == "liquidity")
    event_factor = 1 + 0.12 * sev + 0.05 * liq
    sn = stats.get(min(HORIZONS, key=lambda x: abs(x - h)), {})
    stop_pct = max(STOP_K * s_h * event_factor, -(sn.get("p15") or 0), 0.005)
    tgt_pct = m_h
    stretch_pct = max(sn.get("p75") or 0, tgt_pct + 0.5 * s_h)
    p = sn.get("hit")
    if p is None:
        p = _norm_cdf(m_h / s_h)
    b = tgt_pct / stop_pct if stop_pct else 0
    kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
    size_cut = 1 / event_factor ** 2
    ev_net = m_h - cost
    if ev_net > 0 and b < MIN_RR:
        return {"edge": False, "side": side, "note": f"expected fade too small vs risk (R/R {b:.2f})",
                "mean5": curve.get(5), "n": stats.get(5, {}).get("n", 0), "cost": cost, "events": window, "notes": notes}
    if ev_net <= 0:
        return {"edge": False, "side": side, "note": "expected fade too small to cover costs and event risk",
                "mean5": curve.get(5), "n": stats.get(5, {}).get("n", 0), "cost": cost, "events": window,
                "notes": notes}
    n_eff = (sn.get("n") or 0) + (stats.get(5, {}).get("n_own") or 0)
    t = (ev_net / s_h) * math.sqrt(max(n_eff, 1)) if s_h else 0
    score = int(round(50 + 50 * math.tanh(1.5 * (ev_net / s_h if s_h else 0)) * min(1.0, max(t, 0) / 3)))
    return {
        "edge": ev_net > 0, "side": side, "hold": h, "unit": "d", "exit_by": exit_day,
        "half_life": half_life, "A": A, "kappa": k,
        "entry": price, "target": price * (1 + sgn * tgt_pct), "stretch": price * (1 + sgn * stretch_pct),
        "stop": price * (1 - sgn * stop_pct), "tgt_pct": tgt_pct, "stop_pct": stop_pct,
        "hit": p, "rr": b, "ev": ev_net, "cost": cost, "sigma_h": s_h, "sig0": sig0, "sig_lr": sig_lr,
        "kelly_half": min(0.25, kelly / 2) * size_cut, "risk_size": min(0.25, RISK_PER_TRADE / stop_pct) * size_cut,
        "event_factor": event_factor, "events": window, "notes": notes,
        "n": sn.get("n", 0), "t": t, "sharpe": ev_net / s_h if s_h else 0, "score": max(0, min(100, score)),
    }


def equity_plan(row, closes: pd.DataFrame, base: dict, intraday_close: pd.Series | None = None) -> dict:
    sym = row["yf_symbol"]
    bench = row["etf"] if row.get("etf") in closes else None
    direction = int(np.sign(row["idio_z"])) or 1
    mkt = market_of(sym)
    regime = regime_now(base)
    feats = setup_features(closes[sym], intraday_close) if sym in closes else {}
    g = row.get("gap")
    if "since_open" not in feats and g is not None and not pd.isna(g):
        feats["since_open"] = float((1 + row["ret"]) / (1 + g) - 1)
    dv = row.get("adv_usd")
    dv = float(dv) if dv is not None and not pd.isna(dv) else None
    x = model_x(row, feats, regime, dv)
    ev = base.get("ev")
    ridge = ridge_predict(base.get("ridge", {}), x)
    knn = knn_stats(ev, x) if isinstance(ev, pd.DataFrame) and len(ev) else {}
    gapf = x.get("gapf")
    bucket, label = lookup(base, direction, float(row["idio_z"]), row.get("rvol"), gapf)
    own = stock_analogs(sym, bench or ("^STOXX50E" if mkt != "us" else "SPY"), direction)
    stats = ensemble(ridge, knn, bucket, own)
    gb = gbm_predict_one(base.get("gbm"), x)
    for h, v in gb.items():                    # trees blended 50/50 with the ensemble mean
        if h in stats:
            stats[h]["mean"] = 0.5 * stats[h]["mean"] + 0.5 * v
    for h in stats:
        stats[h]["n_own"] = own.get(h, {}).get("n", 0)
    # vol: post-shock idio vol (today's |idio| enters the EWMA) decaying to the long-run level
    sig_lr = own.get("sig_lr") or feats.get("vol250") or feats.get("vol_d") or 0.02
    shock = abs(math.log1p(row.get("idio_ret", row["ret"])))
    sig0 = math.sqrt(EWMA_PHI * sig_lr ** 2 + (1 - EWMA_PHI) * shock ** 2) * 1.25
    events = calendar_events(date.today(), date.today() + timedelta(days=20), mkt, ff_high_impact())
    earn = next_earnings(sym)
    plan = make_plan_v2(sym, float(row["price"]), float(row["ret"]), stats, sig0, sig_lr, dv, mkt, events, earn)
    plan.update({"bucket": label, "features": feats, "regime": regime, "earnings": earn,
                 "model": {"ridge5": ridge.get(5), "gbm5": gb.get(5), "knn5": knn.get(5, {}).get("mean"), "knn_k": knn.get("k"),
                           "bucket5": bucket.get(5, {}).get("mean"), "own5": own.get(5, {}).get("mean"),
                           "own_n": own.get(5, {}).get("n", 0)},
                 "own": {h: own.get(h, {}) for h in HORIZONS}, "x": x})
    plan["prior"] = backtest_prior("us" if mkt == "us" else "eu")
    if plan.get("edge") and plan.get("score", 0) < MIN_PLAN_SCORE:
        plan.update({"edge": False, "score_raw": plan["score"],
                     "note": f"edge score {plan['score']}/100 is below {MIN_PLAN_SCORE} — in the backtest, "
                             f"plans under {MIN_PLAN_SCORE} lost money after costs, so no trade"})
    return plan


def backtest_prior(market: str) -> dict | None:
    """Headline numbers from the last walk-forward backtest (written by backtest.py), for the LLM and alerts."""
    try:
        return json.loads((CACHE / f"backtest_summary_{market}.json").read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------
# Macro markets: own hourly history, OU path, EWMA vol, event window
# --------------------------------------------------------------------------
MACRO_H = [1, 2, 4, 8, 24, 48, 72]


def macro_plan(src: str, sym: str, st: dict) -> dict:
    try:
        if src == "kraken":
            from macro import kraken_closes
            s = kraken_closes(sym, interval=60)
        elif src == "basket":
            return {"edge": False, "n": 0, "note": "basket proxy — no plan"}
        else:
            import yfinance as yf
            from macro import OTC_FUT
            d = yf.download(OTC_FUT.get(sym, sym), period="60d", interval="60m", progress=False, auto_adjust=False)["Close"]
            s = (d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d).dropna()
        if len(s) < 200:
            return {"edge": False, "n": 0, "note": "not enough history"}
        lr = np.log(s).diff()
        win = 1 if st["window"] == "1h" else 4
        mv = lr.rolling(win).sum()
        sd = lr.rolling(240).std().shift(win) * math.sqrt(win)
        z = mv / sd
        direction = 1 if st["ret"] > 0 else -1
        ev_idx = z.index[(np.sign(z) == direction) & (z.abs() >= 0.75 * abs(st["z"]))][:-1]
        kept, last = [], None
        for t in ev_idx:
            if last is None or (t - last) >= pd.Timedelta(hours=6):
                kept.append(t)
                last = t
        stats = {}
        for n in MACRO_H:
            vals = []
            for t in kept:
                i = s.index.get_loc(t)
                if i + n < len(s):
                    vals.append(-direction * (s.iloc[i + n] / s.iloc[i] - 1))
            stats[n] = _stats(pd.Series(vals))
        curve = {h: v["mean"] for h, v in stats.items() if v.get("mean") is not None}
        side = "long" if direction < 0 else "short"
        if not curve or max(curve.values()) <= 0:
            return {"edge": False, "n": len(kept), "side": side, "note": "this market's recent history favours continuation",
                    "mean5": curve.get(24)}
        ou = fit_ou(curve)
        A, k = (ou[0], ou[1]) if ou else (max(curve.values()), 0.1)
        sig_lr = float(lr.iloc[-500:].std())
        shock = abs(math.log1p(st["ret"])) / math.sqrt(win)
        sig0 = math.sqrt(EWMA_PHI * sig_lr ** 2 + (1 - EWMA_PHI) * shock ** 2) * 1.25
        best = None
        for h in MACRO_H:
            m_h = A * (1 - math.exp(-k * h))
            s_h = 0.5 * sigma_path(sig0, sig_lr, h) + 0.5 * (stats[h].get("std") or sigma_path(sig0, sig_lr, h))
            sc = m_h / s_h if s_h else -1
            if best is None or sc > best[1]:
                best = (h, sc, m_h, s_h)
        h, sharpe, m_h, s_h = best
        now = datetime.now(timezone.utc)
        exit_t = now + timedelta(hours=h)
        events = [e for e in calendar_events(now.date(), exit_t.date(), "us", ff_high_impact())
                  if e["kind"] == "macro"]
        ef = 1 + 0.12 * sum(e.get("severity", 1) for e in events)
        sn = stats[h]
        sgn = 1 if side == "long" else -1
        stop_pct = max(STOP_K * s_h * ef, -(sn.get("p15") or 0), 0.001)
        p = sn.get("hit") or _norm_cdf(m_h / s_h)
        b = m_h / stop_pct
        if b < MIN_RR:
            return {"edge": False, "n": len(kept), "side": side, "events": events,
                    "note": f"expected reversion too small vs risk (R/R {b:.2f})", "mean5": curve.get(24)}
        kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
        n_eff = sn.get("n", 0)
        t = sharpe * math.sqrt(max(n_eff, 1))
        price = float(st["price"])
        return {"edge": True, "side": side, "hold": h, "unit": "h", "exit_by": exit_t, "half_life": math.log(2) / k,
                "entry": price, "target": price * (1 + sgn * m_h),
                "stretch": price * (1 + sgn * max(sn.get("p75") or 0, m_h + 0.5 * s_h)),
                "stop": price * (1 - sgn * stop_pct), "tgt_pct": m_h, "stop_pct": stop_pct, "hit": p, "rr": b,
                "ev": m_h, "cost": 0.0, "sigma_h": s_h, "kelly_half": min(0.25, kelly / 2) / ef ** 2,
                "risk_size": min(0.25, RISK_PER_TRADE / stop_pct) / ef ** 2, "events": events, "notes": [],
                "event_factor": ef, "n": n_eff, "n_events": len(kept), "t": t,
                "score": max(0, min(100, int(round(50 + 50 * math.tanh(1.5 * sharpe) * min(1.0, max(t, 0) / 3)))))}
    except Exception as e:
        print(f"  macro plan failed for {sym}: {e}", file=sys.stderr)
        return {"edge": False, "n": 0, "note": "unavailable"}


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
def _p(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x*100:+.{d}f}%"


def fmt_px(p: float) -> str:
    return f"{p:,.0f}" if p >= 1000 else (f"{p:,.2f}" if p >= 10 else f"{p:.4f}")


def hold_label(plan: dict) -> str:
    h, u = plan.get("hold"), plan.get("unit", "d")
    if u == "h":
        return f"{h}h" if h < 24 else f"{h // 24}d"
    return f"{h} trading day" + ("s" if h != 1 else "")


def _exit_label(plan: dict) -> str:
    x = plan.get("exit_by")
    if isinstance(x, datetime):
        from zoneinfo import ZoneInfo
        return x.astimezone(ZoneInfo("Europe/London")).strftime("%a %H:%M UK")
    if isinstance(x, date):
        return x.strftime("%a %d %b")
    return ""


def plan_lines(plan: dict, ccy_note: str = "") -> list[str]:
    import html as _h
    e = _h.escape
    out = []
    if plan.get("edge"):
        hl = plan.get("half_life")
        hl_txt = (f"{hl:.1f}{'h' if plan.get('unit') == 'h' else 'd'}" if hl is not None else "")
        out.append(f"📐 <b>Quant: {plan['side'].upper()} fade</b> · edge {plan['score']}/100 · "
                   f"exit by <b>{_exit_label(plan)}</b> ({hold_label(plan)}" + (f", half-life {hl_txt}" if hl_txt else "") + ")")
        out.append(f"   Entry {fmt_px(plan['entry'])} → <b>target {fmt_px(plan['target'])}</b> ({_p(plan['tgt_pct'])}) · "
                   f"stretch {fmt_px(plan['stretch'])} · <b>stop {fmt_px(plan['stop'])}</b> (−{plan['stop_pct']*100:.1f}%){ccy_note}")
        out.append(f"   Win {plan['hit']*100:.0f}% · R/R {plan['rr']:.2f} · EV {_p(plan['ev'], 2)} net"
                   + (f" of ~{plan['cost']*100:.2f}% costs" if plan.get("cost") else "")
                   + f" · size: ½-Kelly {plan['kelly_half']*100:.0f}% / risk-{RISK_PER_TRADE*100:.1f}% {plan['risk_size']*100:.0f}% of capital")
        m = plan.get("model") or {}
        if m:
            bits = []
            if m.get("ridge5") is not None:
                bits.append(f"ridge {_p(m['ridge5'])}")
            if m.get("gbm5") is not None:
                bits.append(f"trees {_p(m['gbm5'])}")
            if m.get("knn5") is not None:
                bits.append(f"{m.get('knn_k')} nearest analogs {_p(m['knn5'])}")
            if m.get("bucket5") is not None:
                bits.append(f"bucket {_p(m['bucket5'])}")
            if m.get("own_n"):
                bits.append(f"own history {_p(m.get('own5'))} (n={m['own_n']})")
            out.append("   Model (5d fade): " + " · ".join(bits) + f" · t={plan['t']:.1f}")
        if plan.get("sigma_h"):
            out.append(f"   Vol: σ over hold {plan['sigma_h']*100:.1f}% (post-shock EWMA"
                       + (f" {plan['sig0']*100:.1f}%→{plan['sig_lr']*100:.1f}%/day" if plan.get("sig0") else "") + ")"
                       + (f" · stop widened ×{plan['event_factor']:.2f} for events" if plan.get("event_factor", 1) > 1.001 else ""))
    else:
        mean5 = plan.get("mean5")
        out.append(f"📐 <b>Quant: no fade edge</b> — {e(plan.get('note', ''))}"
                   + (f" (avg {_p(mean5)}, n={plan.get('n', 0)})" if mean5 is not None else ""))
    r = plan.get("regime") or {}
    if r.get("vix") is not None:
        out.append(f"   Regime: <b>{e(r['label'])}</b> · VIX {r['vix']:.1f}"
                   + (f" ({'backwardation' if r['vixts'] > 1 else 'contango'})" if r.get("vixts") else "")
                   + (f" · index 20d {_p(r['mkt20d'])}" if r.get("mkt20d") is not None else "")
                   + (f" · breadth {r['breadth']*100:.0f}% >50DMA" if r.get("breadth") is not None else ""))
    evs = plan.get("events") or []
    earn = plan.get("earnings")
    ev_txt = [f"{x['date']:%a} {x['label']}" for x in evs[:5]]
    if earn and not any("earnings" in t for t in ev_txt):
        ev_txt.append(f"earnings {earn:%a %d %b}")
    if ev_txt or plan.get("notes"):
        out.append("   Events in window: " + e(" · ".join(ev_txt) if ev_txt else "none")
                   + ("" if not plan.get("notes") else " · " + e("; ".join(plan["notes"]))))
    f = plan.get("features") or {}
    if f:
        bits = []
        for k, lab in (("rsi2", "RSI2"), ("rsi14", "RSI14")):
            if not pd.isna(f.get(k, np.nan)):
                bits.append(f"{lab} {f[k]:.0f}")
        for n in (20, 50, 200):
            v = f.get(f"ma{n}_sig")
            if v is not None and not pd.isna(v):
                bits.append(f"{v:+.1f}σ vs {n}DMA")
        if not pd.isna(f.get("ret20", np.nan)):
            bits.append(f"20d {_p(f['ret20'], 0)}")
        if not pd.isna(f.get("pos52", np.nan)):
            bits.append(f"52w pos {f['pos52']*100:.0f}%")
        if f.get("vol250") and f.get("vol20"):
            bits.append(f"vol regime {f['vol20']/f['vol250']:.1f}×")
        if not pd.isna(f.get("clv", np.nan)):
            bits.append(f"at {f['clv']*100:.0f}% of day range")
        if not pd.isna(f.get("since_open", np.nan)):
            bits.append(f"since open {_p(f['since_open'])}")
        if bits:
            out.append("   Setup: " + " · ".join(bits))
    return out


def plan_for_prompt(plan: dict) -> dict:
    keep = {k: plan.get(k) for k in ("edge", "side", "hold", "unit", "half_life", "tgt_pct", "stop_pct", "hit", "rr",
                                     "ev", "cost", "n", "t", "score", "bucket", "note", "mean5", "event_factor",
                                     "kelly_half", "risk_size")}
    keep["exit_by"] = str(plan.get("exit_by")) if plan.get("exit_by") else None
    keep["events_in_window"] = [f"{e['date']} {e['label']}" for e in plan.get("events") or []]
    keep["earnings"] = str(plan.get("earnings")) if plan.get("earnings") else None
    keep["notes"] = plan.get("notes")
    keep["regime"] = plan.get("regime")
    keep["model_5d"] = plan.get("model")
    if plan.get("prior"):
        keep["backtest_prior"] = plan["prior"]
    if plan.get("score_raw") is not None:
        keep["score_below_threshold"] = plan["score_raw"]
    keep["features"] = {k: (round(v, 3) if isinstance(v, float) and not np.isnan(v) else None)
                        for k, v in (plan.get("features") or {}).items()}
    return json.loads(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in keep.items()}, default=str))

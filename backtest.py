"""
Walk-forward backtest of the fade models — run on the server.

  python backtest.py                    # US, 3 years
  python backtest.py --market eu        # UK & Europe
  python backtest.py --years 4 --telegram

Method
  * Universe: today's liquid list (note: survivorship — delisted names are missing).
  * Events: every |idio z| >= 3 and |move| >= 3% day, with the same features the live model uses
    (all computed from data available at that day's close).
  * Walk-forward: for each month M, every model is trained only on events dated more than 15
    calendar days before M starts (embargo so 10-day outcomes can't leak), then trades M's events.
  * Entry at the event-day close (the alert fires during the move; a close entry is the realistic
    approximation). Exit at the close after the chosen hold, or at target/stop for the plan
    strategies. P&L is the raw (unhedged) stock return in the fade direction, net of estimated
    round-trip costs; a beta-hedged version is reported too.
Strategies
  baseline   fade every event, hold 3 days
  bucket     bucket base rate (direction x size x volume x gap)
  ridge      ridge regression on 14 features
  knn        nearest-neighbour analogs
  gbm        gradient-boosted trees (sklearn HistGradientBoosting, shallow, early stopping)
  ensemble   ridge + knn + bucket (what the live bot uses)
  ens+gbm    ensemble with the trees added
  plan       ensemble -> full live plan logic (OU hold, target/stop, R/R and cost filters)
  plan+gbm   same with trees in the ensemble
A strategy trades an event only if its best predicted horizon beats costs; hold = that horizon.
"""

from __future__ import annotations

import argparse
import json
import html
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import quant as Q

HERE = Path(__file__).resolve().parent
EMBARGO_DAYS = 15
MIN_TRAIN = 400


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(market: str, years: int):
    import scanner as S
    path = HERE / "cache" / f"bt_{market}_{years}y.pkl"
    try:
        d = pd.read_pickle(path)
        if d.get("asof") == date.today():
            print("Using cached backtest data.")
            return d
    except Exception:
        pass
    if market == "us":
        uni = S.load_universe()
        bench_of = dict(zip(uni["yf_symbol"], uni["etf"]))
        syms = list(uni["yf_symbol"]) + sorted(set(uni["etf"]) | {S.MARKET_ETF})
        index, min_dv = S.MARKET_ETF, S.MIN_DOLLAR_VOL
    else:
        import europe as E
        uni = E.load_universe()
        bench_of = dict(zip(uni["yf_symbol"], uni["bench"]))
        syms = list(uni["yf_symbol"]) + sorted(set(uni["bench"]))
        index, min_dv = "^STOXX50E", 0.0
    print(f"Downloading {years}y of daily data for {len(syms)} symbols (one-off, 10-30 min)…", flush=True)
    got = S.yf_chunked(syms, ("Open", "Close", "Volume"), period=f"{years}y", interval="1d", auto_adjust=True)
    for k, f in got.items():
        f.index = pd.to_datetime(f.index).tz_localize(None) if getattr(f.index, "tz", None) else pd.to_datetime(f.index)
    import yfinance as yf
    vx = yf.download(["^VIX", "^VIX3M"], period=f"{years}y", interval="1d", progress=False, auto_adjust=False)["Close"]
    vx.index = pd.to_datetime(vx.index.date)
    vix = pd.DataFrame({"vix": vx["^VIX"], "vixts": vx["^VIX"] / vx["^VIX3M"]}).ffill()
    d = {"asof": date.today(), "close": got["close"], "open": got["open"], "volume": got["volume"],
         "bench_of": bench_of, "index": index, "min_dv": min_dv, "vix": vix, "market": market}
    path.parent.mkdir(exist_ok=True)
    pd.to_pickle(d, path)
    return d


def build_events(d: dict) -> pd.DataFrame:
    close = d["close"]
    ev = Q.events_table(close, d["open"], d["volume"], d["bench_of"], min_dv=d["min_dv"],
                        market=d["index"] if d["index"] in close else None, vix=d["vix"])
    if ev.empty:
        return ev
    # raw forward path (10 closes after the event) and long-run vol, for P&L and the plan's vol model
    pos = {t: i for i, t in enumerate(close.index)}
    r = np.log(close).diff()
    vol250 = r.rolling(250, min_periods=120).std().shift(1)
    paths, vols, prices = [], [], []
    for t, s in zip(ev["date"], ev["symbol"]):
        i = pos[t]
        c = close[s].values
        p0 = c[i]
        seg = c[i + 1:i + 11] / p0 - 1
        paths.append(np.pad(seg, (0, 10 - len(seg)), constant_values=np.nan))
        vols.append(vol250[s].iloc[i])
        prices.append(p0)
    ev["path"] = paths
    ev["vol250"] = vols
    ev["price"] = prices
    ev["date"] = pd.to_datetime(ev["date"])
    ev = Q._numeric(ev)
    return ev.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
fit_gbm = Q.fit_gbm


def gbm_predict(g, X: pd.DataFrame) -> dict[int, np.ndarray]:
    if not g:
        return {}
    return {h: m.predict(X[g["cols"]].values) for h, m in g["models"].items()}


def choose(preds: dict[int, float], cost: float) -> tuple[int | None, float]:
    best = (None, 0.0)
    for h, p in preds.items():
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        net = p - cost
        if net > best[1]:
            best = (h, net)
    return best


def realise(e, h: int, side: int, cost: float) -> tuple[float, float]:
    raw = e["path"][h - 1]
    hedged = e[f"fade_{h}"]
    return side * raw - cost, (hedged - cost) if not pd.isna(hedged) else np.nan


def simulate_plan(e, plan: dict, cost: float) -> tuple[float, int, str]:
    side = 1 if plan["side"] == "long" else -1
    tgt, stp, h = plan["tgt_pct"], plan["stop_pct"], plan["hold"]
    for i in range(h):
        x = e["path"][i]
        if pd.isna(x):
            break
        if side * x >= tgt:
            return side * x - cost, i + 1, "target"
        if side * x <= -stp:
            return side * x - cost, i + 1, "stop"
    x = e["path"][h - 1]
    return (side * x - cost) if not pd.isna(x) else np.nan, h, "hold"


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------
def run(ev: pd.DataFrame, market: str) -> pd.DataFrame:
    ev = ev.copy()
    ev["month"] = ev["date"].dt.to_period("M")
    months = sorted(ev["month"].unique())
    trades = []
    try:
        import sklearn  # noqa: F401
        has_gbm = True
    except Exception:
        has_gbm = False
        print("  (scikit-learn not available — tree model skipped)", file=sys.stderr)
    t0 = time.time()
    for M in months:
        start = M.to_timestamp()
        train = ev[ev["date"] < start - pd.Timedelta(days=EMBARGO_DAYS)]
        test = ev[ev["month"] == M]
        if len(train) < MIN_TRAIN or test.empty:
            continue
        br = Q.build_base_rates(train)
        ridge = Q.fit_ridge(train)
        gbm = fit_gbm(train) if has_gbm else None
        gp = gbm_predict(gbm, test)
        for j, (_, e) in enumerate(test.iterrows()):
            side = 1 if e["dir"] < 0 else -1          # fade
            sname = "long" if side > 0 else "short"
            dv = 10 ** e["lv"] if not pd.isna(e["lv"]) else None
            cost = Q.cost_estimate(e["symbol"], dv, sname)
            x = {c: (None if pd.isna(e.get(c)) else float(e[c])) for c in Q.FEATURES}
            rp = Q.ridge_predict(ridge, x)
            kn = Q.knn_stats(train, x)
            rv = math.exp(e["lrvol"]) if not pd.isna(e["lrvol"]) else None
            bk, _ = Q.lookup(br, int(e["dir"]), float(e["z"]), rv, None if pd.isna(e["gapf"]) else float(e["gapf"]))
            ens = Q.ensemble(rp, kn, bk, {})
            g = {h: float(v[j]) for h, v in gp.items()}
            ensg = {h: {**s, "mean": 0.5 * s["mean"] + 0.5 * g[h]} if h in g else s for h, s in ens.items()}
            preds = {
                "baseline": {3: 1.0},
                "bucket": {h: bk.get(h, {}).get("mean") for h in Q.HORIZONS},
                "ridge": rp,
                "knn": {h: kn.get(h, {}).get("mean") for h in Q.HORIZONS},
                "gbm": g,
                "ensemble": {h: s["mean"] for h, s in ens.items()},
                "ens+gbm": {h: s["mean"] for h, s in ensg.items()},
            }
            base = {"date": e["date"], "symbol": e["symbol"], "dir": e["dir"], "z": e["z"], "cost": cost}
            # momentum: trade WITH the move (the inverse of a fade)
            fcost = Q.cost_estimate(e["symbol"], dv, "long" if side < 0 else "short")
            follow = {"follow": {3: 1.0}}
            if g:
                follow["follow+gbm"] = {h: -v for h, v in g.items()}
                follow["follow+ens"] = {h: -s["mean"] for h, s in ensg.items()}
            for name, p in follow.items():
                h, net = (3, 1.0) if name == "follow" else choose(p, fcost)
                if h is None:
                    continue
                pnl, _ = realise(e, h, -side, fcost)
                if not pd.isna(pnl):
                    trades.append({**base, "strategy": name, "hold": h, "pred": net, "pnl": pnl, "hedged": np.nan,
                                   "exit": "hold"})
            for name, p in preds.items():
                if not p:
                    continue
                if name == "baseline":
                    h, net = 3, 1.0
                else:
                    h, net = choose(p, cost)
                if h is None:
                    continue
                pnl, hedged = realise(e, h, side, cost)
                if pd.isna(pnl):
                    continue
                trades.append({**base, "strategy": name, "hold": h, "pred": net, "pnl": pnl, "hedged": hedged,
                               "exit": "hold"})
            # full plan logic
            for name, stats in (("plan", ens), ("plan+gbm", ensg if g else None)):
                if not stats:
                    continue
                sig_lr = e["vol250"] if not pd.isna(e["vol250"]) else 0.02
                shock = abs(math.log1p(e["ret"]))
                sig0 = math.sqrt(Q.EWMA_PHI * sig_lr ** 2 + (1 - Q.EWMA_PHI) * shock ** 2) * 1.25
                mkt = Q.market_of(e["symbol"])
                plan = Q.make_plan_v2(e["symbol"], float(e["price"]), float(e["ret"]), stats, sig0, sig_lr, dv,
                                      mkt, [], None, today=e["date"].date())
                if not plan.get("edge"):
                    continue
                pnl, held, how = simulate_plan(e, plan, plan["cost"])
                if pd.isna(pnl):
                    continue
                trades.append({**base, "strategy": name, "hold": held, "pred": plan["ev"], "pnl": pnl,
                               "hedged": np.nan, "exit": how, "score": plan["score"]})
        print(f"  {M}: trained on {len(train)} events, tested {len(test)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(trades)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def max_drawdown(x: pd.Series) -> float:
    c = x.cumsum()
    return float((c - c.cummax()).min()) if len(c) else 0.0


def summarise(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in tr.groupby("strategy"):
        g = g.sort_values("date")
        m = g.groupby(g["date"].dt.to_period("M"))["pnl"].mean()
        rows.append({
            "strategy": name, "trades": len(g), "hit": (g["pnl"] > 0).mean(), "avg": g["pnl"].mean(),
            "median": g["pnl"].median(), "t": g["pnl"].mean() / g["pnl"].std() * math.sqrt(len(g)) if len(g) > 2 else np.nan,
            "sharpe_m": m.mean() / m.std() * math.sqrt(12) if len(m) > 2 and m.std() > 0 else np.nan,
            "pos_months": (m > 0).mean() if len(m) else np.nan, "max_dd": max_drawdown(g["pnl"]),
            "avg_hold": g["hold"].mean(), "hedged_avg": g["hedged"].mean(),
        })
    order = ["baseline", "bucket", "ridge", "knn", "gbm", "ensemble", "ens+gbm", "plan", "plan+gbm",
             "follow", "follow+gbm", "follow+ens"]
    df = pd.DataFrame(rows)
    df["o"] = df["strategy"].map({k: i for i, k in enumerate(order)}).fillna(99)
    return df.sort_values("o").drop(columns="o")


def report(tr: pd.DataFrame, market: str) -> str:
    s = summarise(tr)
    lines = [f"BACKTEST ({market.upper()}) {tr['date'].min():%b %Y} – {tr['date'].max():%b %Y} · walk-forward, net of costs",
             "", f"{'strategy':<10}{'trades':>7}{'hit':>6}{'avg':>8}{'t':>6}{'Shrp':>6}{'+mo':>5}{'maxDD':>8}{'hold':>5}"]
    for _, r in s.iterrows():
        lines.append(f"{r['strategy']:<10}{r['trades']:>7}{r['hit']*100:>5.0f}%{r['avg']*100:>+7.2f}%{r['t']:>6.1f}"
                     f"{r['sharpe_m']:>6.2f}{r['pos_months']*100:>4.0f}%{r['max_dd']*100:>+7.0f}%{r['avg_hold']:>5.1f}")
    lines += ["", "avg = mean net P&L per trade (unhedged); t = t-stat; Shrp = annualised Sharpe of monthly",
              "average trade; +mo = % of months positive; maxDD = worst cumulative drawdown in 1-unit trades."]
    for name in ("plan", "plan+gbm"):
        p = tr[tr["strategy"] == name].copy()
        if len(p) >= 40 and "score" in p:
            p["bucket"] = pd.cut(p["score"], [0, 55, 65, 75, 101], labels=["<55", "55-64", "65-74", "75+"], right=False)
            lines += ["", f"{name}: by edge score", f"{'score':<7}{'trades':>7}{'hit':>6}{'avg':>8}{'t':>6}{'target':>8}{'stop':>6}"]
            for b, g in p.groupby("bucket", observed=True):
                t = g["pnl"].mean() / g["pnl"].std() * math.sqrt(len(g)) if len(g) > 2 and g["pnl"].std() > 0 else float("nan")
                lines.append(f"{str(b):<7}{len(g):>7}{(g['pnl']>0).mean()*100:>5.0f}%{g['pnl'].mean()*100:>+7.2f}%{t:>6.1f}"
                             f"{(g['exit']=='target').mean()*100:>7.0f}%{(g['exit']=='stop').mean()*100:>5.0f}%")
    best = s[s["strategy"] != "baseline"].sort_values("t", ascending=False).head(1)
    if len(best):
        lines += ["", f"Best by t-stat: {best.iloc[0]['strategy']}"]
    yrs = tr[tr["strategy"].isin(["baseline", "ens+gbm", "plan+gbm", "follow", "follow+gbm"])].copy()
    yrs["year"] = yrs["date"].dt.year
    lines += ["", "By year (avg net P&L per trade, trades):"]
    for (st, y), g in yrs.groupby(["strategy", "year"]):
        lines.append(f"  {st:<9} {y}: {g['pnl'].mean()*100:+.2f}% ({len(g)})")
    return "\n".join(lines)


def save_summary(tr: pd.DataFrame, market: str) -> None:
    """Compact headline numbers the live bot feeds to the analyst (quant.backtest_prior)."""
    def st(g):
        if len(g) < 3:
            return None
        return {"avg": round(float(g["pnl"].mean()), 4), "t": round(float(g["pnl"].mean() / g["pnl"].std() * math.sqrt(len(g))), 1),
                "hit": round(float((g["pnl"] > 0).mean()), 2), "n": int(len(g))}
    out = {"period": f"{tr['date'].min():%b %Y} - {tr['date'].max():%b %Y}", "market": market}
    for name, key in (("baseline", "fade_all"), ("follow", "follow_all"), ("follow+gbm", "follow_model")):
        s = st(tr[tr["strategy"] == name])
        if s:
            out[f"{key}_avg"], out[f"{key}_t"], out[f"{key}_n"] = s["avg"], s["t"], s["n"]
    pg = tr[tr["strategy"] == "plan+gbm"]
    if "score" in pg:
        for lo, hi, lab in ((0, Q.MIN_PLAN_SCORE, "below"), (Q.MIN_PLAN_SCORE, 101, "above")):
            s = st(pg[(pg["score"] >= lo) & (pg["score"] < hi)])
            if s:
                out[f"plan_score_{lab}_{Q.MIN_PLAN_SCORE}"] = s
    (HERE / "cache" / f"backtest_summary_{market}.json").write_text(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "eu"], default="us")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--telegram", action="store_true", help="also send the summary to Telegram")
    args = ap.parse_args()
    d = load_data(args.market, args.years)
    print("Building events…", flush=True)
    ev = build_events(d)
    print(f"{len(ev)} events from {ev['date'].min():%Y-%m-%d} to {ev['date'].max():%Y-%m-%d}", flush=True)
    tr = run(ev, args.market)
    if tr.empty:
        print("No trades — not enough history.")
        return 1
    out = HERE / "cache" / f"backtest_trades_{args.market}.csv"
    tr.to_csv(out, index=False)
    rep = report(tr, args.market)
    save_summary(tr, args.market)
    print("\n" + rep + f"\n\nTrades saved to {out}")
    if args.telegram:
        from scanner import send_telegram
        send_telegram("🧪 <pre>" + html.escape(rep) + "</pre>", force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

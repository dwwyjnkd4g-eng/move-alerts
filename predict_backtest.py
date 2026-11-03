"""
Backtest: do prediction-market repricings help the fade decision?

Uses Polymarket's official APIs, no key needed:
  Gamma API   https://gamma-api.polymarket.com/events   the biggest macro/geopolitical events since --since
  CLOB API    https://clob.polymarket.com/prices-history hourly odds history for each market
and hourly prices (Yahoo, ~2 years max) for gold, DXY, US 10y yield, S&P futures, WTI and bitcoin.

Three questions, each answered out-of-sample:

  Q1  INFORMATION   When a linked event's odds jump >= 10 pts in an hour, do the assets move abnormally
                    in that same hour? (share of shocks with an asset |z| >= 2, vs the base rate)
  Q2  FADE FILTER   Take every big hourly asset move (|z| >= 3, like the macro alerts) and split it:
                    "odds-confirmed" (a linked market repriced >= 10 pts within -2h..+1h) vs not.
                    Compare the fade P&L at 4h and 24h. If confirmed moves revert less, the live rule
                    (halve size / lean don't-fade) is validated.
  Q3  LEAD-LAG      After an odds shock, does the asset keep drifting the way the odds imply over the
                    next 4h / 24h? Direction comes from the sign of that market's past co-movement with
                    the asset, estimated ONLY on data before the shock (no look-ahead).

  python predict_backtest.py                   default: since 2024-11-01, top ~150 markets
  python predict_backtest.py --telegram        also send the summary
  python predict_backtest.py --max-markets 80  quicker run

Results: cache/predict_bt_{shocks,moves}.csv and cache/predict_backtest_summary.json (read by quant.py).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
UA = {"User-Agent": "move-alerts/1.0 (research backtest)", "Accept": "application/json"}

ASSETS = {"Gold": "GC=F", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "S&P fut": "ES=F", "WTI": "CL=F", "Bitcoin": "BTC-USD"}
LINKS = [  # keyword -> assets the event plausibly moves
    (r"\b(fed|fomc|rate cut|rate hike|interest rate|powell|cpi|inflation|jobs|payrolls?|unemployment|recession|gdp)\b",
     ["Gold", "DXY", "US10Y", "S&P fut", "Bitcoin"]),
    (r"\b(tariffs?|trade deal|trade war|china|export controls?)\b", ["S&P fut", "DXY", "Gold", "WTI"]),
    (r"\b(war|ceasefire|invade|invasion|strike|missile|iran|israel|gaza|russia|ukraine|houthi|taiwan|nato)\b",
     ["WTI", "Gold", "S&P fut"]),
    (r"\b(opec|oil|crude)\b", ["WTI"]),
    (r"\b(election|president|senate|house|congress|prime minister|chancellor|parliament)\b", ["DXY", "S&P fut", "Gold", "Bitcoin"]),
    (r"\b(shutdown|debt ceiling|default|treasury)\b", ["Gold", "DXY", "US10Y"]),
    (r"\b(bitcoin|crypto|stablecoin|sec|etf)\b", ["Bitcoin"]),
]
SKIP = re.compile(r"\b(nba|nfl|nhl|mlb|ufc|super bowl|oscar|grammy|tweet|album|movie|temperature|up or down|"
                  r"price of bitcoin|bitcoin (above|below|hit|reach)|eth(ereum)? (above|below))\b", re.I)
SHOCK = 0.10         # 10 pts in one hour
MOVE_Z = 3.0         # asset move size, in sigmas of hourly returns
INFO_Z = 2.0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def linked_assets(text: str) -> list[str]:
    out = []
    for rx, assets in LINKS:
        if re.search(rx, text, re.I):
            out += [a for a in assets if a not in out]
    return out


def gamma_events(since: str, max_events: int) -> list[dict]:
    rows, seen = [], set()
    for closed in ("true", "false"):
        for k in range(12):
            params = {"closed": closed, "order": "volume", "ascending": "false", "limit": 100, "offset": 100 * k}
            try:
                r = requests.get("https://gamma-api.polymarket.com/events", params=params, headers=UA, timeout=30)
                if r.status_code == 422 or r.status_code == 400:      # older API: no ordering by volume
                    params.pop("order"), params.pop("ascending")
                    r = requests.get("https://gamma-api.polymarket.com/events", params=params, headers=UA, timeout=30)
                r.raise_for_status()
                evs = r.json()
            except Exception as e:
                print(f"  gamma {closed} page {k}: {e}", file=sys.stderr)
                break
            if not evs:
                break
            for ev in evs:
                if ev.get("id") in seen:
                    continue
                seen.add(ev.get("id"))
                end = str(ev.get("endDate") or "")[:10]
                if end and end < since:
                    continue
                title = ev.get("title") or ""
                if SKIP.search(title) or not linked_assets(title):
                    continue
                rows.append(ev)
            time.sleep(0.25)
    rows.sort(key=lambda e: -float(e.get("volume") or 0))
    return rows[:max_events]


def markets_of(ev: dict, per_event: int = 3) -> list[dict]:
    ms = []
    for m in ev.get("markets") or []:
        try:
            tok = json.loads(m.get("clobTokenIds") or "[]")[0]
        except Exception:
            continue
        vol = float(m.get("volumeNum") or m.get("volume") or 0)
        if vol < 250_000:
            continue
        ms.append({"token": tok, "event": ev.get("title"), "question": m.get("question") or "",
                   "label": m.get("groupItemTitle") or "", "vol": vol})
    return sorted(ms, key=lambda m: -m["vol"])[:per_event]


def odds_history(token: str, start_ts: int) -> pd.Series:
    """Hourly YES price. Tries the full history, then 14-day chunks (some resolved markets need it)."""
    url = "https://clob.polymarket.com/prices-history"
    pts = []
    try:
        r = requests.get(url, params={"market": token, "interval": "max", "fidelity": 60}, headers=UA, timeout=30)
        if r.ok:
            pts = r.json().get("history") or []
    except Exception:
        pts = []
    if len(pts) < 48:
        pts = []
        t = start_ts
        now = int(time.time())
        for _ in range(60):
            if t >= now:
                break
            try:
                r = requests.get(url, params={"market": token, "startTs": t, "endTs": min(t + 14 * 86400, now),
                                              "fidelity": 60}, headers=UA, timeout=30)
                if r.ok:
                    pts += r.json().get("history") or []
            except Exception:
                pass
            t += 14 * 86400
            time.sleep(0.15)
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(p["t"]), unit="s", tz="UTC").floor("h"): float(p["p"]) for p in pts})
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[s.index >= pd.Timestamp(start_ts, unit="s", tz="UTC")]


def asset_history() -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(list(ASSETS.values()), period="730d", interval="60m", progress=False, auto_adjust=False)["Close"]
    d.index = pd.to_datetime(d.index, utc=True).floor("h")
    d = d[~d.index.duplicated(keep="last")]
    return d.rename(columns={v: k for k, v in ASSETS.items()})


# --------------------------------------------------------------------------
# Analysis (pure functions, testable offline)
# --------------------------------------------------------------------------
def asset_z(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = np.log(px).diff()
    sd = r.rolling(240, min_periods=100).std().shift(1)
    return r, r / sd


def find_shocks(odds: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for key, s in odds.items():
        s = s.asfreq("h").ffill(limit=3)
        d = s.diff()
        for t, v in d[d.abs() >= SHOCK].items():
            p0 = s.shift(1).get(t)
            if p0 is None or pd.isna(p0) or not 0.02 < p0 < 0.98:        # ignore resolution jumps at the edges
                continue
            rows.append({"t": t, "key": key, "dp": float(v), "p0": float(p0)})
    return pd.DataFrame(rows)


def co_sign(odds: pd.Series, ret: pd.Series, before: pd.Timestamp) -> float | None:
    """Sign of past co-movement between hourly odds changes and asset returns, using data before `before`."""
    d = odds.asfreq("h").ffill(limit=3).diff()
    j = pd.concat([d, ret], axis=1, keys=["d", "r"]).dropna()
    j = j[(j.index < before - pd.Timedelta(hours=24)) & (j["d"].abs() >= 0.02)]
    if len(j) < 15:
        return None
    c = np.corrcoef(j["d"], j["r"])[0, 1]
    return float(np.sign(c)) if abs(c) >= 0.05 else None


def analyse(odds: dict[str, pd.Series], meta: dict[str, dict], px: pd.DataFrame) -> dict:
    r, z = asset_z(px)
    shocks = find_shocks(odds)
    res = {"n_markets": len(odds), "n_shocks": int(len(shocks)), "assets": list(px.columns)}
    if shocks.empty:
        return res
    # Q1 + Q3
    q1, q3 = [], []
    logp = np.log(px)
    for _, sh in shocks.iterrows():
        for a in meta[sh["key"]]["assets"]:
            if a not in px:
                continue
            zt = z[a].get(sh["t"])
            if zt is None or pd.isna(zt):
                continue
            q1.append({"asset": a, "abn": abs(zt) >= INFO_Z})
            sg = co_sign(odds[sh["key"]], r[a], sh["t"])
            if sg is None:
                continue
            implied = sg * np.sign(sh["dp"])
            for h in (4, 24):
                t1 = sh["t"] + pd.Timedelta(hours=h)
                p0, p1 = logp[a].get(sh["t"]), logp[a].asof(t1) if t1 <= logp.index[-1] else np.nan
                if pd.isna(p0) or pd.isna(p1):
                    continue
                q3.append({"asset": a, "h": h, "follow": float(implied * (p1 - p0)), "t": sh["t"], "key": sh["key"]})
    base_abn = float((z.abs() >= INFO_Z).stack().mean())
    q1 = pd.DataFrame(q1)
    if len(q1):
        res["q1"] = {"shock_asset_pairs": int(len(q1)), "abnormal_share": float(q1["abn"].mean()), "base_rate": base_abn,
                     "by_asset": {a: float(g["abn"].mean()) for a, g in q1.groupby("asset")}}
    q3 = pd.DataFrame(q3)
    if len(q3):
        res["q3"] = {}
        for h, g in q3.groupby("h"):
            res["q3"][int(h)] = _stat(g["follow"])
            res["q3"][f"{int(h)}_by_asset"] = {a: _stat(gg["follow"]) for a, gg in g.groupby("asset")}
    # Q2: big asset moves, confirmed vs not
    shock_times: dict[str, list[pd.Timestamp]] = {}
    for _, sh in shocks.iterrows():
        for a in meta[sh["key"]]["assets"]:
            shock_times.setdefault(a, []).append(sh["t"])
    q2 = []
    for a in px.columns:
        zz = z[a]
        big = zz[zz.abs() >= MOVE_Z].dropna()
        st = pd.DatetimeIndex(sorted(shock_times.get(a, [])))
        last = None
        for t, zt in big.items():
            if last is not None and t - last < pd.Timedelta(hours=6):
                continue
            last = t
            conf = bool(len(st)) and bool(((st >= t - pd.Timedelta(hours=2)) & (st <= t + pd.Timedelta(hours=1))).any())
            rec = {"asset": a, "t": t, "z": float(zt), "confirmed": conf}
            for h in (4, 24):
                t1 = t + pd.Timedelta(hours=h)
                if t1 > logp.index[-1]:
                    continue
                p1 = logp[a].asof(t1)
                rec[f"fade_{h}h"] = float(-np.sign(zt) * (p1 - logp[a][t]))
            q2.append(rec)
    q2 = pd.DataFrame(q2)
    if len(q2):
        res["q2"] = {}
        for h in (4, 24):
            col = f"fade_{h}h"
            if col not in q2:
                continue
            c, u = q2[q2["confirmed"]][col].dropna(), q2[~q2["confirmed"]][col].dropna()
            res["q2"][h] = {"confirmed": _stat(c), "unconfirmed": _stat(u), "diff_t": _welch(c, u)}
    res["_frames"] = {"shocks": shocks, "moves": q2, "q3": q3}
    return res


def _stat(x: pd.Series) -> dict:
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 3:
        return {"n": n}
    sd = x.std()
    return {"n": int(n), "avg": float(x.mean()), "hit": float((x > 0).mean()), "t": float(x.mean() / sd * math.sqrt(n)) if sd > 0 else 0.0}


def _welch(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    se = math.sqrt(a.var() / len(a) + b.var() / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None


def report(res: dict, since: str) -> str:
    L = [f"PREDICTION-MARKET BACKTEST · Polymarket odds vs assets · since {since}",
         f"{res['n_markets']} markets · {res['n_shocks']} odds shocks (>= {SHOCK*100:.0f} pts in 1h)", ""]
    if "q1" in res:
        q = res["q1"]
        L += ["Q1 Information: do assets move in the shock hour?",
              f"  abnormal move (|z|>={INFO_Z:.0f}) in {q['abnormal_share']*100:.0f}% of {q['shock_asset_pairs']} "
              f"shock-asset pairs vs {q['base_rate']*100:.1f}% of all hours",
              "  " + " · ".join(f"{a} {v*100:.0f}%" for a, v in q["by_asset"].items()), ""]
    if "q2" in res:
        L += ["Q2 Fade filter: big asset moves (|z|>=3), fade P&L", f"{'':<12}{'n':>5}{'hit':>6}{'avg':>9}{'t':>6}"]
        for h, d in res["q2"].items():
            for lab in ("confirmed", "unconfirmed"):
                s = d[lab]
                if s.get("n", 0) >= 3:
                    L.append(f"{lab[:9]+' '+str(h)+'h':<12}{s['n']:>5}{s['hit']*100:>5.0f}%{s['avg']*100:>+8.3f}%{s['t']:>6.1f}")
            if d.get("diff_t") is not None:
                L.append(f"  {h}h: confirmed minus unconfirmed, t = {d['diff_t']:+.1f}")
        L += ["  (confirmed = a linked prediction market repriced >= 10 pts within -2h..+1h of the move)", ""]
    if "q3" in res:
        L += ["Q3 Lead-lag: asset drift after an odds shock, in the direction the odds imply"]
        for h in (4, 24):
            s = res["q3"].get(h)
            if s and s.get("n", 0) >= 3:
                L.append(f"  next {h}h: n={s['n']} hit {s['hit']*100:.0f}% avg {s['avg']*100:+.3f}% t={s['t']:.1f}")
                ba = res["q3"].get(f"{h}_by_asset", {})
                L.append("    " + " · ".join(f"{a} {v['avg']*100:+.2f}% (t {v['t']:.1f})" for a, v in ba.items() if v.get("n", 0) >= 5))
    L += ["", "Read: Q2 'confirmed' fade P&L well below 'unconfirmed' (t <= -2) supports the live rule (halve size,",
          "lean don't-fade). Q3 t >= 2 would justify a 'follow the odds' signal. Costs not deducted (hourly futures)."]
    return "\n".join(L)


def verdict(res: dict) -> dict:
    out = {"asof": datetime.now(timezone.utc).date().isoformat()}
    q2 = res.get("q2", {}).get(24) or res.get("q2", {}).get(4) or {}
    if q2:
        out["fade_confirmed_avg"] = q2["confirmed"].get("avg")
        out["fade_unconfirmed_avg"] = q2["unconfirmed"].get("avg")
        out["confirmed_vs_unconfirmed_t"] = q2.get("diff_t")
        out["odds_confirm_rule_supported"] = bool(q2.get("diff_t") is not None and q2["diff_t"] <= -2)
    q3 = res.get("q3", {}).get(24) or {}
    if q3:
        out["follow_odds_24h_avg"], out["follow_odds_24h_t"] = q3.get("avg"), q3.get("t")
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2024-11-01")
    ap.add_argument("--max-markets", type=int, default=150)
    ap.add_argument("--telegram", action="store_true")
    a = ap.parse_args()
    start_ts = int(pd.Timestamp(a.since, tz="UTC").timestamp())
    print("Finding events…", flush=True)
    evs = gamma_events(a.since, max_events=a.max_markets)
    mk = [m for ev in evs for m in markets_of(ev)][: a.max_markets]
    print(f"{len(evs)} events, {len(mk)} markets. Downloading hourly odds…", flush=True)
    odds, meta = {}, {}
    for i, m in enumerate(mk, 1):
        s = odds_history(m["token"], start_ts)
        if len(s) >= 48:
            key = m["token"][:16]
            odds[key] = s
            meta[key] = {"event": m["event"], "label": m["label"], "assets": linked_assets(m["event"] + " " + m["question"])}
        if i % 25 == 0:
            print(f"  {i}/{len(mk)} ({len(odds)} with history)", flush=True)
        time.sleep(0.1)
    if not odds:
        print("No odds history returned — check that clob.polymarket.com is reachable.")
        return 1
    print("Downloading hourly asset prices…", flush=True)
    px = asset_history()
    res = analyse(odds, meta, px)
    frames = res.pop("_frames", {})
    CACHE.mkdir(exist_ok=True)
    for name, df in frames.items():
        if isinstance(df, pd.DataFrame) and len(df):
            df.to_csv(CACHE / f"predict_bt_{name}.csv", index=False)
    (CACHE / "predict_backtest_summary.json").write_text(json.dumps(verdict(res), indent=1, default=str))
    rep = report(res, a.since)
    print("\n" + rep)
    if a.telegram:
        from scanner import send_telegram
        send_telegram("🧪 <pre>" + html.escape(rep) + "</pre>", force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

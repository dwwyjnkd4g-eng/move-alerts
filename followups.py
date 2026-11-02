"""
Fade follow-ups and the scorecard.

  python followups.py              send +1 and +5 trading-day follow-ups on FADE CANDIDATE alerts
  python followups.py --scorecard  send the scorecard (also included in the Sunday week-ahead)

Reads alerts_log.csv (US stocks) and macro_log.csv (macro markets).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATE = HERE / "followup_state.json"
HORIZONS = [1, 5, 20]


def load_alerts() -> pd.DataFrame:
    frames = []
    eq = HERE / "alerts_log.csv"
    if eq.exists():
        d = pd.read_csv(eq)
        frames.append(pd.DataFrame({
            "id": "eq:" + d["date"].astype(str) + ":" + d["symbol"].astype(str) + ":" + d["mode"].astype(str),
            "kind": "stock", "label": d["symbol"], "source": "yf",
            "symbol": [s if m == "eu" else str(s).replace(".", "-") for s, m in zip(d["symbol"], d["mode"])],
            "date": d["date"], "price": d["price"], "ret": d["ret"],
            "verdict": d["verdict"], "nature": d["move_nature"], "catalyst": d["catalyst_type"],
            **{c: d[c] if c in d else None for c in ("plan_side", "plan_target", "plan_stop", "plan_hold", "plan_score")}}))
    mc = HERE / "macro_log.csv"
    if mc.exists():
        d = pd.read_csv(mc)
        frames.append(pd.DataFrame({
            "id": "mc:" + d["run_ts_utc"].astype(str) + ":" + d["symbol"].astype(str),
            "kind": "macro", "label": d["instrument"], "source": d["source"].replace("yf_prepost", "yf"), "symbol": d["symbol"],
            "date": d["date"], "price": d["price"], "ret": d["ret"],
            "verdict": d["verdict"], "nature": d["move_nature"], "catalyst": d["catalyst_type"]}))
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    a["date"] = pd.to_datetime(a["date"])
    return a


def price_history(alerts: pd.DataFrame) -> dict[str, pd.Series]:
    """Daily closes for every alerted symbol (index = date)."""
    out: dict[str, pd.Series] = {}
    if alerts.empty:
        return out
    start = (alerts["date"].min() - pd.Timedelta(days=7)).date().isoformat()
    yf_syms = sorted(alerts.loc[alerts["source"] == "yf", "symbol"].unique())
    if yf_syms:
        import yfinance as yf
        px = yf.download(yf_syms, start=start, interval="1d", auto_adjust=False, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(yf_syms[0])
        for s in yf_syms:
            if s in px:
                ser = px[s].dropna()
                ser.index = pd.to_datetime(ser.index.date)
                out[s] = ser
    from macro import kraken_closes
    for s in sorted(alerts.loc[alerts["source"] == "kraken", "symbol"].unique()):
        try:
            ser = kraken_closes(s, interval=1440)
            ser.index = pd.to_datetime(ser.index.date)
            out[s] = ser
        except Exception as e:
            print(f"kraken {s}: {e}", file=sys.stderr)
    return out


def forward(alerts: pd.DataFrame, hist: dict[str, pd.Series]) -> pd.DataFrame:
    """Add fwd_{h}d and fade_{h}d (positive = the move reverted) from the alert price."""
    rows = []
    for _, a in alerts.iterrows():
        s = hist.get(a["symbol"])
        rec = a.to_dict()
        if s is not None:
            after = s[s.index > a["date"]]
            rec["days_since"] = len(after)
            rec["last"] = float(s.iloc[-1])
            for h in HORIZONS:
                if len(after) >= h:
                    fwd = float(after.iloc[h - 1]) / a["price"] - 1
                    rec[f"fwd_{h}d"] = fwd
                    rec[f"fade_{h}d"] = -np.sign(a["ret"]) * fwd
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def plan_outcome(r, s: pd.Series, upto: int) -> tuple[str, int | None, float]:
    """First touch of target/stop on daily closes within `upto` days. Returns (status, day, side-adjusted P&L)."""
    after = s[s.index > r["date"]].iloc[:upto]
    side = 1 if r["plan_side"] == "long" else -1
    for i, px in enumerate(after, 1):
        if side * (px - r["plan_target"]) >= 0:
            return "target", i, side * (px / r["price"] - 1)
        if side * (px - r["plan_stop"]) <= 0:
            return "stop", i, side * (px / r["price"] - 1)
    last = after.iloc[-1] if len(after) else r["price"]
    return "open", None, side * (last / r["price"] - 1)


def run_followups(dry: bool = False) -> None:
    from scanner import send_telegram
    alerts = load_alerts()
    if alerts.empty:
        return
    has_plan = alerts.get("plan_side", pd.Series(index=alerts.index, dtype=object)).isin(["long", "short"])
    fades = alerts[(alerts["verdict"] == "fade_candidate") | (has_plan & (alerts["verdict"] != "do_not_fade"))]
    fades = fades[fades["date"] >= pd.Timestamp.now() - pd.Timedelta(days=21)]
    if fades.empty:
        return
    hist = price_history(fades)
    df = forward(fades, hist)
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}
    lines = []
    for _, r in df.iterrows():
        if "days_since" not in r or pd.isna(r.get("days_since")):
            continue
        sent = state.get(r["id"], [])
        if r.get("plan_side") in ("long", "short") and not pd.isna(r.get("plan_hold")):
            hold = int(r["plan_hold"])
            s = hist.get(r["symbol"])
            if s is None:
                continue
            status, day, pnl = plan_outcome(r, s, min(int(r["days_since"]), hold))
            done = status in ("target", "stop") or r["days_since"] >= hold
            key = "final" if done else "d1"
            if key in sent or (key == "d1" and r["days_since"] < 1):
                state[r["id"]] = sent
                continue
            icon = {"target": "✅", "stop": "❌"}.get(status, "⏱" if done else "➖")
            what = {"target": f"target hit day {day}", "stop": f"stopped day {day}"}.get(
                status, f"hold ended ({hold}d)" if done else f"day {int(r['days_since'])} of {hold}")
            lines.append(f"{icon} {html.escape(str(r['label']))} {r['plan_side']} (edge {int(r['plan_score'] or 0)}): "
                         f"{what} · P&amp;L {pnl*100:+.1f}%")
            sent.append(key)
            if done and "d1" not in sent:
                sent.append("d1")
            state[r["id"]] = sent
            continue
        due = [h for h in (1, 5) if r["days_since"] >= h and h not in sent]
        if due:
            h = max(due)                      # if both are due (e.g. after downtime) send only the latest
            fwd = r.get(f"fwd_{h}d")
            if fwd is not None and not pd.isna(fwd):
                px_h = r["price"] * (1 + fwd)
                pre = r["price"] / (1 + r["ret"])
                rev = (r["price"] - px_h) / (r["price"] - pre) if r["price"] != pre else 0
                icon = "✅" if rev > 0.1 else ("❌" if rev < -0.1 else "➖")
                lines.append(f"{icon} {html.escape(str(r['label']))}: flagged {r['ret']*100:+.1f}% on "
                             f"{r['date']:%a %d %b} · {h}d later {fwd*100:+.1f}% "
                             f"(reverted {rev*100:.0f}% of the move)")
            sent += due
        state[r["id"]] = sent
    if lines:
        text = "🔁 <b>Fade follow-ups</b>\n" + "\n".join(lines)
        print(text) if dry else send_telegram(text)
    if not dry:
        STATE.write_text(json.dumps(state))


def scorecard_text(weeks: int = 12) -> str:
    alerts = load_alerts()
    if alerts.empty:
        return "📊 <b>Scorecard</b>\nNo alerts logged yet."
    alerts = alerts[alerts["date"] >= pd.Timestamp.now() - pd.Timedelta(weeks=weeks)]
    df = forward(alerts, price_history(alerts))
    lines = [f"📊 <b>Scorecard</b> (last {weeks} weeks · fade P&amp;L = profit from betting against the move)"]
    tbl = [f"{'verdict':<15}{'n':>4}{'hit 1d':>8}{'avg 1d':>8}{'hit 5d':>8}{'avg 5d':>8}"]
    for v, g in df.groupby("verdict"):
        def st(h):
            col = f"fade_{h}d"
            x = g[col].dropna() if col in g else pd.Series(dtype=float)
            return ((x > 0).mean() if len(x) else np.nan, x.mean() if len(x) else np.nan, len(x))
        h1, a1, _ = st(1)
        h5, a5, _ = st(5)
        f = lambda x, pct=True: "  -" if pd.isna(x) else (f"{x*100:.0f}%" if pct else f"{x*100:+.1f}%")
        tbl.append(f"{str(v):<15}{len(g):>4}{f(h1):>8}{f(a1, False):>8}{f(h5):>8}{f(a5, False):>8}")
    lines.append("<pre>" + html.escape("\n".join(tbl)) + "</pre>")
    lines.append("<i>hit = share of alerts that moved back the other way. For DON'T FADE, a low hit "
                 "rate means the call was right (the move kept going).</i>")
    # quant plans: outcome at the planned hold, by edge score
    if "plan_side" in df and df["plan_side"].isin(["long", "short"]).any():
        hist = price_history(alerts)
        rows = []
        for _, r in df[df["plan_side"].isin(["long", "short"])].iterrows():
            s = hist.get(r["symbol"])
            if s is None or pd.isna(r.get("plan_hold")) or r.get("days_since", 0) < 1:
                continue
            status, _, pnl = plan_outcome(r, s, int(r["plan_hold"]))
            rows.append({"bucket": "edge 65+" if r["plan_score"] >= 65 else ("edge 50-64" if r["plan_score"] >= 50 else "edge <50"),
                         "status": status, "pnl": pnl})
        if rows:
            q = pd.DataFrame(rows)
            t2 = [f"{'plans':<11}{'n':>4}{'target':>8}{'stop':>7}{'win':>6}{'avg P&L':>9}"]
            for b, g in q.groupby("bucket"):
                t2.append(f"{b:<11}{len(g):>4}{(g['status']=='target').mean()*100:>7.0f}%{(g['status']=='stop').mean()*100:>6.0f}%"
                          f"{(g['pnl']>0).mean()*100:>5.0f}%{g['pnl'].mean()*100:>+8.1f}%")
            lines += ["<b>Quant plans</b> (target/stop on daily closes, P&amp;L at hold end)",
                      "<pre>" + html.escape("\n".join(t2)) + "</pre>"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.scorecard:
        from scanner import send_telegram
        t = scorecard_text()
        print(t) if args.dry_run else send_telegram(t)
    else:
        run_followups(dry=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

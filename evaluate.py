"""
Score past alerts: did fade candidates actually revert?

For each logged alert (first per symbol per day), measures the return from the
alert day's close over the next 1, 5 and 20 trading days, and the "fade P&L"
= -sign(original move) * forward return (positive = the move reverted).

Usage:  python evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
HORIZONS = [1, 5, 20]


def main() -> None:
    log = pd.read_csv(HERE / "alerts_log.csv")
    log = log.sort_values("run_ts_utc").drop_duplicates(["date", "symbol"])
    log["yf_symbol"] = log["symbol"].str.replace(".", "-", regex=False)

    start = pd.to_datetime(log["date"]).min() - pd.Timedelta(days=5)
    px = yf.download(sorted(log["yf_symbol"].unique()), start=start.date().isoformat(),
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(log["yf_symbol"].iloc[0])

    rows = []
    for _, a in log.iterrows():
        s = px.get(a["yf_symbol"])
        if s is None:
            continue
        s = s.dropna()
        d = pd.Timestamp(a["date"])
        if d not in s.index:
            continue
        i = s.index.get_loc(d)
        rec = a.to_dict()
        for h in HORIZONS:
            if i + h < len(s):
                fwd = s.iloc[i + h] / s.iloc[i] - 1
                rec[f"fwd_{h}d"] = fwd
                rec[f"fade_{h}d"] = -np.sign(a["ret"]) * fwd
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No scorable alerts yet.")
        return

    def stats(g: pd.DataFrame) -> pd.Series:
        out = {"n": len(g)}
        for h in HORIZONS:
            col = f"fade_{h}d"
            if col in g and g[col].notna().any():
                x = g[col].dropna()
                out[f"fade_{h}d_mean_%"] = round(x.mean() * 100, 2)
                out[f"fade_{h}d_hit"] = round((x > 0).mean(), 2)
        return pd.Series(out)

    pd.set_option("display.width", 160)
    for key in ["verdict", "move_nature", "catalyst_type"]:
        print(f"\n=== by {key} ===")
        print(df.groupby(key).apply(stats, include_groups=False))
    print("\nNote: entry = alert-day close. Intraday alerts could have been traded at a different level.")
    df.to_csv(HERE / "alerts_scored.csv", index=False)


if __name__ == "__main__":
    main()

"""
Street view for a stock: sell-side consensus and price targets, plus rating changes in the last few days.

Source: Yahoo Finance via yfinance (free, covers US and most large European names); Finnhub's free
recommendation trend fills in the rating mix when Yahoo has none. Cached per symbol per day.

  get(yf_symbol, price) -> {
      n, rec (e.g. "buy"), rec_mean (1 strong buy .. 5 sell), pt_mean, pt_median, pt_high, pt_low,
      upside (mean PT vs price), changes: [{date, firm, action, from, to, pt_from, pt_to}]
  }
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "street"
REC_LABEL = {"strong_buy": "Strong buy", "buy": "Buy", "hold": "Hold", "underperform": "Underperform",
             "sell": "Sell", "strongbuy": "Strong buy"}


def _num(x):
    try:
        v = float(x)
        return v if v == v and v > 0 else None
    except (TypeError, ValueError):
        return None


def _fetch(sym: str) -> dict:
    import yfinance as yf
    t = yf.Ticker(sym)
    out: dict = {"changes": []}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    out["n"] = int(info.get("numberOfAnalystOpinions") or 0)
    out["rec"] = info.get("recommendationKey") if info.get("recommendationKey") not in (None, "none") else None
    out["rec_mean"] = _num(info.get("recommendationMean"))
    out["pt_mean"] = _num(info.get("targetMeanPrice"))
    out["pt_median"] = _num(info.get("targetMedianPrice"))
    out["pt_high"] = _num(info.get("targetHighPrice"))
    out["pt_low"] = _num(info.get("targetLowPrice"))
    out["ccy"] = info.get("financialCurrency") or info.get("currency")
    try:
        apt = t.analyst_price_targets or {}
        for k_src, k_dst in (("mean", "pt_mean"), ("median", "pt_median"), ("high", "pt_high"), ("low", "pt_low")):
            if out.get(k_dst) is None:
                out[k_dst] = _num(apt.get(k_src))
    except Exception:
        pass
    try:
        ud = t.upgrades_downgrades
        if ud is not None and len(ud):
            ud = ud.copy()
            ud.index = pd.to_datetime(ud.index, utc=True)
            recent = ud[ud.index >= datetime.now(timezone.utc) - timedelta(days=7)].sort_index(ascending=False)
            for ts, r in recent.head(6).iterrows():
                out["changes"].append({
                    "date": ts.strftime("%a %d %b"), "firm": str(r.get("Firm", ""))[:40],
                    "action": str(r.get("Action", "")).lower(),
                    "from": str(r.get("FromGrade", "") or ""), "to": str(r.get("ToGrade", "") or ""),
                    "pt_from": _num(r.get("priorPriceTarget")), "pt_to": _num(r.get("currentPriceTarget")),
                })
    except Exception:
        pass
    if not out["rec"] and "." not in sym:
        try:
            from scanner import _finnhub
            rt = _finnhub("stock/recommendation", {"symbol": sym}) or []
            if rt:
                x = rt[0]
                tot = sum(int(x.get(k, 0)) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
                if tot:
                    score = (1 * x.get("strongBuy", 0) + 2 * x.get("buy", 0) + 3 * x.get("hold", 0)
                             + 4 * x.get("sell", 0) + 5 * x.get("strongSell", 0)) / tot
                    out["rec_mean"] = round(score, 2)
                    out["rec"] = "strong_buy" if score < 1.5 else "buy" if score < 2.5 else "hold" if score < 3.5 else "sell"
                    out["n"] = out["n"] or tot
        except Exception:
            pass
    return out


def get(sym: str, price: float | None = None) -> dict:
    """Cached daily. Returns {} when there's no coverage (e.g. ETFs)."""
    today = datetime.now(timezone.utc).date().isoformat()
    path = CACHE / f"{sym.replace('/', '_')}.json"
    data = None
    try:
        c = json.loads(path.read_text())
        if c.get("day") == today:
            data = c["data"]
    except Exception:
        pass
    if data is None:
        try:
            data = _fetch(sym)
        except Exception as e:
            print(f"  street view failed for {sym}: {e}", file=sys.stderr)
            data = {}
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"day": today, "data": data}))
        except Exception:
            pass
    if not data or not (data.get("pt_mean") or data.get("rec") or data.get("changes")):
        return {}
    d = dict(data)
    if price and d.get("pt_mean"):
        ratio = d["pt_mean"] / price
        scale = 100.0 if ratio < 0.05 else (0.01 if ratio > 20 else 1.0)   # London: pence vs pounds
        if scale != 1.0:
            for k in ("pt_mean", "pt_median", "pt_high", "pt_low"):
                if d.get(k):
                    d[k] *= scale
            for c in d.get("changes", []):
                for k in ("pt_from", "pt_to"):
                    if c.get(k):
                        c[k] *= scale
        d["upside"] = d["pt_mean"] / price - 1
    if price and d.get("pt_low") and d.get("pt_high"):
        d["pos_in_range"] = (price - d["pt_low"]) / (d["pt_high"] - d["pt_low"]) if d["pt_high"] > d["pt_low"] else None
    return d


def for_prompt(d: dict) -> dict:
    if not d:
        return {}
    keep = {k: d.get(k) for k in ("n", "rec", "rec_mean", "pt_mean", "pt_low", "pt_high", "upside")}
    keep["rating_changes_last_7d"] = d.get("changes", [])[:4]
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in keep.items() if v not in (None, [], "")}


def line(d: dict, plan_target: float | None = None) -> str | None:
    import html
    if not d:
        return None
    e = html.escape
    bits = []
    if d.get("n"):
        bits.append(f"{d['n']} analysts")
    if d.get("rec"):
        bits.append(REC_LABEL.get(d["rec"], d["rec"].replace("_", " ").title())
                    + (f" ({d['rec_mean']:.1f})" if d.get("rec_mean") else ""))
    if d.get("pt_mean"):
        bits.append(f"PT {d['pt_mean']:,.2f}" + (f" ({d['upside']*100:+.0f}% from here)" if d.get("upside") is not None else ""))
    if d.get("pt_low") and d.get("pt_high"):
        bits.append(f"range {d['pt_low']:,.0f}–{d['pt_high']:,.0f}")
    out = "🏦 <b>Street:</b> " + " · ".join(bits) if bits else "🏦 <b>Street:</b>"
    if plan_target and d.get("pt_mean"):
        out += f" · our fade target {plan_target:,.2f} vs street {d['pt_mean']:,.2f}"
    ch = d.get("changes") or []
    if ch:
        def one(c):
            grade = f"{c['from']}→{c['to']}" if c["from"] and c["to"] and c["from"] != c["to"] else (c["to"] or "")
            pt = (f", PT {c['pt_from']:,.0f}→{c['pt_to']:,.0f}" if c.get("pt_from") and c.get("pt_to")
                  else (f", PT {c['pt_to']:,.0f}" if c.get("pt_to") else ""))
            act = {"up": "upgrade", "down": "downgrade", "init": "initiates", "main": "maintains", "reit": "reiterates"}.get(c["action"], c["action"])
            return f"{c['date']} {c['firm']} {act} {grade}{pt}".strip()
        out += "\n   Rating changes (7d): " + "; ".join(e(one(c)) for c in ch[:3])
    return out

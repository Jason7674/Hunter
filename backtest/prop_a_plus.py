"""Backtest of the Hunter Prop A+ model (midnight range sweep & reclaim) on Databento data.

Mirrors pine/hunter_prop_a_plus.pine, with more honest fills than TradingView:
  - entry at the reclaim bar's close + slippage
  - stop fills at stop - slippage; stop wins if stop and target hit in the same bar
  - target is a resting limit and only fills if price trades THROUGH it by a tick
  - commission per contract round trip
  - anything open is flattened at the end of the management window

Usage:
  python backtest/prop_a_plus.py --data ~/es_data/*.zip --contract MES --risk 250
  python backtest/prop_a_plus.py --data ~/es_data --compare-targets
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import load

CONTRACTS = {  # point value, tick size, default round-trip commission
    "ES": (50.0, 0.25, 4.50),
    "MES": (5.0, 0.25, 1.50),
    "NQ": (20.0, 0.25, 4.50),
    "MNQ": (2.0, 0.25, 1.50),
}
TARGETS = ("mid", "midnight", "opposite", "fixed")


@dataclass
class Params:
    range_start: time = time(0, 0)
    range_end: time = time(9, 30)
    entry_end: time = time(11, 0)
    mgmt_end: time = time(12, 0)
    sweep_ticks: int = 4
    reclaim_bars: int = 3
    disp_pct: float = 0.5
    min_rvol: float = 1.3
    rvol_len: int = 20
    pd_filter: bool = True
    target: str = "mid"
    min_rr: float = 2.0
    stop_ticks: int = 2
    risk_usd: float = 250.0
    max_contracts: int = 10
    max_trades: int = 2
    done_on_win: bool = True
    max_day_loss_r: float = 2.0
    contract: str = "MES"
    slippage_ticks: int = 1
    commission: float | None = None  # round trip per contract; None = contract default


def _target(p: Params, d: int, entry: float, risk: float, mo: float, rh: float, rl: float) -> float:
    if p.target == "midnight":
        return mo
    if p.target == "mid":
        return (rh + rl) / 2
    if p.target == "opposite":
        return rh if d == 1 else rl
    return entry + d * p.min_rr * risk


def run(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    pv, tick, default_comm = CONTRACTS[p.contract]
    comm = default_comm if p.commission is None else p.commission
    slip = p.slippage_ticks * tick

    vol_avg = df["volume"].rolling(p.rvol_len, min_periods=p.rvol_len).mean()
    rvol = (df["volume"] / vol_avg).where(vol_avg > 0, 0.0).fillna(0.0).to_numpy()

    t = df.index
    tod = t.time
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    dates = t.date

    trades = []
    # positions of each calendar day's bars
    day_bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for a, b in zip(day_bounds[:-1], day_bounds[1:]):
        day_tod = tod[a:b]
        pre = [i for i in range(a, b) if p.range_start <= day_tod[i - a] < p.range_end]
        sess = [i for i in range(a, b) if p.range_end <= day_tod[i - a] < p.mgmt_end]
        if not pre or not sess or (t[pre[0]] - t[pre[0]].normalize()) > pd.Timedelta(minutes=15):
            continue  # need a range that starts near the midnight bar
        mo = o[pre[0]]
        rh = max(h[i] for i in pre)
        rl = min(l[i] for i in pre)

        sw_low = sw_high = None  # (price, bar position, rvol)
        pos = None
        day_trades, day_r, day_won = 0, 0.0, False

        for i in sess:
            in_entry = tod[i] < p.entry_end

            # 1) manage open position
            if pos is not None and i > pos["bar"]:
                d = pos["dir"]
                hit_s = l[i] <= pos["stop"] if d == 1 else h[i] >= pos["stop"]
                hit_t = h[i] >= pos["tgt"] + tick if d == 1 else l[i] <= pos["tgt"] - tick
                last = i == sess[-1]
                if hit_s or hit_t or last:
                    if hit_s:
                        exit_px, why = pos["stop"] - d * slip, "stop"
                    elif hit_t:
                        exit_px, why = pos["tgt"], "target"
                    else:
                        exit_px, why = c[i] - d * slip, "flat"
                    pnl = (exit_px - pos["entry"]) * d * pv * pos["cts"] - comm * pos["cts"]
                    r = pnl / pos["risk_usd"]
                    trades.append({**pos["rec"], "exit_time": t[i], "exit": exit_px, "result": why,
                                   "pnl": pnl, "r": r})
                    day_r += r
                    day_won |= r > 0
                    pos = None

            # 2) sweep tracking
            if in_entry:
                if l[i] < rl - p.sweep_ticks * tick and (sw_low is None or l[i] < sw_low[0]):
                    sw_low = (l[i], i, rvol[i])
                if h[i] > rh + p.sweep_ticks * tick and (sw_high is None or h[i] > sw_high[0]):
                    sw_high = (h[i], i, rvol[i])

            # 3) setups
            rng = h[i] - l[i]
            disp = rng > 0 and abs(c[i] - o[i]) >= p.disp_pct * rng
            long_ok = (in_entry and sw_low is not None and i - sw_low[1] <= p.reclaim_bars and c[i] > rl
                       and c[i] > o[i] and disp and max(rvol[i], sw_low[2]) >= p.min_rvol
                       and (not p.pd_filter or c[i] < mo))
            short_ok = (in_entry and sw_high is not None and i - sw_high[1] <= p.reclaim_bars and c[i] < rh
                        and c[i] < o[i] and disp and max(rvol[i], sw_high[2]) >= p.min_rvol
                        and (not p.pd_filter or c[i] > mo))
            if long_ok == short_ok:
                continue

            d = 1 if long_ok else -1
            sw = sw_low if long_ok else sw_high
            if long_ok:
                sw_low = None
            else:
                sw_high = None
            day_done = day_trades >= p.max_trades or (p.done_on_win and day_won) or day_r <= -p.max_day_loss_r
            if pos is not None or day_done:
                continue
            stop = sw[0] - d * p.stop_ticks * tick
            risk = (c[i] - stop) * d
            if risk <= 0:
                continue
            tgt = _target(p, d, c[i], risk, mo, rh, rl)
            rr = (tgt - c[i]) * d / risk
            cts = min(p.max_contracts, int(p.risk_usd // (risk * pv)))
            if rr < p.min_rr - 1e-9 or cts < 1:
                continue
            entry = c[i] + d * slip
            day_trades += 1
            pos = {"dir": d, "entry": entry, "stop": stop, "tgt": tgt, "bar": i, "cts": cts,
                   "risk_usd": risk * pv * cts,
                   "rec": {"date": dates[i], "entry_time": t[i], "side": "long" if d == 1 else "short",
                           "entry": entry, "stop": stop, "target": tgt, "planned_rr": rr, "contracts": cts,
                           "range_high": rh, "range_low": rl, "midnight_open": mo}}
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    r = tr["r"]
    eq = r.cumsum()
    dd = (eq.cummax().clip(lower=0) - eq).max()
    losing = (r <= 0).astype(int)
    streak = losing.groupby((losing == 0).cumsum()).sum().max()
    daily = tr.groupby("date")["pnl"].sum()
    years = (pd.Timestamp(tr["date"].max()) - pd.Timestamp(tr["date"].min())).days / 365.25
    return {
        "trades": len(tr),
        "trades/week": round(len(tr) / max(years * 52, 1.0), 2),
        "win%": round(100 * (r > 0).mean(), 1),
        "exp_R": round(r.mean(), 3),
        "net_R": round(r.sum(), 1),
        "max_dd_R": round(dd, 1),
        "max_loss_streak": int(streak),
        "net_$": round(tr["pnl"].sum()),
        "worst_day_$": round(daily.min()),
        "green_days%": round(100 * (daily > 0).mean(), 1),
    }


def prop_eval(tr: pd.DataFrame, all_days, target=3000.0, max_loss=2000.0, max_days=60) -> dict:
    """Start an evaluation on every trading day; EOD trailing drawdown that locks at the start balance."""
    daily = tr.groupby("date")["pnl"].sum() if not tr.empty else pd.Series(dtype=float)
    pnl = pd.Series(0.0, index=pd.Index(sorted(all_days))).add(daily, fill_value=0.0).to_numpy()
    passed = failed = open_ = 0
    days_to_pass = []
    for s in range(len(pnl)):
        bal, peak = 0.0, 0.0
        outcome = None
        for k, x in enumerate(pnl[s:s + max_days]):
            bal += x
            floor = min(peak - max_loss, 0.0)
            if bal <= floor:
                outcome = "fail"
                break
            if bal >= target:
                outcome = "pass"
                days_to_pass.append(k + 1)
                break
            peak = max(peak, bal)
        if outcome == "pass":
            passed += 1
        elif outcome == "fail":
            failed += 1
        elif s + max_days <= len(pnl):
            open_ += 1  # ran out of days
    n = passed + failed + open_
    return {
        "evals": n,
        "pass%": round(100 * passed / n, 1) if n else 0.0,
        "fail%": round(100 * failed / n, 1) if n else 0.0,
        "timeout%": round(100 * open_ / n, 1) if n else 0.0,
        "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="zip files, folders, or .dbn/.csv files")
    ap.add_argument("--contract", default="MES", choices=list(CONTRACTS))
    ap.add_argument("--risk", type=float, default=250.0, help="$ risk per trade")
    ap.add_argument("--target", default="mid", choices=TARGETS)
    ap.add_argument("--min-rr", type=float, default=2.0)
    ap.add_argument("--timeframe", default="1min", help="pandas offset, e.g. 1min, 2min, 5min")
    ap.add_argument("--split", default="2024-01-01", help="out-of-sample start date")
    ap.add_argument("--compare-targets", action="store_true")
    ap.add_argument("--eval-target", type=float, default=3000.0)
    ap.add_argument("--eval-max-loss", type=float, default=2000.0)
    ap.add_argument("--eval-days", type=int, default=60)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = load(a.data, a.timeframe)
    print(f"Loaded {len(df):,} bars  {df.index[0]} → {df.index[-1]}")
    all_days = sorted(set(df.index.date))
    split = pd.Timestamp(a.split).date()

    base = Params(contract=a.contract, risk_usd=a.risk, target=a.target, min_rr=a.min_rr)
    variants = [replace(base, target=tg) for tg in TARGETS] if a.compare_targets else [base]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in variants:
        tr = run(df, p)
        tr.to_csv(out / f"trades_{p.contract}_{p.target}.csv", index=False)
        ins = tr[tr["date"] < split] if not tr.empty else tr
        oos = tr[tr["date"] >= split] if not tr.empty else tr
        for label, part in (("ALL", tr), (f"IN  <{split}", ins), (f"OUT >={split}", oos)):
            rows.append({"target": p.target, "sample": label, **summarize(part)})
        ev = prop_eval(tr, all_days, a.eval_target, a.eval_max_loss, a.eval_days)
        print(f"\n[{p.target}] prop eval (+${a.eval_target:,.0f} target, ${a.eval_max_loss:,.0f} trailing, "
              f"{a.eval_days} days): {ev}")
        if not tr.empty:
            by_year = tr.groupby(pd.to_datetime(tr["date"]).dt.year).apply(lambda g: pd.Series(summarize(g)))
            print(by_year[["trades", "win%", "exp_R", "net_R", "max_dd_R", "net_$"]].to_string())

    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\nParams: {asdict(base)}")


if __name__ == "__main__":
    main()

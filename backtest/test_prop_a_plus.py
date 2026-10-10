"""Synthetic-data tests: plant a sweep & reclaim and check the backtest trades it correctly."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import data
from prop_a_plus import Params, prop_eval, run, summarize

NY = "America/New_York"


def _day(date: str, *, sweep_low: bool = True, hit_target: bool = True) -> pd.DataFrame:
    """One NY day of 1m bars: flat range 5000-5010 overnight, sweep at 09:35, reclaim at 09:36."""
    idx = pd.date_range(f"{date} 00:00", f"{date} 12:59", freq="1min", tz=NY)
    n = len(idx)
    o = np.full(n, 5005.0)
    h = np.full(n, 5006.0)
    l = np.full(n, 5004.0)
    c = np.full(n, 5005.0)
    v = np.full(n, 100.0)
    pos = {ts.strftime("%H:%M"): i for i, ts in enumerate(idx)}
    # Range edges overnight; midnight open high in range so longs are in discount
    h[pos["03:00"]] = 5010.0
    l[pos["04:00"]] = 5000.0
    o[0] = c[0] = 5008.0
    if sweep_low:
        i = pos["09:35"]
        o[i], h[i], l[i], c[i], v[i] = 5001.0, 5001.5, 4997.0, 4998.0, 400.0  # sweep below 5000
        j = i + 1
        o[j], h[j], l[j], c[j], v[j] = 4998.0, 5002.5, 4997.5, 5002.0, 300.0  # strong reclaim
        # after entry: stay above stop, then rally to target (range mid 5005) or dump
        for k in range(j + 1, j + 30):
            o[k], h[k], l[k], c[k] = 5002.0, 5003.0, 5001.0, 5002.0
        k = j + 30
        if hit_target:
            o[k], h[k], l[k], c[k] = 5003.0, 5006.0, 5002.5, 5005.5
        else:
            o[k], h[k], l[k], c[k] = 5001.0, 5001.5, 4990.0, 4991.0
        for m in range(k + 1, n):
            o[m] = h[m] = l[m] = c[m] = c[k]
    return pd.DataFrame({"symbol": "ESH4", "open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def test_long_hits_target():
    tr = run(_day("2024-03-05"), Params(min_rr=0.5))
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t.side == "long" and t.result == "target"
    # stop = sweep low 4997 - 2 ticks; entry = 5002 + 1 tick slippage
    assert t.stop == pytest.approx(4996.5)
    assert t.entry == pytest.approx(5002.25)
    assert t.r > 0


def test_long_stops_out():
    tr = run(_day("2024-03-05", hit_target=False), Params(min_rr=0.5))
    assert len(tr) == 1 and tr.iloc[0].result == "stop" and tr.iloc[0].r < -1.0  # slippage + fees


def test_min_rr_skips_trade():
    # target mid 5005 is ~0.55R away from 5002 with stop 4996.5 -> skipped at 2R
    assert run(_day("2024-03-05"), Params(min_rr=2.0)).empty


def test_no_sweep_no_trade():
    assert run(_day("2024-03-05", sweep_low=False), Params(min_rr=0.5)).empty


def test_sizing_respects_risk():
    tr = run(_day("2024-03-05"), Params(min_rr=0.5, contract="MES", risk_usd=250))
    t = tr.iloc[0]
    assert t.contracts == int(250 // ((5002.0 - 4996.5) * 5.0))


def test_summary_and_eval():
    df = pd.concat([_day("2024-03-05"), _day("2024-03-06", hit_target=False), _day("2024-03-07")])
    tr = run(df, Params(min_rr=0.5))
    s = summarize(tr)
    assert s["trades"] == 3 and s["max_loss_streak"] == 1
    ev = prop_eval(tr, sorted(set(df.index.date)), target=100.0, max_loss=10_000.0, max_days=3)
    assert ev["evals"] >= 1


def test_loader_front_month_from_zip(tmp_path):
    a = _day("2024-03-05").reset_index(names="ts_event")
    b = a.copy()
    b["symbol"] = "ESM4"
    b["volume"] = 1.0  # back month, low volume
    s = a.copy()
    s["symbol"] = "ESH4-ESM4"  # spread, must be dropped
    raw = pd.concat([a, b, s])
    raw["ts_event"] = raw["ts_event"].dt.tz_convert("UTC")
    buf = io.StringIO()
    raw.to_csv(buf, index=False)
    zpath = tmp_path / "glbx.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("glbx-mdp3-20240305.ohlcv-1m.csv", buf.getvalue())
        zf.writestr("metadata.json", "{}")
    df = data.load([zpath])
    assert set(df["symbol"]) == {"ESH4"}
    assert len(df) == len(a)
    assert str(df.index.tz) == NY

# Hunter
My AI

## Pine scripts (`pine/`)

- `hunter_ribbon.pine` — 8-line MA ribbon (7/15/20/25/35/40/50/55), compression → expansion
  LONG/SHORT signals with optional HTF + session filter, fair value gaps that delete once
  price closes through them, and a non-repainting MTF trend table. Paste into TradingView's
  Pine editor (v6).
- `hunter_es_nq_signals.pine` — ES/NQ BUY/SELL signals: level test-and-hold (prior RTH H/L/C,
  overnight H/L, opening range, floor pivots, RTH VWAP ± bands) + reversal candle
  (engulf/hammer/star) + relative volume, filtered by time window and quarter of the hour.
  Candles colored by relative volume. Every signal is tracked to target/stop/timeout, and the
  win % on each signal uses only earlier signals (side × quarter-hour). Hover a label for details.
- `hunter_prop_a_plus.pine` — prop-eval model, one setup only: midnight→09:30 range (midnight
  open + 08:30 open marked), sweep beyond range + strong-body reclaim with volume, longs below
  midnight open / shorts above. Skips trades under min R:R or too wide for $ risk. Prop guardrails
  (max trades/day, done after win, done at -R) and stats: win %, expectancy, max DD, loss streak,
  net $, green days.

## Backtester (`backtest/`)

Python backtest of the Prop A+ model on Databento data (ES/NQ, any schema: `ohlcv-1m`, `ohlcv-1s`,
`trades`; DBN or CSV; zipped or not). Raw multi-contract data is reduced to the highest-volume
outright per trading day. Fills are stricter than TradingView: slippage on entry/stop/flatten,
targets need a trade-through, commissions included.

```bash
pip install -r backtest/requirements.txt
cd backtest
python prop_a_plus.py --data /path/to/es_zips --contract MES --risk 250 --compare-targets
python -m pytest -q
```

Prints stats overall, in-sample (< `--split`, default 2024-01-01) and out-of-sample, by year,
plus a prop-eval simulation (eval started on every day; profit target, EOD trailing max loss
that locks at the starting balance, day limit). Trades go to `results/`.

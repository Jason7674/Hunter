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

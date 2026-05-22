# SPY Directional Strategy: An Honest Backtest

A study in whether a simple machine learning model can predict the
next-day direction of SPY and translate that into a strategy that
beats buy-and-hold after costs. The goal is rigor and honest
evaluation, not a profitable strategy.

## Why this project


## Approach
1. Pull raw daily OHLCV for SPY (yfinance). Establish buy-and-hold baseline.
2. Build a custom backtester (plain pandas), with transaction costs,
   verified on a trivial strategy before any ML.
3. Engineer features (rolling volatility, moving averages, lagged
   returns) using only information available at T-1. Chronological split.
4. Train a logistic regression to predict next-day direction.
5. Evaluate out-of-sample with Monte Carlo resampling against the
   baseline, using pre-committed success criteria.

## Success criteria (committed before seeing results)


## Key decisions and their rationale
- Single asset (SPY) to avoid survivorship bias from a multi-stock universe.
- Custom backtester rather than a framework, to understand the mechanics.
- Direction prediction with a simple model, for interpretability.

## Current backtest sanity check
`src/backtest.py` converts a daily target-position signal into a P&L curve.
The first rule is intentionally trivial: hold SPY long when yesterday's
close-to-close return was positive, otherwise stay flat. Costs are charged
on every position change, so a long/flat round trip pays once on entry and
once on exit.

Run it with:

```bash
python3 src/backtest.py
```

Outputs are written under `data/`:
- `trivial_momentum_backtest.csv`: daily close, signal, turnover, costs,
  returns, equity curve, and P&L curve.
- `trivial_momentum_metrics.csv`: summary metrics after transaction costs.

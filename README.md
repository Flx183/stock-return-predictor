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

## Leakage-safe feature pipeline
`src/features.py` builds one supervised row per target close. Each row carries
both timestamps:

- `feature_timestamp`: the previous trading close, when every feature is known.
- `target_timestamp`: the close whose close-to-close return is being predicted.

All rolling volatility, moving-average, lagged-return, drawdown, RSI, intraday
range, and volume features are calculated as-of each close first, then shifted
forward by one trading row. That makes the row for target day T use only data
available at the close of T-1 or earlier.

`src/features.py` also adds `target_volatility_5d`, a future realized-volatility
label with `target_volatility_start_timestamp` and
`target_volatility_end_timestamp`. That label is not used as a direction-model
feature; it is included so volatility forecasting or risk sizing can be tested
separately without blurring feature and target time.

`src/ml_pipeline.py` then sorts by `target_timestamp`, trains on the earliest
80% of rows, tests on the latest 20%, and has no shuffle option. The
`StandardScaler` is fit with `fit_transform` on the training matrix only; the
test matrix only receives `transform`. The feature-correlation report is also
computed from the training rows only.

Run it with:

```bash
python3 src/ml_pipeline.py
```

Outputs are written under `data/`:
- `ml_supervised_dataset.csv`: feature rows with explicit feature and target
  timestamps.
- `ml_test_predictions.csv`: out-of-sample predicted direction probabilities.
- `ml_test_metrics.csv`: out-of-sample classification metrics and split dates.
- `train_feature_correlations.csv`: train-only feature pairs whose absolute
  correlation is at least 0.8.

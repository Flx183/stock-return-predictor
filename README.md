# SPY Directional Strategy: An Honest Backtest

A study in whether a simple machine learning model can predict the
next-day direction of SPY and translate that into a strategy that
beats buy-and-hold after costs. The goal is rigor and honest
evaluation, not a profitable strategy.

## Why this project
 To learn what doing machine learning in a quantitative finance context actually feels like in practice, from raw data to a costed strategy and an honest comparison against a benchmark. The goal is to build the apparatus around the question (leakage-safe features, a costed backtester, a chronological split, a Monte Carlo comparison to buy-and-hold) and to report whatever the apparatus returns, including a null.

## Approach
1. Pull raw daily OHLCV for SPY (yfinance). Establish buy-and-hold baseline.
2. Build a custom backtester (plain pandas), with transaction costs,
   verified on a trivial strategy before any ML.
3. Engineer features (rolling volatility, moving averages, lagged
   returns) using only information available at T-1. Chronological split.
4. Train a logistic regression to predict next-day direction.
5. Evaluate out-of-sample with Monte Carlo resampling against the
   baseline, using pre-committed success criteria.

## Key decisions and their rationale
- Single asset (SPY) to avoid survivorship bias from a multi-stock universe.
- Custom backtester rather than a framework, to understand the mechanics.
- Direction prediction with a simple model, for interpretability.

## Current backtest sanity check
`src/backtest.py` converts a daily target-position signal into a P&L curve. The first rule is intentionally trivial: hold SPY long when yesterday's close-to-close return was positive, otherwise stay flat. Costs are charged on every position change, so a long/flat round trip pays once on entry and once on exit.

Run it with:

```bash
python3 src/backtest.py
```

Outputs are written under `data/`:
- `trivial_momentum_backtest.csv`: daily close, signal, turnover, costs,
  returns, equity curve, and P&L curve.
- `trivial_momentum_metrics.csv`: summary metrics after transaction costs.

## Leakage-safe feature pipeline
`src/features.py` builds one supervised row per target close. Each row carries both timestamps:
- `feature_timestamp`: the previous trading close, when every feature is known.
- `target_timestamp`: the close whose close-to-close return is being predicted.

All rolling volatility, moving-average, lagged-return, drawdown, RSI, intraday range, and volume features are calculated as-of each close first, then shifted forward by one trading row. That makes the row for target day T use only data available at the close of T-1 or earlier.

`src/features.py` also adds `target_volatility_5d`, a future realized-volatility label with `target_volatility_start_timestamp` and
`target_volatility_end_timestamp`. That label is not used as a direction-model feature; it is included so volatility forecasting or risk sizing can be tested separately without blurring feature and target time.

`src/ml_pipeline.py` then sorts by `target_timestamp`, trains on the earliest 80% of rows, tests on the latest 20%, and has no shuffle option. The `StandardScaler` is fit with `fit_transform` on the training matrix only; the test matrix only receives `transform`. The feature-correlation report is also computed from the training rows only.

Run it with:

```bash
python3 src/ml_pipeline.py
```

Outputs are written under `data/`:
- `ml_supervised_dataset.csv`: feature rows with explicit feature and target timestamps.
- `ml_test_predictions.csv`: out-of-sample predicted direction probabilities.
- `ml_test_metrics.csv`: out-of-sample classification metrics and split dates.
- `train_feature_correlations.csv`: train-only feature pairs whose absolute correlation is at least 0.8.

To convert the model predictions into a costed strategy and compare it with buy-and-hold, run:

```bash
python3 src/model_backtest.py
```

This uses the same backtester as the trivial strategy. The first model test row is anchored at its `feature_timestamp`, so the first predicted close-to-close return is included without leaking earlier information. Outputs are written under `data/`:

- `ml_strategy_backtest.csv`: model signal, turnover, costs, net strategy
  returns, strategy equity curve, buy-and-hold returns, and buy-and-hold equity curve.
- `ml_strategy_metrics.csv`: out-of-sample strategy metrics after costs and same-period buy-and-hold metrics.
- `ml_monte_carlo_comparison.csv`: paired bootstrap comparison summary.
- `ml_monte_carlo_samples.csv`: individual Monte Carlo bootstrap samples.

## Walk-forward validation and pre-registered experiments

The single 80/20 split above produces one number per question: one accuracy, one strategy-vs-buy-and-hold verdict. That is a single draw from a noisy process. The experiments here replace each point estimate with a *distribution*, and they fix their success thresholds *before* any result is seen. Those thresholds live in `PREREGISTRATION.md`, which is committed before the experiment code is run so the verdicts cannot be reverse-engineered from the data.

`src/walkforward.py` is the shared harness, with no model or threshold baked in:

- **Expanding-window folds.** Eight contiguous out-of-sample blocks of 126
  trading days (~6 months) tile the most recent ~4 years. Each fold re-fits the model on every row strictly *before* its test block, so no future row informs a fit, and all scaling is fit on training rows only.
- **Moving-block bootstrap.** The pooled out-of-sample daily series is resampledin overlapping 21-day (~1-month) blocks (10000 resamples, seed 42). Unlike the i.i.d. paired bootstrap in `src/model_backtest.py`, this preserves the serial correlation of daily data, so the confidence intervals are honest. A statistic passes only if its 90% CI excludes zero on the side that matters.
- **Embargo.** When a target looks forward several days, the last few training rows of each fold carry labels that cross into the test block. The harness can drop them (`embargo`), which can only weaken a positive result, never create one.

Run the two direction experiments (logistic regression and XGBoost) with:

```bash
python3 src/experiment_direction.py
```

Outputs under `data/`:
- `walkforward_direction_folds.csv`: per-fold train/test dates, train size, and per-fold accuracy and excess return for each model.
- `walkforward_direction_oos.csv`: pooled out-of-sample daily positions and net returns for both models alongside buy-and-hold.
- `walkforward_direction_model_metrics.csv`: pooled accuracy, ROC AUC, and log loss for each model.
- `walkforward_direction_bootstrap.csv`: block-bootstrap CIs and pre-registered verdicts for LR vs buy-and-hold, XGBoost vs LR, and XGBoost vs buy-and-hold.

Run the volatility experiment with:

```bash
python3 src/experiment_volatility.py
```

Outputs under `data/`:
- `walkforward_volatility_folds.csv`: per-fold RMSE for the OLS model and for the persistence baseline.
- `walkforward_volatility_oos.csv`: pooled out-of-sample forecasts and squared errors for both.
- `walkforward_volatility_summary.csv`: pooled RMSE, RMSE improvement, QLIKE, out-of-sample R^2, the block-bootstrap CI, and the pre-registered verdict.

### Results (pre-registered, out-of-sample 2020-06 to 2024-06, 1008 days)

These are the verdicts the pre-registered criteria returned, reported as-is.

- **Experiment 1 — logistic regression direction vs buy-and-hold: NULL.** Pooled accuracy 0.538 sits *below* the always-up base rate of 0.540 (ROC AUC 0.504). Mean daily excess return is -1.3e-05 with a 90% CI of [-2.0e-04, +1.4e-04] that straddles zero; total compounded excess is -1.2%. The distribution agrees with the single-split null: the strategy does not beat buy-and-hold.
- **Experiment 2 — XGBoost vs logistic regression (economic test): NULL.** The gradient-boosted strategy does not beat the linear one net of costs; if anything it is worse (mean daily excess -2.3e-04, 90% CI [-4.5e-04, +4.1e-05], total compounded excess -37.6% over the window from over-trading).Nonlinear structure adds nothing economically here.
- **Experiment 3 — volatility forecast vs persistence: BEATS PERSISTENCE.** The OLS model cuts pooled RMSE by 20.2% versus the naive persistence baseline (0.067 vs 0.085), with a 90% block-bootstrap CI on the per-day squared-error reduction of [1.7e-03, 3.6e-03] that excludes zero. QLIKE improves (0.48 vs 1.27) and out-of-sample R^2 versus persistence is 0.36. The improvement is positive in all eight folds and survives the 4-day embargo essentially unchanged, so it is not a fold-boundary artifact.

The honest summary: next-day *direction* is not predictable on SPY with these features, and a nonlinear model does not rescue it — but near-term *volatility* is forecastable well beyond naive persistence, consistent with volatility clustering. Predictable risk, unpredictable return.

For an interactive visualization report, open:

```text
notebooks/visualization_report.ipynb
```

The notebook loads the generated CSVs, shows the model/backtest summary tables, regenerates the charts, and displays them inline.

For a terminal-only chart refresh, run:

```bash
python3 src/visualize.py
```

Charts are written under `data/figures/`:

- `equity_curves.png`: ML strategy net of costs vs buy-and-hold.
- `drawdowns.png`: drawdowns for both equity curves.
- `prediction_probabilities.png`: out-of-sample predicted up probabilities, split by correct and incorrect predictions.
- `confusion_matrix.png`: predicted vs actual direction counts.
- `train_feature_correlation_heatmap.png`: train-only feature correlations.
- `monte_carlo_excess_return.png`: paired bootstrap excess-return distribution.

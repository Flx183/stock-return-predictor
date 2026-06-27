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

### Single-split verdict (the first null, stated plainly)

On the one chronological 80/20 split (out-of-sample 2021-09-30 to 2024-06-28, 690 days) the logistic model scored **accuracy 0.507, balanced accuracy 0.490, and ROC AUC 0.476** — at or below chance, and slightly worse than a coin flip. Its predicted-up rate was 0.893, so it was mostly just saying "up." Turned into a costed strategy it returned **+27.2% against buy-and-hold's +30.5%, an excess of −3.3%.** A paired bootstrap put the probability the strategy beats buy-and-hold at **38.6% on total return** (42.5% on Sharpe). The plain conclusion: **the strategy lost to buy-and-hold.** This single number is one draw from a noisy process, which is exactly why the walk-forward experiments below replace it with a distribution.

## Walk-forward validation and pre-registered experiments

The single 80/20 split above produces one number per question: one accuracy, one strategy-vs-buy-and-hold verdict. That is a single draw from a noisy process. The experiments here replace each point estimate with a *distribution*, and they fix their success thresholds *before* any result is seen. Those thresholds live in `PREREGISTRATION.md`, which is committed before the experiment code is run so the verdicts cannot be reverse-engineered from the data.

`src/walkforward.py` is the shared harness, with no model or threshold baked in:

- **Expanding-window folds.** Eight contiguous out-of-sample blocks of 126
  trading days (~6 months) tile the most recent ~4 years. Each fold re-fits the model on every row strictly *before* its test block, so no future row informs a fit, and all scaling is fit on training rows only.
- **Moving-block bootstrap.** The pooled out-of-sample daily series is resampled in overlapping 21-day (~1-month) blocks (10000 resamples, seed 42). Unlike the i.i.d. paired bootstrap in `src/model_backtest.py`, this preserves the serial correlation of daily data, so the confidence intervals are honest. A statistic passes only if its 90% CI excludes zero on the side that matters.
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
- **Experiment 2 — XGBoost vs logistic regression (economic test): NULL.** The gradient-boosted strategy does not beat the linear one net of costs; if anything it is worse (mean daily excess -2.3e-04, 90% CI [-4.5e-04, +4.1e-05], total compounded excess -37.6% over the window from over-trading). Nonlinear structure adds nothing economically here.
- **Experiment 3 — volatility forecast vs persistence: BEATS PERSISTENCE.** The OLS model cuts pooled RMSE by 20.2% versus the naive persistence baseline (0.067 vs 0.085), with a 90% block-bootstrap CI on the per-day squared-error reduction of [1.7e-03, 3.6e-03] that excludes zero. QLIKE improves (0.48 vs 1.27) and out-of-sample R^2 versus persistence is 0.36. The improvement is positive in all eight folds and survives the 4-day embargo essentially unchanged, so it is not a fold-boundary artifact.

The honest summary: next-day *direction* is not predictable on SPY with these features, and a nonlinear model does not rescue it — but near-term *volatility* is forecastable well beyond naive persistence, consistent with volatility clustering. Predictable risk, unpredictable return.

## Limitations (the scope is the defense)

Every result above holds only inside these boundaries. Naming them is not a hedge; it is what keeps the claims true.

- **Single asset.** Everything is SPY. A single liquid index was chosen to avoid the survivorship bias of a multi-stock universe, but the price is zero cross-sectional breadth: nothing here generalizes to single names, other indices, or other asset classes.
- **Price and volume features only.** All 26 features are functions of OHLCV (returns, realized vol, moving-average ratios, drawdowns, RSI, intraday range, volume z-scores). No fundamentals, macro, options-implied vol, order flow, or sentiment. A null on direction is a statement about *this feature set*, not about all possible signals.
- **Weak-form efficiency only.** Because the inputs are past prices and volume, the direction null bears only on **weak-form** market efficiency. It says nothing about semi-strong (public information) or strong (private information) forms.
- **One window, regime-bound.** Features begin ~2010-10 (after the 200-day SMA warm-up); the pooled out-of-sample window is 2020-06 to 2024-06 (~1008 trading days). That window contains the COVID crash and the 2022 bear market — informative, but not a guarantee the results survive into other regimes.
- **Daily, next-day horizon.** Direction is next-day close-to-close; the volatility target is 5-day-ahead realized vol. Nothing here speaks to intraday or multi-month horizons.
- **One cost model.** Results are net of a single linear cost of `0.0001` per unit turnover. They are not robust to a materially higher cost assumption, slippage, or market impact.
- **The volatility win is relative to one baseline.** "Beats persistence" means it beats *naive last-value persistence* on 5-day realized vol. It is **not** a claim to beat a tuned GARCH/HAR model or the implied volatility surface.

## What this project does and does not claim

**Failing to reject the null is not the same as proving the null true.** A non-significant direction result means the test — at this sample size, with this feature set, on this asset and window — *did not find* an edge, not that no edge exists. The pre-registered bar (a 90% CI excluding zero) controls false positives, not false negatives, and with ~1008 noisy daily observations the test has limited power to detect a *small* true edge. So the defensible statement is "we did not find next-day direction predictable here," which is strictly weaker than "next-day direction is unpredictable." Absence of evidence is not evidence of absence.

**What this project does not claim.** It does not claim markets are efficient, that SPY's direction is unpredictable in general, or that the EMH is true. It does not claim volatility forecasting is easy, or that the OLS model would beat a tuned GARCH/HAR or implied vol. It does not claim any strategy here is profitable. It claims exactly, and only, this: under a pre-registered, leakage-controlled, costed protocol on SPY daily data from 2020-06 to 2024-06, using price and volume features alone — (1) a logistic and a gradient-boosted next-day **direction** model did not beat buy-and-hold, and (2) an OLS forecast of 5-day-ahead realized **volatility** beat naive persistence by ~20% out of sample. Predictable risk, unpredictable return — for this asset, these features, this window.

## How each artifact threat is controlled (threat → code)

A backtest result is only as trustworthy as the specific mechanism that rules out each way it could be an artifact. If the mechanism cannot be pointed at a line, the threat is not controlled, only hoped away.

1. **Lookahead leakage** (using information the model could not have had).
   - Every feature is computed as of a close and then **shifted forward one row** so the target row for day `T` sees only data through `T-1`: [features.py:138](src/features.py#L138).
   - The invariant `feature_timestamp < target_timestamp` is asserted both at build time ([features.py:158](src/features.py#L158)) and again inside the harness ([walkforward.py:103](src/walkforward.py#L103)).
   - The 5-day-forward volatility label can leak its last 4 observations across a fold boundary, so each fold **embargoes** its final `5-1 = 4` training rows: [experiment_volatility.py:119](src/experiment_volatility.py#L119) feeding [walkforward.py:149-152](src/walkforward.py#L149-L152). Embargo can only weaken a positive result, never manufacture one.
   - The costed backtest anchors the first out-of-sample day at its prior close with zero exposure, so the first predicted return is earned without pretending foreknowledge: [model_backtest.py:62-70](src/model_backtest.py#L62-L70).
2. **In-sample overfitting** (fitting noise and grading on the same data).
   - Expanding-window walk-forward: each fold re-fits on `[0, test_start)` and is graded on the next, unseen block — the model never touches its own test rows: [walkforward.py:44-84](src/walkforward.py#L44-L84).
   - The `StandardScaler` is `fit_transform`-ed on training rows only and merely `transform`-ed on test rows: [experiment_direction.py:84-86](src/experiment_direction.py#L84-L86), [experiment_volatility.py:76-78](src/experiment_volatility.py#L76-L78).
   - XGBoost runs on **fixed, pre-registered hyperparameters with no tuning and no early stopping**, so there is no tuning-leakage: [experiment_direction.py:67-80](src/experiment_direction.py#L67-L80). Its null (total compounded excess −37.6%) is the *positive demonstration* that a flexible learner does not invent signal where there is none.
3. **Base-rate illusion** (mistaking "mostly up" for skill).
   - Accuracy is judged against the **always-up base rate**, reported explicitly as `actual_positive_rate` ([experiment_direction.py:130](src/experiment_direction.py#L130)): pooled accuracy 0.538 sits *below* the 0.540 base rate, and ROC AUC 0.504 ≈ chance. The model's 0.875 predicted-up rate is what an accuracy-only view would have hidden.
4. **Serial-correlation-inflated significance** (i.i.d. tests overstating confidence on autocorrelated returns).
   - Significance uses a **moving-block bootstrap** in overlapping 21-day blocks, preserving within-month autocorrelation that the i.i.d. paired bootstrap ([model_backtest.py:202](src/model_backtest.py#L202)) destroys: [walkforward.py:199-229](src/walkforward.py#L199-L229). The single pass/fail flag is whether the 90% CI lower bound clears zero: [walkforward.py:257](src/walkforward.py#L257).

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

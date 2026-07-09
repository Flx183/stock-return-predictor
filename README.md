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

Run the multi-horizon experiment (Experiment 4 — does predictability appear further out?) with:

```bash
python3 src/experiment_horizons.py
```

It re-uses the same harness with a forward-`h`-day return target (`build_leakage_safe_features(ohlcv, forward_horizon=h)`) across `h = 5, 21, 63, 252` trading days, adds longer look-back features, sets the bootstrap block length to `>= 2h` and the embargo to `h - 1`, and tests two pre-registered rules per horizon: beat a constant-drift forecast as a regression, and beat buy-and-hold as a non-overlapping direction strategy. Outputs under `data/`:
- `horizon_regression.csv`: per horizon and model, OOS R^2 vs drift, RMSEs, and the block-bootstrap CI on the squared-error reduction.
- `horizon_classification.csv`: per horizon and model, accuracy vs base rate, and the non-overlapping strategy vs buy-and-hold with its CI.
- `horizon_ladder.csv`: the headline table (effective non-overlapping N, base rate, OOS R^2, accuracy, excess, verdict) per horizon.

Run the volatility-baseline experiments (Experiment 5 — is the vol win real? and Experiment 6 — where does it come from?) with:

```bash
python3 src/experiment_vol_baselines.py
```

Experiment 3 measured the volatility model against *persistence*, a weak baseline. These two stress it against a strong one and decompose it, through the identical harness (same folds, `embargo = 4`, same block bootstrap, the same OLS). Experiment 5 swaps in **HAR-RV** (Corsi 2009 — an OLS of forward 5-day RV on lagged daily/weekly/monthly realized vol, kept out of the canonical feature set); Experiment 6 runs three nested models (persistence, vol-lags-only, full 26). Outputs under `data/`:
- `vol_har_comparisons.csv`: 26-feature model vs HAR-RV (gating), plus HAR-RV vs persistence and model vs persistence (an Experiment-3 reproduction check).
- `vol_ablation_comparisons.csv`: full vs vol-lags-only (gating), plus each vs persistence.

Run the volatility-targeting backtest (Experiment 7 — does a better forecast buy a better Sharpe?) with:

```bash
python3 src/experiment_vol_targeting.py
```

One vol-targeting strategy (daily exposure = `target_vol / forecast_vol`, long-only, 15% target vol, 2× cap, costed) is run three times through the same backtester, fed by persistence, HAR-RV, and the 26-feature model — changing only the forecast, so any difference is the forecast's doing. Timing is leakage-safe: the position for the return ending on day t is sized by the forecast made as of day t-1. The pre-registered statistic (Experiment-1 style) is the annualized Sharpe difference vs buy-and-hold with a block-bootstrap 90% CI. Outputs under `data/`:
- `vol_targeting_comparisons.csv`: per source, strategy Sharpe, Sharpe difference and its CI, excess return, turnover, cost, and capped/floored fractions.
- `vol_targeting_oos.csv`: pooled out-of-sample positions and daily returns for each source.

Run the VIX experiment (Experiment 8 — does option-implied vol help?). This is the one experiment that uses information beyond SPY OHLCV, so it first pulls `^VIX`:

```bash
python3 src/vix_data.py        # pulls ^VIX into data/VIX_data.csv (external data source)
python3 src/experiment_vix.py
```

VIX is the option-implied 30-day volatility of the S&P 500; dividing by 100 gives decimal annualized vol, directly comparable to `target_volatility_5d`. Two questions, VIX taken as of the feature close (leakage-safe): Q1 — does VIX beat the 26-feature model as a *forecast* (same 10%-and-CI bar as Experiment 5)? Q2 — does adding lagged VIX as a *feature* improve the model (same incremental-CI test as Experiment 6)? Outputs under `data/`:
- `vix_forecast_comparisons.csv`: VIX vs model / persistence / HAR as forecasts, plus a model-vs-persistence Experiment-3 reproduction check.
- `vix_feature_comparisons.csv`: model+VIX vs model (and vs persistence).

### Results (pre-registered, out-of-sample 2020-06 to 2024-06, 1008 days)

These are the verdicts the pre-registered criteria returned, reported as-is.

- **Experiment 1 — logistic regression direction vs buy-and-hold: NULL.** Pooled accuracy 0.538 sits *below* the always-up base rate of 0.540 (ROC AUC 0.504). Mean daily excess return is -1.3e-05 with a 90% CI of [-2.0e-04, +1.4e-04] that straddles zero; total compounded excess is -1.2%. The distribution agrees with the single-split null: the strategy does not beat buy-and-hold.
- **Experiment 2 — XGBoost vs logistic regression (economic test): NULL.** The gradient-boosted strategy does not beat the linear one net of costs; if anything it is worse (mean daily excess -2.3e-04, 90% CI [-4.5e-04, +4.1e-05], total compounded excess -37.6% over the window from over-trading). Nonlinear structure adds nothing economically here.
- **Experiment 3 — volatility forecast vs persistence: BEATS PERSISTENCE.** The OLS model cuts pooled RMSE by 20.2% versus the naive persistence baseline (0.067 vs 0.085), with a 90% block-bootstrap CI on the per-day squared-error reduction of [1.7e-03, 3.6e-03] that excludes zero. QLIKE improves (0.48 vs 1.27) and out-of-sample R^2 versus persistence is 0.36. The improvement is positive in all eight folds and survives the 4-day embargo essentially unchanged, so it is not a fold-boundary artifact.
- **Experiment 4 — longer horizons (week/month/quarter/year): NULL.** The 1-day null does not automatically extend, so it was tested. At `h = 5, 21, 63` trading days every model's out-of-sample R^2 versus a prevailing-mean drift forecast is *negative* (linear −1.13, −2.65, −6.18; regularized XGBoost −0.36, −0.68, −0.68), with squared-error-reduction CIs lying entirely below zero at the rigorous horizons — the features are worse than forecasting the mean. Direction accuracy (0.56–0.58) stays below the always-up base rate at every horizon, even as that base rate climbs 0.61 → 0.68 → 0.75 → 0.84; that rising number is the equity risk premium, not a forecastable edge, which is exactly why the benchmark is drift/buy-and-hold rather than 50%. The yearly horizon (~8 non-overlapping windows) is reported illustrative-only. The null broadens: OHLCV features do not beat drift or buy-and-hold from a day out to a quarter.
- **Experiment 5 — 26-feature model vs HAR-RV: NULL (matches HAR-RV).** Experiment 3's baseline was weak. Against HAR-RV (Corsi 2009), the standard strong baseline, the 26-feature model's RMSE gain falls to **5.0%** (0.06749 vs 0.07105) with a 90% CI on the squared-error reduction of [−5.9e-05, +1.0e-03] that straddles zero — below the pre-registered 10% bar and not significant. HAR-RV already beats persistence by 16.0%. The model-vs-persistence row reproduces Experiment 3 exactly (20.20%, 0.06749/0.08458), a built-in check that the harness is unchanged. Honest claim: the model *matches* HAR-RV, it does not beat it.
- **Experiment 6 — feature ablation: NULL (the win is the volatility lags).** Three nested OLS models show the four realized-vol lags alone recover **17.3%** of the 20.2% win over persistence; adding the other 22 features (returns, SMAs, RSI, drawdowns, volume) contributes only ~3.5% more, and that increment's CI [−1.5e-04, +9.2e-04] straddles zero. The volatility result is a volatility-lag (HAR-style) result, not a "26 clever features" result.
- **Experiment 7 — volatility targeting, does forecast accuracy pay? NULL, and inverted.** The same vol-targeting strategy fed by three forecasts: buy-and-hold Sharpe over the window is 1.010, and **no source beats it significantly** (Sharpe differences +0.137 persistence, +0.118 HAR-RV, +0.0001 model — every 90% CI straddles zero). Strikingly, the ordering is *reversed* from RMSE: the crudest forecast (persistence) gives the largest point Sharpe gain because it is the most reactive (it hits the 2× cap 16.7% of the time and times exposure most aggressively), while the best-RMSE 26-feature model produces a Sharpe identical to buy-and-hold. Better RMSE did not buy a better Sharpe after costs — it bought a worse one.
- **Experiment 8 — option-implied volatility (VIX): NULL both ways.** The one experiment that uses information beyond SPY OHLCV (VIX is built from option prices, so it steps outside weak-form efficiency). As a raw *forecast* of 5-day realized vol, VIX/100 is **27.8% worse** than the model (and slightly worse than persistence) — the variance risk premium biases it high. As an added *feature*, where the OLS can de-bias it, lagged VIX adds **nothing** (incremental RMSE −0.37%, CI straddling zero): for a 5-day horizon the historical realized-vol lags already contain what implied vol would add. Stepping outside weak-form did not help.

The honest summary: next-day *direction* is not predictable on SPY with these features, a nonlinear model does not rescue it, and the null holds out to weekly, monthly, and quarterly horizons. Near-term *volatility* **is** forecastable — but stated precisely: a linear combination of volatility lags beats last-value persistence by ~20% and **matches, but does not beat, HAR-RV**, with the other 22 features adding nothing significant; when that forecast is put to work, **no vol-targeting variant beats buy-and-hold on Sharpe, and the most accurate forecast delivers the least economic value**; and even **option-implied VIX does not improve the forecast**, as an input or a feature. Predictable risk, unpredictable return — and even the predictable risk is just HAR, does not pay, and is not helped by the options market.

## Limitations (the scope is the defense)

Every result above holds only inside these boundaries. Naming them is not a hedge; it is what keeps the claims true.

- **Single asset.** Everything is SPY. A single liquid index was chosen to avoid the survivorship bias of a multi-stock universe, but the price is zero cross-sectional breadth: nothing here generalizes to single names, other indices, or other asset classes.
- **Price and volume features only, with one exception.** The 26 features and all direction/horizon experiments are functions of OHLCV alone (returns, realized vol, moving-average ratios, drawdowns, RSI, intraday range, volume z-scores) — no fundamentals, macro, order flow, or sentiment. The single exception is Experiment 8, which adds **option-implied VIX** to the *volatility* forecast (and finds it does not help). A null on direction is a statement about *this feature set*, not about all possible signals.
- **Weak-form efficiency (except Experiment 8).** Because their inputs are past prices and volume, the direction and horizon nulls bear only on **weak-form** market efficiency; they say nothing about semi-strong (public information) or strong (private information) forms. Experiment 8 deliberately steps outside weak-form by adding VIX — public option-market information — to the volatility forecast, and finds no improvement, so on this target the step across the boundary changed nothing.
- **One window, regime-bound.** Features begin ~2010-10 (after the 200-day SMA warm-up); the pooled out-of-sample window is 2020-06 to 2024-06 (~1008 trading days). That window contains the COVID crash and the 2022 bear market — informative, but not a guarantee the results survive into other regimes.
- **Horizons from one day to one year — but not intraday, and yearly only illustratively.** Direction was tested at 1, 5, 21, and 63 trading days (all null) and 252 days (illustrative-only, ~8 non-overlapping windows); the volatility target is 5-day-ahead realized vol. Nothing here speaks to intraday horizons, and the yearly result is not evidence either way. The long-horizon predictability documented in the literature lives mostly in *valuation fundamentals* (dividend yield, CAPE), which are outside this OHLCV-only feature set by construction — so this is a null on price/volume, not on all long-horizon predictors.
- **One cost model.** Results are net of a single linear cost of `0.0001` per unit turnover. They are not robust to a materially higher cost assumption, slippage, or market impact.
- **The volatility win is a HAR-style win, checked against HAR-RV.** The 20% edge is over *naive last-value persistence*. Experiment 5 shows it does **not** clear a standard HAR-RV baseline (only ~5%, CI straddling zero), and Experiment 6 shows the volatility lags — not the other 22 features — carry it. So the honest claim is "matches HAR-RV," and it is still **not** a claim to beat a *tuned* GARCH/HAR or the implied volatility surface.

## What this project does and does not claim

**Failing to reject the null is not the same as proving the null true.** A non-significant direction result means the test — at this sample size, with this feature set, on this asset and window — *did not find* an edge, not that no edge exists. The pre-registered bar (a 90% CI excluding zero) controls false positives, not false negatives, and with ~1008 noisy daily observations the test has limited power to detect a *small* true edge. So the defensible statement is "we did not find next-day direction predictable here," which is strictly weaker than "next-day direction is unpredictable." Absence of evidence is not evidence of absence.

**What this project does not claim.** It does not claim markets are efficient, that SPY's direction is unpredictable in general, or that the EMH is true. It does not claim volatility forecasting is easy, or that the OLS model would beat a tuned GARCH/HAR or implied vol. It does not claim any strategy here is profitable. It claims exactly, and only, this: under a pre-registered, leakage-controlled, costed protocol on SPY daily data from 2020-06 to 2024-06, using price and volume features alone — (1) a logistic and a gradient-boosted next-day **direction** model did not beat buy-and-hold, (2) that direction null also held at 5-, 21-, and 63-day **horizons** (with the 1-year horizon under-powered and reported illustratively), against both a drift and a buy-and-hold benchmark, (3) an OLS forecast of 5-day-ahead realized **volatility** beat naive persistence by ~20% out of sample but **matched, and did not beat, a standard HAR-RV baseline** — with an ablation showing the volatility lags, not the other features, carry that win; (4) when those forecasts drove a costed **volatility-targeting** strategy, **none beat buy-and-hold on Sharpe**, and the most accurate forecast produced the least economic value; and (5) adding **option-implied VIX** — stepping outside weak-form efficiency — did not improve the volatility forecast, neither as a raw input nor as a feature. Predictable risk, unpredictable return — for this asset, these features, these horizons, this window.

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
5. **Overlapping-target inflation at long horizons** (a `k`-day forward return sampled daily overlaps its neighbours by `k-1` days, so naive R^2 and standard errors are wildly overstated — the "myth of long-horizon predictability").
   - The horizon experiment scales the bootstrap block length to `>= 2k` so the overlap-induced autocorrelation survives resampling, and embargoes `k-1` training rows per fold: [experiment_horizons.py](src/experiment_horizons.py) (`_horizon_folds`, `embargo = horizon - 1`).
   - The number of *non-overlapping* windows (`n_oos / k`) is reported for every horizon (202 → 48 → 16 → 8), and horizons are tiered in advance — 5/21 rigorous, 63 suggestive, 252 illustrative-only — so a thin-sample "positive" at a year is never dressed up as evidence: [experiment_horizons.py](src/experiment_horizons.py) (`RIGOR_TIER`).

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
- `horizon_ladder.png`: OOS R² vs drift and accuracy vs base rate across the 5/21/63/252-day horizons (rendered only after `python3 src/experiment_horizons.py` has produced `data/horizon_ladder.csv`).

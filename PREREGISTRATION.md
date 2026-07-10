# Pre-Registration: Walk-Forward Evaluation of the SPY Models

This document fixes the hypotheses, evaluation procedure, and success
thresholds **before** the experiment code is run or any result is seen. It
exists so the verdicts in this repository cannot be reverse-engineered from the
data. Nothing below may be edited after results are produced; if the apparatus
turns out to be wrong, fix it in a new, dated entry and re-run, rather than
moving a goalpost.

Author: project owner. Committed: 2026-05-29.

## Why walk-forward replaces the single split

The original pipeline (`src/ml_pipeline.py`) reports one chronological 80/20
split: one accuracy, one strategy-vs-buy-and-hold verdict. A single split is a
single draw from a noisy process. These experiments replace that point estimate
with a **distribution** over expanding-window folds and a **block bootstrap**
that respects the serial correlation of daily returns (unlike the i.i.d. paired
bootstrap in `src/model_backtest.py`).

## Shared harness (fixed for all three experiments)

- **Scheme:** expanding-window walk-forward. Each fold re-fits the model on all
  rows strictly before its out-of-sample (OOS) test block, so no future row
  ever informs a fit. All preprocessing (e.g. `StandardScaler`) is fit on the
  fold's training rows only. Inference uses only the leakage-safe features,
  which are already known as of `feature_timestamp` (= the prior close).
- **Folds:** `n_folds = 8`, each OOS test block = `126` trading days (~6 months).
  The 8 blocks tile the most recent `8 * 126 = 1008` trading days (~4 years)
  contiguously. The first fold therefore trains on all rows before that window
  (~2400+ rows, ~9.5 years), satisfying a minimum train size of ~3 years.
- **Block bootstrap:** moving (overlapping) block bootstrap on the **pooled OOS
  daily series**. Block length = `21` trading days (~1 month), `n_resamples =
  10000`, `seed = 42`. Confidence intervals are **90%** (5th to 95th
  percentile). A statistic is "significant" only if its 90% CI excludes zero on
  the side that matters (lower bound > 0 for an improvement).
- **Reproducibility:** all RNG seeded with `42`. XGBoost run single-threaded
  with a fixed seed.

The pooled OOS window is contiguous, so it is also evaluated as one costed
backtest using the existing `src/backtest.py` engine (cost per unit turnover =
`0.0001`), with a prior-close anchor on the first OOS day exactly as in
`src/model_backtest.py`.

---

## Experiment 1 — Logistic regression direction, as a distribution

**Question.** Does the existing logistic-regression direction strategy beat SPY
buy-and-hold out of sample, once we look at a distribution instead of one split?

**Model (unchanged from `src/ml_pipeline.py`).** `StandardScaler` +
`LogisticRegression(C=1.0, max_iter=1000)`. Decision threshold 0.5: predicted up
=> long one unit, otherwise flat. Costed with the existing backtester.

**Headline statistic.** Mean daily excess return =
`LR_net_strategy_return - SPY_buy_and_hold_return`, over the pooled OOS window.

**Pre-registered success criterion.** The LR strategy is declared to beat
buy-and-hold **iff the block-bootstrap 90% CI of the mean daily excess return
excludes zero on the low side (5th percentile > 0).** This single number decides.

**Reported alongside (not gating):** per-fold direction accuracy (8 values),
per-fold excess return (8 values), pooled accuracy, pooled ROC AUC, bootstrap
P(mean daily excess > 0), total compounded excess return, annualized Sharpe
difference.

**Prior / expectation.** The single-split result was already a null (accuracy
0.507 vs a 0.551 always-up base rate; strategy excess -3.3%; P(beat) 0.386). The
honest expectation is that the distribution also fails this criterion.

---

## Experiment 2 — XGBoost vs logistic regression (economic test only)

**Question.** Does a nonlinear model (gradient-boosted trees) extract structure
the linear model misses, measured where it would actually matter — the
costed strategy?

**Model.** `XGBClassifier` with **fixed, pre-registered hyperparameters (no
tuning, no early stopping, so there is no tuning-leakage):**

```
n_estimators      = 300
max_depth         = 3
learning_rate     = 0.05
subsample         = 0.8
colsample_bytree  = 0.8
min_child_weight  = 5
reg_lambda        = 1.0
objective         = "binary:logistic"
eval_metric       = "logloss"
tree_method       = "hist"
random_state      = 42
n_jobs            = 1
```

No feature scaling (trees are scale-invariant). Same 0.5 decision threshold,
same costed backtester, same 8 OOS folds as Experiment 1.

**Headline statistic.** Mean daily excess return =
`XGB_net_strategy_return - LR_net_strategy_return`, paired by day over the pooled
OOS window (block bootstrap on the paired daily difference).

**Pre-registered success criterion.** XGBoost is declared to add economically
useful nonlinear structure **iff the block-bootstrap 90% CI of the mean daily
(XGB - LR) excess return excludes zero on the low side (5th percentile > 0).**
This single number decides. (Statistical-accuracy improvement is explicitly
*not* the test; the chosen bar is economic.)

**Reported alongside (not gating):** XGB-vs-buy-and-hold mean daily excess and
CI, bootstrap P(XGB beats LR), total compounded (XGB - LR) excess, annualized
Sharpe difference, per-fold XGB accuracy and excess return, pooled XGB accuracy
and log loss.

**Prior / expectation.** On a single liquid index with weak daily
predictability, the honest expectation is no economically significant edge over
the linear model.

---

## Experiment 3 — `target_volatility_5d` regression vs persistence

**Question.** Can a model forecast SPY's 5-day-ahead realized volatility better
than naive persistence?

**Target.** `target_volatility_5d`: forward 5-trading-day annualized realized
volatility over `[target_timestamp, target_volatility_end_timestamp]`. Rows
whose forward window is incomplete (trailing NaN target) are dropped before
folding; the harness layout is otherwise identical.

**Model.** `StandardScaler` + `LinearRegression` (ordinary least squares) on the
26 leakage-safe features. OLS has no hyperparameters, so there is no tuning and
no tuning-leakage. For the QLIKE metric only, forecasts are floored at `1e-6` to
keep them positive.

**Baseline.** Naive persistence: forecast = `realized_vol_5d`, the trailing
5-day annualized realized volatility already in the feature set (known as of
`feature_timestamp`). Both model and baseline use only information available
before the target window, so the comparison is fair.

**Primary error metric.** Root mean squared error (RMSE) on annualized vol over
the pooled OOS window. RMSE improvement =
`(RMSE_persistence - RMSE_model) / RMSE_persistence`.

**Pre-registered success criterion (both conditions must hold).**
1. Pooled OOS RMSE improvement over persistence is **>= 10%**, and
2. the block-bootstrap 90% CI of the mean per-day squared-error reduction
   (`persistence_sq_err - model_sq_err`) **excludes zero on the low side (5th
   percentile > 0).**

**Reported alongside (not gating):** QLIKE for model and persistence, per-fold
RMSE for both (8 values each), pooled RMSE for both, out-of-sample R^2 of the
model relative to the persistence forecast.

**Prior / expectation.** Persistence is a strong baseline for realized
volatility. Beating it by >= 10% with significance on daily SPY data is a high
bar; the honest expectation is that the linear model roughly matches persistence
and does **not** clear the bar.

---

## What "honest" means here

For every experiment, whatever the criterion returns is reported as the verdict,
including a null. The criteria above are the only ones that decide pass/fail.
Secondary metrics are described as context, never used to rescue a failed
headline test.

---

## Apparatus refinements (dated, append-only)

### 2026-05-29 — embargo for the forward volatility target (Experiment 3)

`target_volatility_5d` looks forward 5 trading days, so the last 4 training rows
in each fold carry labels whose window crosses into the test block. The original
harness trained on all rows `[0, test_start)`, leaking up to 4 future
observations into each fold's fit. Fix: embargo the final `5 - 1 = 4` training
rows of every fold (`run_walk_forward(..., embargo=4)`). This **only removes**
information from the model and so can never inflate a positive result; it can
only shrink one. Experiments 1 and 2 use a 1-day target and need no embargo
(`embargo = 0`, unchanged). The pre-registered success criterion for Experiment
3 is otherwise unchanged. This entry was added before the embargoed result was
adopted as the reported verdict.

### 2026-06-27 — disclosure: what was forward-registered vs re-tested

The opening claim of this document — that the criteria were fixed "before the
experiment code is run or any result is seen" — is **fully true for Experiment 3
and only partly true for Experiments 1 and 2.** Honesty requires saying so
plainly, and this document's own rule is to correct by dated append, never by
editing the original text. The git history is the supporting evidence:

- `c15fcda`, `6333057` (both **2026-05-27**): the single-split logistic pipeline
  and its costed backtest were built and run. Its null was therefore **already
  known** — accuracy 0.507, ROC AUC 0.476, P(strategy beats buy-and-hold) 0.386.
- `1a32126` (**2026-05-29**): this pre-registration and the walk-forward
  experiment code were committed together.

What follows from those dates:

- **Experiments 1 and 2 are pre-registered *re-tests*, not blind first looks.**
  When this document was written, the single-split direction null was already in
  hand; the "Prior / expectation" sections above disclose it openly. What was
  genuinely fixed in advance here is the *distributional* decision rule (the
  90% block-bootstrap CI excluding zero), applied to models whose single-split
  behaviour was already seen. I do **not** claim these models were unseen. I make
  no positive claim on direction, so a re-test that confirms a known null carries
  no risk of a pre-registration violation favouring me.
- **Experiment 3 is the one genuine forward registration.** No volatility
  *forecasting model* — OLS or any other — existed in any commit before this
  document. The `target_volatility_5d` *label* was added on 2026-05-27, but it
  was never fit, scored, or eyeballed until the experiment code that shipped
  alongside this file. So its success criterion (>= 10% RMSE improvement over
  persistence **and** a block-bootstrap 90% CI excluding zero) was set before any
  volatility result existed. This is the experiment whose result is **positive**,
  and it is therefore the one where pre-registration actually does protective
  work. The ordering is git-verifiable: this document (2026-05-29) precedes the
  committed results snapshot in `RESULTS.md` (2026-06-27).

The short version, the one to say out loud: *I failed to pre-register Experiments
1 and 2 — they were run first as a single split, found null, then re-cast as a
distribution. I did pre-register Experiment 3, and the commit dates prove the
criterion came before the result.* A demonstrated correction beats a hidden
gap.

### 2026-07-02 — Experiment 4: does predictability appear at longer horizons?

**Motivation.** The 1-day null (Experiments 1-2) is a statement about the 1-day
horizon only. Weak-form efficiency is weakest, not strongest, at short horizons:
the literature finds *more* structure further out (3-12 month momentum, multi-year
mean reversion, valuation-ratio predictability whose R^2 rises with horizon). This
experiment asks whether SPY returns become predictable from OHLCV-only features at
weekly, monthly, quarterly, or yearly horizons.

**Target parameterization.** `build_leakage_safe_features(ohlcv, forward_horizon=h)`
sets `target_return` to the forward `h`-trading-day return measured from the
`feature_timestamp` close; `h = 1` reproduces the original next-close target
exactly. Features are the 26 canonical columns plus longer look-backs
(`return_126d`, `return_252d`, `realized_vol_126d/252d`, `drawdown_126d/252d`), all
still OHLCV-derived (`LONG_HORIZON_FEATURE_COLUMNS`). The canonical `FEATURE_COLUMNS`
used by Experiments 1-3 is unchanged, so their verdicts are untouched.

**Horizon ladder and rigor tiers (fixed in advance).** `h in {5, 21, 63, 252}`
trading days = week / month / quarter / year. Because the number of *non-overlapping*
windows (`n_oos / h`) collapses with `h`, tiers are declared before results:
`h = 5, 21` **rigorous**; `h = 63` **suggestive**; `h = 252` **illustrative-only** —
a positive at 252 is reported, never counted as evidence.

**Two framings, two benchmarks.**
1. *Regression* — `StandardScaler + LinearRegression` and `XGBRegressor`
   (Experiment-2 hyperparameters, `objective="reg:squarederror"`) forecast
   `target_return`. Benchmark = constant prevailing-mean *drift*. Headline =
   Campbell-Thompson out-of-sample R^2 vs drift and the block-bootstrap CI of the
   mean per-period squared-error reduction (`drift_sq_err - model_sq_err`).
2. *Direction* — `LogisticRegression` and `XGBClassifier` forecast
   `target_direction`. Benchmark = the drift-aware always-majority base rate
   (reported next to accuracy so the base-rate illusion is explicit). Economic
   headline = a long/flat strategy over **non-overlapping** `h`-day holding periods
   vs buy-and-hold, and the block-bootstrap CI of the mean per-window excess.

**Overlapping-data and sample-size controls.** Bootstrap block length =
`max(21, 2h)` so overlapping-target autocorrelation survives resampling; walk-forward
embargo = `h - 1`; test block = `max(126, 2h)` so every OOS block spans at least one
horizon; effective non-overlapping `N` is reported for every horizon.

**Pre-registered success criteria (per horizon).**
- *Beats drift* iff OOS R^2 vs drift `> 0` **and** the 90% block-bootstrap CI of the
  mean per-period squared-error reduction excludes zero on the low side **and** the
  per-fold reduction is positive in a majority of folds.
- *Beats buy-and-hold* iff the 90% block-bootstrap CI of the mean per-window excess
  return excludes zero on the low side.
- `h = 252` verdict is fixed to "illustrative-only" regardless of the numbers.

**Prior / expectation.** OHLCV-only, single asset, one regime, no valuation
fundamentals (where the literature's long-horizon signal actually lives): the honest
expectation is a null at the rigorous horizons.

**Timing disclosure (same standard as the 2026-06-27 entry).** This registration is
committed **in the same change as its results and code** (`src/experiment_horizons.py`),
so it is *not* a blind forward registration. What was genuinely fixed ahead of the
code is the design: the horizon ladder, the tiers, both benchmarks, and the two
CI-excludes-zero decision rules were set in the approved implementation plan before
the experiment was written or run. Unlike Experiments 1-2 there was no prior
single-split peek, but I make **no** claim of temporal separation between criterion
and result here. As before, I make no positive claim (the rigorous-horizon result is
a null), so a confirmed null carries no pre-registration violation favouring me.

### 2026-07-09 — Experiments 5 & 6: is the volatility win real, and where is it from?

**Motivation.** Experiment 3's headline — the 26-feature OLS beats naive persistence
on 5-day realized volatility by ~20% RMSE — is measured against a *weak* baseline.
Persistence (last-value) is the easiest thing to beat. Two nested follow-ups test
whether the win survives a strong baseline and where inside the feature set it comes
from. Both run through the identical Experiment-3 apparatus: expanding 8×126 folds,
`embargo = 4`, `StandardScaler + LinearRegression` (`_ols_fit_predict`), moving-block
bootstrap (block 21, 10000 resamples, seed 42), 90% CI. The target is unchanged
(`target_volatility_5d`). New regressors are OHLCV-derived and kept out of the
canonical `FEATURE_COLUMNS`, so Experiments 1-4 are untouched.

**Experiment 5 — 26-feature OLS vs HAR-RV.** HAR-RV (Corsi 2009) is the standard
strong baseline: an OLS of forward 5-day RV on three lagged realized-vol components,
annualized daily / weekly / monthly volatility from daily squared returns
(`HAR_RV_COLUMNS`, windows 1 / 5 / 22). No hyperparameters, so no tuning-leakage —
the same argument that licensed the Experiment-3 OLS.
- *Headline statistic.* Pooled RMSE improvement of the 26-feature model over HAR-RV,
  and the block-bootstrap CI of the mean per-day squared-error reduction
  (`har_sq_err - model_sq_err`).
- *Pre-registered success criterion (both must hold).* RMSE improvement over HAR-RV
  `>= 10%` **and** the 90% CI excludes zero on the low side.
- *Reported alongside (not gating).* HAR-RV vs persistence, model vs persistence (a
  built-in reproduction check of Experiment 3), QLIKE and OOS R² for each pair.
- *Prior / expectation.* HAR-RV is strong; the honest expectation is that the 20%
  edge over persistence shrinks a lot, possibly to nothing. "Beats persistence ~20%,
  matches HAR-RV" is the defensible claim; "beats persistence ~20%" while never
  checking HAR-RV is the claim that does not survive scrutiny.

**Experiment 6 — feature ablation (nested models).** Three models against the same
target: persistence (`realized_vol_5d` as the forecast), an OLS on the four
realized-vol lags only (`VOLATILITY_FEATURE_COLUMNS` = realized_vol 5/10/21/63), and
the full 26-feature OLS.
- *Pre-registered question / criterion.* Does the full set beat vol-lags-only? Pass
  iff the 90% CI of the mean per-day *incremental* squared-error reduction
  (`vollag_sq_err - full_sq_err`) excludes zero on the low side.
- *Reported alongside (not gating).* Full vs persistence and vol-lags-only vs
  persistence, so the decomposition of the ~20% win is visible.
- *Language rule fixed in advance.* If vol lags carry the win, the honest claim
  collapses to "a linear combination of volatility lags beats last-value persistence"
  (≈ rediscovering HAR). If the non-vol features add signal beyond vol lags with a CI
  excluding zero, that is the genuinely interesting result, stated no more strongly
  than the CI supports.

**Timing disclosure (same standard as the 2026-06-27 and 2026-07-02 entries).** Both
experiments are committed **in the same change as their code and results**
(`src/experiment_vol_baselines.py`); this is not a blind forward registration. The
model set, the baselines, and the CI-excludes-zero decision rules were fixed in the
approved plan before the code was written. These are re-tests that can only *weaken*
the Experiment-3 positive (a stronger baseline, a stricter nested comparison), so the
direction of any bias is against my own prior headline, not for it.

### 2026-07-09 — Experiment 7: does a better volatility forecast buy a better Sharpe?

**Motivation.** Experiments 3-6 are about statistical accuracy (RMSE). The question a
practitioner asks is whether accuracy converts into economic value. This experiment
runs one volatility-targeting strategy through the existing costed backtester, fed by
three forecasts of the same 5-day-ahead realized vol -- naive persistence, HAR-RV,
and the 26-feature OLS -- and asks, per source, whether it beats SPY buy-and-hold on
risk-adjusted return.

**Strategy (fixed in advance).** Daily target exposure = `target_vol / forecast_vol`,
long-only, floored and capped: `TARGET_ANNUAL_VOL = 0.15`, `POSITION_CAP = 2.0`,
`FORECAST_FLOOR = 1e-6` (an OLS vol forecast can be non-positive; the floor + cap
bound it). Costed at `0.0001` per unit turnover, the same backtester
(`backtest_daily_signal`) and prior-close anchor used everywhere else: the position
held for the close-to-close return ending on day t is sized by the forecast made as
of day t-1 (`feature_timestamp`), so it is leakage-safe. The strategy, target vol,
cap, and cost are **identical** across the three forecasts; only the forecast changes,
so any difference is attributable to the forecast alone. The Sharpe comparison is
invariant to `TARGET_ANNUAL_VOL` up to the cap, so that constant is a scale, not a
tuned knob.

**Headline statistic.** Annualized Sharpe difference (strategy - buy-and-hold),
computed on the pooled out-of-sample daily returns, with a moving-block bootstrap
(block 21, 10000 resamples, seed 42) paired by day.

**Pre-registered success criterion (per source, Experiment-1 style).** A source's
vol-targeting strategy is declared to beat buy-and-hold **iff the block-bootstrap 90%
CI of the Sharpe difference excludes zero on the low side.**

**Reported alongside (not gating).** Mean daily excess return and its CI, total
return vs buy-and-hold, turnover and transaction cost, average position, capped and
floored fractions, and the same statistics for all three sources side by side (the
comparison that ties forecast accuracy to economic value).

**Prior / expectation.** Volatility targeting can raise Sharpe (Moreira-Muir 2017),
but Experiments 5-6 showed the three forecasts are near-identical in accuracy, so the
honest expectation is that the sources land close together and that better RMSE does
not obviously buy a better Sharpe after costs. Reported either way.

**Timing disclosure (same standard as the entries above).** Committed in the same
change as its code and results (`src/experiment_vol_targeting.py`); not a blind
forward registration. The strategy definition, the three sources, and the
Sharpe-CI decision rule were fixed in the approved plan before the code was written.
The experiment makes no positive claim I have an interest in — a null (no source
beats buy-and-hold) is the expected and reported outcome.

### 2026-07-09 — Experiment 8: does option-implied volatility (VIX) help?

**Motivation and scope change.** Every experiment so far used SPY OHLCV only, so the
direction nulls bear on *weak-form* efficiency. This one deliberately steps outside
it: the CBOE VIX index is built from S&P 500 option prices and embeds forward-looking
information no function of past prices and volume can hold. VIX is quoted in
annualized percentage points; dividing by 100 puts it in the same decimal annualized
units as `target_volatility_5d`. Data source: `^VIX` via yfinance into
`data/VIX_data.csv` (`src/vix_data.py`), same 2010-2024 range as SPY.

**Two questions, both through the Experiment-3 harness** (expanding 8×126 folds,
`embargo = 4`, `_ols_fit_predict`, block bootstrap 21/10000/seed 42, 90% CI). VIX is
taken as of the feature close (the prior close), so it is known before the target
window and shares the models' information set — leakage-safe.

**Q1 — VIX as a standalone forecast.** Forecast = VIX/100 as of the feature close.
- *Criterion (same two-part bar as Experiment 5).* VIX beats the 26-feature model
  iff its RMSE improvement over the model is `>= 10%` **and** the 90% block-bootstrap
  CI of the mean per-day squared-error reduction excludes zero.
- *Reported alongside.* VIX vs persistence, VIX vs HAR-RV, and model vs persistence
  (an Experiment-3 reproduction check on this dataset).
- *Prior.* VIX carries a variance risk premium (it sits above realized vol on
  average) and is a 30-day measure used on a 5-day target, so as a raw point forecast
  it is biased high and expected to lose on RMSE. Its value, if any, is as a feature.

**Q2 — VIX as an added feature.** Add lagged VIX (as of the feature close) to the 26
features and refit the OLS.
- *Criterion (same as Experiment 6).* The augmented model beats the 26-feature model
  iff the 90% CI of the mean per-day *incremental* squared-error reduction excludes
  zero on the low side.
- *Prior.* Uncertain: implied vol can carry incremental information, but for a 5-day
  horizon the historical realized-vol lags are already informative, so the honest
  expectation is a small, possibly non-significant, contribution.

**Timing disclosure (same standard as the entries above).** Committed in the same
change as its code and results (`src/experiment_vix.py`); not a blind forward
registration. The two questions, the VIX construction (as-of feature close, /100),
and both decision rules were fixed in the approved plan before the code was written.

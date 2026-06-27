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

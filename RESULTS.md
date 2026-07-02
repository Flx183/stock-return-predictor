# Results Snapshot

These are the verdicts the pre-registered criteria in `PREREGISTRATION.md`
returned, reported exactly as produced. This file is committed **after** that
pre-registration (pre-registration commit: 2026-05-29; this snapshot committed:
2026-06-27) so the ordering "criteria fixed before results seen" is verifiable
from git history, not merely asserted.

Every number below is reproducible from raw data with a fixed seed (42):

```bash
python3 src/experiment_direction.py    # Experiments 1 and 2
python3 src/experiment_volatility.py   # Experiment 3
python3 src/experiment_horizons.py     # Experiment 4 (added 2026-07-02)
```

The raw output CSVs that back this snapshot are committed alongside it under
`data/walkforward_*.csv` and `data/horizon_*.csv` (force-added past the `data/`
gitignore precisely because they are the pre-registered evidence). The
single-split null is backed by `data/ml_test_metrics.csv` and
`data/ml_monte_carlo_comparison.csv`.

Pooled out-of-sample window: ~2020-06 to 2024-06, 1008 trading days, expanding
walk-forward over 8 contiguous 126-day folds.

## Experiment 1 — Logistic-regression direction vs buy-and-hold: NULL

The single pre-registered decision statistic is the block-bootstrap 90% CI of
the mean daily excess return; it passes only if the low bound clears zero.

| Statistic | Value |
| --- | --- |
| Pooled accuracy | 0.5377 |
| Always-up base rate | 0.5397 |
| Pooled ROC AUC | 0.5038 |
| Predicted-up rate | 0.875 |
| Mean daily excess return | -1.27e-05 |
| 90% block-bootstrap CI | [-1.96e-04, +1.38e-04] |
| P(mean daily excess > 0) | 0.366 |
| Total compounded excess | -1.17% |
| **CI excludes zero (low side)?** | **No** |

**Verdict: NULL.** Accuracy is below the always-up base rate, AUC is at chance,
and the CI straddles zero. The strategy does not beat buy-and-hold.

## Experiment 2 — XGBoost vs logistic regression (economic test): NULL

Decision statistic: block-bootstrap 90% CI of the mean daily (XGB − LR) net
return, paired by day.

| Statistic | Value |
| --- | --- |
| Pooled accuracy (XGB) | 0.5179 |
| Pooled ROC AUC (XGB) | 0.5002 |
| Mean daily (XGB − LR) excess | -2.28e-04 |
| 90% block-bootstrap CI | [-4.51e-04, +4.06e-05] |
| P(XGB beats LR) | 0.0871 |
| Total compounded (XGB − LR) excess | -37.57% |
| **CI excludes zero (low side)?** | **No** |

**Verdict: NULL.** The gradient-boosted strategy does not beat the linear one net
of costs; over-trading makes it materially worse. A flexible learner does not
rescue near-zero signal.

## Experiment 3 — Volatility forecast vs persistence: BEATS PERSISTENCE

Decision rule (both conditions must hold): pooled RMSE improvement >= 10% **and**
block-bootstrap 90% CI of the mean per-day squared-error reduction excludes zero
on the low side.

| Statistic | Value |
| --- | --- |
| Model RMSE | 0.06749 |
| Persistence RMSE | 0.08458 |
| RMSE improvement | 20.20% (threshold 10%) |
| Mean per-day squared-error reduction | 2.60e-03 |
| 90% block-bootstrap CI | [1.68e-03, 3.62e-03] |
| P(reduction > 0) | 1.000 |
| QLIKE (model vs persistence) | 0.480 vs 1.267 |
| Out-of-sample R² vs persistence | 0.363 |
| RMSE improvement positive in all 8 folds | Yes |
| Survives 4-day embargo | Yes |
| **Both conditions met?** | **Yes** |

**Verdict: BEATS PERSISTENCE.** The OLS forecast (effectively a HAR-RV-style
combination of 5/10/21/63-day realized volatility) cuts RMSE by ~20% out of
sample, with a CI that excludes zero, improvement in every fold, and no
sensitivity to the embargo. This is the result the forward pre-registration
protects (see `PREREGISTRATION.md`, 2026-06-27 disclosure entry).

## Experiment 4 — Longer horizons (week / month / quarter / year): NULL

Does predictability appear further out, where the 1-day null cannot reach? Same
walk-forward harness, forward-`h`-day return target, OHLCV-only features (26
canonical + longer look-backs). Two pre-registered decision rules: beat a
constant-drift forecast as a regression (Campbell-Thompson OOS R^2 > 0 **and** a
90% block-bootstrap CI on the mean squared-error reduction excluding zero), and
beat buy-and-hold as a non-overlapping direction strategy (90% CI on mean
per-window excess excluding zero). Bootstrap block length `>= 2h`, embargo
`h - 1`.

| Horizon | Tier | Non-overlap windows | Up-rate (base) | Linear OOS R² vs drift | XGB OOS R² vs drift | Direction acc / base-rate acc | Strategy excess vs B&H | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5d (week) | rigorous | 202 | 0.609 | −1.13 | −0.36 | 0.564 / 0.609 | −33.7% | **NULL** |
| 21d (month) | rigorous | 48 | 0.679 | −2.65 | −0.68 | 0.569 / 0.679 | −23.4% | **NULL** |
| 63d (quarter) | suggestive | 16 | 0.750 | −6.18 | −0.68 | 0.565 / 0.750 | −54.7% | **NULL** |
| 252d (year) | illustrative | 8 | 0.844 | −0.20 | −0.10 | 0.834 / 0.844 | −0.02% | illustrative-only |

**Verdict: NULL at every rigorous and suggestive horizon.** Three things, all
pointing the same way:

- **Regression is worse than drift, significantly so.** OOS R^2 vs a
  prevailing-mean forecast is negative at every horizon and every model; for the
  rigorous horizons the block-bootstrap CI of the squared-error reduction lies
  *entirely below zero* (e.g. h=5 linear [−1.1e-03, −2.4e-04]; h=21 linear
  [−1.0e-02, −1.0e-03]). Adding features to forecast a near-random target only
  adds variance. The regularized XGBoost is less bad than OLS but still never
  beats the mean.
- **Direction never beats the base rate.** Pooled accuracy (0.56–0.58) sits below
  the always-up base rate at every horizon, even as that base rate climbs from
  0.61 to 0.84. The models predict "up" 78–99% of the time and still lose to
  simply *always* predicting up.
- **The rising up-rate is the point.** 0.609 → 0.679 → 0.750 → 0.844 is the equity
  risk premium showing through: over a year SPY rose in ~84% of windows. That is
  compensation for risk, not a forecastable edge — and it is why "direction
  accuracy" is a trap at long horizons and why the benchmark here is drift /
  buy-and-hold, not 50%. At h=252 the "strategy" collapses into always-long
  (windows long 100% of the time) and merely reproduces buy-and-hold.

The yearly row is reported for completeness only: with ~8 non-overlapping windows
its CI straddles zero and it is not evidence either way. This does not prove
returns are unpredictable — only that OHLCV-only features do not beat drift or
buy-and-hold at these horizons on SPY, 2010–2024. The long-horizon predictability
in the literature lives mostly in valuation fundamentals, which are outside this
feature set by design.

## One-line summary

Next-day **direction** on SPY is not predictable with these price/volume
features, a nonlinear model does not change that, and **the null holds out to
weekly, monthly, and quarterly horizons too** — while near-term **realized
volatility** is forecastable well beyond naive persistence. Predictable risk,
unpredictable return, at every horizon tested.

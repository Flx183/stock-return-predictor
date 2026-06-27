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
```

The raw output CSVs that back this snapshot are committed alongside it under
`data/walkforward_*.csv` (force-added past the `data/` gitignore precisely
because they are the pre-registered evidence). The single-split null is backed
by `data/ml_test_metrics.csv` and `data/ml_monte_carlo_comparison.csv`.

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

## One-line summary

Next-day **direction** on SPY is not predictable with these price/volume
features, and a nonlinear model does not change that — but near-term **realized
volatility** is forecastable well beyond naive persistence. Predictable risk,
unpredictable return.

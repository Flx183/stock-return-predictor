# Results Snapshot

These are the verdicts the pre-registered criteria in `PREREGISTRATION.md`
returned, reported exactly as produced. This file is committed **after** that
pre-registration (pre-registration commit: 2026-05-29; this snapshot committed:
2026-06-27) so the ordering "criteria fixed before results seen" is verifiable
from git history, not merely asserted.

Every number below is reproducible from raw data with a fixed seed (42):

```bash
python3 src/experiment_direction.py     # Experiments 1 and 2
python3 src/experiment_volatility.py    # Experiment 3
python3 src/experiment_horizons.py      # Experiment 4 (added 2026-07-02)
python3 src/experiment_vol_baselines.py # Experiments 5 and 6 (added 2026-07-09)
python3 src/experiment_vol_targeting.py # Experiment 7 (added 2026-07-09)
python3 src/vix_data.py                  # pull ^VIX (external data, Experiment 8)
python3 src/experiment_vix.py           # Experiment 8 (added 2026-07-09)
```

The raw output CSVs that back this snapshot are committed alongside it under
`data/walkforward_*.csv`, `data/horizon_*.csv`, `data/vol_*_comparisons.csv`,
`data/vol_targeting_*.csv`, and `data/vix_*_comparisons.csv` (force-added past the
`data/` gitignore precisely because they are the pre-registered evidence). The
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

**Read this with Experiments 5 and 6 below, never alone.** Persistence is a weak
baseline. Against the standard strong baseline (HAR-RV) the ~20% edge shrinks to
~5% and is no longer significant, and an ablation shows the win is carried by the
volatility lags, not the other 22 features. The defensible one-liner is "beats
last-value persistence ~20%, matches HAR-RV," not "beats persistence ~20%" in
isolation.

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

## Experiment 5 — 26-feature model vs HAR-RV: NULL (matches HAR-RV)

The Experiment-3 win was measured against *persistence*, a weak baseline. HAR-RV
(Corsi 2009) — an OLS of forward 5-day RV on lagged daily/weekly/monthly realized
vol — is the standard strong baseline. Decision rule (both must hold): the
26-feature model's RMSE improvement over HAR-RV is >= 10% **and** the
block-bootstrap 90% CI of the mean per-day squared-error reduction excludes zero.

| Comparison | Model RMSE | Baseline RMSE | RMSE improvement | OOS R² | 90% CI of sq-err reduction | Excludes zero? |
| --- | --- | --- | --- | --- | --- | --- |
| **26-feature vs HAR-RV** (gating) | 0.06749 | 0.07105 | **5.02%** | 0.098 | [−5.9e-05, +1.0e-03] | **No** |
| HAR-RV vs persistence (context) | 0.07105 | 0.08458 | 15.99% | 0.294 | [1.4e-03, 2.8e-03] | Yes |
| model vs persistence (Exp 3 check) | 0.06749 | 0.08458 | 20.20% | 0.363 | [1.7e-03, 3.6e-03] | Yes |

**Verdict: NULL.** The 26-feature model does **not** beat HAR-RV: the RMSE gain is
~5% (below the 10% bar) and the CI straddles zero. HAR-RV alone captures most of
the edge (16% over persistence). The bottom row reproduces Experiment 3 to the
digit (20.20%, 0.06749/0.08458), confirming the harness is unchanged. Honest
claim: **the model matches HAR-RV; it does not beat it.**

## Experiment 6 — feature ablation (persistence < vol-lags-only < full 26): NULL

Three nested OLS models against the same target. Decision rule: the full 26-feature
set beats vol-lags-only iff the 90% CI of the mean per-day *incremental*
squared-error reduction excludes zero.

| Comparison | Model RMSE | Baseline RMSE | RMSE improvement | OOS R² | 90% CI of sq-err reduction | Excludes zero? |
| --- | --- | --- | --- | --- | --- | --- |
| **full vs vol-lags-only** (gating) | 0.06749 | 0.06993 | **3.49%** | 0.069 | [−1.5e-04, +9.2e-04] | **No** |
| vol-lags-only vs persistence (context) | 0.06993 | 0.08458 | 17.32% | 0.316 | [1.5e-03, 3.0e-03] | Yes |
| full vs persistence (context) | 0.06749 | 0.08458 | 20.20% | 0.363 | [1.7e-03, 3.6e-03] | Yes |

**Verdict: NULL.** The four volatility lags alone recover 17.3% of the 20.2% win;
the other 22 features (returns, SMAs, RSI, drawdowns, volume) add ~3% more, and
that increment's CI straddles zero. So the volatility result is a **volatility-lag
(HAR-style) result**, not a "26 clever features" result. Honest claim: **a linear
combination of volatility lags beats last-value persistence — essentially
rediscovering HAR.**

## Experiment 7 — volatility targeting: does a better forecast buy a better Sharpe? NULL

The same vol-targeting strategy (daily exposure = target_vol / forecast_vol,
long-only, 15% target, 2× cap, costed) run three times, changing only the forecast.
Buy-and-hold Sharpe over the window is **1.0095**. Decision rule (Experiment-1
style): a source beats buy-and-hold iff the block-bootstrap 90% CI of the Sharpe
difference excludes zero on the low side.

| Forecast feeding the strategy | Strategy Sharpe | Sharpe diff vs B&H | 90% CI of Sharpe diff | Excess total return | Turnover | Beats B&H? |
| --- | --- | --- | --- | --- | --- | --- |
| persistence (worst RMSE) | 1.146 | +0.137 | [−0.200, +0.536] | +25.0% | 170.6 | **No** |
| HAR-RV | 1.127 | +0.118 | [−0.168, +0.415] | +16.7% | 132.7 | **No** |
| 26-feature model (best RMSE) | 1.010 | +0.0001 | [−0.296, +0.251] | +0.2% | 133.4 | **No** |

**Verdict: NULL for all three — and the accuracy→value link is inverted.** No
forecast's vol-targeting strategy significantly beats buy-and-hold on Sharpe (every
CI straddles zero, over a window where buy-and-hold already had Sharpe ~1.0). More
pointedly: the **best-RMSE forecast produced the *worst* economic outcome** — the
26-feature model's vol-targeted Sharpe is identical to buy-and-hold (+0.0001), while
the crudest forecast (persistence) gave the largest point improvement (+0.137). The
mechanism is visible in the diagnostics: persistence is the most reactive forecast
(hits the 2× cap 16.7% of the time, highest turnover), so it times exposure most
aggressively — which happened to help over 2020–2024 but is not statistically
reliable. The lesson practitioners repeat: **better RMSE does not translate into a
better Sharpe after costs.** Here the RMSE ranking and the Sharpe ranking are
literally reversed.

## Experiment 8 — VIX (option-implied vol): NULL both ways (steps outside weak-form)

The only experiment that uses information beyond SPY OHLCV: VIX is built from S&P 500
option prices. Two questions on the same 5-day realized-vol target, VIX taken as of
the feature close (VIX/100, decimal annualized). The model-vs-persistence row
reproduces Experiment 3 exactly (20.20%, n=1008), confirming the row set is unchanged.

| Question | Comparison | RMSE | Baseline RMSE | Improvement | 90% CI of reduction | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **Q1** VIX as a forecast | VIX vs model | 0.08626 | 0.06749 | **−27.8%** | [−4.2e-03, −1.5e-03] | **NULL** |
| Q1 (context) | VIX vs persistence | 0.08626 | 0.08458 | −1.99% | [−2.0e-03, +1.6e-03] | worse |
| **Q2** VIX as a feature | model+VIX vs model | 0.06774 | 0.06749 | **−0.37%** | [−4.2e-04, +3.7e-04] | **NULL** |

**Verdict: NULL both ways.** As a raw forecast (Q1) VIX is **27.8% worse** than the
model and even slightly worse than naive persistence — exactly what the **variance
risk premium** predicts: VIX (median ~16.6% annualized) systematically sits above
realized vol, so as an uncalibrated point forecast of near-term RV it is biased high.
As an added feature (Q2), where the OLS can de-bias and scale it, lagged VIX still
adds **nothing** beyond the 26 historical-vol-inclusive features (incremental CI
straddles zero). For this 5-day horizon, past realized volatility already contains
what option-implied vol would contribute. Stepping outside weak-form efficiency did
not buy a better forecast.

## One-line summary

Next-day **direction** on SPY is not predictable with these price/volume
features, a nonlinear model does not change that, and **the null holds out to
weekly, monthly, and quarterly horizons too**. Near-term **realized volatility**
*is* forecastable — but the honest version is narrow: a linear combination of
volatility lags beats last-value persistence by ~20% and **matches, but does not
beat, HAR-RV**; the extra 22 features add nothing significant; when the forecast is
put to work in a vol-targeting strategy **no source beats buy-and-hold on Sharpe,
and the best-RMSE forecast delivers the least economic value**; and even **option-
implied volatility (VIX) does not improve the forecast**, as a raw input or a
feature. Predictable risk, unpredictable return — and even the predictable risk is
just HAR, does not pay, and is not helped by the options market.

"""
Experiments 5 and 6 from PREREGISTRATION.md: is the volatility win real, and
where does it come from?

Experiment 3 showed the 26-feature OLS beats naive last-value persistence on
5-day-ahead realized volatility by ~20% RMSE. Persistence is a weak baseline, so
that headline is only half the story. Two nested follow-ups, both run through the
identical Experiment-3 harness (same expanding folds, same 4-day embargo, same
block bootstrap, the same `_ols_fit_predict`), pin down what the number means:

- **Experiment 5 (HAR-RV).** Replace persistence with HAR-RV (Corsi 2009), the
  standard strong baseline: one OLS of forward 5-day RV on three lagged realized-
  vol components (daily / weekly / monthly). Pre-registered two-part bar, now
  against HAR-RV: the 26-feature model's pooled RMSE improvement over HAR-RV is
  >= 10% **and** the block-bootstrap 90% CI of the mean per-day squared-error
  reduction excludes zero. HAR-RV-vs-persistence is reported alongside, not
  gating. Honest prior: HAR-RV is strong, so the 20% edge over persistence likely
  shrinks a lot, possibly to nothing.

- **Experiment 6 (feature ablation).** Three nested models: persistence, an OLS on
  the four realized-vol lags only, and the full 26-feature OLS. Pre-registered
  question: does the full set beat vol-lags-only, i.e. does the 90% CI of the mean
  per-day *incremental* squared-error reduction exclude zero? The answer fixes the
  language. If vol lags do all the work, the honest claim is "a linear combination
  of volatility lags beats last-value persistence" -- essentially rediscovering
  HAR. If the other features add signal on top of vol lags, that is the
  interesting result, stated no more strongly than the CI supports.

All models and thresholds are pre-registered; nothing here is tuned on the result.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
	from src.baseline import load_spy_ohlcv
	from src.experiment_volatility import (
		PERSISTENCE_FEATURE,
		RMSE_IMPROVEMENT_THRESHOLD,
		_ols_fit_predict,
		_out_of_sample_r2,
		_qlike,
		_rmse,
	)
	from src.features import (
		FEATURE_COLUMNS,
		HAR_RV_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		TARGET_VOLATILITY_WINDOW,
		VOLATILITY_FEATURE_COLUMNS,
		build_leakage_safe_features,
	)
	from src.walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		bootstrap_mean,
		expanding_window_folds,
		run_walk_forward,
		summarize_bootstrap,
	)
except ModuleNotFoundError:  # Allows `python src/experiment_vol_baselines.py`.
	from baseline import load_spy_ohlcv
	from experiment_volatility import (
		PERSISTENCE_FEATURE,
		RMSE_IMPROVEMENT_THRESHOLD,
		_ols_fit_predict,
		_out_of_sample_r2,
		_qlike,
		_rmse,
	)
	from features import (
		FEATURE_COLUMNS,
		HAR_RV_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		TARGET_VOLATILITY_WINDOW,
		VOLATILITY_FEATURE_COLUMNS,
		build_leakage_safe_features,
	)
	from walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		bootstrap_mean,
		expanding_window_folds,
		run_walk_forward,
		summarize_bootstrap,
	)


REPO_ROOT = Path(__file__).resolve().parent.parent
HAR_FILE = REPO_ROOT / "data" / "vol_har_comparisons.csv"
ABLATION_FILE = REPO_ROOT / "data" / "vol_ablation_comparisons.csv"

EMBARGO = TARGET_VOLATILITY_WINDOW - 1


def _pooled_forecast(dataset, feature_columns, folds):
	"""One OLS model through the shared harness; pooled OOS forecasts + persistence."""
	return run_walk_forward(
		dataset,
		_ols_fit_predict,
		feature_columns,
		TARGET_VOLATILITY_COLUMN,
		folds=folds,
		metadata_columns=[PERSISTENCE_FEATURE],
		embargo=EMBARGO,
	)


def _compare(name, y_true, model_pred, baseline_pred, indices):
	"""RMSE improvement, OOS R^2, and the block-bootstrap CI of the mean per-day
	squared-error reduction of `model` over `baseline` (positive = model better)."""
	model_sq_err = (y_true - model_pred) ** 2
	baseline_sq_err = (y_true - baseline_pred) ** 2
	reduction = baseline_sq_err - model_sq_err
	bootstrap = summarize_bootstrap(
		bootstrap_mean(reduction, indices), point_estimate=float(np.mean(reduction))
	)
	model_rmse = _rmse(y_true, model_pred)
	baseline_rmse = _rmse(y_true, baseline_pred)
	return {
		"comparison": name,
		"model_rmse": model_rmse,
		"baseline_rmse": baseline_rmse,
		"rmse_improvement": float((baseline_rmse - model_rmse) / baseline_rmse),
		"oos_r2_vs_baseline": _out_of_sample_r2(y_true, model_pred, baseline_pred),
		"model_qlike": _qlike(y_true, model_pred),
		"baseline_qlike": _qlike(y_true, baseline_pred),
		"mean_sq_err_reduction": bootstrap["point_estimate"],
		"reduction_ci_low": bootstrap["ci_low"],
		"reduction_ci_high": bootstrap["ci_high"],
		"reduction_prob_positive": bootstrap["prob_positive"],
		"reduction_excludes_zero_low": bootstrap["excludes_zero_low"],
	}


def _prepare(feature_columns, ohlcv=None):
	if ohlcv is None:
		ohlcv = load_spy_ohlcv()
	dataset = build_leakage_safe_features(ohlcv, feature_columns=feature_columns)
	dataset = dataset.loc[dataset[TARGET_VOLATILITY_COLUMN].notna()].copy()
	folds = expanding_window_folds(len(dataset))
	return dataset, folds


def _aligned_truth(*pooled_frames):
	reference = pooled_frames[0]["target_timestamp"].to_numpy()
	for frame in pooled_frames[1:]:
		if not np.array_equal(frame["target_timestamp"].to_numpy(), reference):
			raise ValueError("pooled out-of-sample dates do not align across models.")
	return pooled_frames[0]["y_true"].to_numpy(dtype=float)


def run_har_experiment(block_length=DEFAULT_BLOCK_LENGTH, n_resamples=DEFAULT_N_RESAMPLES, seed=DEFAULT_SEED, ohlcv=None):
	dataset, folds = _prepare([*FEATURE_COLUMNS, *HAR_RV_COLUMNS], ohlcv=ohlcv)
	model = _pooled_forecast(dataset, FEATURE_COLUMNS, folds)
	har = _pooled_forecast(dataset, HAR_RV_COLUMNS, folds)

	y_true = _aligned_truth(model, har)
	model_pred = model["prediction"].to_numpy(dtype=float)
	har_pred = har["prediction"].to_numpy(dtype=float)
	persistence_pred = model[PERSISTENCE_FEATURE].to_numpy(dtype=float)

	indices = block_bootstrap_indices(len(y_true), block_length=block_length, n_resamples=n_resamples, seed=seed)
	comparisons = pd.DataFrame(
		[
			_compare("model_vs_har", y_true, model_pred, har_pred, indices),
			_compare("har_vs_persistence", y_true, har_pred, persistence_pred, indices),
			_compare("model_vs_persistence", y_true, model_pred, persistence_pred, indices),
		]
	)
	comparisons["n_oos_days"] = int(len(y_true))
	comparisons["block_length"] = int(block_length)

	gating = comparisons.set_index("comparison").loc["model_vs_har"]
	beats_har = bool(
		gating["rmse_improvement"] >= RMSE_IMPROVEMENT_THRESHOLD and gating["reduction_excludes_zero_low"]
	)
	comparisons["is_gating"] = comparisons["comparison"] == "model_vs_har"
	comparisons["rmse_improvement_threshold"] = float(RMSE_IMPROVEMENT_THRESHOLD)
	comparisons["beats_baseline_verdict"] = comparisons["comparison"].map(
		{"model_vs_har": beats_har}
	)
	return {"comparisons": comparisons, "beats_har": beats_har}


def run_ablation_experiment(block_length=DEFAULT_BLOCK_LENGTH, n_resamples=DEFAULT_N_RESAMPLES, seed=DEFAULT_SEED, ohlcv=None):
	dataset, folds = _prepare(FEATURE_COLUMNS, ohlcv=ohlcv)
	full = _pooled_forecast(dataset, FEATURE_COLUMNS, folds)
	vollag = _pooled_forecast(dataset, VOLATILITY_FEATURE_COLUMNS, folds)

	y_true = _aligned_truth(full, vollag)
	full_pred = full["prediction"].to_numpy(dtype=float)
	vollag_pred = vollag["prediction"].to_numpy(dtype=float)
	persistence_pred = full[PERSISTENCE_FEATURE].to_numpy(dtype=float)

	indices = block_bootstrap_indices(len(y_true), block_length=block_length, n_resamples=n_resamples, seed=seed)
	comparisons = pd.DataFrame(
		[
			_compare("full_vs_vollag", y_true, full_pred, vollag_pred, indices),
			_compare("full_vs_persistence", y_true, full_pred, persistence_pred, indices),
			_compare("vollag_vs_persistence", y_true, vollag_pred, persistence_pred, indices),
		]
	)
	comparisons["n_oos_days"] = int(len(y_true))
	comparisons["block_length"] = int(block_length)

	gating = comparisons.set_index("comparison").loc["full_vs_vollag"]
	full_adds_signal = bool(gating["reduction_excludes_zero_low"])
	comparisons["is_gating"] = comparisons["comparison"] == "full_vs_vollag"
	comparisons["adds_signal_verdict"] = comparisons["comparison"].map(
		{"full_vs_vollag": full_adds_signal}
	)
	return {"comparisons": comparisons, "full_adds_signal": full_adds_signal}


def _print_comparison(row):
	print(
		f"    RMSE model {row['model_rmse']:.6f} vs baseline {row['baseline_rmse']:.6f} "
		f"(improvement {row['rmse_improvement']:.4%}); OOS R^2 {row['oos_r2_vs_baseline']:.4f}"
	)
	print(
		"    mean per-day squared-error reduction "
		f"{row['mean_sq_err_reduction']:.6e}, 90% CI "
		f"[{row['reduction_ci_low']:.6e}, {row['reduction_ci_high']:.6e}], "
		f"excludes zero (low): {bool(row['reduction_excludes_zero_low'])}"
	)


def main():
	try:
		har = run_har_experiment()
		ablation = run_ablation_experiment()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	HAR_FILE.parent.mkdir(parents=True, exist_ok=True)
	har["comparisons"].to_csv(HAR_FILE, index=False)
	ablation["comparisons"].to_csv(ABLATION_FILE, index=False)

	har_rows = har["comparisons"].set_index("comparison")
	print("Experiment 5 - 26-feature OLS vs HAR-RV (forward 5-day realized vol):")
	print(f"  Outputs: {HAR_FILE.parent}")
	print("  Model vs HAR-RV (GATING, needs >=10% RMSE gain AND CI excluding zero):")
	_print_comparison(har_rows.loc["model_vs_har"])
	print("  HAR-RV vs persistence (context):")
	_print_comparison(har_rows.loc["har_vs_persistence"])
	print("  Model vs persistence (context; should reproduce Experiment 3):")
	_print_comparison(har_rows.loc["model_vs_persistence"])
	print(
		"  PRE-REGISTERED VERDICT: "
		+ (
			"the 26-feature model beats HAR-RV."
			if har["beats_har"]
			else "the 26-feature model does NOT beat HAR-RV (null); report it as matching HAR-RV."
		)
	)
	print("")

	ablation_rows = ablation["comparisons"].set_index("comparison")
	print("Experiment 6 - feature ablation (persistence < vol-lags-only < full 26):")
	print("  Full vs vol-lags-only (GATING, needs incremental CI excluding zero):")
	_print_comparison(ablation_rows.loc["full_vs_vollag"])
	print("  Full vs persistence (context):")
	_print_comparison(ablation_rows.loc["full_vs_persistence"])
	print("  Vol-lags-only vs persistence (context):")
	_print_comparison(ablation_rows.loc["vollag_vs_persistence"])
	print(
		"  PRE-REGISTERED VERDICT: "
		+ (
			"the non-vol features add signal beyond volatility lags."
			if ablation["full_adds_signal"]
			else "the non-vol features add NOTHING beyond volatility lags (the win is the vol lags)."
		)
	)


if __name__ == "__main__":
	main()

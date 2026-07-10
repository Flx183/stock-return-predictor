"""
Experiment 8 from PREREGISTRATION.md: does option-implied volatility (VIX) help?

This is the one experiment that steps **outside weak-form efficiency**: VIX is
built from S&P 500 option prices, so it embeds forward-looking information no
function of past SPY prices and volume can contain. Two pre-registered questions
about forecasting the same 5-day-ahead realized volatility, both through the
Experiment-3 harness (expanding folds, `embargo = 4`, block bootstrap, the same
`_ols_fit_predict`):

- **Q1 — VIX as a standalone forecast.** Use VIX as of the feature close (in
  decimal annualized units, VIX/100) directly as the forecast. Does it beat the
  26-feature model? Pre-registered bar (same as Experiment 5): RMSE improvement
  over the model >= 10% **and** the block-bootstrap 90% CI of the mean per-day
  squared-error reduction excludes zero. Honest prior: VIX carries a variance risk
  premium (it sits above realized vol on average), so as a raw point forecast it
  is biased high and likely loses on RMSE — its value, if any, is as a feature a
  regression can de-bias.
- **Q2 — VIX as an added feature.** Add lagged VIX (as of the feature close) to
  the 26 features and refit. Does the augmented model beat the 26-feature model?
  Pre-registered criterion (same as Experiment 6): the 90% CI of the mean per-day
  *incremental* squared-error reduction excludes zero.

VIX as of the feature close is known before the target window, so both uses are
leakage-safe and share the models' information set. All choices are pre-registered.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
	from src.baseline import load_spy_ohlcv
	from src.experiment_vol_baselines import _compare
	from src.experiment_volatility import PERSISTENCE_FEATURE, RMSE_IMPROVEMENT_THRESHOLD, _ols_fit_predict
	from src.features import (
		FEATURE_COLUMNS,
		HAR_RV_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		TARGET_VOLATILITY_WINDOW,
		build_leakage_safe_features,
	)
	from src.vix_data import load_vix_close
	from src.walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		expanding_window_folds,
		run_walk_forward,
	)
except ModuleNotFoundError:  # Allows `python src/experiment_vix.py`.
	from baseline import load_spy_ohlcv
	from experiment_vol_baselines import _compare
	from experiment_volatility import PERSISTENCE_FEATURE, RMSE_IMPROVEMENT_THRESHOLD, _ols_fit_predict
	from features import (
		FEATURE_COLUMNS,
		HAR_RV_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		TARGET_VOLATILITY_WINDOW,
		build_leakage_safe_features,
	)
	from vix_data import load_vix_close
	from walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		expanding_window_folds,
		run_walk_forward,
	)


REPO_ROOT = Path(__file__).resolve().parent.parent
FORECAST_FILE = REPO_ROOT / "data" / "vix_forecast_comparisons.csv"
FEATURE_FILE = REPO_ROOT / "data" / "vix_feature_comparisons.csv"

VIX_FEATURE = "vix_lag"
VIX_FORECAST = "vix_forecast"
EMBARGO = TARGET_VOLATILITY_WINDOW - 1


def _prepare_with_vix(ohlcv=None, vix=None):
	if ohlcv is None:
		ohlcv = load_spy_ohlcv()
	if vix is None:
		vix = load_vix_close()
	dataset = build_leakage_safe_features(ohlcv, feature_columns=[*FEATURE_COLUMNS, *HAR_RV_COLUMNS])
	# VIX as of the feature close (prior close) — known before the target window.
	feature_dates = pd.DatetimeIndex(dataset["feature_timestamp"])
	dataset[VIX_FEATURE] = vix.reindex(feature_dates).to_numpy()
	dataset[VIX_FORECAST] = dataset[VIX_FEATURE]  # VIX/100 as a direct RV forecast
	dataset = dataset.loc[dataset[TARGET_VOLATILITY_COLUMN].notna() & dataset[VIX_FEATURE].notna()].copy()
	folds = expanding_window_folds(len(dataset))
	return dataset, folds


def run_vix_experiment(block_length=DEFAULT_BLOCK_LENGTH, n_resamples=DEFAULT_N_RESAMPLES, seed=DEFAULT_SEED, ohlcv=None, vix=None):
	dataset, folds = _prepare_with_vix(ohlcv=ohlcv, vix=vix)

	model = run_walk_forward(
		dataset, _ols_fit_predict, FEATURE_COLUMNS, TARGET_VOLATILITY_COLUMN,
		folds=folds, metadata_columns=[PERSISTENCE_FEATURE, VIX_FORECAST], embargo=EMBARGO,
	)
	model_plus_vix = run_walk_forward(
		dataset, _ols_fit_predict, [*FEATURE_COLUMNS, VIX_FEATURE], TARGET_VOLATILITY_COLUMN,
		folds=folds, embargo=EMBARGO,
	)
	har = run_walk_forward(
		dataset, _ols_fit_predict, HAR_RV_COLUMNS, TARGET_VOLATILITY_COLUMN,
		folds=folds, embargo=EMBARGO,
	)
	if not (
		np.array_equal(model["target_timestamp"].to_numpy(), model_plus_vix["target_timestamp"].to_numpy())
		and np.array_equal(model["target_timestamp"].to_numpy(), har["target_timestamp"].to_numpy())
	):
		raise ValueError("out-of-sample dates do not align across models.")

	y_true = model["y_true"].to_numpy(dtype=float)
	model_pred = model["prediction"].to_numpy(dtype=float)
	model_vix_pred = model_plus_vix["prediction"].to_numpy(dtype=float)
	har_pred = har["prediction"].to_numpy(dtype=float)
	persistence_pred = model[PERSISTENCE_FEATURE].to_numpy(dtype=float)
	vix_pred = model[VIX_FORECAST].to_numpy(dtype=float)

	indices = block_bootstrap_indices(len(y_true), block_length=block_length, n_resamples=n_resamples, seed=seed)

	# Q1 — VIX as a standalone forecast.
	forecast = pd.DataFrame(
		[
			_compare("vix_vs_model", y_true, vix_pred, model_pred, indices),
			_compare("vix_vs_persistence", y_true, vix_pred, persistence_pred, indices),
			_compare("vix_vs_har", y_true, vix_pred, har_pred, indices),
			_compare("model_vs_persistence", y_true, model_pred, persistence_pred, indices),
		]
	)
	gate_q1 = forecast.set_index("comparison").loc["vix_vs_model"]
	vix_beats_model = bool(
		gate_q1["rmse_improvement"] >= RMSE_IMPROVEMENT_THRESHOLD and gate_q1["reduction_excludes_zero_low"]
	)

	# Q2 — VIX as an added feature.
	feature = pd.DataFrame(
		[
			_compare("model_plus_vix_vs_model", y_true, model_vix_pred, model_pred, indices),
			_compare("model_plus_vix_vs_persistence", y_true, model_vix_pred, persistence_pred, indices),
		]
	)
	gate_q2 = feature.set_index("comparison").loc["model_plus_vix_vs_model"]
	vix_adds_signal = bool(gate_q2["reduction_excludes_zero_low"])

	for frame in (forecast, feature):
		frame["n_oos_days"] = int(len(y_true))
		frame["block_length"] = int(block_length)
	forecast["is_gating"] = forecast["comparison"] == "vix_vs_model"
	feature["is_gating"] = feature["comparison"] == "model_plus_vix_vs_model"

	return {
		"forecast": forecast,
		"feature": feature,
		"vix_beats_model": vix_beats_model,
		"vix_adds_signal": vix_adds_signal,
	}


def _print_comparison(row):
	print(
		f"    RMSE {row['model_rmse']:.6f} vs baseline {row['baseline_rmse']:.6f} "
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
		results = run_vix_experiment()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	FORECAST_FILE.parent.mkdir(parents=True, exist_ok=True)
	results["forecast"].to_csv(FORECAST_FILE, index=False)
	results["feature"].to_csv(FEATURE_FILE, index=False)

	forecast = results["forecast"].set_index("comparison")
	feature = results["feature"].set_index("comparison")
	print("Experiment 8 - VIX (option-implied vol) for 5-day realized-vol forecasting:")
	print(f"  Outputs: {FORECAST_FILE.parent}")
	print("  NOTE: VIX embeds option-market information -> this steps outside weak-form efficiency.")
	print("")
	print("  Q1 - VIX as a standalone forecast:")
	print("    VIX vs model (GATING, needs >=10% RMSE gain AND CI excluding zero):")
	_print_comparison(forecast.loc["vix_vs_model"])
	print("    VIX vs persistence (context):")
	_print_comparison(forecast.loc["vix_vs_persistence"])
	print(
		"    PRE-REGISTERED VERDICT: "
		+ (
			"VIX beats the model as a forecast."
			if results["vix_beats_model"]
			else "VIX does NOT beat the model as a raw forecast (null; variance risk premium biases it high)."
		)
	)
	print("")
	print("  Q2 - VIX as an added feature (26 features + lagged VIX):")
	print("    model+VIX vs model (GATING, needs incremental CI excluding zero):")
	_print_comparison(feature.loc["model_plus_vix_vs_model"])
	print(
		"    PRE-REGISTERED VERDICT: "
		+ (
			"lagged VIX adds forecasting signal beyond the 26 features."
			if results["vix_adds_signal"]
			else "lagged VIX adds NOTHING beyond the 26 features (null)."
		)
	)


if __name__ == "__main__":
	main()

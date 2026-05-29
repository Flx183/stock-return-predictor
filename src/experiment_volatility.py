"""
Experiment 3 from PREREGISTRATION.md.

Forecast `target_volatility_5d` (forward 5-day annualized realized volatility)
with an OLS model on the leakage-safe features, and ask whether it beats naive
persistence (forecast = trailing `realized_vol_5d`). The pre-registered bar has
two parts that must both hold:

1. pooled out-of-sample RMSE improvement over persistence >= 10%, and
2. the block-bootstrap 90% CI of the mean per-day squared-error reduction
   excludes zero on the low side.

Persistence is a strong baseline for realized volatility, so the honest
expectation is a null. QLIKE, per-fold RMSE, and out-of-sample R^2 are reported
as context only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
	from src.baseline import load_spy_ohlcv
	from src.features import (
		FEATURE_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		TARGET_VOLATILITY_WINDOW,
		build_leakage_safe_features,
	)
	from src.walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		bootstrap_mean,
		describe_folds,
		expanding_window_folds,
		run_walk_forward,
		summarize_bootstrap,
	)
except ModuleNotFoundError:  # Allows `python src/experiment_volatility.py`.
	from baseline import load_spy_ohlcv
	from features import (
		FEATURE_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		TARGET_VOLATILITY_WINDOW,
		build_leakage_safe_features,
	)
	from walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		bootstrap_mean,
		describe_folds,
		expanding_window_folds,
		run_walk_forward,
		summarize_bootstrap,
	)


REPO_ROOT = Path(__file__).resolve().parent.parent
VOL_FOLDS_FILE = REPO_ROOT / "data" / "walkforward_volatility_folds.csv"
VOL_OOS_FILE = REPO_ROOT / "data" / "walkforward_volatility_oos.csv"
VOL_SUMMARY_FILE = REPO_ROOT / "data" / "walkforward_volatility_summary.csv"

PERSISTENCE_FEATURE = "realized_vol_5d"
RMSE_IMPROVEMENT_THRESHOLD = 0.10
QLIKE_FLOOR = 1e-6


def _ols_fit_predict(X_train, y_train, X_test):
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	model = LinearRegression()
	model.fit(X_train_scaled, y_train.astype(float))
	return model.predict(X_test_scaled)


def _rmse(y_true, y_pred):
	error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
	return float(np.sqrt(np.mean(error**2)))


def _qlike(y_true, y_pred, floor=QLIKE_FLOOR):
	"""Mean QLIKE loss on the variance proxy; lower is better."""
	variance_true = np.asarray(y_true, dtype=float) ** 2
	variance_pred = np.clip(np.asarray(y_pred, dtype=float), floor, None) ** 2
	ratio = variance_true / variance_pred
	return float(np.mean(ratio - np.log(ratio) - 1.0))


def _out_of_sample_r2(y_true, model_pred, persistence_pred):
	"""Campbell-Thompson OOS R^2: model SSE relative to persistence SSE."""
	y_true = np.asarray(y_true, dtype=float)
	model_sse = float(np.sum((y_true - np.asarray(model_pred, dtype=float)) ** 2))
	persistence_sse = float(np.sum((y_true - np.asarray(persistence_pred, dtype=float)) ** 2))
	if persistence_sse <= 0:
		return float("nan")
	return 1.0 - model_sse / persistence_sse


def run_volatility_experiment(
	block_length=DEFAULT_BLOCK_LENGTH,
	n_resamples=DEFAULT_N_RESAMPLES,
	seed=DEFAULT_SEED,
):
	ohlcv = load_spy_ohlcv()
	dataset = build_leakage_safe_features(ohlcv)
	# Drop rows whose forward volatility window is incomplete (trailing NaN target).
	dataset = dataset.loc[dataset[TARGET_VOLATILITY_COLUMN].notna()].copy()
	folds = expanding_window_folds(len(dataset))
	# The target looks forward TARGET_VOLATILITY_WINDOW days, so embargo the last
	# (window - 1) training rows whose label window would cross into the test block.
	embargo = TARGET_VOLATILITY_WINDOW - 1

	pooled = run_walk_forward(
		dataset,
		_ols_fit_predict,
		FEATURE_COLUMNS,
		TARGET_VOLATILITY_COLUMN,
		folds=folds,
		metadata_columns=[PERSISTENCE_FEATURE],
		embargo=embargo,
	)

	oos = pd.DataFrame(
		{
			"target_timestamp": pooled["target_timestamp"].to_numpy(),
			"fold_id": pooled["fold_id"].to_numpy(),
			"y_true": pooled["y_true"].to_numpy(),
			"model_pred": pooled["prediction"].to_numpy(),
			"persistence_pred": pooled[PERSISTENCE_FEATURE].to_numpy(),
		}
	)
	oos["model_sq_err"] = (oos["y_true"] - oos["model_pred"]) ** 2
	oos["persistence_sq_err"] = (oos["y_true"] - oos["persistence_pred"]) ** 2

	model_rmse = _rmse(oos["y_true"], oos["model_pred"])
	persistence_rmse = _rmse(oos["y_true"], oos["persistence_pred"])
	rmse_improvement = (persistence_rmse - model_rmse) / persistence_rmse

	reduction = (oos["persistence_sq_err"] - oos["model_sq_err"]).to_numpy(dtype=float)
	indices = block_bootstrap_indices(
		len(reduction), block_length=block_length, n_resamples=n_resamples, seed=seed
	)
	bootstrap = summarize_bootstrap(
		bootstrap_mean(reduction, indices), point_estimate=float(np.mean(reduction))
	)

	meets_rmse_threshold = bool(rmse_improvement >= RMSE_IMPROVEMENT_THRESHOLD)
	beats_persistence = bool(meets_rmse_threshold and bootstrap["excludes_zero_low"])

	summary = {
		"n_oos_days": int(len(oos)),
		"oos_start_target": oos["target_timestamp"].iloc[0].date().isoformat(),
		"oos_end_target": oos["target_timestamp"].iloc[-1].date().isoformat(),
		"model_rmse": model_rmse,
		"persistence_rmse": persistence_rmse,
		"rmse_improvement": float(rmse_improvement),
		"rmse_improvement_threshold": float(RMSE_IMPROVEMENT_THRESHOLD),
		"meets_rmse_threshold": meets_rmse_threshold,
		"model_qlike": _qlike(oos["y_true"], oos["model_pred"]),
		"persistence_qlike": _qlike(oos["y_true"], oos["persistence_pred"]),
		"out_of_sample_r2_vs_persistence": _out_of_sample_r2(
			oos["y_true"], oos["model_pred"], oos["persistence_pred"]
		),
		"mean_sq_err_reduction": bootstrap["point_estimate"],
		"sq_err_reduction_ci_low": bootstrap["ci_low"],
		"sq_err_reduction_median": bootstrap["median"],
		"sq_err_reduction_ci_high": bootstrap["ci_high"],
		"sq_err_reduction_prob_positive": bootstrap["prob_positive"],
		"sq_err_reduction_excludes_zero_low": bootstrap["excludes_zero_low"],
		"beats_persistence": beats_persistence,
		"embargo": int(embargo),
		"block_length": int(block_length),
		"n_resamples": int(n_resamples),
		"seed": int(seed),
	}

	folds_table = describe_folds(dataset, folds, TARGET_VOLATILITY_COLUMN, embargo=embargo)
	model_rmse_by_fold = oos.groupby("fold_id").apply(
		lambda group: _rmse(group["y_true"], group["model_pred"]), include_groups=False
	)
	persistence_rmse_by_fold = oos.groupby("fold_id").apply(
		lambda group: _rmse(group["y_true"], group["persistence_pred"]), include_groups=False
	)
	folds_table["model_rmse"] = folds_table["fold_id"].map(model_rmse_by_fold)
	folds_table["persistence_rmse"] = folds_table["fold_id"].map(persistence_rmse_by_fold)
	folds_table["rmse_improvement"] = (
		folds_table["persistence_rmse"] - folds_table["model_rmse"]
	) / folds_table["persistence_rmse"]

	return {"folds": folds_table, "oos": oos, "summary": pd.DataFrame([summary])}


def main():
	try:
		results = run_volatility_experiment()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	VOL_FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
	results["folds"].to_csv(VOL_FOLDS_FILE, index=False)
	results["oos"].to_csv(VOL_OOS_FILE, index=False)
	results["summary"].to_csv(VOL_SUMMARY_FILE, index=False)

	row = results["summary"].iloc[0]
	print("Walk-forward volatility experiment (expanding window, 8 folds):")
	print(f"  Outputs: {VOL_SUMMARY_FILE.parent}")
	print(
		"  OOS window: "
		f"{row['oos_start_target']} to {row['oos_end_target']} ({int(row['n_oos_days'])} days)"
	)
	print(f"  Model RMSE:       {row['model_rmse']:.6f}")
	print(f"  Persistence RMSE: {row['persistence_rmse']:.6f}")
	print(
		f"  RMSE improvement: {row['rmse_improvement']:.4%} "
		f"(threshold {row['rmse_improvement_threshold']:.0%}, "
		f"met: {bool(row['meets_rmse_threshold'])})"
	)
	print(
		"  Mean per-day squared-error reduction: "
		f"{row['mean_sq_err_reduction']:.6e}, "
		f"90% CI [{row['sq_err_reduction_ci_low']:.6e}, {row['sq_err_reduction_ci_high']:.6e}]"
	)
	print(
		f"  QLIKE model {row['model_qlike']:.4f} vs persistence {row['persistence_qlike']:.4f}; "
		f"OOS R^2 vs persistence {row['out_of_sample_r2_vs_persistence']:.4f}"
	)
	print(
		"  PRE-REGISTERED VERDICT: "
		+ (
			"model beats persistence."
			if bool(row["beats_persistence"])
			else "model does NOT beat persistence (null)."
		)
	)


if __name__ == "__main__":
	main()

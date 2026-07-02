"""
Experiment 4 from PREREGISTRATION.md: does return predictability appear at
longer horizons?

The 1-day study (Experiments 1-2) found next-day direction unpredictable on SPY.
That is a statement about the 1-day horizon and says nothing on its own about
weekly, monthly, or yearly returns -- the empirical-finance literature actually
finds *more* structure at longer horizons (momentum, mean reversion, valuation
predictability), though mostly from fundamentals rather than price history. This
experiment extends the exact same leakage-controlled walk-forward harness across
a horizon ladder and reports, per horizon, whether OHLCV-only features:

1. beat a constant-drift benchmark as a return *regression* (Campbell-Thompson
   out-of-sample R^2, with a block-bootstrap CI on the mean per-period
   squared-error reduction), and
2. beat SPY buy-and-hold as a *direction* strategy over non-overlapping holding
   periods (block-bootstrap CI on the mean per-window excess return).

Two honesty controls are built in and matter more as the horizon grows:

- Overlapping targets. A k-day forward return sampled daily overlaps its
  neighbours by k-1 days, so the bootstrap block length is set to >= 2*k to keep
  the serial correlation, and the walk-forward embargo drops the k-1 training
  rows whose window would cross into the test block.
- Sample-size collapse. The number of *non-overlapping* windows (n_oos / k)
  falls from a few hundred at a week to a handful at a year. Horizons are tiered
  in advance: 5 and 21 rigorous, 63 suggestive, 252 illustrative-only -- a
  "positive" at 252 is reported, never counted as evidence.

All model choices and thresholds are pre-registered; nothing here is tuned on the
result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

try:
	from src.backtest import DEFAULT_COST_PER_TRADE
	from src.baseline import TRADING_DAYS_PER_YEAR, load_spy_ohlcv
	from src.features import LONG_HORIZON_FEATURE_COLUMNS, build_leakage_safe_features
	from src.walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_MIN_TRAIN_SIZE,
		DEFAULT_N_FOLDS,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		DEFAULT_TEST_SIZE,
		block_bootstrap_indices,
		bootstrap_mean,
		expanding_window_folds,
		run_walk_forward,
		summarize_bootstrap,
	)
except ModuleNotFoundError:  # Allows `python src/experiment_horizons.py`.
	from backtest import DEFAULT_COST_PER_TRADE
	from baseline import TRADING_DAYS_PER_YEAR, load_spy_ohlcv
	from features import LONG_HORIZON_FEATURE_COLUMNS, build_leakage_safe_features
	from walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_MIN_TRAIN_SIZE,
		DEFAULT_N_FOLDS,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		DEFAULT_TEST_SIZE,
		block_bootstrap_indices,
		bootstrap_mean,
		expanding_window_folds,
		run_walk_forward,
		summarize_bootstrap,
	)


REPO_ROOT = Path(__file__).resolve().parent.parent
REGRESSION_FILE = REPO_ROOT / "data" / "horizon_regression.csv"
CLASSIFICATION_FILE = REPO_ROOT / "data" / "horizon_classification.csv"
LADDER_FILE = REPO_ROOT / "data" / "horizon_ladder.csv"

# Trading-day horizons: week, month, quarter (bridge), year.
HORIZONS = (5, 21, 63, 252)
RIGOR_TIER = {5: "rigorous", 21: "rigorous", 63: "suggestive", 252: "illustrative"}

DECISION_THRESHOLD = 0.5
LR_PARAMS = {"C": 1.0, "max_iter": 1000}
# Fixed before any result was seen; shared shape with Experiment 2.
_XGB_COMMON = {
	"n_estimators": 300,
	"max_depth": 3,
	"learning_rate": 0.05,
	"subsample": 0.8,
	"colsample_bytree": 0.8,
	"min_child_weight": 5,
	"reg_lambda": 1.0,
	"tree_method": "hist",
	"random_state": DEFAULT_SEED,
	"n_jobs": 1,
}
XGB_CLASSIFIER_PARAMS = {**_XGB_COMMON, "objective": "binary:logistic", "eval_metric": "logloss"}
XGB_REGRESSOR_PARAMS = {**_XGB_COMMON, "objective": "reg:squarederror", "eval_metric": "rmse"}


# --- fit_predict callbacks -------------------------------------------------

def _scaled_linear_fit_predict(X_train, y_train, X_test):
	scaler = StandardScaler()
	model = LinearRegression()
	model.fit(scaler.fit_transform(X_train), y_train.astype(float))
	return model.predict(scaler.transform(X_test))


def _xgb_regressor_fit_predict(X_train, y_train, X_test):
	model = XGBRegressor(**XGB_REGRESSOR_PARAMS)
	model.fit(X_train, y_train.astype(float))
	return model.predict(X_test)


def _drift_fit_predict(X_train, y_train, X_test):
	"""Constant prevailing-mean forecast: the honest 'no signal' benchmark."""
	return np.full(len(X_test), float(np.asarray(y_train, dtype=float).mean()))


def _logistic_fit_predict(X_train, y_train, X_test):
	scaler = StandardScaler()
	classifier = LogisticRegression(**LR_PARAMS)
	classifier.fit(scaler.fit_transform(X_train), y_train.astype(int))
	return classifier.predict_proba(scaler.transform(X_test))[:, 1]


def _xgb_classifier_fit_predict(X_train, y_train, X_test):
	classifier = XGBClassifier(**XGB_CLASSIFIER_PARAMS)
	classifier.fit(X_train, y_train.astype(int))
	return classifier.predict_proba(X_test)[:, 1]


def _base_rate_fit_predict(X_train, y_train, X_test):
	"""Predict the training up-rate for every test row (always-majority class)."""
	return np.full(len(X_test), float(np.asarray(y_train, dtype=float).mean()))


# --- per-horizon walk-forward configuration --------------------------------

def _horizon_folds(n_rows, horizon):
	"""
	Fold, block-length, and resample settings scaled to the horizon.

	The out-of-sample test block must be at least one full horizon wide, and the
	bootstrap block must be at least two horizons wide so overlapping-target
	autocorrelation survives resampling. Both grow with the horizon, which forces
	few folds (and a wide CI) at the longest horizons -- by design.
	"""
	test_size = max(DEFAULT_TEST_SIZE, 2 * horizon)
	block_length = max(DEFAULT_BLOCK_LENGTH, 2 * horizon)
	max_folds = (n_rows - DEFAULT_MIN_TRAIN_SIZE) // test_size
	n_folds = int(min(DEFAULT_N_FOLDS, max_folds))
	if n_folds < 1:
		raise ValueError(
			f"horizon {horizon}: {n_rows} rows cannot supply a test block of "
			f"{test_size} beyond a {DEFAULT_MIN_TRAIN_SIZE}-row minimum training set."
		)
	folds = expanding_window_folds(
		n_rows, n_folds=n_folds, test_size=test_size, min_train_size=DEFAULT_MIN_TRAIN_SIZE
	)
	return folds, test_size, block_length


# --- evaluation ------------------------------------------------------------

def _rmse(y_true, y_pred):
	error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
	return float(np.sqrt(np.mean(error**2)))


def _out_of_sample_r2(y_true, model_pred, drift_pred):
	"""Campbell-Thompson OOS R^2: model SSE relative to the drift benchmark SSE."""
	y_true = np.asarray(y_true, dtype=float)
	model_sse = float(np.sum((y_true - np.asarray(model_pred, dtype=float)) ** 2))
	drift_sse = float(np.sum((y_true - np.asarray(drift_pred, dtype=float)) ** 2))
	if drift_sse <= 0:
		return float("nan")
	return 1.0 - model_sse / drift_sse


def _regression_metrics(name, horizon, pooled_model, pooled_drift, block_length, n_resamples, seed):
	if not np.array_equal(
		pooled_model["target_timestamp"].to_numpy(), pooled_drift["target_timestamp"].to_numpy()
	):
		raise ValueError("model and drift out-of-sample dates do not align.")

	y_true = pooled_model["y_true"].to_numpy(dtype=float)
	model_pred = pooled_model["prediction"].to_numpy(dtype=float)
	drift_pred = pooled_drift["prediction"].to_numpy(dtype=float)

	model_sq_err = (y_true - model_pred) ** 2
	drift_sq_err = (y_true - drift_pred) ** 2
	reduction = drift_sq_err - model_sq_err

	indices = block_bootstrap_indices(
		len(reduction), block_length=block_length, n_resamples=n_resamples, seed=seed
	)
	bootstrap = summarize_bootstrap(
		bootstrap_mean(reduction, indices), point_estimate=float(np.mean(reduction))
	)

	fold_ids = pooled_model["fold_id"].to_numpy()
	fold_reduction = pd.Series(reduction).groupby(fold_ids).mean()
	folds_positive_fraction = float((fold_reduction > 0.0).mean())

	oos_r2 = _out_of_sample_r2(y_true, model_pred, drift_pred)
	beats_drift = bool(
		oos_r2 > 0.0 and bootstrap["excludes_zero_low"] and folds_positive_fraction > 0.5
	)
	return {
		"horizon": horizon,
		"rigor_tier": RIGOR_TIER[horizon],
		"model": name,
		"n_oos_days": int(len(y_true)),
		"model_rmse": _rmse(y_true, model_pred),
		"drift_rmse": _rmse(y_true, drift_pred),
		"oos_r2_vs_drift": float(oos_r2),
		"mean_sq_err_reduction": bootstrap["point_estimate"],
		"reduction_ci_low": bootstrap["ci_low"],
		"reduction_ci_high": bootstrap["ci_high"],
		"reduction_prob_positive": bootstrap["prob_positive"],
		"reduction_excludes_zero_low": bootstrap["excludes_zero_low"],
		"folds_positive_fraction": folds_positive_fraction,
		"block_length": int(block_length),
		"beats_drift": beats_drift,
	}


def _nonoverlapping_economic(name, horizon, pooled_pred, dataset, cost_per_trade, n_resamples, seed):
	"""
	Direction strategy vs buy-and-hold over non-overlapping holding periods.

	The pooled out-of-sample rows are contiguous trading days, so taking every
	`horizon`-th row yields non-overlapping windows (each spans exactly one
	holding period). Each window earns the realised `horizon`-day return if the
	model is long, minus a cost on every change in exposure; buy-and-hold simply
	earns the window return. Non-overlapping windows are ~independent, so a plain
	(unit-block) bootstrap on the per-window excess is appropriate.
	"""
	ordered = pooled_pred.sort_values("target_timestamp").reset_index(drop=True)
	windows = ordered.iloc[::horizon].copy()

	realised_return = dataset["target_return"].reindex(windows["target_timestamp"]).to_numpy(dtype=float)
	position = (windows["prediction"].to_numpy() >= DECISION_THRESHOLD).astype(float)
	previous_position = np.concatenate([[0.0], position[:-1]])
	cost = np.abs(position - previous_position) * cost_per_trade

	strategy_return = position * realised_return - cost
	buy_and_hold_return = realised_return
	excess = strategy_return - buy_and_hold_return

	strategy_total = float(np.prod(1.0 + strategy_return) - 1.0)
	buy_and_hold_total = float(np.prod(1.0 + buy_and_hold_return) - 1.0)

	block_length = 1 if len(excess) < 2 else min(DEFAULT_BLOCK_LENGTH, len(excess))
	indices = block_bootstrap_indices(
		len(excess), block_length=block_length, n_resamples=n_resamples, seed=seed
	)
	bootstrap = summarize_bootstrap(
		bootstrap_mean(excess, indices), point_estimate=float(np.mean(excess))
	)
	return {
		"n_windows": int(len(windows)),
		"long_window_fraction": float(np.mean(position)),
		"strategy_total_return": strategy_total,
		"buy_and_hold_total_return": buy_and_hold_total,
		"excess_total_return": strategy_total - buy_and_hold_total,
		"mean_window_excess": bootstrap["point_estimate"],
		"excess_ci_low": bootstrap["ci_low"],
		"excess_ci_high": bootstrap["ci_high"],
		"excess_prob_positive": bootstrap["prob_positive"],
		"beats_buy_and_hold": bool(bootstrap["excludes_zero_low"]),
	}


def _classification_metrics(name, horizon, pooled_pred, pooled_base_rate, dataset, cost_per_trade, n_resamples, seed):
	predicted = (pooled_pred["prediction"].to_numpy() >= DECISION_THRESHOLD).astype(int)
	base_predicted = (pooled_base_rate["prediction"].to_numpy() >= DECISION_THRESHOLD).astype(int)
	actual = pooled_pred["y_true"].astype(int).to_numpy()

	economic = _nonoverlapping_economic(
		name, horizon, pooled_pred, dataset, cost_per_trade, n_resamples, seed
	)
	row = {
		"horizon": horizon,
		"rigor_tier": RIGOR_TIER[horizon],
		"model": name,
		"n_oos_days": int(len(actual)),
		"direction_base_rate": float(np.mean(actual)),
		"pooled_accuracy": float(accuracy_score(actual, predicted)),
		"base_rate_accuracy": float(accuracy_score(actual, base_predicted)),
		"predicted_up_rate": float(np.mean(predicted)),
	}
	row.update(economic)
	return row


# --- orchestration ---------------------------------------------------------

def _run_one_horizon(ohlcv, horizon, cost_per_trade, n_resamples, seed):
	dataset = build_leakage_safe_features(
		ohlcv, forward_horizon=horizon, feature_columns=LONG_HORIZON_FEATURE_COLUMNS
	)
	dataset = dataset.loc[dataset["target_return"].notna()].copy()
	folds, _, block_length = _horizon_folds(len(dataset), horizon)
	embargo = horizon - 1
	features = LONG_HORIZON_FEATURE_COLUMNS

	def wf(fit_predict, target_column):
		return run_walk_forward(
			dataset, fit_predict, features, target_column, folds=folds, embargo=embargo
		)

	# Regression framing.
	drift = wf(_drift_fit_predict, "target_return")
	linear = wf(_scaled_linear_fit_predict, "target_return")
	xgb_reg = wf(_xgb_regressor_fit_predict, "target_return")
	regression_rows = [
		_regression_metrics("linear", horizon, linear, drift, block_length, n_resamples, seed),
		_regression_metrics("xgboost", horizon, xgb_reg, drift, block_length, n_resamples, seed),
	]

	# Direction framing.
	base_rate = wf(_base_rate_fit_predict, "target_direction")
	logistic = wf(_logistic_fit_predict, "target_direction")
	xgb_clf = wf(_xgb_classifier_fit_predict, "target_direction")
	classification_rows = [
		_classification_metrics("logistic", horizon, logistic, base_rate, dataset, cost_per_trade, n_resamples, seed),
		_classification_metrics("xgboost", horizon, xgb_clf, base_rate, dataset, cost_per_trade, n_resamples, seed),
	]
	return regression_rows, classification_rows


def run_horizon_experiment(
	horizons=HORIZONS,
	cost_per_trade=DEFAULT_COST_PER_TRADE,
	n_resamples=DEFAULT_N_RESAMPLES,
	seed=DEFAULT_SEED,
):
	ohlcv = load_spy_ohlcv()
	regression_rows = []
	classification_rows = []
	for horizon in horizons:
		reg, clf = _run_one_horizon(ohlcv, horizon, cost_per_trade, n_resamples, seed)
		regression_rows.extend(reg)
		classification_rows.extend(clf)

	regression = pd.DataFrame(regression_rows)
	classification = pd.DataFrame(classification_rows)
	ladder = _build_ladder(regression, classification)
	return {"regression": regression, "classification": classification, "ladder": ladder}


def _build_ladder(regression, classification):
	linear = regression[regression["model"] == "linear"].set_index("horizon")
	logistic = classification[classification["model"] == "logistic"].set_index("horizon")
	rows = []
	for horizon in sorted(linear.index):
		reg = linear.loc[horizon]
		clf = logistic.loc[horizon]
		illustrative = RIGOR_TIER[horizon] == "illustrative"
		beats = bool(reg["beats_drift"] or clf["beats_buy_and_hold"])
		if illustrative:
			verdict = "illustrative-only (insufficient independent windows)"
		elif beats:
			verdict = "PREDICTABILITY DETECTED - stress-test before any claim"
		else:
			verdict = "null (no OHLCV predictability beyond drift / buy-and-hold)"
		rows.append(
			{
				"horizon": int(horizon),
				"rigor_tier": RIGOR_TIER[horizon],
				"n_oos_days": int(reg["n_oos_days"]),
				"n_nonoverlap_windows": int(clf["n_windows"]),
				"direction_base_rate": float(clf["direction_base_rate"]),
				"linear_oos_r2_vs_drift": float(reg["oos_r2_vs_drift"]),
				"linear_beats_drift": bool(reg["beats_drift"]),
				"logistic_accuracy": float(clf["pooled_accuracy"]),
				"base_rate_accuracy": float(clf["base_rate_accuracy"]),
				"logistic_excess_total_vs_bh": float(clf["excess_total_return"]),
				"logistic_beats_buy_and_hold": bool(clf["beats_buy_and_hold"]),
				"verdict": verdict,
			}
		)
	return pd.DataFrame(rows)


def main():
	try:
		results = run_horizon_experiment()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	REGRESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
	results["regression"].to_csv(REGRESSION_FILE, index=False)
	results["classification"].to_csv(CLASSIFICATION_FILE, index=False)
	results["ladder"].to_csv(LADDER_FILE, index=False)

	print("Walk-forward multi-horizon experiment (expanding window):")
	print(f"  Outputs: {LADDER_FILE.parent}")
	print("")
	header = (
		f"  {'horizon':>7} {'tier':>12} {'eff_N':>6} {'up_rate':>8} "
		f"{'lin_R2':>9} {'acc':>6} {'base_acc':>8} {'excess%':>9}  verdict"
	)
	print(header)
	for _, row in results["ladder"].iterrows():
		print(
			f"  {int(row['horizon']):>7} {row['rigor_tier']:>12} "
			f"{int(row['n_nonoverlap_windows']):>6} {row['direction_base_rate']:>8.3f} "
			f"{row['linear_oos_r2_vs_drift']:>9.4f} {row['logistic_accuracy']:>6.3f} "
			f"{row['base_rate_accuracy']:>8.3f} {row['logistic_excess_total_vs_bh']*100:>9.2f}  "
			f"{row['verdict']}"
		)
	print("")
	print("  Reading it: lin_R2 is Campbell-Thompson OOS R^2 vs a drift forecast")
	print("  (>0 and a CI excluding zero would beat drift); acc vs base_acc shows")
	print("  the always-up base rate rising with horizon (the base-rate illusion);")
	print("  excess% is the non-overlapping direction strategy minus buy-and-hold.")


if __name__ == "__main__":
	main()

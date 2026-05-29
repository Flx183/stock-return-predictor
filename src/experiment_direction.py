"""
Experiments 1 and 2 from PREREGISTRATION.md.

1. Logistic-regression direction strategy, evaluated as a distribution over
   expanding-window folds and a block bootstrap, replacing the single 80/20
   verdict. Headline: does it beat SPY buy-and-hold?
2. XGBoost vs logistic regression, judged purely economically: does the
   gradient-boosted strategy earn more, net of costs, than the linear one?

Both models run through the same `src/walkforward.py` harness and the same
costed backtester as `src/model_backtest.py`. The only pass/fail tests are the
pre-registered block-bootstrap CIs; everything else is reported as context.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
	from src.backtest import DEFAULT_COST_PER_TRADE
	from src.baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices, load_spy_ohlcv
	from src.features import FEATURE_COLUMNS, build_leakage_safe_features
	from src.model_backtest import backtest_model_predictions
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
except ModuleNotFoundError:  # Allows `python src/experiment_direction.py`.
	from backtest import DEFAULT_COST_PER_TRADE
	from baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices, load_spy_ohlcv
	from features import FEATURE_COLUMNS, build_leakage_safe_features
	from model_backtest import backtest_model_predictions
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
FOLDS_FILE = REPO_ROOT / "data" / "walkforward_direction_folds.csv"
OOS_FILE = REPO_ROOT / "data" / "walkforward_direction_oos.csv"
MODEL_METRICS_FILE = REPO_ROOT / "data" / "walkforward_direction_model_metrics.csv"
BOOTSTRAP_FILE = REPO_ROOT / "data" / "walkforward_direction_bootstrap.csv"

DECISION_THRESHOLD = 0.5
LR_PARAMS = {"C": 1.0, "max_iter": 1000}
# Fixed before any result was seen; see PREREGISTRATION.md Experiment 2.
XGB_PARAMS = {
	"n_estimators": 300,
	"max_depth": 3,
	"learning_rate": 0.05,
	"subsample": 0.8,
	"colsample_bytree": 0.8,
	"min_child_weight": 5,
	"reg_lambda": 1.0,
	"objective": "binary:logistic",
	"eval_metric": "logloss",
	"tree_method": "hist",
	"random_state": DEFAULT_SEED,
	"n_jobs": 1,
}


def _logistic_fit_predict(X_train, y_train, X_test):
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	classifier = LogisticRegression(**LR_PARAMS)
	classifier.fit(X_train_scaled, y_train.astype(int))
	return classifier.predict_proba(X_test_scaled)[:, 1]


def _xgboost_fit_predict(X_train, y_train, X_test):
	# Trees are scale-invariant, so the features are passed unscaled.
	classifier = XGBClassifier(**XGB_PARAMS)
	classifier.fit(X_train, y_train.astype(int))
	return classifier.predict_proba(X_test)[:, 1]


def _predictions_frame(pooled):
	predictions = pooled.loc[:, ["feature_timestamp", "target_timestamp"]].copy()
	predictions["probability_up"] = pooled["prediction"].to_numpy()
	predictions["predicted_direction"] = (
		pooled["prediction"].to_numpy() >= DECISION_THRESHOLD
	).astype(int)
	predictions["actual_direction"] = pooled["y_true"].astype(int).to_numpy()
	return predictions


def _oos_returns(prices, pooled, cost_per_trade):
	"""Costed backtest of one model's pooled predictions; return the OOS rows."""
	predictions = _predictions_frame(pooled)
	backtest = backtest_model_predictions(prices, predictions, cost_per_trade=cost_per_trade)
	oos = backtest.loc[backtest["is_test_target_date"]].copy()
	return oos


def _pooled_model_metrics(pooled, name):
	probability_up = pooled["prediction"].to_numpy()
	predicted = (probability_up >= DECISION_THRESHOLD).astype(int)
	actual = pooled["y_true"].astype(int).to_numpy()
	roc_auc = float("nan")
	if len(np.unique(actual)) == 2:
		roc_auc = float(roc_auc_score(actual, probability_up))
	return {
		"model": name,
		"pooled_accuracy": float(accuracy_score(actual, predicted)),
		"pooled_roc_auc": roc_auc,
		"pooled_log_loss": float(log_loss(actual, probability_up, labels=[0, 1])),
		"predicted_positive_rate": float(np.mean(predicted)),
		"actual_positive_rate": float(np.mean(actual)),
		"n_oos_days": int(len(pooled)),
		"oos_start_target": pooled["target_timestamp"].iloc[0].date().isoformat(),
		"oos_end_target": pooled["target_timestamp"].iloc[-1].date().isoformat(),
	}


def _annualized_sharpe(returns):
	std = returns.std()
	if std <= 0:
		return float("nan")
	return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _compounded(returns):
	return float(np.prod(1.0 + returns.to_numpy()) - 1.0)


def _compare(name, series_a, series_b, indices):
	"""Block-bootstrap the mean daily excess (a - b) and summarize it."""
	excess = (series_a - series_b).to_numpy(dtype=float)
	samples = bootstrap_mean(excess, indices)
	summary = summarize_bootstrap(samples, point_estimate=float(np.mean(excess)))
	return {
		"comparison": name,
		"mean_daily_excess": summary["point_estimate"],
		"ci_low": summary["ci_low"],
		"median": summary["median"],
		"ci_high": summary["ci_high"],
		"prob_positive": summary["prob_positive"],
		"excludes_zero_low": summary["excludes_zero_low"],
		"total_compounded_excess": _compounded(series_a) - _compounded(series_b),
		"sharpe_difference": _annualized_sharpe(series_a) - _annualized_sharpe(series_b),
	}


def run_direction_experiment(
	cost_per_trade=DEFAULT_COST_PER_TRADE,
	block_length=DEFAULT_BLOCK_LENGTH,
	n_resamples=DEFAULT_N_RESAMPLES,
	seed=DEFAULT_SEED,
):
	ohlcv = load_spy_ohlcv()
	dataset = build_leakage_safe_features(ohlcv)
	prices = load_spy_close_prices()
	folds = expanding_window_folds(len(dataset))

	lr_pooled = run_walk_forward(dataset, _logistic_fit_predict, FEATURE_COLUMNS, "target_direction", folds=folds)
	xgb_pooled = run_walk_forward(dataset, _xgboost_fit_predict, FEATURE_COLUMNS, "target_direction", folds=folds)

	lr_oos = _oos_returns(prices, lr_pooled, cost_per_trade)
	xgb_oos = _oos_returns(prices, xgb_pooled, cost_per_trade)
	if not lr_oos.index.equals(xgb_oos.index):
		raise ValueError("LR and XGB out-of-sample dates do not align.")

	lr_net = lr_oos["net_strategy_return"]
	xgb_net = xgb_oos["net_strategy_return"]
	buy_and_hold = lr_oos["buy_and_hold_return"]

	indices = block_bootstrap_indices(
		len(buy_and_hold), block_length=block_length, n_resamples=n_resamples, seed=seed
	)
	comparisons = pd.DataFrame(
		[
			_compare("LR_vs_buy_and_hold", lr_net, buy_and_hold, indices),
			_compare("XGB_vs_LR", xgb_net, lr_net, indices),
			_compare("XGB_vs_buy_and_hold", xgb_net, buy_and_hold, indices),
		]
	)
	comparisons["n_oos_days"] = int(len(buy_and_hold))
	comparisons["block_length"] = int(block_length)
	comparisons["n_resamples"] = int(n_resamples)
	comparisons["seed"] = int(seed)

	oos = pd.DataFrame(
		{
			"target_timestamp": lr_oos.index,
			"asset_return": lr_oos["asset_return"].to_numpy(),
			"buy_and_hold_return": buy_and_hold.to_numpy(),
			"lr_position": lr_oos["position"].to_numpy(),
			"lr_net_return": lr_net.to_numpy(),
			"xgb_position": xgb_oos["position"].to_numpy(),
			"xgb_net_return": xgb_net.to_numpy(),
		}
	)
	fold_lookup = lr_pooled.set_index("target_timestamp")["fold_id"]
	oos["fold_id"] = oos["target_timestamp"].map(fold_lookup).astype(int)

	folds_table = _build_fold_table(dataset, folds, oos, lr_pooled, xgb_pooled)
	model_metrics = pd.DataFrame(
		[_pooled_model_metrics(lr_pooled, "logistic_regression"), _pooled_model_metrics(xgb_pooled, "xgboost")]
	)
	return {
		"folds": folds_table,
		"oos": oos,
		"model_metrics": model_metrics,
		"comparisons": comparisons,
		"lr_pooled": lr_pooled,
		"xgb_pooled": xgb_pooled,
	}


def _fold_accuracy(pooled):
	predicted = (pooled["prediction"] >= DECISION_THRESHOLD).astype(int)
	correct = (predicted == pooled["y_true"].astype(int)).astype(float)
	return correct.groupby(pooled["fold_id"]).mean()


def _build_fold_table(dataset, folds, oos, lr_pooled, xgb_pooled):
	base = describe_folds(dataset, folds, "target_direction")
	lr_excess = oos.groupby("fold_id").apply(
		lambda group: _compounded(group["lr_net_return"]) - _compounded(group["buy_and_hold_return"]),
		include_groups=False,
	)
	xgb_excess = oos.groupby("fold_id").apply(
		lambda group: _compounded(group["xgb_net_return"]) - _compounded(group["buy_and_hold_return"]),
		include_groups=False,
	)
	base["lr_accuracy"] = base["fold_id"].map(_fold_accuracy(lr_pooled))
	base["xgb_accuracy"] = base["fold_id"].map(_fold_accuracy(xgb_pooled))
	base["lr_excess_total"] = base["fold_id"].map(lr_excess)
	base["xgb_excess_total"] = base["fold_id"].map(xgb_excess)
	return base


def main():
	try:
		results = run_direction_experiment()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
	results["folds"].to_csv(FOLDS_FILE, index=False)
	results["oos"].to_csv(OOS_FILE, index=False)
	results["model_metrics"].to_csv(MODEL_METRICS_FILE, index=False)
	results["comparisons"].to_csv(BOOTSTRAP_FILE, index=False)

	comparisons = results["comparisons"].set_index("comparison")
	lr_metrics = results["model_metrics"].set_index("model").loc["logistic_regression"]
	xgb_metrics = results["model_metrics"].set_index("model").loc["xgboost"]
	lr_row = comparisons.loc["LR_vs_buy_and_hold"]
	xgb_row = comparisons.loc["XGB_vs_LR"]

	print("Walk-forward direction experiments (expanding window, 8 folds):")
	print(f"  Outputs: {BOOTSTRAP_FILE.parent}")
	print(
		"  OOS window: "
		f"{lr_metrics['oos_start_target']} to {lr_metrics['oos_end_target']} "
		f"({int(lr_metrics['n_oos_days'])} days)"
	)
	print(
		"  Pooled accuracy: "
		f"LR {lr_metrics['pooled_accuracy']:.4f} (AUC {lr_metrics['pooled_roc_auc']:.4f}), "
		f"XGB {xgb_metrics['pooled_accuracy']:.4f} (AUC {xgb_metrics['pooled_roc_auc']:.4f}), "
		f"always-up base rate {lr_metrics['actual_positive_rate']:.4f}"
	)
	print("")
	print("  Experiment 1 - LR strategy vs buy-and-hold (mean daily excess return):")
	print(
		f"    point {lr_row['mean_daily_excess']:.6e}, "
		f"90% CI [{lr_row['ci_low']:.6e}, {lr_row['ci_high']:.6e}], "
		f"P(>0) {lr_row['prob_positive']:.4f}"
	)
	print(f"    total compounded excess {lr_row['total_compounded_excess']:.4f}")
	print(
		"    PRE-REGISTERED VERDICT: "
		+ ("LR beats buy-and-hold." if lr_row["excludes_zero_low"] else "LR does NOT beat buy-and-hold (null).")
	)
	print("")
	print("  Experiment 2 - XGBoost strategy vs LR strategy (mean daily excess return, economic test):")
	print(
		f"    point {xgb_row['mean_daily_excess']:.6e}, "
		f"90% CI [{xgb_row['ci_low']:.6e}, {xgb_row['ci_high']:.6e}], "
		f"P(>0) {xgb_row['prob_positive']:.4f}"
	)
	print(f"    total compounded excess {xgb_row['total_compounded_excess']:.4f}")
	print(
		"    PRE-REGISTERED VERDICT: "
		+ (
			"XGBoost adds economically useful nonlinear structure."
			if xgb_row["excludes_zero_low"]
			else "XGBoost does NOT beat the linear model economically (null)."
		)
	)


if __name__ == "__main__":
	main()

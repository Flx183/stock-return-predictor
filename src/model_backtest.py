from pathlib import Path

import numpy as np
import pandas as pd

try:
	from src.backtest import DEFAULT_COST_PER_TRADE, backtest_daily_signal
	from src.baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices
	from src.ml_pipeline import run_pipeline
except ModuleNotFoundError:  # Allows `python src/model_backtest.py`.
	from backtest import DEFAULT_COST_PER_TRADE, backtest_daily_signal
	from baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices
	from ml_pipeline import run_pipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_BACKTEST_FILE = REPO_ROOT / "data" / "ml_strategy_backtest.csv"
STRATEGY_METRICS_FILE = REPO_ROOT / "data" / "ml_strategy_metrics.csv"
MONTE_CARLO_COMPARISON_FILE = REPO_ROOT / "data" / "ml_monte_carlo_comparison.csv"
MONTE_CARLO_SAMPLES_FILE = REPO_ROOT / "data" / "ml_monte_carlo_samples.csv"
DEFAULT_MONTE_CARLO_SIMULATIONS = 10_000
DEFAULT_MONTE_CARLO_SEED = 42


def _validate_predictions(predictions):
	required_columns = {
		"feature_timestamp",
		"target_timestamp",
		"probability_up",
		"predicted_direction",
	}
	missing_columns = required_columns - set(predictions.columns)
	if missing_columns:
		raise ValueError(f"predictions is missing column(s): {', '.join(sorted(missing_columns))}")

	data = predictions.copy()
	data.index = data.index.rename(None)
	data["feature_timestamp"] = pd.to_datetime(data["feature_timestamp"], errors="coerce")
	data["target_timestamp"] = pd.to_datetime(data["target_timestamp"], errors="coerce")
	if data[["feature_timestamp", "target_timestamp"]].isna().any().any():
		raise ValueError("predictions contains invalid feature or target timestamp values.")
	if data["target_timestamp"].duplicated().any():
		raise ValueError("predictions cannot contain duplicate target timestamps.")
	if not data["predicted_direction"].isin([0, 1]).all():
		raise ValueError("predicted_direction must contain only 0 or 1.")

	data = data.sort_values("target_timestamp")
	if not (data["feature_timestamp"] < data["target_timestamp"]).all():
		raise ValueError("feature_timestamp must be strictly earlier than target_timestamp.")
	return data


def build_signal_from_predictions(predictions):
	"""
	Convert test-set predictions into daily target exposure.

	The first row is the prior close used as an anchor with zero exposure. That
	anchor lets the first target day earn its real close-to-close return without
	pretending the model knew anything before `feature_timestamp`.
	"""
	predictions = _validate_predictions(predictions)
	first_feature_timestamp = predictions["feature_timestamp"].iloc[0]
	signal = pd.Series(
		[0.0, *predictions["predicted_direction"].astype(float).to_list()],
		index=pd.DatetimeIndex(
			[first_feature_timestamp, *predictions["target_timestamp"].to_list()],
			name="Date",
		),
		name="signal",
	)
	if signal.index.has_duplicates:
		raise ValueError("prediction signal cannot have duplicate dates.")
	return signal


def _slice_prices_for_signal(prices, signal):
	prices = prices.sort_index().astype(float)
	missing_dates = signal.index.difference(prices.index)
	if not missing_dates.empty:
		raise ValueError(
			"prices is missing prediction signal date(s): "
			f"{', '.join(date.date().isoformat() for date in missing_dates[:5])}"
		)

	start_date = signal.index.min()
	end_date = signal.index.max()
	return prices.loc[start_date:end_date]


def backtest_model_predictions(prices, predictions, cost_per_trade=DEFAULT_COST_PER_TRADE):
	predictions = _validate_predictions(predictions)
	signal = build_signal_from_predictions(predictions)
	price_window = _slice_prices_for_signal(prices, signal)
	backtest = backtest_daily_signal(price_window, signal, cost_per_trade=cost_per_trade)

	prediction_lookup = predictions.set_index("target_timestamp")
	backtest["probability_up"] = prediction_lookup["probability_up"].reindex(backtest.index)
	backtest["predicted_direction"] = prediction_lookup["predicted_direction"].reindex(backtest.index)
	backtest["actual_direction"] = prediction_lookup.get(
		"actual_direction",
		pd.Series(dtype=float),
	).reindex(backtest.index)
	backtest["is_test_target_date"] = backtest.index.isin(prediction_lookup.index)
	backtest["buy_and_hold_return"] = backtest["asset_return"]
	backtest["buy_and_hold_equity_curve"] = (1.0 + backtest["buy_and_hold_return"]).cumprod()
	backtest["excess_return_vs_buy_and_hold"] = (
		backtest["net_strategy_return"] - backtest["buy_and_hold_return"]
	)
	return backtest


def _return_metrics(returns, equity_curve, prefix):
	if returns.empty:
		raise ValueError("returns cannot be empty.")

	total_return = equity_curve.iloc[-1] - 1.0
	years = len(returns) / TRADING_DAYS_PER_YEAR
	annualized_return = np.nan
	if equity_curve.iloc[-1] > 0 and years > 0:
		annualized_return = equity_curve.iloc[-1] ** (1 / years) - 1.0

	daily_volatility = returns.std()
	sharpe = np.nan
	if daily_volatility > 0:
		sharpe = returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

	drawdown = equity_curve / equity_curve.cummax() - 1.0
	return {
		f"{prefix}_total_return": float(total_return),
		f"{prefix}_annualized_return": float(annualized_return),
		f"{prefix}_sharpe": float(sharpe),
		f"{prefix}_max_drawdown": float(drawdown.min()),
	}


def compute_model_strategy_metrics(backtest, cost_per_trade=DEFAULT_COST_PER_TRADE):
	test_backtest = backtest.loc[backtest["is_test_target_date"]].copy()
	if test_backtest.empty:
		raise ValueError("backtest does not contain any test target dates.")

	strategy_metrics = _return_metrics(
		test_backtest["net_strategy_return"],
		test_backtest["equity_curve"],
		"strategy",
	)
	benchmark_metrics = _return_metrics(
		test_backtest["buy_and_hold_return"],
		test_backtest["buy_and_hold_equity_curve"],
		"buy_and_hold",
	)
	metrics = {
		**strategy_metrics,
		**benchmark_metrics,
		"excess_total_return": (
			strategy_metrics["strategy_total_return"]
			- benchmark_metrics["buy_and_hold_total_return"]
		),
		"excess_sharpe": strategy_metrics["strategy_sharpe"] - benchmark_metrics["buy_and_hold_sharpe"],
		"total_transaction_cost": float(test_backtest["transaction_cost"].sum()),
		"total_turnover": float(test_backtest["turnover"].sum()),
		"n_trade_days": int((test_backtest["turnover"] > 0).sum()),
		"average_position": float(test_backtest["position"].mean()),
		"cost_per_trade": float(cost_per_trade),
		"test_start_target_timestamp": test_backtest.index[0].date().isoformat(),
		"test_end_target_timestamp": test_backtest.index[-1].date().isoformat(),
		"n_test_days": int(len(test_backtest)),
	}
	return metrics


def _bootstrap_sharpe(sampled_returns):
	mean = sampled_returns.mean(axis=1)
	std = sampled_returns.std(axis=1, ddof=1)
	sharpe = np.full_like(mean, np.nan, dtype=float)
	nonzero = std > 0
	sharpe[nonzero] = mean[nonzero] / std[nonzero] * np.sqrt(TRADING_DAYS_PER_YEAR)
	return sharpe


def monte_carlo_compare_to_buy_and_hold(
	backtest,
	n_simulations=DEFAULT_MONTE_CARLO_SIMULATIONS,
	seed=DEFAULT_MONTE_CARLO_SEED,
):
	"""
	Paired bootstrap over out-of-sample daily returns.

	Each draw samples the same day index for strategy and buy-and-hold, preserving
	the empirical relationship between the two return streams on each test day.
	"""
	if n_simulations <= 0:
		raise ValueError("n_simulations must be positive.")

	test_backtest = backtest.loc[backtest["is_test_target_date"]].copy()
	if test_backtest.empty:
		raise ValueError("backtest does not contain any test target dates.")

	strategy_returns = test_backtest["net_strategy_return"].to_numpy(dtype=float)
	benchmark_returns = test_backtest["buy_and_hold_return"].to_numpy(dtype=float)
	n_days = len(test_backtest)
	rng = np.random.default_rng(seed)
	sampled_indices = rng.integers(0, n_days, size=(n_simulations, n_days))

	sampled_strategy_returns = strategy_returns[sampled_indices]
	sampled_benchmark_returns = benchmark_returns[sampled_indices]
	strategy_total_return = np.prod(1.0 + sampled_strategy_returns, axis=1) - 1.0
	benchmark_total_return = np.prod(1.0 + sampled_benchmark_returns, axis=1) - 1.0
	strategy_sharpe = _bootstrap_sharpe(sampled_strategy_returns)
	benchmark_sharpe = _bootstrap_sharpe(sampled_benchmark_returns)

	samples = pd.DataFrame(
		{
			"simulation": np.arange(n_simulations),
			"strategy_total_return": strategy_total_return,
			"buy_and_hold_total_return": benchmark_total_return,
			"excess_total_return": strategy_total_return - benchmark_total_return,
			"strategy_sharpe": strategy_sharpe,
			"buy_and_hold_sharpe": benchmark_sharpe,
			"excess_sharpe": strategy_sharpe - benchmark_sharpe,
		}
	)
	samples["strategy_beats_buy_and_hold_total_return"] = samples["excess_total_return"] > 0.0
	samples["strategy_beats_buy_and_hold_sharpe"] = samples["excess_sharpe"] > 0.0

	actual_strategy_total_return = np.prod(1.0 + strategy_returns) - 1.0
	actual_benchmark_total_return = np.prod(1.0 + benchmark_returns) - 1.0
	summary = {
		"n_simulations": int(n_simulations),
		"n_test_days": int(n_days),
		"seed": int(seed),
		"actual_strategy_total_return": float(actual_strategy_total_return),
		"actual_buy_and_hold_total_return": float(actual_benchmark_total_return),
		"actual_excess_total_return": float(actual_strategy_total_return - actual_benchmark_total_return),
		"strategy_beats_buy_and_hold_total_return_probability": float(
			samples["strategy_beats_buy_and_hold_total_return"].mean()
		),
		"strategy_beats_buy_and_hold_sharpe_probability": float(
			samples["strategy_beats_buy_and_hold_sharpe"].mean()
		),
		"excess_total_return_p05": float(samples["excess_total_return"].quantile(0.05)),
		"excess_total_return_p50": float(samples["excess_total_return"].quantile(0.50)),
		"excess_total_return_p95": float(samples["excess_total_return"].quantile(0.95)),
		"strategy_total_return_p50": float(samples["strategy_total_return"].quantile(0.50)),
		"buy_and_hold_total_return_p50": float(samples["buy_and_hold_total_return"].quantile(0.50)),
		"excess_sharpe_p05": float(samples["excess_sharpe"].quantile(0.05)),
		"excess_sharpe_p50": float(samples["excess_sharpe"].quantile(0.50)),
		"excess_sharpe_p95": float(samples["excess_sharpe"].quantile(0.95)),
	}
	return pd.DataFrame([summary]), samples


def run_model_backtest(
	cost_per_trade=DEFAULT_COST_PER_TRADE,
	n_simulations=DEFAULT_MONTE_CARLO_SIMULATIONS,
	seed=DEFAULT_MONTE_CARLO_SEED,
):
	_, predictions, _, _ = run_pipeline()
	prices = load_spy_close_prices()
	backtest = backtest_model_predictions(prices, predictions, cost_per_trade=cost_per_trade)
	metrics = compute_model_strategy_metrics(backtest, cost_per_trade=cost_per_trade)
	monte_carlo_summary, monte_carlo_samples = monte_carlo_compare_to_buy_and_hold(
		backtest,
		n_simulations=n_simulations,
		seed=seed,
	)
	return backtest, metrics, monte_carlo_summary, monte_carlo_samples


def main():
	try:
		backtest, metrics, monte_carlo_summary, monte_carlo_samples = run_model_backtest()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	STRATEGY_BACKTEST_FILE.parent.mkdir(parents=True, exist_ok=True)
	backtest.to_csv(STRATEGY_BACKTEST_FILE, index_label="Date")
	pd.DataFrame([metrics]).to_csv(STRATEGY_METRICS_FILE, index=False)
	monte_carlo_summary.to_csv(MONTE_CARLO_COMPARISON_FILE, index=False)
	monte_carlo_samples.to_csv(MONTE_CARLO_SAMPLES_FILE, index=False)

	summary = monte_carlo_summary.iloc[0]
	print("ML prediction strategy backtest:")
	print(f"  Strategy equity curve: {STRATEGY_BACKTEST_FILE}")
	print(f"  Strategy metrics: {STRATEGY_METRICS_FILE}")
	print(f"  Monte Carlo comparison: {MONTE_CARLO_COMPARISON_FILE}")
	print(f"  Monte Carlo samples: {MONTE_CARLO_SAMPLES_FILE}")
	print(
		"  Test target dates: "
		f"{metrics['test_start_target_timestamp']} to {metrics['test_end_target_timestamp']}"
	)
	print(f"  Strategy total return after costs: {metrics['strategy_total_return']:.6f}")
	print(f"  Buy-and-hold total return: {metrics['buy_and_hold_total_return']:.6f}")
	print(f"  Excess total return: {metrics['excess_total_return']:.6f}")
	print(
		"  Monte Carlo P(strategy beats buy-and-hold total return): "
		f"{summary['strategy_beats_buy_and_hold_total_return_probability']:.4f}"
	)


if __name__ == "__main__":
	main()

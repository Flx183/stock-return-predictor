"""
Experiment 7 from PREREGISTRATION.md: does a better volatility forecast buy a
better Sharpe?

Experiments 5 and 6 showed the volatility model matches HAR-RV on RMSE and that
the win is carried by the volatility lags. This experiment asks the question a
practitioner actually cares about: does forecast *accuracy* translate into
economic *value*? It runs one volatility-targeting strategy -- daily SPY exposure
= target_vol / forecast_vol, floored, capped, long-only -- through the existing
costed backtester, fed by three different forecasts of the same 5-day-ahead
realized vol: naive persistence, HAR-RV, and the 26-feature OLS. The strategy,
target vol, cap, and cost are identical across the three; only the forecast
changes, so any Sharpe difference is attributable to the forecast alone.

Timing is leakage-safe by construction: the position held for the close-to-close
return ending on day t is sized by the forecast made as of day t-1
(`feature_timestamp`), exactly the anchoring the direction backtester already
uses. The pre-registered decision statistic, in the style of Experiment 1, is the
annualized Sharpe difference (strategy - buy-and-hold) with a moving-block
bootstrap 90% CI; a source "beats buy-and-hold" only if that CI excludes zero on
the low side.

Honest prior: volatility targeting can raise Sharpe (Moreira-Muir 2017), but
Experiments 5-6 make near-identical forecasts, so the honest expectation is that
the three sources land close together and that better RMSE does not obviously buy
a better Sharpe after costs. Reported either way.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
	from src.backtest import DEFAULT_COST_PER_TRADE, backtest_daily_signal
	from src.baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices
	from src.experiment_vol_baselines import _pooled_forecast, _prepare
	from src.experiment_volatility import PERSISTENCE_FEATURE
	from src.features import FEATURE_COLUMNS, HAR_RV_COLUMNS
	from src.walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		bootstrap_mean,
		summarize_bootstrap,
	)
except ModuleNotFoundError:  # Allows `python src/experiment_vol_targeting.py`.
	from backtest import DEFAULT_COST_PER_TRADE, backtest_daily_signal
	from baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices
	from experiment_vol_baselines import _pooled_forecast, _prepare
	from experiment_volatility import PERSISTENCE_FEATURE
	from features import FEATURE_COLUMNS, HAR_RV_COLUMNS
	from walkforward import (
		DEFAULT_BLOCK_LENGTH,
		DEFAULT_N_RESAMPLES,
		DEFAULT_SEED,
		block_bootstrap_indices,
		bootstrap_mean,
		summarize_bootstrap,
	)


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARISONS_FILE = REPO_ROOT / "data" / "vol_targeting_comparisons.csv"
OOS_FILE = REPO_ROOT / "data" / "vol_targeting_oos.csv"

# Pre-registered strategy constants (fixed before the result was seen). The
# Sharpe comparison is invariant to TARGET_ANNUAL_VOL up to the cap, so this
# choice is a scale, not a tuned knob; the cap and floor bound an OLS forecast
# that can come out non-positive.
TARGET_ANNUAL_VOL = 0.15
POSITION_CAP = 2.0
FORECAST_FLOOR = 1e-6


def _vol_target_weights(forecast_vol, target_vol=TARGET_ANNUAL_VOL, cap=POSITION_CAP, floor=FORECAST_FLOOR):
	forecast_vol = np.asarray(forecast_vol, dtype=float)
	floored = forecast_vol <= floor
	safe = np.where(floored, floor, forecast_vol)
	weights = np.clip(target_vol / safe, 0.0, cap)
	return weights, floored


def _run_strategy(prices, predictions, target_vol, cap, cost_per_trade):
	"""Vol-targeted long-only backtest of one forecast; return OOS rows + diagnostics."""
	weights, floored = _vol_target_weights(predictions["forecast_vol"], target_vol=target_vol, cap=cap)
	first_feature = pd.Timestamp(predictions["feature_timestamp"].iloc[0])
	target_dates = pd.DatetimeIndex(pd.to_datetime(predictions["target_timestamp"]))
	# Anchor at the first feature close with zero exposure so the first sized
	# position is only applied to a strictly future close-to-close return.
	signal = pd.Series([0.0, *weights], index=pd.DatetimeIndex([first_feature, *target_dates], name="Date"), name="signal")
	if signal.index.has_duplicates:
		raise ValueError("vol-target signal cannot have duplicate dates.")

	price_window = prices.sort_index().astype(float).loc[signal.index.min():signal.index.max()]
	missing = signal.index.difference(price_window.index)
	if not missing.empty:
		raise ValueError(f"prices is missing {len(missing)} signal date(s), e.g. {missing[0].date().isoformat()}.")

	backtest = backtest_daily_signal(price_window, signal, cost_per_trade=cost_per_trade)
	oos = backtest.loc[backtest.index.isin(target_dates)].copy()
	oos["buy_and_hold_return"] = oos["asset_return"]

	diagnostics = {
		"avg_position": float(oos["position"].mean()),
		"capped_fraction": float((oos["position"] >= cap - 1e-9).mean()),
		"floored_forecast_fraction": float(np.mean(floored)),
		"total_turnover": float(oos["turnover"].sum()),
		"n_rebalance_days": int((oos["turnover"] > 1e-9).sum()),
		"total_transaction_cost": float(oos["transaction_cost"].sum()),
	}
	return oos, diagnostics


def _annualized_sharpe(returns):
	returns = np.asarray(returns, dtype=float)
	std = returns.std(ddof=1)
	if std <= 0 or not np.isfinite(std):
		return float("nan")
	return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe_difference_samples(strategy, benchmark, indices):
	"""Block-bootstrap distribution of (strategy Sharpe - benchmark Sharpe), paired by day."""

	def sharpe_rows(values):
		sampled = values[indices]
		mean = sampled.mean(axis=1)
		std = sampled.std(axis=1, ddof=1)
		out = np.full(mean.shape, np.nan)
		nonzero = std > 0
		out[nonzero] = mean[nonzero] / std[nonzero] * np.sqrt(TRADING_DAYS_PER_YEAR)
		return out

	return sharpe_rows(strategy) - sharpe_rows(benchmark)


def _compare_sharpe(name, oos, diagnostics, indices):
	net = oos["net_strategy_return"].to_numpy(dtype=float)
	buy_and_hold = oos["buy_and_hold_return"].to_numpy(dtype=float)

	strategy_sharpe = _annualized_sharpe(net)
	benchmark_sharpe = _annualized_sharpe(buy_and_hold)
	sharpe_samples = _sharpe_difference_samples(net, buy_and_hold, indices)
	sharpe_samples = sharpe_samples[np.isfinite(sharpe_samples)]
	sharpe_summary = summarize_bootstrap(sharpe_samples, point_estimate=strategy_sharpe - benchmark_sharpe)

	excess = net - buy_and_hold
	excess_summary = summarize_bootstrap(bootstrap_mean(excess, indices), point_estimate=float(np.mean(excess)))

	strategy_total = float(np.prod(1.0 + net) - 1.0)
	benchmark_total = float(np.prod(1.0 + buy_and_hold) - 1.0)
	return {
		"forecast_source": name,
		"strategy_sharpe": strategy_sharpe,
		"buy_and_hold_sharpe": benchmark_sharpe,
		"sharpe_difference": strategy_sharpe - benchmark_sharpe,
		"sharpe_diff_ci_low": sharpe_summary["ci_low"],
		"sharpe_diff_ci_high": sharpe_summary["ci_high"],
		"sharpe_diff_prob_positive": sharpe_summary["prob_positive"],
		"beats_buy_and_hold_sharpe": sharpe_summary["excludes_zero_low"],
		"mean_daily_excess": excess_summary["point_estimate"],
		"excess_ci_low": excess_summary["ci_low"],
		"excess_ci_high": excess_summary["ci_high"],
		"strategy_total_return": strategy_total,
		"buy_and_hold_total_return": benchmark_total,
		"excess_total_return": strategy_total - benchmark_total,
		**diagnostics,
	}


def _forecast_predictions(pooled, forecast_values):
	return pd.DataFrame(
		{
			"feature_timestamp": pooled["feature_timestamp"].to_numpy(),
			"target_timestamp": pooled["target_timestamp"].to_numpy(),
			"forecast_vol": np.asarray(forecast_values, dtype=float),
		}
	)


def run_vol_targeting_experiment(
	target_vol=TARGET_ANNUAL_VOL,
	cap=POSITION_CAP,
	cost_per_trade=DEFAULT_COST_PER_TRADE,
	block_length=DEFAULT_BLOCK_LENGTH,
	n_resamples=DEFAULT_N_RESAMPLES,
	seed=DEFAULT_SEED,
	ohlcv=None,
):
	dataset, folds = _prepare([*FEATURE_COLUMNS, *HAR_RV_COLUMNS], ohlcv=ohlcv)
	model = _pooled_forecast(dataset, FEATURE_COLUMNS, folds)
	har = _pooled_forecast(dataset, HAR_RV_COLUMNS, folds)
	if not np.array_equal(model["target_timestamp"].to_numpy(), har["target_timestamp"].to_numpy()):
		raise ValueError("model and HAR out-of-sample dates do not align.")

	prices = load_spy_close_prices() if ohlcv is None else ohlcv["Close"]
	forecast_sources = {
		"persistence": model[PERSISTENCE_FEATURE].to_numpy(dtype=float),
		"har_rv": har["prediction"].to_numpy(dtype=float),
		"model_26_feature": model["prediction"].to_numpy(dtype=float),
	}

	indices = block_bootstrap_indices(len(model), block_length=block_length, n_resamples=n_resamples, seed=seed)
	rows = []
	oos_frames = []
	for name, forecast in forecast_sources.items():
		predictions = _forecast_predictions(model, forecast)
		oos, diagnostics = _run_strategy(prices, predictions, target_vol, cap, cost_per_trade)
		rows.append(_compare_sharpe(name, oos, diagnostics, indices))
		oos_frames.append(
			pd.DataFrame(
				{
					"target_timestamp": oos.index,
					"forecast_source": name,
					"position": oos["position"].to_numpy(),
					"net_strategy_return": oos["net_strategy_return"].to_numpy(),
					"buy_and_hold_return": oos["buy_and_hold_return"].to_numpy(),
				}
			)
		)

	comparisons = pd.DataFrame(rows)
	comparisons["n_oos_days"] = int(len(model))
	comparisons["target_annual_vol"] = float(target_vol)
	comparisons["position_cap"] = float(cap)
	comparisons["cost_per_trade"] = float(cost_per_trade)
	comparisons["block_length"] = int(block_length)
	comparisons["oos_start_target"] = model["target_timestamp"].iloc[0].date().isoformat()
	comparisons["oos_end_target"] = model["target_timestamp"].iloc[-1].date().isoformat()
	return {"comparisons": comparisons, "oos": pd.concat(oos_frames, ignore_index=True)}


def main():
	try:
		results = run_vol_targeting_experiment()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	COMPARISONS_FILE.parent.mkdir(parents=True, exist_ok=True)
	results["comparisons"].to_csv(COMPARISONS_FILE, index=False)
	results["oos"].to_csv(OOS_FILE, index=False)

	comparisons = results["comparisons"]
	first = comparisons.iloc[0]
	print("Experiment 7 - volatility targeting: does a better forecast buy a better Sharpe?")
	print(f"  Outputs: {COMPARISONS_FILE.parent}")
	print(
		f"  OOS window: {first['oos_start_target']} to {first['oos_end_target']} "
		f"({int(first['n_oos_days'])} days); target vol {first['target_annual_vol']:.0%}, "
		f"cap {first['position_cap']:.1f}x"
	)
	print(f"  Buy-and-hold Sharpe (same for all sources): {first['buy_and_hold_sharpe']:.4f}")
	print("")
	print(f"  {'source':>18} {'strat_SR':>9} {'SR_diff':>8} {'90% CI of SR diff':>26} {'turnover':>9}  beats B&H?")
	for _, row in comparisons.iterrows():
		print(
			f"  {row['forecast_source']:>18} {row['strategy_sharpe']:>9.4f} {row['sharpe_difference']:>8.4f} "
			f"[{row['sharpe_diff_ci_low']:>10.4f}, {row['sharpe_diff_ci_high']:>10.4f}] "
			f"{row['total_turnover']:>9.1f}  "
			+ ("YES" if row["beats_buy_and_hold_sharpe"] else "no (null)")
		)
	print("")
	print("  PRE-REGISTERED VERDICT (per source): a vol-targeting strategy beats")
	print("  buy-and-hold only if its Sharpe-difference 90% CI excludes zero on the low side.")
	print("  Comparing across sources ties forecast accuracy to economic value.")


if __name__ == "__main__":
	main()

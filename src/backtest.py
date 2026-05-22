from pathlib import Path

import numpy as np
import pandas as pd

try:
	from src.baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices
except ModuleNotFoundError:  # Allows `python src/backtest.py`.
	from baseline import TRADING_DAYS_PER_YEAR, load_spy_close_prices


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "trivial_momentum_backtest.csv"
METRICS_FILE = REPO_ROOT / "data" / "trivial_momentum_metrics.csv"
DEFAULT_COST_PER_TRADE = 0.0001


def _validate_prices(prices):
	if not isinstance(prices, pd.Series):
		raise TypeError("prices must be a pandas Series.")
	if prices.empty:
		raise ValueError("prices cannot be empty.")
	if prices.index.has_duplicates:
		raise ValueError("prices cannot have duplicate dates.")

	prices = prices.sort_index().astype(float)
	if prices.isna().any():
		raise ValueError("prices cannot contain missing values.")
	if (prices <= 0).any():
		raise ValueError("prices must be positive.")

	return prices


def previous_return_long_flat_signal(prices):
	"""Target exposure: long when yesterday's close-to-close return was positive."""
	prices = _validate_prices(prices)
	daily_returns = prices.pct_change()
	signal = (daily_returns.shift(1) > 0).astype(float)
	signal.name = "signal"
	return signal


def backtest_daily_signal(prices, signal, cost_per_trade=DEFAULT_COST_PER_TRADE):
	"""
	Convert a daily target-exposure signal into a net P&L curve.

	`signal` is the position held for the close-to-close return ending on that
	date. Transaction cost is charged on absolute changes in exposure, so a
	long/flat round trip pays cost once on entry and once on exit.
	"""
	prices = _validate_prices(prices)
	if cost_per_trade < 0:
		raise ValueError("cost_per_trade cannot be negative.")

	position = pd.Series(signal, index=prices.index, name="position").astype(float)
	if position.isna().any():
		raise ValueError("signal must provide a numeric target position for every price date.")

	asset_return = prices.pct_change().fillna(0.0)
	previous_position = position.shift(1).fillna(0.0)
	turnover = (position - previous_position).abs()
	transaction_cost = turnover * cost_per_trade
	gross_strategy_return = position * asset_return
	net_strategy_return = gross_strategy_return - transaction_cost
	equity_curve = (1.0 + net_strategy_return).cumprod()

	return pd.DataFrame(
		{
			"close": prices,
			"asset_return": asset_return,
			"signal": position,
			"position": position,
			"turnover": turnover,
			"transaction_cost": transaction_cost,
			"gross_strategy_return": gross_strategy_return,
			"net_strategy_return": net_strategy_return,
			"equity_curve": equity_curve,
			"pnl_curve": equity_curve - 1.0,
		}
	)


def compute_strategy_metrics(backtest):
	if backtest.empty:
		raise ValueError("backtest cannot be empty.")

	equity_curve = backtest["equity_curve"]
	net_returns = backtest["net_strategy_return"]
	total_return = equity_curve.iloc[-1] - 1.0
	years = len(backtest) / TRADING_DAYS_PER_YEAR
	annualized_return = np.nan
	if equity_curve.iloc[-1] > 0 and years > 0:
		annualized_return = equity_curve.iloc[-1] ** (1 / years) - 1.0

	daily_volatility = net_returns.std()
	sharpe = np.nan
	if daily_volatility > 0:
		sharpe = net_returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

	drawdown = equity_curve / equity_curve.cummax() - 1.0

	return {
		"total_return": float(total_return),
		"annualized_return": float(annualized_return),
		"sharpe": float(sharpe),
		"max_drawdown": float(drawdown.min()),
		"total_transaction_cost": float(backtest["transaction_cost"].sum()),
		"total_turnover": float(backtest["turnover"].sum()),
		"n_trade_days": int((backtest["turnover"] > 0).sum()),
		"average_position": float(backtest["position"].mean()),
		"cost_per_trade": float(backtest["transaction_cost"].sum() / backtest["turnover"].sum())
		if backtest["turnover"].sum() > 0
		else 0.0,
		"start_date": backtest.index[0].date().isoformat(),
		"end_date": backtest.index[-1].date().isoformat(),
		"n_days": int(len(backtest)),
	}


def main():
	try:
		prices = load_spy_close_prices()
		signal = previous_return_long_flat_signal(prices)
		backtest = backtest_daily_signal(prices, signal, cost_per_trade=DEFAULT_COST_PER_TRADE)
		metrics = compute_strategy_metrics(backtest)
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	backtest.to_csv(OUTPUT_FILE, index_label="Date")
	pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)

	sharpe = metrics["sharpe"]
	sharpe_text = "n/a" if pd.isna(sharpe) else f"{sharpe:.4f}"

	print("Trivial previous-return long/flat backtest:")
	print(f"  Data file: {OUTPUT_FILE}")
	print(f"  Metrics file: {METRICS_FILE}")
	print(f"  Cost per 1.0 notional trade: {DEFAULT_COST_PER_TRADE:.6f}")
	print(f"  Start date: {metrics['start_date']}")
	print(f"  End date: {metrics['end_date']}")
	print(f"  Total return after costs: {metrics['total_return']:.6f}")
	print(f"  Annualized return after costs: {metrics['annualized_return']:.6f}")
	print(f"  Sharpe ratio after costs (ann.): {sharpe_text}")
	print(f"  Max drawdown: {metrics['max_drawdown']:.6f}")
	print(f"  Total transaction cost: {metrics['total_transaction_cost']:.6f}")
	print(f"  Trade days: {metrics['n_trade_days']}")


if __name__ == "__main__":
	main()

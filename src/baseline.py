from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "SPY_data.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "baseline_metrics.csv"
TRADING_DAYS_PER_YEAR = 252


def load_spy_close_prices():
	if not DATA_FILE.exists():
		raise FileNotFoundError(f"{DATA_FILE} does not exist. Run src/pulldata.py first.")

	df = pd.read_csv(DATA_FILE, skiprows=[1, 2])
	missing_columns = {"Price", "Close"} - set(df.columns)
	if missing_columns:
		raise ValueError(
			"SPY_data.csv should match the yfinance SPY export used in this project. "
			f"Missing column(s): {', '.join(sorted(missing_columns))}"
		)

	dates = pd.to_datetime(df["Price"], errors="coerce")
	close = pd.to_numeric(df["Close"], errors="coerce")

	bad_rows = dates.isna() | close.isna()
	if bad_rows.any():
		line_numbers = (df.index[bad_rows] + 4).astype(str).tolist()
		raise ValueError(
			"SPY_data.csv has invalid date or close price value(s) on CSV line(s): "
			f"{', '.join(line_numbers[:5])}"
		)

	prices = pd.Series(close.to_numpy(), index=dates, name="Close").sort_index()
	if prices.index.has_duplicates:
		raise ValueError("SPY_data.csv has duplicate dates.")
	if len(prices) < 2:
		raise ValueError("SPY_data.csv needs at least two rows to calculate returns.")
	if (prices <= 0).any():
		raise ValueError("SPY_data.csv has zero or negative close prices.")

	return prices


def compute_buy_and_hold_metrics(prices):
	total_return = prices.iloc[-1] / prices.iloc[0] - 1
	years = len(prices) / TRADING_DAYS_PER_YEAR
	annualized_return = (1 + total_return) ** (1 / years) - 1

	daily_returns = prices.pct_change().dropna()
	daily_volatility = daily_returns.std()
	sharpe = np.nan
	if daily_volatility > 0:
		sharpe = daily_returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

	cumulative = prices / prices.iloc[0]
	drawdown = cumulative / cumulative.cummax() - 1

	return {
		"total_return": float(total_return),
		"annualized_return": float(annualized_return),
		"sharpe": float(sharpe),
		"max_drawdown": float(drawdown.min()),
		"start_date": prices.index[0].date().isoformat(),
		"end_date": prices.index[-1].date().isoformat(),
		"n_days": int(len(prices)),
	}


def main():
	try:
		prices = load_spy_close_prices()
		metrics = compute_buy_and_hold_metrics(prices)
	except (FileNotFoundError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame([metrics]).to_csv(OUTPUT_FILE, index=False)

	sharpe = metrics["sharpe"]
	sharpe_text = "n/a" if pd.isna(sharpe) else f"{sharpe:.4f}"

	print("Buy-and-hold baseline metrics:")
	print(f"  Data file: {DATA_FILE}")
	print(f"  Start date: {metrics['start_date']}")
	print(f"  End date: {metrics['end_date']}")
	print(f"  Total return: {metrics['total_return']:.6f}")
	print(f"  Annualized return: {metrics['annualized_return']:.6f}")
	print(f"  Sharpe ratio (ann.): {sharpe_text}")
	print(f"  Max drawdown: {metrics['max_drawdown']:.6f}")
	print(f"Saved metrics to: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()

"""
CBOE VIX index loader for Experiment 8.

VIX is the option-implied 30-day volatility of the S&P 500, quoted in annualized
percentage points (VIX = 20 means 20% annualized). Dividing by 100 puts it in the
same decimal, annualized units as `target_volatility_5d`, so it can be used both
as a direct forecast of realized volatility and as a leakage-safe feature. Because
VIX embeds option-market information, using it steps the project outside weak-form
efficiency; see README's scope statement.

`src/pulldata.py` pulls SPY; this module pulls and loads ^VIX the same way, into
`data/VIX_data.csv` with the identical yfinance three-header-row layout.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
VIX_DATA_FILE = REPO_ROOT / "data" / "VIX_data.csv"


def pull_vix(start="2010-01-01", end="2024-06-30", file_path=VIX_DATA_FILE):
	import yfinance as yf

	data = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
	if data.empty:
		raise ValueError("yfinance returned no ^VIX rows for the requested range.")
	file_path = Path(file_path)
	file_path.parent.mkdir(parents=True, exist_ok=True)
	data.to_csv(file_path)
	return file_path


def load_vix_close(file_path=VIX_DATA_FILE):
	"""Return the daily VIX close as a decimal annualized volatility (VIX / 100)."""
	file_path = Path(file_path)
	if not file_path.exists():
		raise FileNotFoundError(f"{file_path} does not exist. Run `python3 src/vix_data.py` first.")

	df = pd.read_csv(file_path, skiprows=[1, 2])
	missing = {"Price", "Close"} - set(df.columns)
	if missing:
		raise ValueError(
			"VIX_data.csv should match the yfinance ^VIX export. "
			f"Missing column(s): {', '.join(sorted(missing))}"
		)

	dates = pd.to_datetime(df["Price"], errors="coerce")
	close = pd.to_numeric(df["Close"], errors="coerce")
	vix = pd.Series(close.to_numpy(), index=dates, name="vix_close").sort_index()
	vix = vix.dropna()
	if vix.empty:
		raise ValueError("VIX_data.csv has no valid rows after parsing.")
	if vix.index.has_duplicates:
		raise ValueError("VIX_data.csv has duplicate dates.")
	if (vix <= 0).any():
		raise ValueError("VIX_data.csv has non-positive VIX values.")
	return vix / 100.0


def main():
	try:
		path = pull_vix()
	except (ValueError, ImportError) as error:
		raise SystemExit(f"Error: {error}")
	vix = load_vix_close(path)
	print("Pulled CBOE VIX:")
	print(f"  Data file: {path}")
	print(f"  Rows: {len(vix)}  Range: {vix.index[0].date()} to {vix.index[-1].date()}")
	print(f"  Decimal annualized vol range: {vix.min():.4f} to {vix.max():.4f} (median {vix.median():.4f})")


if __name__ == "__main__":
	main()

from pathlib import Path

import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "SPY_data.csv"


def main():
	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	
	data = yf.download("SPY", start="2010-01-01", end="2024-06-30", auto_adjust=True)
	data.to_csv(OUTPUT_FILE)

if __name__ == "__main__":
	main()

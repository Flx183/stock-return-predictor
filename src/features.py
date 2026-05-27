import numpy as np
import pandas as pd

try:
	from src.baseline import TRADING_DAYS_PER_YEAR
except ModuleNotFoundError:  # Allows `python src/features.py`.
	from baseline import TRADING_DAYS_PER_YEAR


RETURN_WINDOWS = (1, 2, 5, 10, 21, 63)
VOLATILITY_WINDOWS = (5, 10, 21, 63)
SMA_WINDOWS = (5, 10, 20, 50, 200)
PRICE_COLUMNS = ("Open", "High", "Low", "Close")
REQUIRED_COLUMNS = (*PRICE_COLUMNS, "Volume")
TARGET_VOLATILITY_WINDOW = 5
TARGET_VOLATILITY_COLUMN = f"target_volatility_{TARGET_VOLATILITY_WINDOW}d"

FEATURE_COLUMNS = [
	*(f"return_{window}d" for window in RETURN_WINDOWS),
	*(f"realized_vol_{window}d" for window in VOLATILITY_WINDOWS),
	*(f"sma_{window}d_ratio" for window in SMA_WINDOWS),
	"sma_5d_20d_spread",
	"sma_20d_50d_spread",
	"sma_50d_200d_spread",
	"drawdown_21d",
	"drawdown_63d",
	"rsi_14d",
	"intraday_return_1d",
	"range_1d",
	"volume_change_1d",
	"volume_zscore_21d",
	"volume_zscore_63d",
]


def _validate_ohlcv(ohlcv):
	if not isinstance(ohlcv, pd.DataFrame):
		raise TypeError("ohlcv must be a pandas DataFrame.")

	missing_columns = set(REQUIRED_COLUMNS) - set(ohlcv.columns)
	if missing_columns:
		raise ValueError(f"ohlcv is missing column(s): {', '.join(sorted(missing_columns))}")

	data = ohlcv.loc[:, REQUIRED_COLUMNS].copy()
	if not isinstance(data.index, pd.DatetimeIndex):
		data.index = pd.to_datetime(data.index, errors="coerce")
	if data.index.isna().any():
		raise ValueError("ohlcv has invalid date index value(s).")
	if data.index.has_duplicates:
		raise ValueError("ohlcv cannot have duplicate dates.")

	for column in REQUIRED_COLUMNS:
		data[column] = pd.to_numeric(data[column], errors="coerce")
	if data.isna().any().any():
		raise ValueError("ohlcv cannot contain missing or non-numeric OHLCV values.")
	if (data.loc[:, PRICE_COLUMNS] <= 0).any().any():
		raise ValueError("ohlcv price columns must be positive.")
	if (data["Volume"] < 0).any():
		raise ValueError("ohlcv volume cannot be negative.")

	return data.sort_index()


def _rolling_zscore(values, window):
	mean = values.rolling(window=window, min_periods=window).mean()
	std = values.rolling(window=window, min_periods=window).std()
	zscore = (values - mean) / std.replace(0.0, np.nan)
	return zscore.mask(std == 0.0, 0.0)


def _rsi(close, window=14):
	delta = close.diff()
	gain = delta.clip(lower=0.0)
	loss = -delta.clip(upper=0.0)
	avg_gain = gain.rolling(window=window, min_periods=window).mean()
	avg_loss = loss.rolling(window=window, min_periods=window).mean()
	rs = avg_gain / avg_loss.replace(0.0, np.nan)
	rsi = 100.0 - (100.0 / (1.0 + rs))
	rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
	rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
	return rsi / 100.0


def build_leakage_safe_features(ohlcv):
	"""
	Build one supervised row per target close.

	For row T, `target_timestamp` is the close whose return is being predicted.
	`feature_timestamp` is the immediately prior trading close. Every feature is
	first calculated as of a close timestamp, then shifted forward one row so that
	the row for target T can only use data available at close T-1 or earlier.
	`target_volatility_5d` is a future realized-volatility label, not a feature:
	it covers returns from `target_timestamp` through
	`target_volatility_end_timestamp`.
	"""
	ohlcv = _validate_ohlcv(ohlcv)
	close = ohlcv["Close"]
	daily_return = close.pct_change()
	timestamps = pd.Series(ohlcv.index, index=ohlcv.index)
	future_realized_volatility = (
		daily_return.rolling(
			window=TARGET_VOLATILITY_WINDOW,
			min_periods=TARGET_VOLATILITY_WINDOW,
		).std()
		.shift(-(TARGET_VOLATILITY_WINDOW - 1))
		* np.sqrt(TRADING_DAYS_PER_YEAR)
	)

	features_as_of_close = pd.DataFrame(index=ohlcv.index)
	for window in RETURN_WINDOWS:
		features_as_of_close[f"return_{window}d"] = close.pct_change(window)
	for window in VOLATILITY_WINDOWS:
		features_as_of_close[f"realized_vol_{window}d"] = (
			daily_return.rolling(window=window, min_periods=window).std()
			* np.sqrt(TRADING_DAYS_PER_YEAR)
		)
	for window in SMA_WINDOWS:
		sma = close.rolling(window=window, min_periods=window).mean()
		features_as_of_close[f"sma_{window}d_ratio"] = close / sma - 1.0

	sma_5 = close.rolling(window=5, min_periods=5).mean()
	sma_20 = close.rolling(window=20, min_periods=20).mean()
	sma_50 = close.rolling(window=50, min_periods=50).mean()
	sma_200 = close.rolling(window=200, min_periods=200).mean()

	features_as_of_close["sma_5d_20d_spread"] = sma_5 / sma_20 - 1.0
	features_as_of_close["sma_20d_50d_spread"] = sma_20 / sma_50 - 1.0
	features_as_of_close["sma_50d_200d_spread"] = sma_50 / sma_200 - 1.0
	features_as_of_close["drawdown_21d"] = close / close.rolling(21, min_periods=21).max() - 1.0
	features_as_of_close["drawdown_63d"] = close / close.rolling(63, min_periods=63).max() - 1.0
	features_as_of_close["rsi_14d"] = _rsi(close, window=14)
	features_as_of_close["intraday_return_1d"] = ohlcv["Close"] / ohlcv["Open"] - 1.0
	features_as_of_close["range_1d"] = ohlcv["High"] / ohlcv["Low"] - 1.0
	features_as_of_close["volume_change_1d"] = ohlcv["Volume"].pct_change()
	features_as_of_close["volume_zscore_21d"] = _rolling_zscore(ohlcv["Volume"], 21)
	features_as_of_close["volume_zscore_63d"] = _rolling_zscore(ohlcv["Volume"], 63)

	features = features_as_of_close.loc[:, FEATURE_COLUMNS].shift(1)
	dataset = pd.concat(
		[
			timestamps.shift(1).rename("feature_timestamp"),
			timestamps.rename("target_timestamp"),
			daily_return.rename("target_return"),
			(daily_return > 0.0).astype(int).rename("target_direction"),
			timestamps.rename("target_volatility_start_timestamp"),
			timestamps.shift(-(TARGET_VOLATILITY_WINDOW - 1)).rename("target_volatility_end_timestamp"),
			future_realized_volatility.rename(TARGET_VOLATILITY_COLUMN),
			features,
		],
		axis=1,
	)
	dataset = dataset.replace([np.inf, -np.inf], np.nan)
	dataset = dataset.dropna(
		subset=["feature_timestamp", "target_timestamp", "target_return", *FEATURE_COLUMNS]
	)
	if dataset.empty:
		raise ValueError("not enough valid OHLCV rows to build the requested features.")
	if not (dataset["feature_timestamp"] < dataset["target_timestamp"]).all():
		raise ValueError("feature_timestamp must be strictly earlier than target_timestamp.")

	dataset.index = pd.DatetimeIndex(dataset["target_timestamp"], name="target_timestamp")
	return dataset

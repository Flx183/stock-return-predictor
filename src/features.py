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

# Longer look-backs used only by the multi-horizon experiment. They are kept out
# of the canonical FEATURE_COLUMNS so the pre-registered 1-day Experiments 1/2/3
# keep their exact feature set. `build_leakage_safe_features` always computes
# these columns; they only enter a run when a caller passes them explicitly.
LONG_RETURN_WINDOWS = (126, 252)
LONG_VOLATILITY_WINDOWS = (126, 252)
LONG_DRAWDOWN_WINDOWS = (126, 252)

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

# The canonical 26 features plus longer-look-back momentum/mean-reversion/vol
# terms, so long-horizon structure can be captured. Still OHLCV-derived only.
LONG_HORIZON_FEATURE_COLUMNS = [
	*FEATURE_COLUMNS,
	*(f"return_{window}d" for window in LONG_RETURN_WINDOWS),
	*(f"realized_vol_{window}d" for window in LONG_VOLATILITY_WINDOWS),
	*(f"drawdown_{window}d" for window in LONG_DRAWDOWN_WINDOWS),
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


def build_leakage_safe_features(ohlcv, forward_horizon=1, feature_columns=None):
	"""
	Build one supervised row per target close.

	`feature_timestamp` is the trading close as of which every feature is known;
	each feature is first calculated as of a close timestamp, then shifted forward
	one row so that a row can only use data available at that prior close or
	earlier. `target_return` is the forward return over the next `forward_horizon`
	trading days measured from the `feature_timestamp` close, and
	`target_timestamp` is the close that closes that window (so a 1-day horizon
	reproduces the original next-close target exactly). `target_direction` is the
	sign of that forward return.

	`feature_columns` selects which pre-computed feature columns are kept and
	required to be non-null (defaults to the canonical FEATURE_COLUMNS). Pass
	LONG_HORIZON_FEATURE_COLUMNS to include the longer look-back terms.

	`target_volatility_5d` is a future realized-volatility label, not a feature:
	it covers returns from `target_timestamp` through
	`target_volatility_end_timestamp`.
	"""
	if not isinstance(forward_horizon, (int, np.integer)) or forward_horizon < 1:
		raise ValueError("forward_horizon must be a positive integer.")
	if feature_columns is None:
		feature_columns = FEATURE_COLUMNS
	feature_columns = list(feature_columns)

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

	# Longer look-backs (kept out of the canonical set; see LONG_* constants).
	for window in LONG_RETURN_WINDOWS:
		features_as_of_close[f"return_{window}d"] = close.pct_change(window)
	for window in LONG_VOLATILITY_WINDOWS:
		features_as_of_close[f"realized_vol_{window}d"] = (
			daily_return.rolling(window=window, min_periods=window).std()
			* np.sqrt(TRADING_DAYS_PER_YEAR)
		)
	for window in LONG_DRAWDOWN_WINDOWS:
		features_as_of_close[f"drawdown_{window}d"] = (
			close / close.rolling(window, min_periods=window).max() - 1.0
		)

	features_as_of_close["rsi_14d"] = _rsi(close, window=14)
	features_as_of_close["intraday_return_1d"] = ohlcv["Close"] / ohlcv["Open"] - 1.0
	features_as_of_close["range_1d"] = ohlcv["High"] / ohlcv["Low"] - 1.0
	features_as_of_close["volume_change_1d"] = ohlcv["Volume"].pct_change()
	features_as_of_close["volume_zscore_21d"] = _rolling_zscore(ohlcv["Volume"], 21)
	features_as_of_close["volume_zscore_63d"] = _rolling_zscore(ohlcv["Volume"], 63)

	# Forward return over `forward_horizon` trading days, measured from the
	# feature close (the prior row). shift(1) is that feature close; shifting by
	# (1 - forward_horizon) reaches the close `forward_horizon` steps ahead of it.
	# forward_horizon == 1 collapses to the original next-close return exactly.
	base_close = close.shift(1)
	forward_close = close.shift(1 - forward_horizon)
	forward_return = forward_close / base_close - 1.0
	target_timestamps = timestamps.shift(1 - forward_horizon)

	features = features_as_of_close.loc[:, feature_columns].shift(1)
	dataset = pd.concat(
		[
			timestamps.shift(1).rename("feature_timestamp"),
			target_timestamps.rename("target_timestamp"),
			forward_return.rename("target_return"),
			(forward_return > 0.0).astype(int).rename("target_direction"),
			timestamps.rename("target_volatility_start_timestamp"),
			timestamps.shift(-(TARGET_VOLATILITY_WINDOW - 1)).rename("target_volatility_end_timestamp"),
			future_realized_volatility.rename(TARGET_VOLATILITY_COLUMN),
			features,
		],
		axis=1,
	)
	dataset = dataset.replace([np.inf, -np.inf], np.nan)
	dataset = dataset.dropna(
		subset=["feature_timestamp", "target_timestamp", "target_return", *feature_columns]
	)
	if dataset.empty:
		raise ValueError("not enough valid OHLCV rows to build the requested features.")
	if not (dataset["feature_timestamp"] < dataset["target_timestamp"]).all():
		raise ValueError("feature_timestamp must be strictly earlier than target_timestamp.")

	dataset.index = pd.DatetimeIndex(dataset["target_timestamp"], name="target_timestamp")
	return dataset

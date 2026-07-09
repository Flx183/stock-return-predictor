import numpy as np
import pandas as pd
import pytest

from src.features import (
	FEATURE_COLUMNS,
	HAR_RV_COLUMNS,
	LONG_HORIZON_FEATURE_COLUMNS,
	TARGET_VOLATILITY_COLUMN,
	build_leakage_safe_features,
)


def make_ohlcv(n_days=260):
	dates = pd.bdate_range("2020-01-01", periods=n_days)
	step = np.arange(n_days, dtype=float)
	wave = np.sin(step / 7.0)
	close = pd.Series(100.0 + step * 0.15 + wave, index=dates)
	open_ = close * (1.0 - 0.001)
	high = close * (1.0 + 0.004)
	low = close * (1.0 - 0.004)
	volume = pd.Series(1_000_000.0 + step * 1_000.0 + (step % 11) * 10_000.0, index=dates)
	return pd.DataFrame(
		{
			"Open": open_,
			"High": high,
			"Low": low,
			"Close": close,
			"Volume": volume,
		},
		index=dates,
	)


def test_features_are_aligned_to_previous_trading_close():
	ohlcv = make_ohlcv()
	dataset = build_leakage_safe_features(ohlcv)

	target_timestamp = ohlcv.index[220]
	feature_timestamp = ohlcv.index[219]
	row = dataset.loc[target_timestamp]
	close = ohlcv["Close"]

	assert row["target_timestamp"] == target_timestamp
	assert row["feature_timestamp"] == feature_timestamp
	assert row["target_return"] == pytest.approx(close.iloc[220] / close.iloc[219] - 1.0)
	assert row["return_1d"] == pytest.approx(close.iloc[219] / close.iloc[218] - 1.0)
	assert row["sma_5d_ratio"] == pytest.approx(close.iloc[219] / close.iloc[215:220].mean() - 1.0)

	feature_return_window = close.pct_change().iloc[215:220]
	expected_vol = feature_return_window.std() * np.sqrt(252)
	assert row["realized_vol_5d"] == pytest.approx(expected_vol)
	assert (dataset["feature_timestamp"] < dataset["target_timestamp"]).all()


def test_future_volatility_target_has_explicit_forward_window():
	ohlcv = make_ohlcv()
	dataset = build_leakage_safe_features(ohlcv)

	target_timestamp = ohlcv.index[220]
	row = dataset.loc[target_timestamp]
	close = ohlcv["Close"]

	expected_window = close.pct_change().iloc[220:225]
	expected_volatility = expected_window.std() * np.sqrt(252)
	assert row["target_volatility_start_timestamp"] == target_timestamp
	assert row["target_volatility_end_timestamp"] == ohlcv.index[224]
	assert row[TARGET_VOLATILITY_COLUMN] == pytest.approx(expected_volatility)


@pytest.mark.parametrize("horizon", [1, 5, 21, 63])
def test_forward_horizon_target_spans_the_full_window(horizon):
	ohlcv = make_ohlcv(n_days=500)
	dataset = build_leakage_safe_features(ohlcv, forward_horizon=horizon)
	close = ohlcv["Close"].to_numpy()

	assert (dataset["feature_timestamp"] < dataset["target_timestamp"]).all()

	feature_pos = ohlcv.index.get_indexer(dataset["feature_timestamp"])
	target_pos = ohlcv.index.get_indexer(dataset["target_timestamp"])
	# The target close is exactly `horizon` trading steps after the feature close.
	assert ((target_pos - feature_pos) == horizon).all()

	expected_return = close[target_pos] / close[feature_pos] - 1.0
	np.testing.assert_allclose(dataset["target_return"].to_numpy(), expected_return, rtol=1e-9)
	assert (dataset["target_direction"].to_numpy() == (expected_return > 0.0).astype(int)).all()


def test_default_horizon_reproduces_next_close_target():
	ohlcv = make_ohlcv()
	dataset = build_leakage_safe_features(ohlcv)  # default forward_horizon=1
	canonical_next_close_return = ohlcv["Close"].pct_change().reindex(dataset.index)
	np.testing.assert_allclose(
		dataset["target_return"].to_numpy(),
		canonical_next_close_return.to_numpy(),
		rtol=1e-12,
	)


def test_long_horizon_features_are_optional_and_complete():
	ohlcv = make_ohlcv(n_days=900)
	default = build_leakage_safe_features(ohlcv)
	assert "return_252d" not in default.columns

	extended = build_leakage_safe_features(
		ohlcv, forward_horizon=21, feature_columns=LONG_HORIZON_FEATURE_COLUMNS
	)
	for column in LONG_HORIZON_FEATURE_COLUMNS:
		assert column in extended.columns
	assert extended[LONG_HORIZON_FEATURE_COLUMNS].notna().all().all()


def test_har_rv_components_are_optional_and_leakage_safe():
	ohlcv = make_ohlcv(n_days=400)
	default = build_leakage_safe_features(ohlcv)
	for column in HAR_RV_COLUMNS:
		assert column not in default.columns

	dataset = build_leakage_safe_features(ohlcv, feature_columns=[*FEATURE_COLUMNS, *HAR_RV_COLUMNS])
	for column in HAR_RV_COLUMNS:
		assert column in dataset.columns
	assert dataset[HAR_RV_COLUMNS].notna().all().all()

	# The daily HAR component as of a feature close is sqrt(252) * |that day's return|.
	target_timestamp = ohlcv.index[300]
	feature_index = 299
	close = ohlcv["Close"]
	prior_return = close.iloc[feature_index] / close.iloc[feature_index - 1] - 1.0
	expected_daily = np.sqrt(252) * abs(prior_return)
	assert dataset.loc[target_timestamp, "har_rv_daily"] == pytest.approx(expected_daily)


def test_forward_horizon_must_be_a_positive_integer():
	ohlcv = make_ohlcv()
	with pytest.raises(ValueError):
		build_leakage_safe_features(ohlcv, forward_horizon=0)
	with pytest.raises(ValueError):
		build_leakage_safe_features(ohlcv, forward_horizon=1.5)


def test_target_day_close_does_not_change_same_row_features():
	ohlcv = make_ohlcv()
	target_timestamp = ohlcv.index[220]
	original = build_leakage_safe_features(ohlcv)

	changed = ohlcv.copy()
	changed.loc[target_timestamp, "Close"] *= 1.25
	changed_dataset = build_leakage_safe_features(changed)

	pd.testing.assert_series_equal(
		original.loc[target_timestamp, FEATURE_COLUMNS],
		changed_dataset.loc[target_timestamp, FEATURE_COLUMNS],
		check_names=False,
	)
	assert changed_dataset.loc[target_timestamp, "target_return"] != pytest.approx(
		original.loc[target_timestamp, "target_return"]
	)

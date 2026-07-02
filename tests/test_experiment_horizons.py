import numpy as np
import pandas as pd
import pytest

from src.experiment_horizons import (
	HORIZONS,
	RIGOR_TIER,
	_horizon_folds,
	_nonoverlapping_economic,
	_regression_metrics,
	_run_one_horizon,
)
from src.walkforward import DEFAULT_BLOCK_LENGTH, DEFAULT_TEST_SIZE


def make_ohlcv(n_days=2000):
	dates = pd.bdate_range("2010-01-01", periods=n_days)
	step = np.arange(n_days, dtype=float)
	wave = np.sin(step / 9.0) + 0.5 * np.sin(step / 40.0)
	close = pd.Series(100.0 + step * 0.05 + wave, index=dates)
	open_ = close * (1.0 - 0.001)
	high = close * (1.0 + 0.004)
	low = close * (1.0 - 0.004)
	volume = pd.Series(1_000_000.0 + step * 500.0 + (step % 13) * 8_000.0, index=dates)
	return pd.DataFrame(
		{"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
		index=dates,
	)


def _pooled(target_timestamps, y_true, prediction, fold_ids):
	return pd.DataFrame(
		{
			"fold_id": fold_ids,
			"feature_timestamp": target_timestamps - pd.Timedelta(days=1),
			"target_timestamp": target_timestamps,
			"y_true": y_true,
			"prediction": prediction,
		}
	)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_horizon_folds_scale_block_and_test_size_with_horizon(horizon):
	folds, test_size, block_length = _horizon_folds(n_rows=3000, horizon=horizon)
	# A test block must hold at least one horizon; the bootstrap block must hold
	# at least two so overlapping-target autocorrelation survives resampling.
	assert test_size >= max(DEFAULT_TEST_SIZE, 2 * horizon)
	assert block_length >= max(DEFAULT_BLOCK_LENGTH, 2 * horizon)
	assert len(folds) >= 1
	# Folds are contiguous and expanding, never reaching into the future.
	for earlier, later in zip(folds, folds[1:]):
		assert earlier.test_end == later.test_start
		assert later.train_end == later.test_start


def test_regression_metrics_reward_a_real_signal_and_reject_noise():
	n = 260
	timestamps = pd.bdate_range("2015-01-01", periods=n)
	fold_ids = np.repeat(np.arange(4), n // 4)[:n]
	rng = np.random.default_rng(0)
	y_true = rng.normal(0.0, 0.02, size=n)
	drift = _pooled(timestamps, y_true, np.zeros(n), fold_ids)

	# A perfect forecast beats the flat drift benchmark in every fold.
	perfect = _pooled(timestamps, y_true, y_true, fold_ids)
	good = _regression_metrics("m", 5, perfect, drift, block_length=21, n_resamples=500, seed=1)
	assert good["oos_r2_vs_drift"] == pytest.approx(1.0)
	assert good["folds_positive_fraction"] == pytest.approx(1.0)
	assert good["beats_drift"] is True

	# A forecast identical to drift cannot beat drift.
	same = _pooled(timestamps, y_true, np.zeros(n), fold_ids)
	null = _regression_metrics("m", 5, same, drift, block_length=21, n_resamples=500, seed=1)
	assert null["oos_r2_vs_drift"] == pytest.approx(0.0)
	assert null["beats_drift"] is False


def test_nonoverlapping_economic_uses_disjoint_windows():
	n = 120
	horizon = 21
	timestamps = pd.bdate_range("2016-01-01", periods=n)
	dataset = pd.DataFrame(
		{"target_return": np.linspace(-0.01, 0.01, n)},
		index=pd.DatetimeIndex(timestamps, name="target_timestamp"),
	)
	pooled = _pooled(timestamps, np.ones(n), np.ones(n), np.zeros(n, dtype=int))
	result = _nonoverlapping_economic("m", horizon, pooled, dataset, cost_per_trade=0.0, n_resamples=200, seed=1)
	# Every horizon-th row -> ceil(n / horizon) disjoint windows.
	assert result["n_windows"] == len(range(0, n, horizon))
	# Always long, no cost -> strategy exactly equals buy-and-hold.
	assert result["excess_total_return"] == pytest.approx(0.0, abs=1e-12)


def test_run_one_horizon_returns_scored_rows_end_to_end():
	ohlcv = make_ohlcv(n_days=2000)
	regression_rows, classification_rows = _run_one_horizon(
		ohlcv, horizon=5, cost_per_trade=0.0001, n_resamples=200, seed=42
	)
	assert {row["model"] for row in regression_rows} == {"linear", "xgboost"}
	assert {row["model"] for row in classification_rows} == {"logistic", "xgboost"}
	for row in regression_rows:
		assert row["rigor_tier"] == RIGOR_TIER[5]
		assert np.isfinite(row["oos_r2_vs_drift"])
		assert isinstance(row["beats_drift"], bool)
	for row in classification_rows:
		assert 0.0 <= row["pooled_accuracy"] <= 1.0
		assert 0.0 <= row["direction_base_rate"] <= 1.0
		assert row["n_windows"] >= 1

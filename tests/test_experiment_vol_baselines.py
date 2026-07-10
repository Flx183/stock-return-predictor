import numpy as np
import pandas as pd
import pytest

from src.experiment_vol_baselines import (
	_compare,
	run_ablation_experiment,
	run_har_experiment,
)
from src.walkforward import block_bootstrap_indices


def make_ohlcv(n_days=2200):
	dates = pd.bdate_range("2010-01-01", periods=n_days)
	step = np.arange(n_days, dtype=float)
	# A slow trend with volatility that clusters, so realized-vol forecasting is
	# non-trivial and HAR-style baselines have something to grip.
	regime = 1.0 + 0.6 * np.sin(step / 120.0) ** 2
	shocks = regime * np.sin(step / 3.0) * 0.6
	close = pd.Series(100.0 * np.cumprod(1.0 + 0.0003 + 0.004 * shocks / regime.mean()), index=dates)
	open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + 0.0005)
	high = pd.concat([open_, close], axis=1).max(axis=1) * (1.0 + 0.003)
	low = pd.concat([open_, close], axis=1).min(axis=1) * (1.0 - 0.003)
	volume = pd.Series(1_000_000.0 + step * 300.0 + (step % 17) * 5_000.0, index=dates)
	return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def test_compare_rewards_a_better_forecast_and_rejects_a_tie():
	y_true = np.linspace(0.1, 0.3, 120)
	baseline = y_true + np.tile([0.05, -0.05], 60)
	indices = block_bootstrap_indices(len(y_true), block_length=5, n_resamples=500, seed=1)

	perfect = _compare("m", y_true, y_true, baseline, indices)
	assert perfect["rmse_improvement"] == pytest.approx(1.0)
	assert perfect["oos_r2_vs_baseline"] == pytest.approx(1.0)
	assert perfect["reduction_excludes_zero_low"] is True

	tie = _compare("m", y_true, baseline, baseline, indices)
	assert tie["rmse_improvement"] == pytest.approx(0.0)
	assert tie["oos_r2_vs_baseline"] == pytest.approx(0.0)
	assert tie["reduction_excludes_zero_low"] is False


def test_har_experiment_is_internally_consistent():
	result = run_har_experiment(ohlcv=make_ohlcv(), n_resamples=200)
	rows = result["comparisons"].set_index("comparison")
	assert set(rows.index) == {"model_vs_har", "har_vs_persistence", "model_vs_persistence"}
	# The same model forecast underlies both model rows, so its RMSE must match.
	assert rows.loc["model_vs_har", "model_rmse"] == pytest.approx(
		rows.loc["model_vs_persistence", "model_rmse"]
	)
	assert isinstance(result["beats_har"], bool)


def test_ablation_nesting_orders_the_persistence_baseline():
	result = run_ablation_experiment(ohlcv=make_ohlcv(), n_resamples=200)
	rows = result["comparisons"].set_index("comparison")
	assert set(rows.index) == {"full_vs_vollag", "full_vs_persistence", "vollag_vs_persistence"}
	# Full and vol-lags-only are scored against the same persistence forecast.
	assert rows.loc["full_vs_persistence", "baseline_rmse"] == pytest.approx(
		rows.loc["vollag_vs_persistence", "baseline_rmse"]
	)
	assert isinstance(result["full_adds_signal"], bool)

import numpy as np
import pandas as pd
import pytest

from src.experiment_vol_targeting import (
	POSITION_CAP,
	TARGET_ANNUAL_VOL,
	_annualized_sharpe,
	_vol_target_weights,
	run_vol_targeting_experiment,
)


def make_ohlcv(n_days=2200):
	dates = pd.bdate_range("2010-01-01", periods=n_days)
	step = np.arange(n_days, dtype=float)
	regime = 1.0 + 0.6 * np.sin(step / 120.0) ** 2
	shocks = regime * np.sin(step / 3.0) * 0.6
	close = pd.Series(100.0 * np.cumprod(1.0 + 0.0003 + 0.004 * shocks / regime.mean()), index=dates)
	open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + 0.0005)
	high = pd.concat([open_, close], axis=1).max(axis=1) * (1.0 + 0.003)
	low = pd.concat([open_, close], axis=1).min(axis=1) * (1.0 - 0.003)
	volume = pd.Series(1_000_000.0 + step * 300.0 + (step % 17) * 5_000.0, index=dates)
	return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def test_vol_target_weights_scale_inversely_and_clip():
	forecast = np.array([TARGET_ANNUAL_VOL, 2.0 * TARGET_ANNUAL_VOL, 0.01 * TARGET_ANNUAL_VOL, -0.2, 0.0])
	weights, floored = _vol_target_weights(forecast, target_vol=TARGET_ANNUAL_VOL, cap=POSITION_CAP)

	assert weights[0] == pytest.approx(1.0)          # forecast == target -> unit exposure
	assert weights[1] == pytest.approx(0.5)          # double the vol -> half the exposure
	assert weights[2] == pytest.approx(POSITION_CAP)  # tiny vol -> capped, not unbounded
	# Non-positive forecasts are floored (never negative exposure) and flagged.
	assert weights[3] == pytest.approx(POSITION_CAP)
	assert weights[4] == pytest.approx(POSITION_CAP)
	assert floored.tolist() == [False, False, False, True, True]
	assert np.all((weights >= 0.0) & (weights <= POSITION_CAP))


def test_annualized_sharpe_matches_hand_computation():
	returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
	expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
	assert _annualized_sharpe(returns) == pytest.approx(expected)


def test_all_sources_share_the_same_buy_and_hold_benchmark():
	result = run_vol_targeting_experiment(ohlcv=make_ohlcv(), n_resamples=200)
	comparisons = result["comparisons"]
	assert set(comparisons["forecast_source"]) == {"persistence", "har_rv", "model_26_feature"}

	# The benchmark is buy-and-hold over the same out-of-sample days for every
	# source, so its Sharpe must be identical across rows.
	assert comparisons["buy_and_hold_sharpe"].nunique() == 1
	assert (comparisons["floored_forecast_fraction"].between(0.0, 1.0)).all()
	assert (comparisons["avg_position"] >= 0.0).all()
	assert comparisons["n_oos_days"].iloc[0] > 0

import numpy as np
import pytest

from src.experiment_volatility import _out_of_sample_r2, _qlike, _rmse


def test_rmse_is_zero_for_a_perfect_forecast():
	y = np.array([0.10, 0.20, 0.30])
	assert _rmse(y, y) == pytest.approx(0.0)


def test_rmse_matches_hand_computation():
	y_true = np.array([0.10, 0.20])
	y_pred = np.array([0.13, 0.16])
	assert _rmse(y_true, y_pred) == pytest.approx(np.sqrt((0.03**2 + 0.04**2) / 2))


def test_qlike_is_zero_for_a_perfect_forecast():
	y = np.array([0.10, 0.25, 0.40])
	assert _qlike(y, y) == pytest.approx(0.0, abs=1e-12)


def test_out_of_sample_r2_rewards_beating_persistence():
	y_true = np.array([0.10, 0.20, 0.30])
	persistence = np.array([0.05, 0.25, 0.40])

	assert _out_of_sample_r2(y_true, y_true, persistence) == pytest.approx(1.0)
	assert _out_of_sample_r2(y_true, persistence, persistence) == pytest.approx(0.0)

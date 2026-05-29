import pandas as pd

from src.experiment_direction import _compare
from src.walkforward import block_bootstrap_indices


def test_compare_flags_a_strict_economic_improvement():
	a = pd.Series([0.011, 0.012, 0.009, 0.010, 0.013, 0.008])
	b = pd.Series([0.001, 0.002, -0.001, 0.000, 0.003, -0.002])
	indices = block_bootstrap_indices(len(a), block_length=2, n_resamples=300, seed=42)

	result = _compare("a_vs_b", a, b, indices)

	assert result["mean_daily_excess"] > 0
	assert result["ci_low"] > 0
	assert result["excludes_zero_low"]
	assert result["prob_positive"] == 1.0
	assert result["total_compounded_excess"] > 0


def test_compare_returns_null_when_series_are_identical():
	a = pd.Series([0.01, -0.02, 0.03, 0.00, 0.015, -0.01])
	indices = block_bootstrap_indices(len(a), block_length=2, n_resamples=300, seed=42)

	result = _compare("a_vs_a", a, a.copy(), indices)

	assert result["mean_daily_excess"] == 0.0
	assert not result["excludes_zero_low"]
	assert result["prob_positive"] == 0.0
	assert result["total_compounded_excess"] == 0.0

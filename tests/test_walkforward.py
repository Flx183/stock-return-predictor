import numpy as np
import pandas as pd
import pytest

from src.walkforward import (
	block_bootstrap_indices,
	bootstrap_mean,
	expanding_window_folds,
	run_walk_forward,
	summarize_bootstrap,
)


def make_dataset(n_rows=12):
	target_timestamps = pd.bdate_range("2024-01-02", periods=n_rows)
	feature_timestamps = pd.bdate_range("2024-01-01", periods=n_rows)
	step = np.arange(n_rows, dtype=float)
	return pd.DataFrame(
		{
			"feature_timestamp": feature_timestamps,
			"target_timestamp": target_timestamps,
			"target_direction": (step % 2 == 0).astype(int),
			"feature_a": step,
		},
		index=target_timestamps,
	)


def test_expanding_window_folds_are_contiguous_and_expanding():
	folds = expanding_window_folds(12, n_folds=3, test_size=2, min_train_size=2)

	assert [f.fold_id for f in folds] == [0, 1, 2]
	assert [(f.test_start, f.test_end) for f in folds] == [(6, 8), (8, 10), (10, 12)]
	# Train is everything before the test block: expanding, never future.
	for fold in folds:
		assert fold.train_start == 0
		assert fold.train_end == fold.test_start


def test_expanding_window_folds_reject_too_many_rows_requested():
	with pytest.raises(ValueError, match="not enough rows"):
		expanding_window_folds(10, n_folds=8, test_size=126)


def test_expanding_window_folds_reject_small_first_train():
	with pytest.raises(ValueError, match="minimum"):
		expanding_window_folds(12, n_folds=3, test_size=2, min_train_size=8)


def test_run_walk_forward_pools_out_of_sample_rows_in_order():
	dataset = make_dataset(12)
	folds = expanding_window_folds(12, n_folds=3, test_size=2, min_train_size=2)
	seen_train_sizes = []

	def fit_predict(X_train, y_train, X_test):
		seen_train_sizes.append(len(X_train))
		return X_test["feature_a"].to_numpy()

	pooled = run_walk_forward(dataset, fit_predict, ["feature_a"], "target_direction", folds=folds)

	assert seen_train_sizes == [6, 8, 10]  # expanding training window
	assert pooled["fold_id"].tolist() == [0, 0, 1, 1, 2, 2]
	assert pooled["prediction"].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
	assert pooled["target_timestamp"].tolist() == dataset["target_timestamp"].iloc[6:].tolist()
	assert pooled["y_true"].tolist() == dataset["target_direction"].iloc[6:].tolist()


def test_embargo_trims_training_rows_without_touching_the_test_block():
	dataset = make_dataset(12)
	folds = expanding_window_folds(12, n_folds=3, test_size=2, min_train_size=2)
	seen_train_sizes = []

	def fit_predict(X_train, y_train, X_test):
		seen_train_sizes.append(len(X_train))
		return X_test["feature_a"].to_numpy()

	pooled = run_walk_forward(
		dataset, fit_predict, ["feature_a"], "target_direction", folds=folds, embargo=1
	)

	assert seen_train_sizes == [5, 7, 9]  # one fewer training row per fold than embargo=0
	assert pooled["prediction"].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]  # test block unchanged


def test_block_bootstrap_indices_preserve_block_structure():
	indices = block_bootstrap_indices(10, block_length=5, n_resamples=50, seed=1)

	assert indices.shape == (50, 10)
	assert indices.min() >= 0 and indices.max() <= 9
	# Each length-5 block is contiguous: consecutive positions differ by exactly 1.
	for row in indices:
		assert np.all(np.diff(row[:5]) == 1)
		assert np.all(np.diff(row[5:]) == 1)


def test_block_bootstrap_indices_are_seed_deterministic():
	a = block_bootstrap_indices(40, block_length=7, n_resamples=100, seed=42)
	b = block_bootstrap_indices(40, block_length=7, n_resamples=100, seed=42)
	c = block_bootstrap_indices(40, block_length=7, n_resamples=100, seed=7)

	assert np.array_equal(a, b)
	assert not np.array_equal(a, c)


def test_block_bootstrap_indices_reject_block_longer_than_series():
	with pytest.raises(ValueError, match="block_length"):
		block_bootstrap_indices(5, block_length=10)


def test_summarize_bootstrap_flags_strictly_positive_ci():
	indices = block_bootstrap_indices(40, block_length=4, n_resamples=500, seed=42)

	positive = summarize_bootstrap(bootstrap_mean(np.full(40, 0.5), indices), point_estimate=0.5)
	assert positive["ci_low"] == pytest.approx(0.5)
	assert positive["excludes_zero_low"]
	assert positive["prob_positive"] == 1.0

	zero = summarize_bootstrap(bootstrap_mean(np.zeros(40), indices), point_estimate=0.0)
	assert not zero["excludes_zero_low"]
	assert zero["prob_positive"] == 0.0

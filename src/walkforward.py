"""
Walk-forward evaluation harness shared by the pre-registered experiments.

This module provides the mechanics only, with no model or success threshold
baked in:

- `expanding_window_folds`: contiguous out-of-sample blocks, each trained on all
  rows strictly before it (expanding window, no future leakage).
- `run_walk_forward`: re-fit a caller-supplied `fit_predict` per fold and collect
  pooled out-of-sample predictions with their leakage-safe timestamps.
- `block_bootstrap_indices` / `bootstrap_mean` / `summarize_bootstrap`: a moving
  (overlapping) block bootstrap that resamples contiguous blocks of the pooled
  out-of-sample daily series, preserving serial correlation that an i.i.d.
  bootstrap would destroy.

Thresholds and model choices live in the experiment modules and in
PREREGISTRATION.md, never here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_N_FOLDS = 8
DEFAULT_TEST_SIZE = 126
DEFAULT_MIN_TRAIN_SIZE = 756
DEFAULT_BLOCK_LENGTH = 21
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_CI = 0.90


@dataclass(frozen=True)
class Fold:
	fold_id: int
	train_start: int
	train_end: int  # exclusive; also the first out-of-sample row
	test_start: int
	test_end: int  # exclusive


def expanding_window_folds(
	n_rows,
	n_folds=DEFAULT_N_FOLDS,
	test_size=DEFAULT_TEST_SIZE,
	min_train_size=DEFAULT_MIN_TRAIN_SIZE,
):
	"""
	Tile the most recent `n_folds * test_size` rows into contiguous test blocks.

	Fold k trains on rows [0, test_start_k) and tests on the next `test_size`
	rows. Training is therefore expanding: each fold sees every row before its
	test block, including earlier folds' test rows, and never a future row.
	"""
	if n_folds <= 0 or test_size <= 0:
		raise ValueError("n_folds and test_size must be positive.")
	total_test = n_folds * test_size
	first_test_start = n_rows - total_test
	if first_test_start <= 0:
		raise ValueError(
			f"not enough rows: {n_rows} rows cannot supply {n_folds} folds of "
			f"{test_size} test rows plus a training set."
		)
	if min_train_size is not None and first_test_start < min_train_size:
		raise ValueError(
			f"first fold would train on {first_test_start} rows, below the "
			f"minimum of {min_train_size}."
		)

	folds = []
	for k in range(n_folds):
		test_start = first_test_start + k * test_size
		folds.append(
			Fold(
				fold_id=k,
				train_start=0,
				train_end=test_start,
				test_start=test_start,
				test_end=test_start + test_size,
			)
		)
	return folds


def _prepare_dataset(dataset, feature_columns, target_column, metadata_columns):
	required = {"feature_timestamp", "target_timestamp", target_column, *feature_columns}
	missing = required - set(dataset.columns)
	if missing:
		raise ValueError(f"dataset is missing column(s): {', '.join(sorted(missing))}")

	data = dataset.copy()
	data.index = data.index.rename(None)
	data["feature_timestamp"] = pd.to_datetime(data["feature_timestamp"], errors="coerce")
	data["target_timestamp"] = pd.to_datetime(data["target_timestamp"], errors="coerce")
	if data[["feature_timestamp", "target_timestamp"]].isna().any().any():
		raise ValueError("dataset contains invalid feature or target timestamp values.")
	if data["target_timestamp"].duplicated().any():
		raise ValueError("dataset cannot contain duplicate target timestamps.")

	data = data.sort_values("target_timestamp").reset_index(drop=True)
	if not (data["feature_timestamp"] < data["target_timestamp"]).all():
		raise ValueError("feature_timestamp must be strictly earlier than target_timestamp.")
	if data[list(feature_columns)].isna().any().any():
		raise ValueError("feature matrix cannot contain missing values.")
	if data[target_column].isna().any():
		raise ValueError(f"target column '{target_column}' cannot contain missing values.")
	for column in metadata_columns:
		if column not in data.columns:
			raise ValueError(f"metadata column '{column}' is not in the dataset.")
	return data


def run_walk_forward(
	dataset,
	fit_predict,
	feature_columns,
	target_column,
	folds=None,
	metadata_columns=(),
	embargo=0,
):
	"""
	Run an expanding-window walk-forward and pool the out-of-sample predictions.

	`fit_predict(X_train, y_train, X_test)` must return one prediction per test
	row (an up-probability for classification, a forecast value for regression).
	The returned frame carries `fold_id`, both leakage-safe timestamps, the true
	target as `y_true`, the model `prediction`, and any requested metadata.

	`embargo` drops that many rows from the end of each fold's training set. It
	is needed when the target looks forward over several days: without it the
	final training rows carry labels whose window crosses into the test block,
	leaking a few future observations into the fit. A 1-day target needs no
	embargo; a k-day forward target needs `embargo = k - 1`.
	"""
	if embargo < 0:
		raise ValueError("embargo cannot be negative.")
	feature_columns = list(feature_columns)
	metadata_columns = list(metadata_columns)
	data = _prepare_dataset(dataset, feature_columns, target_column, metadata_columns)
	if folds is None:
		folds = expanding_window_folds(len(data))

	carry_columns = ["feature_timestamp", "target_timestamp", *metadata_columns]
	pooled_frames = []
	for fold in folds:
		train_end = fold.train_end - embargo
		if train_end <= fold.train_start:
			raise ValueError(f"fold {fold.fold_id}: embargo removes the entire training set.")
		train = data.iloc[fold.train_start:train_end]
		test = data.iloc[fold.test_start:fold.test_end]
		X_train = train.loc[:, feature_columns].astype(float)
		y_train = train[target_column]
		X_test = test.loc[:, feature_columns].astype(float)

		prediction = np.asarray(fit_predict(X_train, y_train, X_test), dtype=float)
		if prediction.shape != (len(test),):
			raise ValueError(
				f"fold {fold.fold_id}: fit_predict returned {prediction.shape}, "
				f"expected ({len(test)},)."
			)

		fold_frame = test.loc[:, carry_columns].copy()
		fold_frame.insert(0, "fold_id", fold.fold_id)
		fold_frame["y_true"] = test[target_column].to_numpy()
		fold_frame["prediction"] = prediction
		pooled_frames.append(fold_frame)

	pooled = pd.concat(pooled_frames, ignore_index=True)
	pooled = pooled.sort_values("target_timestamp").reset_index(drop=True)
	if pooled["target_timestamp"].duplicated().any():
		raise ValueError("walk-forward folds overlap: duplicate out-of-sample dates.")
	return pooled


def describe_folds(dataset, folds, target_column, embargo=0):
	"""One row per fold: train size and the train/test date boundaries."""
	data = _prepare_dataset(dataset, [], target_column, [])
	rows = []
	for fold in folds:
		train = data.iloc[fold.train_start:fold.train_end - embargo]
		test = data.iloc[fold.test_start:fold.test_end]
		rows.append(
			{
				"fold_id": fold.fold_id,
				"n_train": int(len(train)),
				"n_test": int(len(test)),
				"train_start_target": train["target_timestamp"].iloc[0].date().isoformat(),
				"train_end_target": train["target_timestamp"].iloc[-1].date().isoformat(),
				"test_start_target": test["target_timestamp"].iloc[0].date().isoformat(),
				"test_end_target": test["target_timestamp"].iloc[-1].date().isoformat(),
			}
		)
	return pd.DataFrame(rows)


def block_bootstrap_indices(
	n,
	block_length=DEFAULT_BLOCK_LENGTH,
	n_resamples=DEFAULT_N_RESAMPLES,
	seed=DEFAULT_SEED,
):
	"""
	Moving-block bootstrap index matrix of shape (n_resamples, n).

	Each resample concatenates ceil(n / block_length) overlapping contiguous
	blocks of length `block_length` drawn with replacement, then truncates to n.
	Applying the same matrix to two paired series preserves their day-by-day
	pairing, the way the original paired bootstrap did, while also preserving
	within-block serial correlation.
	"""
	if n <= 0:
		raise ValueError("n must be positive.")
	if block_length <= 0:
		raise ValueError("block_length must be positive.")
	if block_length > n:
		raise ValueError("block_length cannot exceed the series length.")
	if n_resamples <= 0:
		raise ValueError("n_resamples must be positive.")

	rng = np.random.default_rng(seed)
	n_blocks = int(np.ceil(n / block_length))
	max_start = n - block_length  # inclusive
	starts = rng.integers(0, max_start + 1, size=(n_resamples, n_blocks))
	offsets = np.arange(block_length)
	indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, n_blocks * block_length)
	return indices[:, :n]


def bootstrap_mean(values, indices):
	"""Mean of `values` along each bootstrap resample (one mean per row of `indices`)."""
	values = np.asarray(values, dtype=float)
	return values[indices].mean(axis=1)


def summarize_bootstrap(samples, point_estimate, ci=DEFAULT_CI):
	"""
	Reduce a bootstrap statistic array to a point estimate, a CI, and P(> 0).

	`excludes_zero_low` is the pre-registered decision flag: the CI lower bound
	is strictly positive, i.e. the statistic is a significant improvement.
	"""
	samples = np.asarray(samples, dtype=float)
	lower_q = (1.0 - ci) / 2.0
	upper_q = 1.0 - lower_q
	ci_low = float(np.quantile(samples, lower_q))
	ci_high = float(np.quantile(samples, upper_q))
	return {
		"point_estimate": float(point_estimate),
		"ci_level": float(ci),
		"ci_low": ci_low,
		"median": float(np.quantile(samples, 0.5)),
		"ci_high": ci_high,
		"prob_positive": float(np.mean(samples > 0.0)),
		"excludes_zero_low": bool(ci_low > 0.0),
	}

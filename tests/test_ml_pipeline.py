import numpy as np
import pandas as pd
import pytest

from src.ml_pipeline import (
	build_feature_correlation_report,
	chronological_train_test_split,
	fit_logistic_regression,
)


def make_supervised_dataset(n_rows=12):
	target_timestamps = pd.bdate_range("2024-01-02", periods=n_rows)
	feature_timestamps = pd.bdate_range("2024-01-01", periods=n_rows)
	step = np.arange(n_rows, dtype=float)
	return pd.DataFrame(
		{
			"feature_timestamp": feature_timestamps,
			"target_timestamp": target_timestamps,
			"target_return": np.where(step % 2 == 0, 0.01, -0.01),
			"target_direction": (step % 2 == 0).astype(int),
			"feature_a": step,
			"feature_b": step * 2.0 + 1.0,
		},
		index=target_timestamps,
	)


def test_chronological_split_uses_earliest_rows_for_train_and_latest_for_test():
	dataset = make_supervised_dataset()
	split = chronological_train_test_split(
		dataset,
		train_size=9,
		feature_columns=("feature_a", "feature_b"),
	)

	assert split.X_train.index.tolist() == dataset.index[:9].tolist()
	assert split.X_test.index.tolist() == dataset.index[9:].tolist()
	assert split.train_metadata["target_timestamp"].max() < split.test_metadata["target_timestamp"].min()
	assert (split.train_metadata["feature_timestamp"] < split.train_metadata["target_timestamp"]).all()
	assert (split.test_metadata["feature_timestamp"] < split.test_metadata["target_timestamp"]).all()


def test_default_split_uses_80_percent_train_and_20_percent_test():
	dataset = make_supervised_dataset(n_rows=10)
	split = chronological_train_test_split(
		dataset,
		feature_columns=("feature_a", "feature_b"),
	)

	assert split.X_train.index.tolist() == dataset.index[:8].tolist()
	assert split.X_test.index.tolist() == dataset.index[8:].tolist()


def test_scaler_is_fit_on_train_rows_only_then_applied_to_test():
	dataset = make_supervised_dataset()
	split = chronological_train_test_split(
		dataset,
		train_size=9,
		feature_columns=("feature_a", "feature_b"),
	)
	fitted = fit_logistic_regression(split)

	np.testing.assert_allclose(fitted.scaler.mean_, split.X_train.mean().to_numpy())
	np.testing.assert_allclose(fitted.scaler.var_, split.X_train.var(ddof=0).to_numpy())
	assert not np.allclose(fitted.scaler.mean_, dataset[["feature_a", "feature_b"]].mean().to_numpy())
	np.testing.assert_allclose(fitted.X_train_scaled.mean(axis=0).to_numpy(), np.zeros(2), atol=1e-12)
	assert not np.allclose(fitted.X_test_scaled.mean(axis=0).to_numpy(), np.zeros(2), atol=1e-3)


def test_feature_correlation_report_uses_train_rows_only():
	dataset = make_supervised_dataset(n_rows=10)
	dataset.loc[dataset.index[8:], "feature_b"] = -dataset.loc[dataset.index[8:], "feature_a"]
	split = chronological_train_test_split(
		dataset,
		train_size=8,
		feature_columns=("feature_a", "feature_b"),
	)

	report = build_feature_correlation_report(split, min_abs_correlation=0.99)

	assert len(report) == 1
	row = report.iloc[0]
	assert row["feature_a"] == "feature_a"
	assert row["feature_b"] == "feature_b"
	assert row["correlation"] == pytest.approx(1.0)
	assert row["n_train"] == 8

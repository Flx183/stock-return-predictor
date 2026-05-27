from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
	from src.baseline import load_spy_ohlcv
	from src.features import FEATURE_COLUMNS, build_leakage_safe_features
except ModuleNotFoundError:  # Allows `python src/ml_pipeline.py`.
	from baseline import load_spy_ohlcv
	from features import FEATURE_COLUMNS, build_leakage_safe_features


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = REPO_ROOT / "data" / "ml_supervised_dataset.csv"
PREDICTIONS_FILE = REPO_ROOT / "data" / "ml_test_predictions.csv"
METRICS_FILE = REPO_ROOT / "data" / "ml_test_metrics.csv"
CORRELATION_FILE = REPO_ROOT / "data" / "train_feature_correlations.csv"
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_CORRELATION_THRESHOLD = 0.8


@dataclass(frozen=True)
class ChronologicalSplit:
	X_train: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_test: pd.Series
	train_metadata: pd.DataFrame
	test_metadata: pd.DataFrame
	feature_columns: tuple
	target_column: str


@dataclass(frozen=True)
class FittedLogisticModel:
	scaler: StandardScaler
	classifier: LogisticRegression
	split: ChronologicalSplit
	X_train_scaled: pd.DataFrame
	X_test_scaled: pd.DataFrame


def _resolve_train_count(n_rows, train_size):
	if isinstance(train_size, bool):
		raise TypeError("train_size must be a float fraction or integer row count.")
	if isinstance(train_size, float):
		if not 0.0 < train_size < 1.0:
			raise ValueError("float train_size must be between 0 and 1.")
		train_count = int(np.floor(n_rows * train_size))
	elif isinstance(train_size, int):
		train_count = train_size
	else:
		raise TypeError("train_size must be a float fraction or integer row count.")

	if train_count <= 0 or train_count >= n_rows:
		raise ValueError("train_size must leave at least one train row and one test row.")
	return train_count


def chronological_train_test_split(
	dataset,
	train_size=DEFAULT_TRAIN_FRACTION,
	feature_columns=None,
	target_column="target_direction",
):
	"""
	Split earliest rows into train and latest rows into test.

	This function intentionally has no shuffle option. The input row timestamp is
	`target_timestamp`; each row also carries the prior `feature_timestamp` so
	the train/test boundary can be audited directly.
	"""
	if dataset.empty:
		raise ValueError("dataset cannot be empty.")

	feature_columns = tuple(feature_columns or FEATURE_COLUMNS)
	required_columns = {"feature_timestamp", "target_timestamp", target_column, *feature_columns}
	missing_columns = required_columns - set(dataset.columns)
	if missing_columns:
		raise ValueError(f"dataset is missing column(s): {', '.join(sorted(missing_columns))}")

	data = dataset.copy()
	data.index = data.index.rename(None)
	data["feature_timestamp"] = pd.to_datetime(data["feature_timestamp"], errors="coerce")
	data["target_timestamp"] = pd.to_datetime(data["target_timestamp"], errors="coerce")
	if data[["feature_timestamp", "target_timestamp"]].isna().any().any():
		raise ValueError("dataset contains invalid feature or target timestamp values.")
	if data["target_timestamp"].duplicated().any():
		raise ValueError("dataset cannot contain duplicate target timestamps.")
	data = data.sort_values("target_timestamp")
	data.index = pd.DatetimeIndex(data["target_timestamp"], name="target_timestamp")
	if not (data["feature_timestamp"] < data["target_timestamp"]).all():
		raise ValueError("feature_timestamp must be strictly earlier than target_timestamp.")

	data = data.dropna(subset=[target_column])
	train_count = _resolve_train_count(len(data), train_size)
	train = data.iloc[:train_count].copy()
	test = data.iloc[train_count:].copy()
	if train["target_timestamp"].max() >= test["target_timestamp"].min():
		raise ValueError("chronological split failed: train dates must end before test dates begin.")

	X_train = train.loc[:, feature_columns].astype(float)
	X_test = test.loc[:, feature_columns].astype(float)
	y_train = pd.to_numeric(train[target_column], errors="raise")
	y_test = pd.to_numeric(test[target_column], errors="raise")
	if X_train.isna().any().any() or X_test.isna().any().any():
		raise ValueError("feature matrix cannot contain missing values.")
	if y_train.isna().any() or y_test.isna().any():
		raise ValueError("target cannot contain missing values.")

	metadata_columns = [
		column
		for column in data.columns
		if column == "feature_timestamp" or column.startswith("target_")
	]
	return ChronologicalSplit(
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		train_metadata=train.loc[:, metadata_columns],
		test_metadata=test.loc[:, metadata_columns],
		feature_columns=feature_columns,
		target_column=target_column,
	)


def fit_logistic_regression(split, C=1.0, max_iter=1000):
	if split.y_train.nunique() < 2:
		raise ValueError("training target must contain both classes.")

	y_train = split.y_train.astype(int)
	scaler = StandardScaler()
	X_train_scaled = pd.DataFrame(
		scaler.fit_transform(split.X_train),
		index=split.X_train.index,
		columns=split.feature_columns,
	)
	X_test_scaled = pd.DataFrame(
		scaler.transform(split.X_test),
		index=split.X_test.index,
		columns=split.feature_columns,
	)

	classifier = LogisticRegression(C=C, max_iter=max_iter)
	classifier.fit(X_train_scaled, y_train)
	return FittedLogisticModel(
		scaler=scaler,
		classifier=classifier,
		split=split,
		X_train_scaled=X_train_scaled,
		X_test_scaled=X_test_scaled,
	)


def predict_test_set(fitted_model, probability_threshold=0.5):
	probability_up = fitted_model.classifier.predict_proba(fitted_model.X_test_scaled)[:, 1]
	predicted_direction = (probability_up >= probability_threshold).astype(int)
	predictions = fitted_model.split.test_metadata.copy()
	predictions["probability_up"] = probability_up
	predictions["predicted_direction"] = predicted_direction
	predictions["actual_direction"] = fitted_model.split.y_test.to_numpy()
	return predictions


def build_feature_correlation_report(split, min_abs_correlation=DEFAULT_CORRELATION_THRESHOLD):
	if not 0.0 <= min_abs_correlation <= 1.0:
		raise ValueError("min_abs_correlation must be between 0 and 1.")

	report_columns = [
		"feature_a",
		"feature_b",
		"correlation",
		"abs_correlation",
		"n_train",
		"train_start_target_timestamp",
		"train_end_target_timestamp",
	]
	correlation = split.X_train.corr(method="pearson")
	rows = []
	for left_index, feature_a in enumerate(split.feature_columns):
		for feature_b in split.feature_columns[left_index + 1:]:
			correlation_value = correlation.loc[feature_a, feature_b]
			if pd.isna(correlation_value):
				continue
			abs_correlation = abs(correlation_value)
			if abs_correlation < min_abs_correlation:
				continue
			rows.append(
				{
					"feature_a": feature_a,
					"feature_b": feature_b,
					"correlation": float(correlation_value),
					"abs_correlation": float(abs_correlation),
					"n_train": int(len(split.X_train)),
					"train_start_target_timestamp": (
						split.train_metadata["target_timestamp"].iloc[0].date().isoformat()
					),
					"train_end_target_timestamp": (
						split.train_metadata["target_timestamp"].iloc[-1].date().isoformat()
					),
				}
			)

	report = pd.DataFrame(rows, columns=report_columns)
	if report.empty:
		return report
	return report.sort_values(["abs_correlation", "feature_a", "feature_b"], ascending=[False, True, True])


def compute_test_metrics(split, predictions):
	probability_up = predictions["probability_up"].to_numpy()
	predicted_direction = predictions["predicted_direction"].to_numpy()
	y_test = split.y_test.to_numpy()
	metrics = {
		"accuracy": float(accuracy_score(y_test, predicted_direction)),
		"balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_direction)),
		"log_loss": float(log_loss(y_test, probability_up, labels=[0, 1])),
		"test_positive_rate": float(np.mean(y_test)),
		"predicted_positive_rate": float(np.mean(predicted_direction)),
		"n_train": int(len(split.X_train)),
		"n_test": int(len(split.X_test)),
		"train_fraction": float(len(split.X_train) / (len(split.X_train) + len(split.X_test))),
		"test_fraction": float(len(split.X_test) / (len(split.X_train) + len(split.X_test))),
		"train_start_target_timestamp": split.train_metadata["target_timestamp"].iloc[0].date().isoformat(),
		"train_end_target_timestamp": split.train_metadata["target_timestamp"].iloc[-1].date().isoformat(),
		"test_start_target_timestamp": split.test_metadata["target_timestamp"].iloc[0].date().isoformat(),
		"test_end_target_timestamp": split.test_metadata["target_timestamp"].iloc[-1].date().isoformat(),
		"train_end_feature_timestamp": split.train_metadata["feature_timestamp"].iloc[-1].date().isoformat(),
		"test_start_feature_timestamp": split.test_metadata["feature_timestamp"].iloc[0].date().isoformat(),
		"n_features": int(len(split.feature_columns)),
	}
	if split.y_test.nunique() == 2:
		metrics["roc_auc"] = float(roc_auc_score(y_test, probability_up))
	else:
		metrics["roc_auc"] = float("nan")
	return metrics


def run_pipeline(train_size=DEFAULT_TRAIN_FRACTION):
	ohlcv = load_spy_ohlcv()
	dataset = build_leakage_safe_features(ohlcv)
	split = chronological_train_test_split(dataset, train_size=train_size)
	fitted_model = fit_logistic_regression(split)
	predictions = predict_test_set(fitted_model)
	correlation_report = build_feature_correlation_report(split)
	metrics = compute_test_metrics(split, predictions)
	metrics["correlation_threshold"] = float(DEFAULT_CORRELATION_THRESHOLD)
	metrics["n_high_abs_feature_correlations"] = int(len(correlation_report))
	return dataset, predictions, metrics, correlation_report


def main():
	try:
		dataset, predictions, metrics, correlation_report = run_pipeline()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
	dataset.to_csv(DATASET_FILE, index=False)
	predictions.to_csv(PREDICTIONS_FILE, index=False)
	pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)
	correlation_report.to_csv(CORRELATION_FILE, index=False)

	print("Leakage-safe SPY direction model:")
	print(f"  Supervised dataset: {DATASET_FILE}")
	print(f"  Test predictions: {PREDICTIONS_FILE}")
	print(f"  Test metrics: {METRICS_FILE}")
	print(f"  Train-only feature correlations: {CORRELATION_FILE}")
	print(
		"  Chronological split: "
		f"{metrics['n_train']} train rows ({metrics['train_fraction']:.1%}), "
		f"{metrics['n_test']} test rows ({metrics['test_fraction']:.1%})"
	)
	print(
		"  Train target dates: "
		f"{metrics['train_start_target_timestamp']} to {metrics['train_end_target_timestamp']}"
	)
	print(
		"  Test target dates: "
		f"{metrics['test_start_target_timestamp']} to {metrics['test_end_target_timestamp']}"
	)
	print(f"  Accuracy: {metrics['accuracy']:.4f}")
	print(f"  ROC AUC: {metrics['roc_auc']:.4f}" if not pd.isna(metrics["roc_auc"]) else "  ROC AUC: n/a")


if __name__ == "__main__":
	main()

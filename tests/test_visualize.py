import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.visualize import (
	plot_confusion_matrix,
	plot_drawdowns,
	plot_equity_curves,
	plot_monte_carlo_excess_return,
	plot_prediction_probabilities,
	plot_train_feature_correlation_heatmap,
)


def assert_png_written(path):
	assert path.exists()
	assert path.stat().st_size > 0


def make_backtest_frame():
	dates = pd.bdate_range("2024-01-01", periods=5)
	strategy_returns = pd.Series([0.0, 0.01, -0.02, 0.015, 0.005], index=dates)
	benchmark_returns = pd.Series([0.0, 0.008, -0.01, 0.012, 0.002], index=dates)
	return pd.DataFrame(
		{
			"Date": dates,
			"equity_curve": (1.0 + strategy_returns).cumprod().to_numpy(),
			"buy_and_hold_equity_curve": (1.0 + benchmark_returns).cumprod().to_numpy(),
		}
	)


def make_metrics_frame():
	return pd.DataFrame(
		[
			{
				"strategy_total_return": 0.01,
				"buy_and_hold_total_return": 0.02,
				"excess_total_return": -0.01,
			}
		]
	)


def make_predictions_frame():
	return pd.DataFrame(
		{
			"target_timestamp": pd.bdate_range("2024-01-01", periods=4),
			"probability_up": [0.6, 0.4, 0.7, 0.3],
			"predicted_direction": [1, 0, 1, 0],
			"actual_direction": [1, 1, 0, 0],
		}
	)


def make_dataset_frame(n_rows=20):
	dates = pd.bdate_range("2024-01-02", periods=n_rows)
	feature_dates = pd.bdate_range("2024-01-01", periods=n_rows)
	data = {
		"feature_timestamp": feature_dates,
		"target_timestamp": dates,
		"target_direction": [0, 1] * (n_rows // 2),
	}
	for index, column in enumerate(FEATURE_COLUMNS):
		data[column] = np.arange(n_rows, dtype=float) + index
	return pd.DataFrame(data)


def test_visualization_plots_write_png_files(tmp_path):
	backtest = make_backtest_frame()
	metrics = make_metrics_frame()
	predictions = make_predictions_frame()
	dataset = make_dataset_frame()
	samples = pd.DataFrame(
		{
			"excess_total_return": np.linspace(-0.1, 0.1, 20),
		}
	)
	summary = pd.DataFrame([{"actual_excess_total_return": -0.02}])

	paths = [
		plot_equity_curves(backtest, metrics, tmp_path / "equity.png"),
		plot_drawdowns(backtest, tmp_path / "drawdowns.png"),
		plot_prediction_probabilities(predictions, tmp_path / "probabilities.png"),
		plot_confusion_matrix(predictions, tmp_path / "confusion.png"),
		plot_train_feature_correlation_heatmap(dataset, tmp_path / "correlation.png"),
		plot_monte_carlo_excess_return(samples, summary, tmp_path / "monte_carlo.png"),
	]

	for path in paths:
		assert_png_written(path)

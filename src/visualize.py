from pathlib import Path
import os


MPL_CACHE_DIR = Path("/private/tmp/stock-return-predictor-matplotlib")
XDG_CACHE_DIR = Path("/private/tmp/stock-return-predictor-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
	from src.features import FEATURE_COLUMNS
	from src.ml_pipeline import DEFAULT_TRAIN_FRACTION, chronological_train_test_split
except ModuleNotFoundError:  # Allows `python src/visualize.py`.
	from features import FEATURE_COLUMNS
	from ml_pipeline import DEFAULT_TRAIN_FRACTION, chronological_train_test_split


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"

STRATEGY_BACKTEST_FILE = DATA_DIR / "ml_strategy_backtest.csv"
STRATEGY_METRICS_FILE = DATA_DIR / "ml_strategy_metrics.csv"
MONTE_CARLO_COMPARISON_FILE = DATA_DIR / "ml_monte_carlo_comparison.csv"
MONTE_CARLO_SAMPLES_FILE = DATA_DIR / "ml_monte_carlo_samples.csv"
SUPERVISED_DATASET_FILE = DATA_DIR / "ml_supervised_dataset.csv"
TEST_PREDICTIONS_FILE = DATA_DIR / "ml_test_predictions.csv"
HORIZON_LADDER_FILE = DATA_DIR / "horizon_ladder.csv"

EQUITY_CURVES_FIGURE = FIGURES_DIR / "equity_curves.png"
DRAWDOWNS_FIGURE = FIGURES_DIR / "drawdowns.png"
PREDICTION_PROBABILITIES_FIGURE = FIGURES_DIR / "prediction_probabilities.png"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix.png"
TRAIN_FEATURE_CORRELATION_HEATMAP_FIGURE = FIGURES_DIR / "train_feature_correlation_heatmap.png"
MONTE_CARLO_EXCESS_RETURN_FIGURE = FIGURES_DIR / "monte_carlo_excess_return.png"
HORIZON_LADDER_FIGURE = FIGURES_DIR / "horizon_ladder.png"


def _read_csv(path, date_columns=None):
	if not path.exists():
		raise FileNotFoundError(f"{path} does not exist. Run src/model_backtest.py first.")
	return pd.read_csv(path, parse_dates=date_columns or [])


def _save_figure(fig, output_path):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	plt.close(fig)
	return output_path


def _drawdown(equity_curve):
	return equity_curve / equity_curve.cummax() - 1.0


def plot_equity_curves(backtest, metrics, output_path=EQUITY_CURVES_FIGURE):
	fig, ax = plt.subplots(figsize=(11, 6))
	ax.plot(backtest["Date"], backtest["equity_curve"], label="ML strategy net of costs", linewidth=2)
	ax.plot(backtest["Date"], backtest["buy_and_hold_equity_curve"], label="Buy and hold", linewidth=2)
	ax.set_title("Out-of-Sample Equity Curves")
	ax.set_ylabel("Growth of $1")
	ax.set_xlabel("Date")
	ax.grid(True, alpha=0.25)
	ax.legend()

	if not metrics.empty:
		row = metrics.iloc[0]
		text = (
			f"Strategy return: {row['strategy_total_return']:.1%}\n"
			f"Buy-hold return: {row['buy_and_hold_total_return']:.1%}\n"
			f"Excess: {row['excess_total_return']:.1%}"
		)
		ax.text(
			0.02,
			0.98,
			text,
			transform=ax.transAxes,
			verticalalignment="top",
			bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
		)

	return _save_figure(fig, output_path)


def plot_drawdowns(backtest, output_path=DRAWDOWNS_FIGURE):
	fig, ax = plt.subplots(figsize=(11, 5))
	ax.plot(backtest["Date"], _drawdown(backtest["equity_curve"]), label="ML strategy net of costs", linewidth=2)
	ax.plot(backtest["Date"], _drawdown(backtest["buy_and_hold_equity_curve"]), label="Buy and hold", linewidth=2)
	ax.set_title("Out-of-Sample Drawdowns")
	ax.set_ylabel("Drawdown")
	ax.set_xlabel("Date")
	ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
	ax.grid(True, alpha=0.25)
	ax.legend()
	return _save_figure(fig, output_path)


def plot_prediction_probabilities(predictions, output_path=PREDICTION_PROBABILITIES_FIGURE):
	fig, ax = plt.subplots(figsize=(11, 5))
	correct = predictions["predicted_direction"] == predictions["actual_direction"]
	ax.scatter(
		predictions.loc[correct, "target_timestamp"],
		predictions.loc[correct, "probability_up"],
		s=12,
		alpha=0.55,
		label="Correct",
	)
	ax.scatter(
		predictions.loc[~correct, "target_timestamp"],
		predictions.loc[~correct, "probability_up"],
		s=12,
		alpha=0.55,
		label="Incorrect",
	)
	ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold")
	ax.set_title("Out-of-Sample Up Probability")
	ax.set_ylabel("Predicted probability of up day")
	ax.set_xlabel("Target date")
	ax.set_ylim(0.0, 1.0)
	ax.grid(True, alpha=0.25)
	ax.legend()
	return _save_figure(fig, output_path)


def plot_confusion_matrix(predictions, output_path=CONFUSION_MATRIX_FIGURE):
	labels = [0, 1]
	matrix = pd.crosstab(
		predictions["actual_direction"],
		predictions["predicted_direction"],
	).reindex(index=labels, columns=labels, fill_value=0)

	fig, ax = plt.subplots(figsize=(5.5, 5))
	image = ax.imshow(matrix.to_numpy(), cmap="Blues")
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
	ax.set_title("Out-of-Sample Confusion Matrix")
	ax.set_xlabel("Predicted direction")
	ax.set_ylabel("Actual direction")
	ax.set_xticks([0, 1], labels=["Down/flat", "Up"])
	ax.set_yticks([0, 1], labels=["Down/flat", "Up"])

	for row_index in range(matrix.shape[0]):
		for column_index in range(matrix.shape[1]):
			value = int(matrix.iloc[row_index, column_index])
			ax.text(column_index, row_index, str(value), ha="center", va="center", color="black")

	return _save_figure(fig, output_path)


def plot_train_feature_correlation_heatmap(dataset, output_path=TRAIN_FEATURE_CORRELATION_HEATMAP_FIGURE):
	split = chronological_train_test_split(
		dataset,
		train_size=DEFAULT_TRAIN_FRACTION,
		feature_columns=FEATURE_COLUMNS,
	)
	correlation = split.X_train.corr(method="pearson")

	fig, ax = plt.subplots(figsize=(12, 10))
	image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
	ax.set_title("Train-Only Feature Correlation Heatmap")
	ax.set_xticks(np.arange(len(correlation.columns)), labels=correlation.columns, rotation=90, fontsize=7)
	ax.set_yticks(np.arange(len(correlation.index)), labels=correlation.index, fontsize=7)
	return _save_figure(fig, output_path)


def plot_monte_carlo_excess_return(samples, summary, output_path=MONTE_CARLO_EXCESS_RETURN_FIGURE):
	fig, ax = plt.subplots(figsize=(10, 5.5))
	ax.hist(samples["excess_total_return"], bins=60, alpha=0.75, color="#4477AA")
	ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Break-even vs buy-hold")
	if not summary.empty:
		actual = summary.loc[0, "actual_excess_total_return"]
		ax.axvline(actual, color="#CC3311", linewidth=2, label=f"Actual excess return: {actual:.1%}")
	ax.set_title("Monte Carlo Paired Bootstrap: Excess Total Return")
	ax.set_xlabel("Strategy total return minus buy-and-hold total return")
	ax.set_ylabel("Simulation count")
	ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
	ax.grid(True, alpha=0.25)
	ax.legend()
	return _save_figure(fig, output_path)


def plot_horizon_ladder(ladder, output_path=HORIZON_LADDER_FIGURE):
	"""Two panels: return regression never beats drift; direction never beats the rising base rate."""
	ladder = ladder.sort_values("horizon").reset_index(drop=True)
	labels = [
		f"{int(h)}d" + ("\n(illustrative)" if tier == "illustrative" else "")
		for h, tier in zip(ladder["horizon"], ladder["rigor_tier"])
	]
	x = np.arange(len(ladder))

	fig, (ax_r2, ax_acc) = plt.subplots(1, 2, figsize=(13, 5.5))

	colors = ["#CC3311" if r2 <= 0 else "#228833" for r2 in ladder["linear_oos_r2_vs_drift"]]
	ax_r2.bar(x, ladder["linear_oos_r2_vs_drift"], color=colors, alpha=0.85)
	ax_r2.axhline(0.0, color="black", linewidth=1.2, label="Beats drift (R² > 0)")
	for xi, r2 in zip(x, ladder["linear_oos_r2_vs_drift"]):
		ax_r2.text(xi, r2, f"{r2:.2f}", ha="center", va="top" if r2 < 0 else "bottom", fontsize=9)
	ax_r2.set_title("Return regression never beats drift\n(out-of-sample R² vs a prevailing-mean forecast)")
	ax_r2.set_ylabel("Campbell-Thompson OOS R² vs drift")
	ax_r2.set_xlabel("Forward horizon (trading days)")
	ax_r2.set_xticks(x, labels=labels)
	ax_r2.grid(True, axis="y", alpha=0.25)
	ax_r2.legend(loc="lower left")

	width = 0.38
	ax_acc.bar(x - width / 2, ladder["logistic_accuracy"], width, label="Model accuracy", color="#4477AA")
	ax_acc.bar(x + width / 2, ladder["base_rate_accuracy"], width, label="Always-up base rate", color="#BBBBBB")
	ax_acc.plot(x, ladder["direction_base_rate"], color="#EE6677", marker="o", linewidth=1.5, label="Up-rate (risk premium)")
	ax_acc.set_ylim(0.0, 1.0)
	ax_acc.set_title("Direction accuracy stays below the rising base rate\n(the base-rate illusion grows with horizon)")
	ax_acc.set_ylabel("Accuracy / up-rate")
	ax_acc.set_xlabel("Forward horizon (trading days)")
	ax_acc.set_xticks(x, labels=labels)
	ax_acc.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
	ax_acc.grid(True, axis="y", alpha=0.25)
	ax_acc.legend(loc="lower right")

	return _save_figure(fig, output_path)


def create_all_visualizations():
	backtest = _read_csv(STRATEGY_BACKTEST_FILE, date_columns=["Date"])
	metrics = _read_csv(STRATEGY_METRICS_FILE)
	monte_carlo_summary = _read_csv(MONTE_CARLO_COMPARISON_FILE)
	monte_carlo_samples = _read_csv(MONTE_CARLO_SAMPLES_FILE)
	dataset = _read_csv(
		SUPERVISED_DATASET_FILE,
		date_columns=[
			"feature_timestamp",
			"target_timestamp",
			"target_volatility_start_timestamp",
			"target_volatility_end_timestamp",
		],
	)
	predictions = _read_csv(
		TEST_PREDICTIONS_FILE,
		date_columns=[
			"feature_timestamp",
			"target_timestamp",
			"target_volatility_start_timestamp",
			"target_volatility_end_timestamp",
		],
	)

	figures = [
		plot_equity_curves(backtest, metrics),
		plot_drawdowns(backtest),
		plot_prediction_probabilities(predictions),
		plot_confusion_matrix(predictions),
		plot_train_feature_correlation_heatmap(dataset),
		plot_monte_carlo_excess_return(monte_carlo_samples, monte_carlo_summary),
	]
	# Only rendered once the multi-horizon experiment (Experiment 4) has been run.
	if HORIZON_LADDER_FILE.exists():
		figures.append(plot_horizon_ladder(_read_csv(HORIZON_LADDER_FILE)))
	return figures


def main():
	try:
		figure_paths = create_all_visualizations()
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise SystemExit(f"Error: {error}")

	print("Saved visualizations:")
	for path in figure_paths:
		print(f"  {path}")


if __name__ == "__main__":
	main()

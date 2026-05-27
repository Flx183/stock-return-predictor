import pandas as pd
import pytest

from src.model_backtest import (
	backtest_model_predictions,
	build_signal_from_predictions,
	compute_model_strategy_metrics,
	monte_carlo_compare_to_buy_and_hold,
)


def make_predictions():
	return pd.DataFrame(
		{
			"feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
			"target_timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
			"probability_up": [0.70, 0.40, 0.65],
			"predicted_direction": [1, 0, 1],
			"actual_direction": [1, 0, 1],
		}
	)


def test_prediction_signal_includes_prior_close_anchor():
	signal = build_signal_from_predictions(make_predictions())

	assert signal.index.tolist() == pd.to_datetime(
		["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
	).tolist()
	assert signal.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_model_prediction_backtest_keeps_first_target_return_and_costs():
	dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
	prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=dates)

	backtest = backtest_model_predictions(prices, make_predictions(), cost_per_trade=0.01)

	assert backtest["asset_return"].tolist() == pytest.approx([0.0, 0.1, -0.1, 0.1])
	assert backtest["position"].tolist() == [0.0, 1.0, 0.0, 1.0]
	assert backtest["turnover"].tolist() == [0.0, 1.0, 1.0, 1.0]
	assert backtest["net_strategy_return"].tolist() == pytest.approx([0.0, 0.09, -0.01, 0.09])
	assert backtest.loc[pd.Timestamp("2024-01-02"), "probability_up"] == pytest.approx(0.70)
	assert not backtest.loc[pd.Timestamp("2024-01-01"), "is_test_target_date"]
	assert backtest.loc[pd.Timestamp("2024-01-02"), "is_test_target_date"]

	metrics = compute_model_strategy_metrics(backtest, cost_per_trade=0.01)
	assert metrics["n_test_days"] == 3
	assert metrics["strategy_total_return"] == pytest.approx((1.09 * 0.99 * 1.09) - 1.0)
	assert metrics["buy_and_hold_total_return"] == pytest.approx((1.1 * 0.9 * 1.1) - 1.0)


def test_monte_carlo_comparison_uses_test_days_only():
	dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
	prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=dates)
	backtest = backtest_model_predictions(prices, make_predictions(), cost_per_trade=0.01)

	summary, samples = monte_carlo_compare_to_buy_and_hold(backtest, n_simulations=50, seed=7)

	assert len(samples) == 50
	assert summary.loc[0, "n_test_days"] == 3
	assert summary.loc[0, "actual_strategy_total_return"] == pytest.approx((1.09 * 0.99 * 1.09) - 1.0)
	assert summary.loc[0, "actual_buy_and_hold_total_return"] == pytest.approx((1.1 * 0.9 * 1.1) - 1.0)
	assert samples["strategy_beats_buy_and_hold_total_return"].isin([True, False]).all()

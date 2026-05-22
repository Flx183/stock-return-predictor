import pandas as pd
import pytest

from src.backtest import backtest_daily_signal, previous_return_long_flat_signal


def test_previous_return_signal_and_costed_equity_curve_are_hand_checkable():
	dates = pd.date_range("2024-01-01", periods=5, freq="D")
	prices = pd.Series([100.0, 110.0, 121.0, 108.9, 119.79], index=dates)

	signal = previous_return_long_flat_signal(prices)
	result = backtest_daily_signal(prices, signal, cost_per_trade=0.01)

	assert result["position"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
	assert result["turnover"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]
	assert result["transaction_cost"].tolist() == [0.0, 0.0, 0.01, 0.0, 0.01]
	assert result["net_strategy_return"].tolist() == pytest.approx([0.0, 0.0, 0.09, -0.1, -0.01])
	assert result["equity_curve"].tolist() == pytest.approx([1.0, 1.0, 1.09, 0.981, 0.97119])


def test_backtest_rejects_negative_transaction_costs():
	dates = pd.date_range("2024-01-01", periods=2, freq="D")
	prices = pd.Series([100.0, 101.0], index=dates)
	signal = pd.Series([0.0, 1.0], index=dates)

	with pytest.raises(ValueError, match="cost_per_trade"):
		backtest_daily_signal(prices, signal, cost_per_trade=-0.01)

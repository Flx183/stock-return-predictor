import numpy as np
import pandas as pd
import pytest

from src.experiment_vix import VIX_FEATURE, _prepare_with_vix, run_vix_experiment
from src.vix_data import load_vix_close


def make_ohlcv(n_days=2200):
	dates = pd.bdate_range("2010-01-01", periods=n_days)
	step = np.arange(n_days, dtype=float)
	regime = 1.0 + 0.6 * np.sin(step / 120.0) ** 2
	shocks = regime * np.sin(step / 3.0) * 0.6
	close = pd.Series(100.0 * np.cumprod(1.0 + 0.0003 + 0.004 * shocks / regime.mean()), index=dates)
	open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + 0.0005)
	high = pd.concat([open_, close], axis=1).max(axis=1) * (1.0 + 0.003)
	low = pd.concat([open_, close], axis=1).min(axis=1) * (1.0 - 0.003)
	volume = pd.Series(1_000_000.0 + step * 300.0, index=dates)
	return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def make_vix(index):
	step = np.arange(len(index), dtype=float)
	return pd.Series(0.15 + 0.05 * np.sin(step / 30.0), index=index, name="vix_close")


def test_load_vix_close_parses_and_rescales_to_decimal(tmp_path):
	path = tmp_path / "VIX_data.csv"
	path.write_text(
		"Price,Close,High,Low,Open,Volume\n"
		"Ticker,^VIX,^VIX,^VIX,^VIX,^VIX\n"
		"Date,,,,,\n"
		"2020-01-02,12.47,13.0,12.0,12.5,0\n"
		"2020-01-03,14.02,15.0,13.5,13.6,0\n"
	)
	vix = load_vix_close(path)
	# VIX is quoted in percentage points; the loader returns decimal annualized vol.
	assert vix.loc["2020-01-02"] == pytest.approx(0.1247)
	assert vix.loc["2020-01-03"] == pytest.approx(0.1402)


def test_vix_feature_is_taken_as_of_the_feature_close():
	ohlcv = make_ohlcv()
	vix = make_vix(ohlcv.index)
	dataset, _ = _prepare_with_vix(ohlcv=ohlcv, vix=vix)

	# The VIX feature on each row must be VIX as of that row's feature_timestamp
	# (the prior close), never the target close — otherwise it would leak.
	sample = dataset.iloc[len(dataset) // 2]
	assert sample[VIX_FEATURE] == pytest.approx(vix.loc[sample["feature_timestamp"]])
	assert (dataset["feature_timestamp"] < dataset["target_timestamp"]).all()


def test_vix_experiment_reports_both_questions():
	ohlcv = make_ohlcv()
	vix = make_vix(ohlcv.index)
	result = run_vix_experiment(ohlcv=ohlcv, vix=vix, n_resamples=200)

	forecast = set(result["forecast"]["comparison"])
	assert {"vix_vs_model", "vix_vs_persistence", "model_vs_persistence"} <= forecast
	assert "model_plus_vix_vs_model" in set(result["feature"]["comparison"])
	assert isinstance(result["vix_beats_model"], bool)
	assert isinstance(result["vix_adds_signal"], bool)

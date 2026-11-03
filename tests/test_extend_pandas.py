"""
Tests for quantstats.extend_pandas functionality
"""

import numpy as np
import pandas as pd
import pytest

import quantstats as qs


@pytest.fixture
def sample_returns():
    """Generate sample returns as a pandas Series."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.randn(252) * 0.02, index=dates, name="Strategy")
    return returns


class TestExtendPandas:
    """Test extend_pandas functionality."""

    def test_extend_pandas_adds_methods(self, sample_returns):
        """Test that extend_pandas adds methods to Series."""
        qs.extend_pandas()

        # Check that quantstats methods are now available on Series
        assert hasattr(sample_returns, "sharpe")
        assert hasattr(sample_returns, "sortino")
        assert hasattr(sample_returns, "max_drawdown")
        assert hasattr(sample_returns, "cagr")

    def test_sharpe_via_pandas(self, sample_returns):
        """Test Sharpe ratio via pandas extension."""
        qs.extend_pandas()
        result = sample_returns.sharpe()
        assert np.isfinite(result)

    def test_sortino_via_pandas(self, sample_returns):
        """Test Sortino ratio via pandas extension."""
        qs.extend_pandas()
        result = sample_returns.sortino()
        assert np.isfinite(result)

    def test_max_drawdown_via_pandas(self, sample_returns):
        """Test max drawdown via pandas extension."""
        qs.extend_pandas()
        result = sample_returns.max_drawdown()
        assert result <= 0

    def test_cagr_via_pandas(self, sample_returns):
        """Test CAGR via pandas extension."""
        qs.extend_pandas()
        result = sample_returns.cagr()
        assert np.isfinite(result)

    def test_volatility_via_pandas(self, sample_returns):
        """Test volatility via pandas extension."""
        qs.extend_pandas()
        result = sample_returns.volatility()
        assert result > 0

    def test_calmar_via_pandas(self, sample_returns):
        """Test Calmar ratio via pandas extension."""
        qs.extend_pandas()
        result = sample_returns.calmar()
        assert np.isfinite(result)


class TestExtendPandasWithParams:
    """Test extend_pandas with parameters."""

    def test_sharpe_with_rf(self, sample_returns):
        """Test Sharpe with risk-free rate via pandas."""
        qs.extend_pandas()
        result_no_rf = sample_returns.sharpe(rf=0)
        result_with_rf = sample_returns.sharpe(rf=0.02)
        # Should be different
        assert result_no_rf != result_with_rf

    def test_cagr_compounded(self, sample_returns):
        """Test CAGR with compounded option via pandas."""
        qs.extend_pandas()
        result_comp = sample_returns.cagr(compounded=True)
        result_simple = sample_returns.cagr(compounded=False)
        # May or may not be different depending on returns
        assert np.isfinite(result_comp)
        assert np.isfinite(result_simple)


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
def test_matched_statistics_through_pandas(metric):
    qs.extend_pandas()
    dates = pd.date_range("2026-09-01", periods=5)
    # The first returns have no known starting date and must not be compared.
    # Later returns satisfy strategy = 2 * benchmark + 0.01: beta=2,
    # R-squared=1 and alpha=0.01 per period (0.52 with periods=52).
    strategy = pd.Series([9, 0.03, -0.03, 0.07, 0.04], index=dates, name="Strategy")
    benchmark = pd.Series([7, 0.01, -0.02, 0.03, 0.015], index=dates)
    original = strategy.copy(deep=True)
    kwargs = {"frequency": "daily", "prepare_returns": False}
    if metric == "greeks":
        kwargs["periods"] = 52
    actual = getattr(strategy, metric)(benchmark, **kwargs)
    if metric == "greeks":
        assert actual["beta"] == pytest.approx(2)
        assert actual["alpha"] == pytest.approx(0.52)
    elif metric in ("r_squared", "r2"):
        assert actual == pytest.approx(1)
    elif metric == "information_ratio":
        active = np.array([0.02, -0.01, 0.04, 0.025])
        assert actual == pytest.approx(active.mean() / active.std(ddof=1))
    pd.testing.assert_series_equal(strategy, original)


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
def test_calendar_options_through_pandas(metric):
    qs.extend_pandas()
    # Monday's observation is supplied on a declared holiday, so it remains
    # part of the weekly product rather than being discarded as a closure.
    dates = pd.to_datetime(
        ["2026-08-30", "2026-09-06", "2026-09-07", "2026-09-13", "2026-09-20"]
    )
    strategy = pd.Series([0, 0.03, 0, -0.03, 0.07], index=dates)
    benchmark = pd.Series([0, 0.01, 0, -0.02, 0.03], index=dates)
    result = getattr(strategy, metric)(
        benchmark,
        prepare_returns=False,
        frequency="weekly",
        holidays=["2026-09-07"],
    )
    if metric == "greeks":
        assert result["beta"] == pytest.approx(2)
        assert result["alpha"] == pytest.approx(0.52)
        assert result.attrs["frequency"] == "weekly"
    elif metric in ("r_squared", "r2"):
        assert result == pytest.approx(1)
    else:
        active = np.array([0.02, -0.01, 0.04])
        assert result == pytest.approx(active.mean() / active.std(ddof=1))

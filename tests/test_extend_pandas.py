"""
Tests for quantstats.extend_pandas functionality
"""

import pytest
import pandas as pd
import numpy as np

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

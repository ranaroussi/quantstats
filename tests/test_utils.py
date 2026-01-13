"""
Tests for quantstats.utils module
"""

import pytest
import pandas as pd
import numpy as np

import quantstats as qs
from quantstats import utils


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.Series(100 * np.cumprod(1 + np.random.randn(100) * 0.02), index=dates)
    return prices


@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
    return returns


class TestPrepareReturns:
    """Test _prepare_returns function."""

    def test_converts_prices_to_returns(self, sample_prices):
        """Test that prices are converted to returns."""
        result = utils._prepare_returns(sample_prices)
        # First value should be NaN or 0 (pct_change result)
        assert len(result) == len(sample_prices)
        # Returns should be small values (not prices)
        assert result.dropna().abs().max() < 1

    def test_passes_through_returns(self, sample_returns):
        """Test that returns pass through unchanged in magnitude."""
        result = utils._prepare_returns(sample_returns)
        # Should still be returns (small values)
        assert result.abs().max() < 1

    def test_handles_dataframe(self, sample_prices):
        """Test DataFrame handling."""
        df = pd.DataFrame({"A": sample_prices, "B": sample_prices * 1.1})
        result = utils._prepare_returns(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2


class TestToReturns:
    """Test to_returns function."""

    def test_to_returns_from_prices(self, sample_prices):
        """Test conversion from prices to returns."""
        result = utils.to_returns(sample_prices)
        assert isinstance(result, pd.Series)
        # Should be small values (returns)
        assert result.dropna().abs().max() < 1


class TestToPrices:
    """Test to_prices function."""

    def test_to_prices_from_returns(self, sample_returns):
        """Test conversion from returns to prices."""
        result = utils.to_prices(sample_returns, base=100)
        assert isinstance(result, pd.Series)
        # First value should be close to base
        assert abs(result.iloc[0] - 100) < 10


class TestLogReturns:
    """Test log returns conversion."""

    def test_log_returns(self, sample_returns):
        """Test log returns calculation."""
        result = utils.log_returns(sample_returns)
        assert isinstance(result, pd.Series)
        # Log returns should be close to simple returns for small values
        np.testing.assert_array_almost_equal(
            result.dropna().values, sample_returns.dropna().values, decimal=2
        )


class TestGroupReturns:
    """Test group_returns function."""

    def test_group_by_year(self, sample_returns):
        """Test grouping returns by year."""
        # Create returns spanning multiple years
        dates = pd.date_range("2019-01-01", periods=400, freq="D")
        returns = pd.Series(np.random.randn(400) * 0.01, index=dates)
        result = utils.group_returns(returns, returns.index.year)
        assert len(result) >= 2  # Should have at least 2 years


class TestAggregateReturns:
    """Test aggregate_returns function."""

    def test_aggregate_monthly(self, sample_returns):
        """Test monthly aggregation."""
        # Need returns spanning multiple months
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        result = utils.aggregate_returns(returns, "month")
        assert len(result) <= len(returns)

    def test_aggregate_yearly(self, sample_returns):
        """Test yearly aggregation."""
        dates = pd.date_range("2019-01-01", periods=400, freq="D")
        returns = pd.Series(np.random.randn(400) * 0.01, index=dates)
        result = utils.aggregate_returns(returns, "year")
        assert len(result) <= len(returns)


class TestRebase:
    """Test rebase function."""

    def test_rebase_to_100(self, sample_prices):
        """Test rebasing prices to 100."""
        result = utils.rebase(sample_prices, base=100)
        assert abs(result.iloc[0] - 100) < 0.01


class TestMakePortfolio:
    """Test make_portfolio function."""

    def test_make_portfolio_comp(self, sample_returns):
        """Test portfolio creation with compounding."""
        result = utils.make_portfolio(sample_returns, start_balance=10000, mode="comp")
        assert isinstance(result, pd.Series)
        # Should start near the initial balance
        assert abs(result.iloc[0] - 10000) < 1000


class TestValidation:
    """Test input validation."""

    def test_validate_series(self, sample_returns):
        """Test validation accepts valid Series."""
        assert utils.validate_input(sample_returns) is True

    def test_validate_dataframe(self, sample_returns):
        """Test validation accepts valid DataFrame."""
        df = pd.DataFrame({"A": sample_returns})
        assert utils.validate_input(df) is True

    def test_validate_none_raises(self):
        """Test validation raises for None input."""
        with pytest.raises(utils.DataValidationError):
            utils.validate_input(None)

    def test_validate_empty_raises(self):
        """Test validation raises for empty input."""
        with pytest.raises(utils.DataValidationError):
            utils.validate_input(pd.Series([], dtype=float))


class TestInNotebook:
    """Test notebook detection."""

    def test_in_notebook_returns_bool(self):
        """Test that _in_notebook returns a boolean."""
        result = utils._in_notebook()
        assert isinstance(result, bool)
        # In pytest, should return False
        assert result is False


class TestFileStream:
    """Test file stream creation."""

    def test_file_stream_creation(self):
        """Test _file_stream returns BytesIO object."""
        result = utils._file_stream()
        assert hasattr(result, "read")
        assert hasattr(result, "write")

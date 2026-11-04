"""
Tests for quantstats.utils module
"""

import numpy as np
import pandas as pd
import pytest

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


class TestReturnCacheOrder:
    """A previous preparation must not supply another row sequence's returns."""

    @pytest.fixture(autouse=True)
    def isolated_cache(self, monkeypatch):
        monkeypatch.setattr(utils, "_PREPARE_RETURNS_CACHE", {})

    @pytest.mark.parametrize("container", ["series", "frame", "mixed_frame"])
    @pytest.mark.parametrize("order", [(3, 2, 1, 0), (2, 0, 3, 1)])
    @pytest.mark.parametrize("reordered_first", [False, True])
    def test_preparation_uses_requested_row_order(
        self, container, order, reordered_first
    ):
        utc = pd.DatetimeIndex(
            [
                "2024-03-09 05:00",
                "2024-03-10 05:00",
                "2024-03-11 04:00",
                "2024-03-12 04:00",
            ],
            name="observed",
        )
        index = utc.tz_localize("UTC").tz_convert("America/New_York")
        prices = pd.Series([100.0, 110.0, 99.0, 118.8], index=index, name="prices")
        if container != "series":
            prices = prices.to_frame()
        if container == "mixed_frame":
            prices[17] = [0.01, -0.02, 0.03, 0.005]
        reordered = prices.iloc[list(order)]
        cases = [reordered, prices] if reordered_first else [prices, reordered]

        for data in cases:
            original = data.copy(deep=True)
            # The return formula uses adjacent prices in the requested order,
            # even when dates are descending or shuffled. Sorting is not a fix.
            values = (
                data.to_numpy() if container == "series" else data["prices"].to_numpy()
            )
            expected_values = np.r_[np.nan, values[1:] / values[:-1] - 1]
            expected_index = utc.take([index.get_loc(label) for label in data.index])
            expected = pd.Series(expected_values, index=expected_index, name="prices")
            if container != "series":
                expected = expected.to_frame()
                if container == "mixed_frame":
                    expected[17] = data[17].to_numpy()

            result = utils._prepare_returns(data)

            check = (
                pd.testing.assert_series_equal
                if container == "series"
                else pd.testing.assert_frame_equal
            )
            check(result, expected)
            check(data, original)

    @pytest.mark.parametrize("apply_rf", [False, True])
    def test_reordered_nullable_prices_with_time_varying_rf(self, apply_rf):
        index = pd.date_range("2024-01-01", periods=4, name="observed")
        prices = pd.DataFrame(
            {"asset": pd.array([100, 110, 99, 118.8], dtype="Float64")}, index=index
        )
        rates = pd.Series([0.01, 0.02, 0.03, 0.04], index=index)
        utils._prepare_returns(prices, rates, 252, apply_rf=apply_rf)
        reversed_prices = prices.iloc[::-1]
        original_prices = reversed_prices.copy(deep=True)
        original_rates = rates.copy(deep=True)
        # Rates follow their labels, while returns follow adjacent prices.
        expected_values = [np.nan, 99 / 118.8 - 1, 110 / 99 - 1, 100 / 110 - 1]
        if apply_rf:
            expected_values = [
                value - ((1 + rate) ** (1 / 252) - 1)
                for value, rate in zip(
                    expected_values, [0.04, 0.03, 0.02, 0.01], strict=True
                )
            ]
        expected = pd.DataFrame(
            {"asset": pd.array(expected_values, dtype="Float64")}, index=index[::-1]
        )

        result = utils._prepare_returns(reversed_prices, rates, 252, apply_rf=apply_rf)

        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(reversed_prices, original_prices)
        pd.testing.assert_series_equal(rates, original_rates)

    @pytest.mark.parametrize("as_frame", [False, True])
    def test_repeated_input_still_hits_and_returns_owned_copies(
        self, monkeypatch, as_frame
    ):
        prices = pd.Series([100.0, 110.0, 99.0], name="asset")
        if as_frame:
            prices = prices.to_frame()
        original = prices.copy(deep=True)
        expected = pd.Series([np.nan, 0.1, -0.1], name="asset")
        if as_frame:
            expected = expected.to_frame()
        calls = []
        pct_change = pd.Series.pct_change

        def counted(data, *args, **kwargs):
            calls.append(1)
            return pct_change(data, *args, **kwargs)

        monkeypatch.setattr(pd.Series, "pct_change", counted)
        check = (
            pd.testing.assert_frame_equal
            if as_frame
            else pd.testing.assert_series_equal
        )
        for _ in range(2):
            result = utils._prepare_returns(prices.copy(deep=True))
            check(result, expected)
            # Challenge both the initial stored copy and a subsequent cache hit.
            result.iloc[1] = 7
            result.index = pd.Index([9, 8, 7], name="changed")
            if as_frame:
                result.columns = ["changed"]
            else:
                result.name = "changed"
        check(prices, original)
        check(utils._prepare_returns(prices), expected)
        assert len(calls) == 1

        prices.iloc[1] = 120
        changed = pd.Series([np.nan, 0.2, 99 / 120 - 1], name="asset")
        if as_frame:
            changed = changed.to_frame()
        check(utils._prepare_returns(prices), changed)
        utils._PREPARE_RETURNS_CACHE.clear()
        check(utils._prepare_returns(prices.copy()), changed)
        assert len(calls) == 3

    def test_hash_failure_keeps_uncached_preparation(self, monkeypatch):
        def unavailable(*args, **kwargs):
            raise MemoryError("hash allocation unavailable")

        monkeypatch.setattr(pd.util, "hash_pandas_object", unavailable)
        prices = pd.Series([100.0, 110.0, 99.0])
        expected = pd.Series([np.nan, 0.1, -0.1])
        pd.testing.assert_series_equal(utils._prepare_returns(prices), expected)
        assert not utils._PREPARE_RETURNS_CACHE


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

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


@pytest.fixture
def interval_pair():
    """A strategy and benchmark with equal returns over three matching periods.

    The benchmark's three weekly returns combine 3, 3 and 4 supplied returns.
    The first values deliberately differ. Their starting dates are unknown,
    so neither first return should be included in a comparison.
    """
    dates = pd.to_datetime(
        [
            "2026-09-06",
            "2026-09-08",
            "2026-09-10",
            "2026-09-13",
            "2026-09-15",
            "2026-09-17",
            "2026-09-20",
            "2026-09-21",
            "2026-09-23",
            "2026-09-25",
            "2026-09-27",
        ]
    )
    strategy = pd.Series(
        [9.0, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.02, -0.01, 0.015],
        index=dates,
        name="Strategy",
    )
    benchmark = pd.Series(
        [0.7, 0.019494, 0.019898, 0.00444806],
        index=dates[[0, 3, 6, 10]],
        name="Benchmark",
    )
    return strategy, benchmark


class TestMatchedBenchmarkPair:
    """Check matching periods and returns without changing the supplied inputs."""

    @pytest.mark.parametrize("kind", ["prices", "returns"])
    @pytest.mark.parametrize("warm_side", ["strategy", "benchmark", "both"])
    def test_matching_is_independent_of_native_cache_history(self, kind, warm_side):
        """Earlier calls on reversed data must not change a later comparison."""
        dates = pd.date_range("2026-01-05", periods=5)
        strategy = pd.Series(
            [100.0, 110.0, 99.0, 118.8, 106.92], index=dates, name="Strategy"
        )
        benchmark = pd.Series(
            [200.0, 220.0, 198.0, 237.6, 213.84], index=dates, name="Benchmark"
        )
        if kind == "returns":
            strategy[:] = [0.0, 0.1, -0.1, 0.2, -0.1]
            benchmark[:] = strategy.to_numpy()
        original_strategy = strategy.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)
        original_cache = utils._PREPARE_RETURNS_CACHE
        try:
            utils._PREPARE_RETURNS_CACHE = {}
            for side, data in (("strategy", strategy), ("benchmark", benchmark)):
                if warm_side in (side, "both"):
                    utils._prepare_benchmark(data.iloc[::-1])
            cached = {
                key: value.copy(deep=True)
                for key, value in utils._PREPARE_RETURNS_CACHE.items()
            }

            actual, reference = utils._prepare_benchmark_pair(
                strategy, benchmark, frequency="daily"
            )

            # Both histories represent the same four adjacent price changes.
            # Checking only beta=1 could miss both histories being wrong the same way.
            expected = pd.Series(
                [0.1, -0.1, 0.2, -0.1], index=dates[1:], name="Strategy"
            )
            pd.testing.assert_series_equal(actual, expected, check_freq=False)
            pd.testing.assert_series_equal(
                reference, expected.rename("Benchmark"), check_freq=False
            )
            assert actual.index.freq is None and reference.index.freq is None
            pd.testing.assert_series_equal(strategy, original_strategy)
            pd.testing.assert_series_equal(benchmark, original_benchmark)
            assert utils._PREPARE_RETURNS_CACHE.keys() == cached.keys()
            for key, value in cached.items():
                pd.testing.assert_series_equal(utils._PREPARE_RETURNS_CACHE[key], value)
        finally:
            utils._PREPARE_RETURNS_CACHE = original_cache

    @pytest.mark.parametrize("apply_rf", [False, True])
    def test_uncached_native_preparation_preserves_conversion_and_rf(
        self, monkeypatch, apply_rf
    ):
        """Caching must not affect price conversion, rate adjustments or dates."""
        dates = pd.date_range("2026-01-05", periods=3, tz="America/New_York")
        prices = pd.Series([100.0, 110.0, 99.0], index=dates, name="Asset")
        original = prices.copy(deep=True)

        def unexpected_hash(*args, **kwargs):
            pytest.fail("Uncached preparation must not inspect shared cache identity")

        monkeypatch.setattr(utils, "_generate_cache_key", unexpected_hash)
        monkeypatch.setattr(utils, "_PREPARE_RETURNS_CACHE", {})
        expected = pd.Series(
            [np.nan, 0.1, -0.1],
            index=pd.DatetimeIndex(
                ["2026-01-05 05:00", "2026-01-06 05:00", "2026-01-07 05:00"]
            ),
            name="Asset",
        )
        if apply_rf:
            expected -= 1.02 ** (1 / 252) - 1

        actual = utils._prepare_returns(
            prices, rf=0.02, nperiods=252, apply_rf=apply_rf, use_cache=False
        )

        pd.testing.assert_series_equal(actual, expected, check_freq=False)
        pd.testing.assert_series_equal(prices, original)
        assert not utils._PREPARE_RETURNS_CACHE

    @pytest.mark.parametrize("swap", [False, True])
    def test_interleaved_calendars_require_complete_shared_boundaries(self, swap):
        """Compare periods even when both inputs have dates the other lacks."""
        strategy_dates = pd.DatetimeIndex(
            ["2026-12-30", "2026-12-31", "2027-01-04", "2027-01-05", "2027-01-08"]
        )
        benchmark_dates = pd.DatetimeIndex(
            ["2026-12-30", "2027-01-01", "2027-01-05", "2027-01-06", "2027-01-08"]
        )
        strategy = pd.Series(
            [7, 0.01, 0.02, 0.03, -0.04], index=strategy_dates, name="Strategy"
        )
        benchmark = pd.Series(
            [9, 0.04, -0.01, 0.02, -0.02], index=benchmark_dates, name="Benchmark"
        )
        # Both inputs have dates the other lacks. Only December 30, January 5
        # and January 8 are shared. Declared holidays with supplied observations
        # still contribute; their returns must never be replaced by zero.
        expected_s = [1.01 * 1.02 * 1.03 - 1, -0.04]
        expected_b = [1.04 * 0.99 - 1, 1.02 * 0.98 - 1]
        left, right = (benchmark, strategy) if swap else (strategy, benchmark)
        expected_left, expected_right = (
            (expected_b, expected_s) if swap else (expected_s, expected_b)
        )

        actual, reference = utils._prepare_benchmark_pair(
            left,
            right,
            prepare_returns=False,
            frequency="daily",
            holidays=[
                "2026-12-31",
                "2027-01-01",
                "2027-01-04",
                "2027-01-06",
                "2027-01-07",
            ],
        )

        labels = pd.DatetimeIndex(["2027-01-05", "2027-01-08"])
        pd.testing.assert_series_equal(
            actual, pd.Series(expected_left, index=labels, name=left.name)
        )
        pd.testing.assert_series_equal(
            reference, pd.Series(expected_right, index=labels, name=right.name)
        )

    def test_compounds_only_complete_shared_intervals(self, interval_pair):
        strategy, benchmark = interval_pair
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=False, frequency="weekly"
        )
        # Three Sunday-ended weeks, each containing all supplied returns.
        # The first values have unknown starts, so neither belongs to a pair.
        expected = pd.Series(
            [0.019494, 0.019898, 0.00444806],
            index=benchmark.index[1:],
            name="Strategy",
        )
        pd.testing.assert_series_equal(actual, expected, atol=1e-14, rtol=1e-12)
        pd.testing.assert_series_equal(
            reference, expected.rename("Benchmark"), atol=1e-14, rtol=1e-12
        )

    @pytest.mark.parametrize("prepare", [False, True])
    @pytest.mark.parametrize("side", ["strategy", "benchmark"])
    @pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, pd.NA])
    def test_explicit_missing_invalidates_interval(
        self, interval_pair, missing, side, prepare
    ):
        strategy, benchmark = interval_pair
        if side == "strategy":
            strategy = strategy.astype("Float64")
            strategy.iloc[2] = missing
        else:
            benchmark = benchmark.astype("Float64")
            benchmark.iloc[1] = missing
        original_s, original_b = strategy.copy(deep=True), benchmark.copy(deep=True)
        # A complete sequence may not skip a week with unknown performance.
        with pytest.raises(ValueError, match="unknown"):
            utils._prepare_benchmark_pair(
                strategy, benchmark, prepare_returns=prepare, frequency="weekly"
            )
        pd.testing.assert_series_equal(strategy, original_s)
        pd.testing.assert_series_equal(benchmark, original_b)

    @pytest.mark.parametrize("prepare", [False, True])
    @pytest.mark.parametrize(
        "zones",
        [("America/New_York", "Asia/Tokyo"), ("Asia/Tokyo", "America/New_York")],
    )
    @pytest.mark.parametrize("clock", ["00:30", "23:30"])
    def test_calendar_keys_precede_utc_conversion(
        self, interval_pair, prepare, zones, clock
    ):
        strategy, benchmark = interval_pair
        hour, minute = map(int, clock.split(":"))
        offset = pd.Timedelta(hours=hour, minutes=minute)
        # This calendar includes the New York DST transition on March 8.
        days = pd.Timestamp("2026-03-01") + pd.to_timedelta(
            [0, 2, 4, 7, 9, 11, 14, 15, 17, 19, 21], unit="D"
        )
        strategy.index = (days + offset).tz_localize(zones[0])
        benchmark.index = (days[[0, 3, 6, 10]] + offset).tz_localize(zones[1])
        originals = strategy.copy(deep=True), benchmark.copy(deep=True)
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=prepare, frequency="weekly"
        )
        labels = strategy.index[[3, 6, 10]].tz_convert("UTC").tz_localize(None)
        expected = pd.Series(
            [0.019494, 0.019898, 0.00444806], index=labels, name="Strategy"
        )
        pd.testing.assert_series_equal(actual, expected, atol=1e-14, rtol=1e-12)
        pd.testing.assert_series_equal(
            reference, expected.rename("Benchmark"), atol=1e-14, rtol=1e-12
        )
        pd.testing.assert_series_equal(strategy, originals[0])
        pd.testing.assert_series_equal(benchmark, originals[1])

    def test_staggered_anchor_and_unmatched_tails(self):
        days = pd.date_range("2026-09-04", periods=9)
        strategy = pd.Series([9, 0.5, 0.1, -0.1, 0.2, -0.2, 0.3, 0.4], index=days[:8])
        benchmark = pd.Series([7, 0.01, 0.02, 0.9], index=days[[1, 3, 5, 8]])
        actual, reference = utils._prepare_benchmark_pair(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="daily",
            holidays=["2026-09-08"],
        )
        expected = pd.Series([-0.01, -0.04], index=days[[3, 5]])
        pd.testing.assert_series_equal(actual, expected, atol=1e-14, rtol=1e-12)
        pd.testing.assert_series_equal(
            reference, pd.Series([0.01, 0.02], index=days[[3, 5]])
        )

    @pytest.mark.parametrize("frequency", ["D", "W-FRI", "MS"])
    def test_arbitrary_two_observation_spans_are_not_a_supported_cadence(
        self, frequency
    ):
        dates = pd.date_range("2025-01-01", periods=5, freq=frequency)
        strategy = pd.Series([8, 0.1, -0.1, 0.2, -0.2], index=dates)
        benchmark = pd.Series([9, -0.01, -0.04], index=dates[[0, 2, 4]])
        before_s, before_b = strategy.copy(deep=True), benchmark.copy(deep=True)
        # Shared dates alone no longer justify mixing unsupported durations.
        with pytest.raises(ValueError, match="comparison periods"):
            utils._prepare_benchmark_pair(
                strategy, benchmark, prepare_returns=False, frequency="daily"
            )
        pd.testing.assert_series_equal(strategy, before_s)
        pd.testing.assert_series_equal(benchmark, before_b)

    @pytest.mark.parametrize("prepare", [False, True])
    def test_empty_matching_preserves_caller_frequency(self, prepare):
        strategy = pd.Series(
            [],
            index=pd.date_range("2026-09-01", periods=0, freq="D", name="Observation"),
            dtype=float,
            name="Strategy",
        )
        benchmark = pd.Series(
            [],
            index=pd.date_range("2026-09-01", periods=0, freq="W-FRI"),
            dtype=float,
            name="Benchmark",
        )
        # Save the frequency values themselves. A Series copy can share its index
        # and would not reliably detect a change to the caller's frequency.
        original_frequencies = strategy.index.freq, benchmark.index.freq
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=prepare, frequency="daily"
        )
        assert (strategy.index.freq, benchmark.index.freq) == original_frequencies
        for result, name in ((actual, "Strategy"), (reference, "Benchmark")):
            assert result.empty and result.index.freq is None
            assert result.name == name and result.index.name == "Observation"

    @pytest.mark.parametrize("dtype", ["int32", "Int64", "Float64"])
    def test_promotes_before_compounding(self, dtype):
        dates = pd.date_range("2026-09-04", periods=3)
        strategy = pd.Series([0, 2**30, 2**30], index=dates, dtype=dtype)
        benchmark = pd.Series([0, 0.01], index=dates[[0, 2]])
        with np.errstate(all="raise"):
            actual, _ = utils._prepare_benchmark_pair(
                strategy, benchmark, prepare_returns=False, frequency="daily"
            )
        assert actual.iloc[0] == pytest.approx(float((2**30 + 1) ** 2 - 1), rel=1e-15)

    def test_total_loss_does_not_poison_later_intervals(self):
        dates = pd.date_range("2026-09-01", periods=4)
        strategy = pd.Series([0, -1, 0.5, 0.2], index=dates)
        benchmark = pd.Series([0, 0.01, 0.02, 0.03], index=dates)
        with np.errstate(all="raise"):
            actual, _ = utils._prepare_benchmark_pair(
                strategy, benchmark, prepare_returns=False, frequency="daily"
            )
        pd.testing.assert_series_equal(
            actual, pd.Series([-1.0, 0.5, 0.2], index=dates[1:]), check_freq=False
        )

    @pytest.mark.parametrize(
        "chunk, expected, atol",
        [
            ([1e-18], 1e-18, 0),
            ([-3e-18], -3e-18, 0),
            ([1e-18, 2e-18], 3e-18, 0),
            ([1e-308, 2e-308], 3e-308, 0),
            ([1e-8, -1e-8], -1e-16, 5e-24),
            ([-1.0, 0.5], -1.0, 0),
            ([-2.0, 0.5], -2.5, 0),
            ([-2.0, -2.0], 0.0, 0),
        ],
    )
    def test_period_compounding_preserves_small_returns_and_signed_growth(
        self, chunk, expected, atol
    ):
        """Small changes must survive consolidation, including offsetting gains.

        For r followed by -r, (1+r)(1-r)-1 is -r squared. Total losses
        and returns below -100% remain valid.
        """
        dates = pd.date_range("2026-01-09", periods=len(chunk) + 1)
        strategy = pd.Series([0.0, *chunk], index=dates)
        benchmark = pd.Series([0.0, 0.01], index=dates[[0, -1]])
        before = strategy.copy(deep=True)
        with np.errstate(all="raise"):
            actual, _ = utils._prepare_benchmark_pair(
                strategy, benchmark, prepare_returns=False, frequency="daily"
            )
            assert all(mode == "raise" for mode in np.geterr().values())
        # Opposite 1e-8 returns leave a 1e-16 loss. Allow rounding on the
        # input scale (about eps * 2e-8), while still decisively excluding zero.
        assert actual.iloc[0] == pytest.approx(expected, rel=1e-14, abs=atol)
        pd.testing.assert_series_equal(strategy, before)

    @pytest.mark.parametrize("pair", [(1.0, -0.5), (3.0, -0.75), (-2.0, -2.0)])
    @pytest.mark.parametrize(
        "order", [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    )
    def test_cancelling_movements_preserve_the_remaining_return(self, pair, order):
        """Exactly neutral growth must leave a small return intact, in any order.

        The growth factors are 2*0.5, 4*0.25 or (-1)*(-1). Each product
        is exactly 1, so the combined return must equal the remaining 1e-18.
        """
        dates = pd.to_datetime(["2026-01-04", "2026-01-06", "2026-01-08", "2026-01-11"])
        chunk = [*pair, 1e-18]
        strategy = pd.Series([0.0, *(chunk[i] for i in order)], index=dates)
        benchmark = pd.Series([0.0, 0.01], index=dates[[0, -1]])
        before_s, before_b = strategy.copy(deep=True), benchmark.copy(deep=True)
        for left, right in ((strategy, benchmark), (benchmark, strategy)):
            with np.errstate(all="raise"):
                actual, reference = utils._prepare_benchmark_pair(
                    left, right, prepare_returns=False, frequency="weekly"
                )
                assert all(mode == "raise" for mode in np.geterr().values())
            result = actual if left is strategy else reference
            assert result.iloc[0] == 1e-18
        pd.testing.assert_series_equal(strategy, before_s)
        pd.testing.assert_series_equal(benchmark, before_b)

    def test_price_preparation_preserves_missing_observations(self):
        dates = pd.date_range("2026-09-01", periods=6)
        prices = pd.Series(
            [100, 102, np.nan, 104.04, 105.0804, 107.182008], index=dates
        )
        benchmark = pd.Series([100, 101, 102, 103, 104, 105], index=dates)
        before_s, before_b = prices.copy(deep=True), benchmark.copy(deep=True)
        # Missing prices also make the following price change unknown; choosing
        # monthly instead of daily cannot restore that missing performance.
        with pytest.raises(ValueError, match="unknown"):
            utils._prepare_benchmark_pair(prices, benchmark, frequency="daily")
        pd.testing.assert_series_equal(prices, before_s)
        pd.testing.assert_series_equal(benchmark, before_b)

    @pytest.mark.parametrize("prepare", [False, True])
    def test_future_values_do_not_change_completed_intervals(
        self, interval_pair, prepare
    ):
        strategy, benchmark = interval_pair
        before = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=prepare, frequency="weekly"
        )
        strategy.iloc[-1] = -0.5
        benchmark.iloc[-1] = 0.8
        after = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=prepare, frequency="weekly"
        )
        for earlier, later in zip(before, after, strict=True):
            pd.testing.assert_series_equal(earlier.iloc[:-1], later.iloc[:-1])

    def test_output_values_and_indexes_are_owned(self, interval_pair):
        """Editing one result must not change either input or the other result."""
        strategy, benchmark = interval_pair
        strategy.index.name = "Observation"
        benchmark = benchmark.to_frame()
        benchmark["Unused"] = np.nan
        original_s, original_b = strategy.copy(deep=True), benchmark.copy(deep=True)
        # Snapshot date storage independently so a shared-index mutation cannot
        # change both the caller and the expected value and escape detection.
        original_s.index = strategy.index.copy(deep=True)
        original_b.index = benchmark.index.copy(deep=True)
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=False, frequency="weekly"
        )
        actual.iloc[0] = 99
        actual.index.array[0] = pd.Timestamp("2000-01-01")
        actual.index.name = "Changed"
        assert reference.index[0] == original_s.index[3]
        reference.iloc[0] = 88
        reference.index.array[-1] = pd.Timestamp("2001-01-01")
        reference.name = "Changed"
        assert actual.index[-1] == original_s.index[10]
        pd.testing.assert_series_equal(strategy, original_s)
        pd.testing.assert_frame_equal(benchmark, original_b)

    @pytest.mark.parametrize(
        "fault", ["duplicate", "unsorted", "nat", "range", "complex"]
    )
    def test_rejections_preserve_caller(self, interval_pair, fault):
        strategy, benchmark = interval_pair
        if fault == "duplicate":
            labels = strategy.index.to_list()
            labels[2] = labels[1]
            strategy.index = pd.DatetimeIndex(labels)
        elif fault == "unsorted":
            strategy = strategy.iloc[::-1]
        elif fault == "nat":
            labels = strategy.index.to_list()
            labels[2] = pd.NaT
            strategy.index = pd.DatetimeIndex(labels)
        elif fault == "range":
            strategy.index = pd.RangeIndex(len(strategy))
        else:
            strategy = strategy.astype(complex)
        original_s, original_b = strategy.copy(deep=True), benchmark.copy(deep=True)
        with pytest.raises(ValueError):
            utils._prepare_benchmark_pair(strategy, benchmark, frequency="daily")
        pd.testing.assert_series_equal(strategy, original_s)
        pd.testing.assert_series_equal(benchmark, original_b)

    def test_insufficient_shared_boundaries_produce_no_pairs(self):
        strategy = pd.Series(
            [0, 0.1], index=pd.to_datetime(["2026-09-01", "2026-09-03"])
        )
        benchmark = pd.Series(
            [0, 0.2], index=pd.to_datetime(["2026-09-02", "2026-09-03"])
        )
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=False, frequency="daily"
        )
        assert actual.empty and reference.empty

    def test_unrepresentable_compounding_raises_without_mutation(self):
        dates = pd.date_range("2026-09-04", periods=3)
        strategy = pd.Series([0, 1e308, 1e308], index=dates)
        benchmark = pd.Series([0, 0.01], index=dates[[0, 2]])
        original = strategy.copy(deep=True)
        with pytest.raises(ValueError, match="finite"):
            utils._prepare_benchmark_pair(
                strategy, benchmark, prepare_returns=False, frequency="daily"
            )
        pd.testing.assert_series_equal(strategy, original)

    @pytest.mark.parametrize("benchmark_fault", ["duplicate", "range", "complex"])
    def test_validate_both_inputs_before_cached_preparation(
        self, interval_pair, benchmark_fault, monkeypatch
    ):
        strategy, benchmark = interval_pair
        if benchmark_fault == "duplicate":
            benchmark.index = pd.DatetimeIndex([benchmark.index[0]] * len(benchmark))
        elif benchmark_fault == "range":
            benchmark.index = pd.RangeIndex(len(benchmark))
        else:
            benchmark = benchmark.astype(complex)
        original_s, original_b = strategy.copy(deep=True), benchmark.copy(deep=True)

        def premature_preparation(*args, **kwargs):
            pytest.fail("Both inputs must be validated before cached preparation")

        monkeypatch.setattr(utils, "_prepare_returns", premature_preparation)
        with pytest.raises(ValueError):
            utils._prepare_benchmark_pair(strategy, benchmark, frequency="daily")
        pd.testing.assert_series_equal(strategy, original_s)
        pd.testing.assert_series_equal(benchmark, original_b)

    @pytest.mark.parametrize(
        "shape", ["strategy-frame", "missing-benchmark", "empty-frame"]
    )
    def test_unsupported_shapes_are_explicit(self, interval_pair, shape):
        strategy, benchmark = interval_pair
        if shape == "strategy-frame":
            strategy = strategy.to_frame()
        elif shape == "missing-benchmark":
            benchmark = None
        else:
            benchmark = pd.DataFrame(index=benchmark.index)
        with pytest.raises(ValueError, match="requires"):
            utils._prepare_benchmark_pair(strategy, benchmark, frequency="daily")

    @pytest.mark.parametrize("aware", [False, True])
    def test_matching_rejects_multiple_observations_per_day(self, aware):
        index = pd.date_range("2026-01-05 09:30", periods=5, freq="h")
        if aware:
            index = index.tz_localize("America/New_York")
        strategy = pd.Series([0, 0.1, -0.1, 0.2, -0.2], index=index)
        benchmark = strategy.iloc[[0, 2, 4]]
        with pytest.raises(ValueError, match="one observation per date"):
            utils._prepare_benchmark_pair(strategy, benchmark, frequency="daily")

    def test_matching_accepts_aware_and_naive_date_labels(self, interval_pair):
        strategy, benchmark = interval_pair
        benchmark.index = benchmark.index.tz_localize("UTC")
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=False, frequency="weekly"
        )
        expected = pd.Series(
            [0.019494, 0.019898, 0.00444806],
            index=strategy.index[[3, 6, 10]],
            name="Strategy",
        )
        pd.testing.assert_series_equal(actual, expected, atol=1e-14, rtol=1e-12)
        pd.testing.assert_series_equal(
            reference, expected.rename("Benchmark"), atol=1e-14, rtol=1e-12
        )

    @pytest.mark.parametrize("enabled", [None, "calendar", "timestamp", 1])
    def test_removed_matching_switch_is_rejected(self, interval_pair, enabled):
        with pytest.raises(TypeError, match="match_periods"):
            utils._prepare_benchmark_pair(*interval_pair, match_periods=enabled)


class TestMatchedCalendarFrequencies:
    """Keep complete calendar periods, including real weekend/holiday returns."""

    def test_weekly_sunday_portfolio_and_weekday_benchmark(self):
        sundays = pd.date_range("2026-01-04", periods=6, freq="W-SUN")
        strategy = pd.Series([9, 0.07, -0.03, 0.02, 0.04, 0.01], index=sundays)
        weekdays = pd.bdate_range("2026-01-02", "2026-02-06")
        benchmark = pd.Series(0.01, index=weekdays)
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, prepare_returns=False, frequency="weekly"
        )
        pd.testing.assert_series_equal(
            actual,
            pd.Series([0.07, -0.03, 0.02, 0.04, 0.01], index=sundays[1:]),
            check_freq=False,
        )
        np.testing.assert_allclose(reference, [1.01**5 - 1] * 5, rtol=1e-12)
        assert actual.attrs["frequency"] == "weekly"
        assert actual.attrs["periods_per_year"] == 52

    def test_calendar_month_end_and_business_month_end(self):
        days = pd.date_range("2026-01-30", "2026-05-31")
        strategy = pd.Series(100 + np.arange(len(days), dtype=float), index=days)
        benchmark = strategy.loc[days.dayofweek < 5] * 2
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, frequency="monthly"
        )
        # February and May end on weekends. Each side keeps its own valuation:
        # the active portfolio uses month end, the closed market Friday.
        s = np.array([101, 129, 160, 190, 221], dtype=float)
        b = np.array([100, 128, 160, 190, 219], dtype=float)
        np.testing.assert_allclose(actual, s[1:] / s[:-1] - 1, atol=1e-14)
        np.testing.assert_allclose(reference, b[1:] / b[:-1] - 1, atol=1e-14)
        assert list(actual.index) == list(
            pd.to_datetime(["2026-02-28", "2026-03-31", "2026-04-30", "2026-05-31"])
        )
        assert actual.attrs["frequency"] == "monthly"

    def test_mixed_history_supports_explicit_monthly_cadence(self):
        dates = pd.date_range("2026-01-01", "2026-01-31").append(
            pd.DatetimeIndex(
                [
                    "2026-02-07",
                    "2026-02-14",
                    "2026-02-21",
                    "2026-02-28",
                    "2026-03-31",
                    "2026-04-30",
                    "2026-05-31",
                    "2026-06-30",
                ]
            )
        )
        strategy = pd.Series(
            100 + (dates - pd.Timestamp("2026-01-01")).days, index=dates, dtype=float
        )
        weekdays = pd.bdate_range("2026-01-01", "2026-06-30")
        benchmark = pd.Series(
            200 + (weekdays - pd.Timestamp("2026-01-01")).days,
            index=weekdays,
            dtype=float,
        )
        actual, reference = utils._prepare_benchmark_pair(
            strategy, benchmark, frequency="monthly"
        )
        s = np.array([130, 158, 189, 219, 250, 280], dtype=float)
        b = np.array([229, 257, 289, 319, 348, 380], dtype=float)
        np.testing.assert_allclose(actual, s[1:] / s[:-1] - 1, atol=1e-14)
        np.testing.assert_allclose(reference, b[1:] / b[:-1] - 1, atol=1e-14)
        assert actual.attrs["frequency"] == "monthly"
        assert len(actual) == 5

    @pytest.mark.parametrize("requested", ["quarterly"])
    def test_requested_quarters_do_not_skip_internal_months(self, requested):
        dates = pd.to_datetime(
            [
                "2026-01-31",
                "2026-03-31",
                "2026-05-31",
                "2026-06-30",
                "2026-08-31",
                "2026-09-30",
                "2026-11-30",
                "2026-12-31",
            ]
        )
        strategy = pd.Series([9, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], index=dates)
        benchmark = strategy * 0.5
        actual, reference = utils._prepare_benchmark_pair(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency=requested,
        )
        np.testing.assert_allclose(
            actual, [1.02 * 1.03 - 1, 1.04 * 1.05 - 1, 1.06 * 1.07 - 1]
        )
        np.testing.assert_allclose(
            reference, [1.01 * 1.015 - 1, 1.02 * 1.025 - 1, 1.03 * 1.035 - 1]
        )
        assert list(actual.index) == list(
            pd.to_datetime(["2026-06-30", "2026-09-30", "2026-12-31"])
        )
        assert actual.attrs["frequency"] == "quarterly"

    def test_explicit_yearly_combines_complete_history(self):
        dates = pd.to_datetime(
            [
                "2022-12-31",
                "2023-07-31",
                "2023-12-31",
                "2024-08-31",
                "2024-12-31",
                "2025-12-31",
            ]
        )
        strategy = pd.Series([9, 0.01, 0.02, 0.03, 0.04, 0.05], index=dates)
        actual, _ = utils._prepare_benchmark_pair(
            strategy, strategy, prepare_returns=False, frequency="yearly"
        )
        np.testing.assert_allclose(actual, [1.01 * 1.02 - 1, 1.03 * 1.04 - 1, 0.05])
        assert actual.attrs["frequency"] == "yearly"

    def test_requested_frequency_does_not_fall_back(self):
        """The supplied monthly returns cannot be split into weekly returns."""
        dates = pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"])
        strategy = pd.Series([0, 0.01, 0.02], index=dates)
        with pytest.raises(ValueError, match="weekly"):
            utils._prepare_benchmark_pair(
                strategy,
                strategy,
                prepare_returns=False,
                frequency="weekly",
            )

    @pytest.mark.parametrize("provided_holiday_return", [False, True])
    def test_four_day_closure_and_supplied_holiday_return(
        self, provided_holiday_return
    ):
        strategy = pd.Series(
            [0, 0.1], index=pd.to_datetime(["2026-09-04", "2026-09-08"])
        )
        benchmark = pd.Series(
            [0, 0.02, 0.03] if provided_holiday_return else [0, 0.03],
            index=pd.to_datetime(
                ["2026-09-04", "2026-09-07", "2026-09-08"]
                if provided_holiday_return
                else ["2026-09-04", "2026-09-08"]
            ),
        )
        actual, reference = utils._prepare_benchmark_pair(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="daily",
            holidays=["2026-09-07"],
        )
        np.testing.assert_allclose(actual, [0.1])
        np.testing.assert_allclose(
            reference, [1.02 * 1.03 - 1 if provided_holiday_return else 0.03]
        )
        with pytest.raises(ValueError, match="daily"):
            utils._prepare_benchmark_pair(
                strategy,
                benchmark,
                prepare_returns=False,
                frequency="daily",
            )

    def test_friday_holiday_allows_thursday_at_sunday_boundary(self):
        sundays = pd.date_range("2026-03-29", periods=3, freq="W-SUN")
        strategy = pd.Series([0, 0.05, 0.1], index=sundays)
        dates = pd.to_datetime(["2026-03-27", "2026-04-02", "2026-04-10"])
        benchmark = pd.Series([0, 0.03, 0.04], index=dates)
        actual, reference = utils._prepare_benchmark_pair(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="weekly",
            holidays=["2026-04-03"],
        )
        np.testing.assert_allclose(actual, [0.05, 0.1])
        np.testing.assert_allclose(reference, [0.03, 0.04])
        with pytest.raises(ValueError, match="weekly"):
            utils._prepare_benchmark_pair(
                strategy,
                benchmark,
                prepare_returns=False,
                frequency="weekly",
            )

    def test_unknown_history_cannot_be_hidden_by_coarsening(self):
        dates = pd.date_range("2026-01-01", "2026-04-30")
        strategy = pd.Series(0.01, index=dates)
        strategy.loc["2026-03-10"] = np.nan
        before = strategy.copy(deep=True)
        with pytest.raises(ValueError, match="unknown"):
            utils._prepare_benchmark_pair(
                strategy,
                pd.Series(0.02, index=dates),
                prepare_returns=False,
                frequency="monthly",
            )
        pd.testing.assert_series_equal(strategy, before)

    @pytest.mark.parametrize("frequency", ["hourly", "D", True, 12, []])
    def test_only_five_named_frequencies(self, frequency):
        dates = pd.date_range("2026-01-01", periods=4)
        series = pd.Series([0, 0.01, 0.02, -0.01], index=dates)
        with pytest.raises(ValueError, match="frequency"):
            utils._prepare_benchmark_pair(series, series, frequency=frequency)


@pytest.mark.parametrize("holidays", ["2026-09-07", [pd.NaT], [42], True])
def test_invalid_holidays_are_rejected_before_preparation(holidays, monkeypatch):
    dates = pd.date_range("2026-09-01", periods=4)
    series = pd.Series([0, 0.01, -0.02, 0.03], index=dates)

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Invalid calendars must not reach preparation")

    monkeypatch.setattr(utils, "_prepare_returns", unexpected_preparation)
    with pytest.raises(ValueError, match="holidays"):
        utils._prepare_benchmark_pair(
            series, series, frequency="daily", holidays=holidays
        )


def test_daily_request_does_not_return_an_arbitrary_interval():
    dates = pd.to_datetime(["2026-01-05", "2026-03-17", "2026-08-11", "2027-02-09"])
    series = pd.Series([0, 0.01, -0.02, 0.03], index=dates)
    with pytest.raises(ValueError, match="daily comparison periods"):
        utils._prepare_benchmark_pair(
            series, series, prepare_returns=False, frequency="daily"
        )


def test_supplied_weekend_holidays_still_establish_calendar_daily():
    dates = pd.date_range("2026-01-09", periods=5)
    series = pd.Series([0, 0.01, -0.02, 0.03, 0.02], index=dates)
    actual, reference = utils._prepare_benchmark_pair(
        series,
        series,
        prepare_returns=False,
        frequency="daily",
        holidays=["2026-01-10", "2026-01-11"],
    )
    np.testing.assert_allclose(actual, [0.01, -0.02, 0.03, 0.02])
    np.testing.assert_allclose(reference, actual)
    assert actual.attrs["periods_per_year"] == 365


def test_supplied_friday_holiday_is_used_at_sunday_boundary():
    sundays = pd.to_datetime(["2026-03-29", "2026-04-05", "2026-04-12"])
    strategy = pd.Series([0, 0.05, 0.1], index=sundays)
    dates = pd.to_datetime(["2026-03-27", "2026-04-02", "2026-04-03", "2026-04-10"])
    benchmark = pd.Series([0, 0.03, 0.02, 0.04], index=dates)
    _, reference = utils._prepare_benchmark_pair(
        strategy,
        benchmark,
        prepare_returns=False,
        frequency="weekly",
        holidays=["2026-04-03"],
    )
    np.testing.assert_allclose(reference, [1.03 * 1.02 - 1, 0.04])


@pytest.mark.parametrize(
    "zone,clock,utc_labels",
    [
        (
            "America/New_York",
            "16:00",
            ["2026-03-08 20:00", "2026-03-15 20:00", "2026-03-22 20:00"],
        ),
        (
            "Asia/Tokyo",
            "00:30",
            ["2026-03-07 15:30", "2026-03-14 15:30", "2026-03-21 15:30"],
        ),
    ],
)
def test_closed_strategy_boundary_preserves_local_clock_across_dst(
    zone, clock, utc_labels
):
    # A Friday valuation carried to Sunday uses Sunday's timezone offset.
    # Tokyo also checks that the public UTC label can fall on the prior date.
    hours, minutes = map(int, clock.split(":"))
    fridays = (
        pd.date_range("2026-02-27", periods=4, freq="W-FRI")
        + pd.Timedelta(hours=hours, minutes=minutes)
    ).tz_localize(zone)
    strategy = pd.Series([0, 0.01, 0.02, 0.03], index=fridays)
    benchmark = pd.Series(
        [0, 0.02, -0.01, 0.03],
        index=pd.date_range("2026-03-01", periods=4, freq="W-SUN"),
    )
    before = strategy.copy(deep=True)
    before.index = strategy.index.copy(deep=True)
    actual, reference = utils._prepare_benchmark_pair(
        strategy,
        benchmark,
        prepare_returns=False,
        frequency="weekly",
    )
    np.testing.assert_allclose(actual, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(reference, [0.02, -0.01, 0.03])
    pd.testing.assert_index_equal(actual.index, pd.to_datetime(utc_labels))
    pd.testing.assert_series_equal(strategy, before)


_CALENDAR_INPUTS = (
    "calendar-daily",
    "business-daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
)


@pytest.fixture
def calendar_frequency_prices():
    """Four years of independently sampled valuations, including a leap year.

    Supplied month/quarter/year-end observations remain present on holidays.
    Business-daily observations omit the declared January 1 and Good Friday
    closures; weekly observations are Sunday valuations.
    """
    days = pd.date_range("2021-01-01", "2024-12-31")
    x = np.arange(len(days), dtype=float)
    portfolio = pd.Series(
        100 + 0.01 * x + 2 * np.sin(0.13 * x), index=days, name="Portfolio"
    )
    benchmark = pd.Series(
        200 + 0.02 * x + 3 * np.cos(0.11 * x), index=days, name="Benchmark"
    )
    holidays = pd.to_datetime(["2024-01-01", "2024-03-29"])
    indexes = {
        "calendar-daily": days,
        "business-daily": days[(days.dayofweek < 5) & ~days.isin(holidays)],
        "weekly": pd.date_range(days[0], days[-1], freq="W-SUN"),
        "monthly": pd.date_range(days[0], days[-1], freq=pd.offsets.MonthEnd()),
        "quarterly": pd.date_range(days[0], days[-1], freq=pd.offsets.QuarterEnd()),
        "yearly": pd.date_range(days[0], days[-1], freq=pd.offsets.YearEnd()),
    }
    return portfolio, benchmark, indexes, holidays


@pytest.mark.parametrize("left_kind", _CALENDAR_INPUTS)
@pytest.mark.parametrize("right_kind", _CALENDAR_INPUTS)
def test_calendar_frequency_pair_matrix(
    calendar_frequency_prices, left_kind, right_kind
):
    """A bounded frequency matrix checks complete values, not just correlations.

    Month/quarter/year boundaries are nested; Sunday weeks generally are not
    nested with them. During these four years, missing open-day valuations
    prevent Sunday-only data from supplying every required calendar year end.
    """
    portfolio, market, indexes, holidays = calendar_frequency_prices
    left = portfolio.loc[indexes[left_kind]]
    right = market.loc[indexes[right_kind]]
    before_left, before_right = left.copy(deep=True), right.copy(deep=True)
    order = {
        "calendar-daily": 0,
        "business-daily": 0,
        "weekly": 1,
        "monthly": 2,
        "quarterly": 3,
        "yearly": 4,
    }
    rank = max(order[left_kind], order[right_kind])
    cadence = ("daily", "weekly", "monthly", "quarterly", "yearly")[rank]
    options = dict(frequency=cadence, holidays=holidays)
    incompatible = "weekly" in (left_kind, right_kind) and rank >= 2
    if incompatible:
        with pytest.raises(ValueError, match="comparison periods"):
            utils._prepare_benchmark_pair(left, right, **options)
    else:
        actual, reference = utils._prepare_benchmark_pair(left, right, **options)
        if cadence == "daily":
            endpoints = indexes[
                "calendar-daily"
                if left_kind == right_kind == "calendar-daily"
                else "business-daily"
            ]
        else:
            endpoints = indexes[cadence]
        # Endpoint prices independently define each return. No production
        # boundary planner, compounding, eligibility or factor inference is reused.
        s = left.reindex(endpoints, method="ffill").to_numpy()
        b = right.reindex(endpoints, method="ffill").to_numpy()
        np.testing.assert_allclose(actual, s[1:] / s[:-1] - 1, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(
            reference, b[1:] / b[:-1] - 1, rtol=1e-11, atol=1e-13
        )
        pd.testing.assert_index_equal(actual.index, endpoints[1:], check_names=False)
        assert actual.index.freq is None and reference.index.freq is None
        assert actual.attrs["frequency"] == cadence
        # The endpoints cover 2021–2023 plus 365 days of leap-year 2024.
        expected_factor = (
            (len(endpoints) - 1) / (3 + 365 / 366)
            if cadence == "daily"
            else {"weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}[cadence]
        )
        assert actual.attrs["periods_per_year"] == pytest.approx(expected_factor)
    pd.testing.assert_series_equal(left, before_left)
    pd.testing.assert_series_equal(right, before_right)


@pytest.mark.parametrize(
    "holiday,dates",
    [
        (
            "2024-03-29",
            ["2024-03-28", "2024-03-29", "2024-03-30", "2024-03-31", "2024-04-01"],
        ),
        (
            "2024-01-01",
            ["2023-12-29", "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"],
        ),
        (
            "2024-09-30",
            ["2024-09-27", "2024-09-28", "2024-09-29", "2024-09-30", "2024-10-01"],
        ),
    ],
)
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("frequency", ["daily"])
def test_four_day_closure_crosses_month_quarter_or_year(
    holiday, dates, swap, frequency
):
    # All four supplied returns belong to the same complete comparison. The
    # Friday/Monday holiday makes the missing interior weekday explainable.
    full = pd.Series([9, 0.01, -0.02, 0.03, -0.01], index=pd.to_datetime(dates))
    sparse = pd.Series([7, 0.015], index=full.index[[0, 4]])
    left, right = (sparse, full) if swap else (full, sparse)
    expected = [1.01 * 0.98 * 1.03 * 0.99 - 1, 0.015]
    if swap:
        expected.reverse()
    actual, reference = utils._prepare_benchmark_pair(
        left,
        right,
        prepare_returns=False,
        frequency=frequency,
        holidays=[holiday],
    )
    np.testing.assert_allclose(
        [actual.iloc[0], reference.iloc[0]], expected, rtol=1e-12, atol=1e-14
    )
    pd.testing.assert_index_equal(actual.index, full.index[[4]])
    # Removing the holiday declaration must not silently retain daily output.
    # This short sample has no complete coarser calendar period either.
    with pytest.raises(ValueError, match="comparison periods"):
        utils._prepare_benchmark_pair(
            left, right, prepare_returns=False, frequency=frequency
        )


@pytest.mark.parametrize("frequency", ["daily", "monthly"])
@pytest.mark.parametrize("side", [0, 1])
def test_provided_missing_holiday_return_is_not_a_closure(frequency, side):
    dates = pd.date_range("2024-01-01", "2024-04-30")
    inputs = [pd.Series(0.01, index=dates), pd.Series(0.02, index=dates)]
    inputs[side].loc["2024-03-29"] = np.nan
    before = [series.copy(deep=True) for series in inputs]
    with pytest.raises(ValueError, match="unknown"):
        utils._prepare_benchmark_pair(
            *inputs,
            prepare_returns=False,
            frequency=frequency,
            holidays=["2024-03-29"],
        )
    for actual, expected in zip(inputs, before, strict=True):
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("fault", ["duplicate-end", "same-day", "backwards"])
def test_conflicting_period_endpoints_rejected_on_either_input(side, fault):
    dates = pd.date_range("2024-03-25", periods=7)
    labels = dates.to_list()
    if fault == "duplicate-end":
        labels[3] = labels[2]
    elif fault == "same-day":
        labels[3] = labels[2] + pd.Timedelta(hours=1)
    else:
        labels[2], labels[3] = labels[3], labels[2]
    inputs = [
        pd.Series([0, 0.01, -0.02, 0.03, 0.02, -0.01, 0.01], index=dates)
        for _ in range(2)
    ]
    inputs[side].index = pd.DatetimeIndex(labels)
    before = [series.copy(deep=True) for series in inputs]
    with pytest.raises(ValueError):
        utils._prepare_benchmark_pair(
            *inputs, prepare_returns=False, frequency="daily", holidays=["2024-03-29"]
        )
    for actual, expected in zip(inputs, before, strict=True):
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("position", ["start", "middle", "end"])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("frequency", ["daily"])
def test_one_shared_anchor_does_not_hide_incompatible_long_history(
    position, swap, frequency
):
    """A shared date must not disguise years of incompatible return intervals."""
    # December 31, 2023 is Sunday; the adjacent year ends are open weekdays or
    # Saturday. A Sunday-only input cannot provide all yearly boundaries.
    first, last = {
        "start": ("2023-12-31", "2024-12-31"),
        "middle": ("2021-01-01", "2024-12-31"),
        "end": ("2021-01-01", "2023-12-31"),
    }[position]
    weekly = pd.Series(100.0, index=pd.date_range(first, last, freq="W-SUN"))
    yearly = pd.Series(
        200.0, index=pd.date_range(first, last, freq=pd.offsets.YearEnd())
    )
    inputs = (yearly, weekly) if swap else (weekly, yearly)
    assert list(weekly.index.intersection(yearly.index)) == [pd.Timestamp("2023-12-31")]
    before = [series.copy(deep=True) for series in inputs]
    with pytest.raises(ValueError, match="comparison periods"):
        utils._prepare_benchmark_pair(*inputs, frequency=frequency)
    for actual, expected in zip(inputs, before, strict=True):
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("sample", ["staggered", "one-observation", "holiday-weekend"])
@pytest.mark.parametrize("frequency", ["daily"])
@pytest.mark.parametrize("prepare", [False, True])
def test_valid_single_anchor_samples_remain_empty(sample, frequency, prepare):
    """Short histories without a shared return interval produce empty results."""
    # Shortness concerns overlapping coverage, not the size of the other input.
    holidays = None
    if sample == "staggered":
        strategy = pd.Series(
            [0.0, 0.1], index=pd.to_datetime(["2026-09-01", "2026-09-03"])
        )
        benchmark = pd.Series(
            [0.0, 0.2], index=pd.to_datetime(["2026-09-02", "2026-09-03"])
        )
    elif sample == "one-observation":
        strategy = pd.Series([0.0], index=pd.to_datetime(["2023-12-31"]))
        benchmark = pd.Series(0.01, index=pd.date_range("2021-01-01", "2024-12-31"))
    else:
        strategy = pd.Series(
            [0.0, 0.1], index=pd.to_datetime(["2024-03-28", "2024-04-01"])
        )
        benchmark = pd.Series(
            [0.0, 0.2], index=pd.to_datetime(["2024-03-29", "2024-04-01"])
        )
        holidays = ["2024-03-29"]
    before = [series.copy(deep=True) for series in (strategy, benchmark)]
    actual, reference = utils._prepare_benchmark_pair(
        strategy,
        benchmark,
        prepare_returns=prepare,
        frequency=frequency,
        holidays=holidays,
    )
    assert actual.empty and reference.empty
    assert actual.attrs["frequency"] == "daily"
    assert actual.index.freq is None and reference.index.freq is None
    for observed, expected in zip((strategy, benchmark), before, strict=True):
        pd.testing.assert_series_equal(observed, expected)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("holiday", [False, True])
def test_missing_closed_tail_extends_either_input_but_supplied_returns_win(
    swap, holiday
):
    days = pd.date_range("2024-01-01", "2024-03-31")
    full = pd.Series(100 + np.arange(len(days), dtype=float), index=days, name="Full")
    # March 31 is Sunday. A Friday holiday permits Thursday's final valuation.
    last = "2024-03-28" if holiday else "2024-03-29"
    cut = full.loc[:last].rename("Cut")
    options = dict(frequency="monthly", holidays=["2024-03-29"] if holiday else None)
    left, right = (full, cut) if swap else (cut, full)
    actual, reference = utils._prepare_benchmark_pair(left, right, **options)
    held = np.array([130.0, 159.0, 187.0 if holiday else 188.0])
    complete = np.array([130.0, 159.0, 190.0])
    expected_left, expected_right = (complete, held) if swap else (held, complete)
    np.testing.assert_allclose(
        actual, expected_left[1:] / expected_left[:-1] - 1, atol=1e-14
    )
    np.testing.assert_allclose(
        reference, expected_right[1:] / expected_right[:-1] - 1, atol=1e-14
    )
    pd.testing.assert_index_equal(
        actual.index, pd.to_datetime(["2024-02-29", "2024-03-31"])
    )
    # When those weekend/holiday observations are actually supplied, include
    # them even though the declared calendar permits their absence.
    supplied, _ = utils._prepare_benchmark_pair(full, full, **options)
    np.testing.assert_allclose(supplied, complete[1:] / complete[:-1] - 1, atol=1e-14)


@pytest.mark.parametrize("swap", [False, True])
def test_open_weekday_blocks_tail_extension_but_not_a_supplied_longer_return(swap):
    days = pd.date_range("2024-01-01", "2024-04-30")
    full = pd.Series(100 + np.arange(len(days), dtype=float), index=days)
    cut = full.loc[:"2024-04-26"]  # Friday; Monday 29 and Tuesday 30 are open.
    left, right = (full, cut) if swap else (cut, full)
    actual, _ = utils._prepare_benchmark_pair(left, right, frequency="monthly")
    assert list(actual.index) == list(pd.to_datetime(["2024-02-29", "2024-03-31"]))
    # A supplied April 30 price/interval return supplies the missing boundary;
    # no daily interpolation is needed for a monthly comparison.
    completed = pd.concat([cut, full.loc["2024-04-30":]])
    left, right = (full, completed) if swap else (completed, full)
    actual, reference = utils._prepare_benchmark_pair(left, right, frequency="monthly")
    expected = np.array([159 / 130 - 1, 190 / 159 - 1, 220 / 190 - 1])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(reference, expected, atol=1e-14)

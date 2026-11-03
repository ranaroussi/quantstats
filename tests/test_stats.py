"""
Tests for quantstats.stats module
"""

import numpy as np
import pandas as pd
import pytest

from quantstats import stats, utils


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("pandas_method", [False, True])
@pytest.mark.parametrize(
    "order", [None, (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
)
def test_frequency_preserves_statistics_for_small_returns(
    prepare, pandas_method, order
):
    """Rescaling valid returns must not erase beta, correlation or active risk.

    With b = [-3, -1, 1, 3]e-18 and s = 2b + 1e-18, beta is 2,
    per-period alpha is 1e-18 and R-squared is 1. Active returns are
    [-2, 0, 2, 4]e-18, giving information ratio sqrt(3/20).
    Weekly cases insert an exactly cancelling 100% gain and 50% loss
    around each small return, in every possible order.
    """
    frequency = "daily" if order is None else "weekly"
    dates = (
        pd.date_range("2026-01-05", periods=5)
        if order is None
        else pd.date_range("2026-01-04", periods=5, freq="W-SUN")
    )
    benchmark = pd.Series([0.0, -3e-18, -1e-18, 1e-18, 3e-18], index=dates)
    strategy = 2 * benchmark + 1e-18
    strategy.iloc[0] = 0.0
    if order is not None:
        labels, values = [dates[0]], [0.0]
        for end, small_return in strategy.iloc[1:].items():
            labels.extend([end - pd.Timedelta(days=5), end - pd.Timedelta(days=3), end])
            chunk = [1.0, -0.5, small_return]
            values.extend(chunk[i] for i in order)
        strategy = pd.Series(values, index=pd.DatetimeIndex(labels))
    before_s, before_b = strategy.copy(deep=True), benchmark.copy(deep=True)
    if pandas_method:
        import quantstats as qs

        qs.extend_pandas()

    def call(name, **kwargs):
        if pandas_method:
            return getattr(strategy, name)(benchmark, **kwargs)
        return getattr(stats, name)(strategy, benchmark, **kwargs)

    kwargs = {"frequency": frequency, "prepare_returns": prepare}
    with np.errstate(all="raise"):
        result = call("greeks", periods=1, **kwargs)
        assert result["beta"] == pytest.approx(2, rel=1e-14, abs=0)
        assert result["alpha"] == pytest.approx(1e-18, rel=1e-14, abs=0)
        assert call("r_squared", **kwargs) == pytest.approx(1, rel=1e-14, abs=0)
        assert call("r2", **kwargs) == pytest.approx(1, rel=1e-14, abs=0)
        assert call("information_ratio", **kwargs) == pytest.approx(
            np.sqrt(3 / 20), rel=1e-14, abs=0
        )
    pd.testing.assert_series_equal(strategy, before_s)
    pd.testing.assert_series_equal(benchmark, before_b)


@pytest.fixture
def sample_returns():
    """Generate sample daily returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.02, index=dates, name="Strategy")
    return returns


@pytest.fixture
def sample_benchmark():
    """Generate sample benchmark returns for testing."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.015, index=dates, name="Benchmark")
    return returns


@pytest.fixture
def positive_returns():
    """Generate strictly positive returns for testing edge cases."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.abs(np.random.randn(100) * 0.01) + 0.001, index=dates)
    return returns


@pytest.fixture
def negative_returns():
    """Generate strictly negative returns for testing edge cases."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(-np.abs(np.random.randn(100) * 0.01) - 0.001, index=dates)
    return returns


class TestBasicStats:
    """Test basic statistical functions."""

    def test_comp(self, sample_returns):
        """Test compound returns calculation."""
        result = stats.comp(sample_returns)
        assert isinstance(result, float)
        # Compound return should be finite
        assert np.isfinite(result)

    def test_compsum(self, sample_returns):
        """Test cumulative compound returns."""
        result = stats.compsum(sample_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        # Final value should match comp()
        np.testing.assert_almost_equal(
            result.iloc[-1], stats.comp(sample_returns), decimal=10
        )

    def test_exposure(self, sample_returns):
        """Test exposure calculation."""
        result = stats.exposure(sample_returns)
        assert 0 <= result <= 1

    def test_win_rate(self, sample_returns):
        """Test win rate calculation."""
        result = stats.win_rate(sample_returns)
        assert 0 <= result <= 1

    def test_win_rate_all_positive(self, positive_returns):
        """Test win rate with all positive returns."""
        result = stats.win_rate(positive_returns)
        assert result == 1.0

    def test_win_rate_all_negative(self, negative_returns):
        """Test win rate with all negative returns."""
        result = stats.win_rate(negative_returns)
        assert result == 0.0


class TestRiskMetrics:
    """Test risk-related metrics."""

    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        result = stats.volatility(sample_returns)
        assert result > 0
        assert np.isfinite(result)

    def test_volatility_annualized(self, sample_returns):
        """Test annualized volatility."""
        daily_vol = stats.volatility(sample_returns, annualize=False)
        annual_vol = stats.volatility(sample_returns, annualize=True)
        # Annualized should be approximately sqrt(252) times daily
        expected_ratio = np.sqrt(252)
        actual_ratio = annual_vol / daily_vol
        np.testing.assert_almost_equal(actual_ratio, expected_ratio, decimal=5)

    def test_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        result = stats.max_drawdown(sample_returns)
        assert result <= 0  # Drawdown is always negative or zero
        assert result >= -1  # Cannot lose more than 100%

    def test_var(self, sample_returns):
        """Test Value at Risk calculation."""
        result = stats.var(sample_returns)
        assert result < 0  # VaR is typically negative (loss)

    def test_cvar(self, sample_returns):
        """Test Conditional Value at Risk (Expected Shortfall)."""
        var = stats.var(sample_returns)
        cvar = stats.cvar(sample_returns)
        # CVaR should be more extreme than VaR
        assert cvar <= var


class TestRatios:
    """Test risk-adjusted return ratios."""

    def test_sharpe(self, sample_returns):
        """Test Sharpe ratio calculation."""
        result = stats.sharpe(sample_returns)
        assert np.isfinite(result)

    def test_sharpe_with_rf(self, sample_returns):
        """Test Sharpe ratio with risk-free rate."""
        result_no_rf = stats.sharpe(sample_returns, rf=0)
        result_with_rf = stats.sharpe(sample_returns, rf=0.02)
        # Handle both scalar and Series results
        if isinstance(result_no_rf, pd.Series):
            result_no_rf = result_no_rf.iloc[0]
            result_with_rf = result_with_rf.iloc[0]
        # Higher rf should lower Sharpe ratio
        assert result_with_rf < result_no_rf

    def test_sortino(self, sample_returns):
        """Test Sortino ratio calculation."""
        result = stats.sortino(sample_returns)
        # Handle both scalar and Series results
        if isinstance(result, pd.Series):
            result = result.iloc[0]
        assert np.isfinite(result)

    def test_sortino_vs_sharpe(self, sample_returns):
        """Test that Sortino uses downside deviation."""
        sharpe = stats.sharpe(sample_returns)
        sortino = stats.sortino(sample_returns)
        # Handle both scalar and Series results
        if hasattr(sharpe, "values"):
            sharpe = float(sharpe.values[0]) if len(sharpe) > 0 else float(sharpe)
        if hasattr(sortino, "values"):
            sortino = float(sortino.values[0]) if len(sortino) > 0 else float(sortino)
        # They should be different (Sortino only penalizes downside)
        assert sharpe != sortino

    def test_calmar(self, sample_returns):
        """Test Calmar ratio calculation."""
        result = stats.calmar(sample_returns)
        assert np.isfinite(result)

    def test_omega(self, sample_returns):
        """Test Omega ratio calculation."""
        result = stats.omega(sample_returns)
        assert result > 0  # Omega is always positive

    def test_cagr(self, sample_returns):
        """Test CAGR calculation."""
        result = stats.cagr(sample_returns)
        assert np.isfinite(result)


class TestBenchmarkComparison:
    """Test benchmark comparison functions."""

    def test_greeks(self, sample_returns, sample_benchmark):
        """Test alpha/beta calculation."""
        result = stats.greeks(sample_returns, sample_benchmark)
        assert "alpha" in result
        assert "beta" in result
        assert np.isfinite(result["alpha"])
        assert np.isfinite(result["beta"])

    def test_r_squared(self, sample_returns, sample_benchmark):
        """Test R-squared calculation."""
        result = stats.r_squared(sample_returns, sample_benchmark)
        assert 0 <= result <= 1

    def test_information_ratio(self, sample_returns, sample_benchmark):
        """Test Information Ratio calculation."""
        result = stats.information_ratio(sample_returns, sample_benchmark)
        assert np.isfinite(result)

    def test_treynor_ratio(self, sample_returns, sample_benchmark):
        """Test Treynor Ratio calculation."""
        result = stats.treynor_ratio(sample_returns, sample_benchmark)
        assert np.isfinite(result)


class TestDrawdown:
    """Test drawdown-related functions."""

    def test_to_drawdown_series(self, sample_returns):
        """Test drawdown series conversion."""
        result = stats.to_drawdown_series(sample_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        # All values should be <= 0 (drawdowns are negative)
        assert (result <= 0).all()

    def test_drawdown_details(self, sample_returns):
        """Test drawdown details extraction."""
        dd = stats.to_drawdown_series(sample_returns)
        result = stats.drawdown_details(dd)
        assert isinstance(result, pd.DataFrame)
        # Should have standard columns
        assert "start" in result.columns
        assert "end" in result.columns
        assert "max drawdown" in result.columns


class TestConsecutive:
    """Test consecutive wins/losses functions."""

    def test_consecutive_wins(self, sample_returns):
        """Test consecutive wins calculation."""
        result = stats.consecutive_wins(sample_returns)
        assert isinstance(result, (int, np.integer))
        assert result >= 0

    def test_consecutive_losses(self, sample_returns):
        """Test consecutive losses calculation."""
        result = stats.consecutive_losses(sample_returns)
        assert isinstance(result, (int, np.integer))
        assert result >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_returns(self):
        """Test handling of empty returns."""
        empty = pd.Series([], dtype=float)
        with pytest.raises(Exception):
            stats.sharpe(empty)

    def test_single_return(self):
        """Test handling of single return value."""
        single = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1))
        # Should not raise error
        result = stats.comp(single)
        assert np.isfinite(result)

    def test_nan_handling(self, sample_returns):
        """Test that NaN values are handled properly."""
        returns_with_nan = sample_returns.copy()
        returns_with_nan.iloc[10:20] = np.nan
        # Should still compute without error
        result = stats.sharpe(returns_with_nan)
        assert np.isfinite(result)

    def test_dataframe_input(self, sample_returns, sample_benchmark):
        """Test DataFrame input handling."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.sharpe(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


@pytest.fixture
def matched_market():
    """Identical growth observed daily by strategy and weekdays by benchmark."""
    dates = pd.date_range("2026-09-04", periods=40, freq="D")
    daily = np.resize(np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.005]), 40)
    daily[0] = 0.0
    strategy = pd.Series(daily, index=dates, name="Strategy")
    prices = (1 + strategy).cumprod()
    benchmark = prices.loc[dates.dayofweek < 5].pct_change(fill_method=None)
    benchmark.iloc[0] = 0.0
    benchmark.name = "Benchmark"
    return strategy, benchmark


class TestMatchedPeriodStatistics:
    """Compare statistics using returns that cover the same time periods."""

    @pytest.mark.parametrize(
        "metric", ["greeks", "r_squared", "r2", "information_ratio"]
    )
    def test_matching_ignores_other_preparation_history(self, monkeypatch, metric):
        """Earlier calculations on reversed prices must not change these statistics."""
        dates = pd.date_range("2026-01-05", periods=5)
        strategy = pd.Series([100.0, 110.0, 99.0, 118.8, 106.92], index=dates)
        benchmark = strategy * 2
        monkeypatch.setattr(utils, "_PREPARE_RETURNS_CACHE", {})
        utils._prepare_benchmark(strategy.iloc[::-1])
        kwargs = {"frequency": "daily"}
        if metric == "greeks":
            kwargs["periods"] = 1

        actual = getattr(stats, metric)(strategy, benchmark, **kwargs)

        # Both inputs yield [.1, -.1, .2, -.1]: beta and R-squared are one,
        # while active returns and alpha are zero.
        if metric == "greeks":
            pd.testing.assert_series_equal(
                actual, pd.Series({"beta": 1.0, "alpha": 0.0}), atol=1e-12, rtol=1e-12
            )
        else:
            assert actual == pytest.approx(
                0.0 if metric == "information_ratio" else 1.0, abs=1e-12
            )

    @pytest.mark.parametrize("metric", ["greeks", "r_squared"])
    def test_identical_market_has_unit_beta_and_r_squared(self, matched_market, metric):
        strategy, benchmark = matched_market
        if metric == "greeks":
            result = stats.greeks(strategy, benchmark, frequency="daily", periods=252)[
                "beta"
            ]
        else:
            result = stats.r_squared(strategy, benchmark, frequency="daily")
        assert result == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("closed", ["weekend", "holiday", "weekend-and-holiday"])
    def test_calendar_closures_do_not_break_exact_replication(
        self, matched_market, closed
    ):
        strategy, _ = matched_market
        prices = (1 + strategy).cumprod()
        mask = np.ones(len(strategy), dtype=bool)
        if "weekend" in closed:
            mask &= strategy.index.dayofweek < 5
        if "holiday" in closed:
            mask &= strategy.index != pd.Timestamp("2026-09-07")
        benchmark = prices.loc[mask].pct_change(fill_method=None)
        benchmark.iloc[0] = 0
        holidays = ["2026-09-07"] if "holiday" in closed else None
        result = stats.greeks(
            strategy, benchmark, frequency="daily", periods=252, holidays=holidays
        )
        assert result["beta"] == pytest.approx(1, abs=1e-12)
        assert result["alpha"] == pytest.approx(0, abs=1e-12)
        assert stats.r_squared(
            strategy, benchmark, frequency="daily", holidays=holidays
        ) == pytest.approx(1)

    def test_independent_moments_and_annualization(self):
        dates = pd.date_range("2026-09-04", periods=9)
        strategy = pd.Series(
            [9, 0.01, 0.02, -0.03, 0.01, 0.04, -0.02, 0.01, 0.015], index=dates
        )
        benchmark = pd.Series(
            [7, 0.02, -0.01, 0.03, 0.005], index=dates[[0, 2, 4, 6, 8]]
        )
        # Declared Monday/Wednesday/Friday closures permit these two-day spans.
        # Calculate four two-day returns directly from the supplied values,
        # independently of the matching helper.
        s = np.array(
            [1.01 * 1.02 - 1, 0.97 * 1.01 - 1, 1.04 * 0.98 - 1, 1.01 * 1.015 - 1]
        )
        b = np.array([0.02, -0.01, 0.03, 0.005])
        beta = np.cov(s, b)[0, 1] / np.var(b, ddof=1)
        alpha = s.mean() - beta * b.mean()
        for factor in (1, 52, 252, 365):
            result = stats.greeks(
                strategy,
                benchmark,
                prepare_returns=False,
                frequency="daily",
                holidays=["2026-09-07", "2026-09-09", "2026-09-11"],
                periods=factor,
            )
            assert result["beta"] == pytest.approx(beta)
            assert result["alpha"] == pytest.approx(alpha * factor)
        expected_r2 = np.corrcoef(s, b)[0, 1] ** 2
        assert stats.r_squared(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="daily",
            holidays=["2026-09-07", "2026-09-09", "2026-09-11"],
        ) == pytest.approx(expected_r2)
        assert stats.r2(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="daily",
            holidays=["2026-09-07", "2026-09-09", "2026-09-11"],
        ) == pytest.approx(expected_r2)
        active = s - b
        assert stats.information_ratio(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="daily",
            holidays=["2026-09-07", "2026-09-09", "2026-09-11"],
        ) == pytest.approx(active.mean() / active.std(ddof=1))

    def test_large_compounded_returns_are_not_reinterpreted_as_prices(self):
        dates = pd.date_range("2026-09-04", periods=7)
        strategy = pd.Series([100, 200, 400, 200, 300, 330, 495], index=dates)
        benchmark = pd.Series([100, 200, 180, 198], index=dates[[0, 2, 4, 6]])
        s = np.array([3.0, -0.25, 0.65])
        b = np.array([1.0, -0.1, 0.1])
        beta = np.cov(s, b)[0, 1] / np.var(b, ddof=1)
        result = stats.greeks(
            strategy,
            benchmark,
            frequency="daily",
            periods=12,
            holidays=["2026-09-07", "2026-09-09"],
        )
        assert result["beta"] == pytest.approx(beta)
        assert result["alpha"] == pytest.approx((s.mean() - beta * b.mean()) * 12)
        active = s - b
        assert stats.information_ratio(
            strategy,
            benchmark,
            frequency="daily",
            holidays=["2026-09-07", "2026-09-09", "2026-09-11"],
        ) == pytest.approx(active.mean() / active.std(ddof=1))

    @pytest.mark.parametrize(
        "factor", [0, -1, -(10**20), np.inf, np.nan, True, "252", np.timedelta64("NaT")]
    )
    def test_matched_alpha_factor_must_be_positive(self, matched_market, factor):
        with pytest.raises(ValueError, match="periods"):
            stats.greeks(*matched_market, frequency="daily", periods=factor)

    @pytest.mark.parametrize(
        "metric", ["greeks", "r_squared", "r2", "information_ratio"]
    )
    def test_matching_rejects_tickers_before_download(
        self, matched_market, metric, monkeypatch
    ):
        def forbidden_download(*args, **kwargs):
            pytest.fail(
                "Matching must not download a ticker and hide missing observations"
            )

        monkeypatch.setattr("quantstats.utils.download_returns", forbidden_download)
        kwargs = {"frequency": "daily"}
        if metric == "greeks":
            kwargs["periods"] = 252
        with pytest.raises(ValueError, match="supplied"):
            getattr(stats, metric)(matched_market[0], "SPY", **kwargs)

    @pytest.mark.parametrize(
        "metric", ["greeks", "r_squared", "r2", "information_ratio"]
    )
    def test_supplied_benchmark_frame_delegates_without_mutation(
        self, matched_market, metric
    ):
        strategy, benchmark = matched_market
        frame = benchmark.to_frame()
        frame["Unused"] = 99
        before_s, before_b = strategy.copy(deep=True), frame.copy(deep=True)
        kwargs = {"frequency": "daily"}
        if metric == "greeks":
            kwargs["periods"] = 252
        actual = getattr(stats, metric)(strategy, frame, **kwargs)
        expected = getattr(stats, metric)(strategy, benchmark, **kwargs)
        if isinstance(actual, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        elif isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            assert actual == pytest.approx(expected, abs=1e-12)
        pd.testing.assert_series_equal(strategy, before_s)
        pd.testing.assert_frame_equal(frame, before_b)

    @pytest.mark.parametrize("metric", ["greeks", "r_squared", "information_ratio"])
    def test_missing_history_cannot_be_skipped(self, metric):
        dates = pd.date_range("2026-09-01", periods=4)
        strategy = pd.Series([0, np.nan, np.nan, np.nan], index=dates)
        benchmark = pd.Series([0, 0.01, -0.02, 0.03], index=dates)
        kwargs = {"frequency": "daily"}
        if metric == "greeks":
            kwargs["periods"] = 252
        with pytest.raises(ValueError, match="unknown"):
            getattr(stats, metric)(strategy, benchmark, **kwargs)

    def test_calendar_declaration_does_not_claim_equal_closing_instants(self):
        dates = pd.date_range("2026-09-01", periods=4)
        strategy = pd.Series(
            [0, 0.03, -0.03, 0.07], index=dates.tz_localize("America/New_York")
        )
        benchmark = pd.Series(
            [0, 0.01, -0.02, 0.03], index=dates.tz_localize("Asia/Tokyo")
        )
        calendar = stats.greeks(strategy, benchmark, frequency="daily", periods=252)
        assert calendar["beta"] == pytest.approx(2)
        # Matching dates is a declaration about daily measurements, not a claim
        # that New York and Tokyo close at the same instant.
        assert stats.r_squared(strategy, benchmark, frequency="daily") == pytest.approx(
            1
        )

    def test_legacy_equal_calendar_and_ticker_controls(self, monkeypatch):
        dates = pd.date_range("2026-09-01", periods=4)
        strategy = pd.Series([0.01, 0.03, -0.03, 0.07], index=dates)
        benchmark = pd.Series([0, 0.01, -0.02, 0.03], index=dates)
        original = strategy.copy(deep=True)
        monkeypatch.setattr(
            "quantstats.utils.download_returns", lambda ticker: benchmark.copy()
        )
        beta = np.cov(strategy, benchmark)[0, 1] / benchmark.var()
        result = stats.greeks(strategy, "SPY")
        assert result["beta"] == pytest.approx(beta)
        assert result["alpha"] == pytest.approx(
            (strategy.mean() - beta * benchmark.mean()) * 252
        )
        assert stats.r_squared(strategy, "SPY") == pytest.approx(
            np.corrcoef(strategy, benchmark)[0, 1] ** 2
        )
        active = strategy - benchmark
        assert stats.information_ratio(strategy, "SPY") == pytest.approx(
            active.mean() / active.std()
        )
        pd.testing.assert_series_equal(strategy, original)

    def test_flat_closed_days_are_valid_observations(self, matched_market):
        strategy, _ = matched_market
        strategy.loc[
            (strategy.index.dayofweek >= 5)
            | (strategy.index == pd.Timestamp("2026-09-07"))
        ] = 0
        prices = (1 + strategy).cumprod()
        benchmark = prices.loc[
            (strategy.index.dayofweek < 5)
            & (strategy.index != pd.Timestamp("2026-09-07"))
        ].pct_change(fill_method=None)
        benchmark.iloc[0] = 0
        result = stats.greeks(
            strategy,
            benchmark,
            frequency="daily",
            periods=252,
            holidays=["2026-09-07"],
        )
        assert result["beta"] == pytest.approx(1)
        assert result["alpha"] == pytest.approx(0, abs=1e-12)

    def test_constant_matched_benchmark_is_undefined(self):
        dates = pd.date_range("2026-09-01", periods=4)
        strategy = pd.Series([0, 0.01, -0.02, 0.03], index=dates)
        benchmark = pd.Series(0.0, index=dates)
        result = stats.greeks(strategy, benchmark, frequency="daily", periods=252)
        assert result.isna().all()
        assert np.isnan(stats.r_squared(strategy, benchmark, frequency="daily"))

    def test_daily_matching_derives_factor_from_observed_coverage(self):
        dates = pd.date_range("2026-09-01", periods=5)
        strategy = pd.Series([9, 0.03, -0.03, 0.07, 0.04], index=dates)
        benchmark = pd.Series([7, 0.01, -0.02, 0.03, 0.015], index=dates)
        # Each complete pair satisfies strategy = 2 * benchmark + 0.01.
        # Four comparisons cover four calendar days. An explicit 252 override
        # remains available; no market calendar is inferred from the sample.
        result = stats.greeks(
            strategy, benchmark, prepare_returns=False, frequency="daily"
        )
        assert result["beta"] == pytest.approx(2)
        assert result["alpha"] == pytest.approx(3.65)

        overridden = stats.greeks(
            strategy, benchmark, periods=252, prepare_returns=False, frequency="daily"
        )
        assert overridden["alpha"] == pytest.approx(2.52)

    @pytest.mark.parametrize(
        "metric", ["greeks", "r_squared", "r2", "information_ratio"]
    )
    def test_later_benchmark_returns_do_not_change_an_earlier_comparison(
        self, matched_market, metric
    ):
        strategy, benchmark = matched_market
        earlier = strategy.iloc[:20]
        kwargs = {"prepare_returns": False, "frequency": "daily"}
        before = getattr(stats, metric)(earlier, benchmark, **kwargs)
        benchmark.iloc[-1] = 0.8
        after = getattr(stats, metric)(earlier, benchmark, **kwargs)
        if metric == "greeks":
            pd.testing.assert_series_equal(before, after)
        else:
            assert before == pytest.approx(after, abs=1e-12)


@pytest.mark.parametrize(
    "cadence,offset,factor",
    [
        ("daily", pd.offsets.Day(), 365),
        ("daily", pd.offsets.BusinessDay(), 4 * 365 / 6),
        ("weekly", pd.offsets.Week(weekday=6), 52),
        ("monthly", pd.offsets.MonthEnd(), 12),
        ("quarterly", pd.offsets.QuarterEnd(), 4),
        ("yearly", pd.offsets.YearEnd(), 1),
    ],
)
def test_matched_alpha_uses_output_cadence_and_explicit_override(
    cadence, offset, factor
):
    # The literal relation s=2*b+.01 gives an intercept of .01 and beta of 2
    # independently of calendar selection. Five observations supply four pairs.
    dates = pd.date_range("2026-09-01", periods=5, freq=offset)
    strategy = pd.Series([9, 0.03, -0.03, 0.07, 0.04], index=dates)
    benchmark = pd.Series([7, 0.01, -0.02, 0.03, 0.015], index=dates)
    kwargs = dict(prepare_returns=False, frequency=cadence)
    for argument in ({}, {"periods": None}, {"periods": 1}, {"periods": 252}):
        result = stats.greeks(strategy, benchmark, **kwargs, **argument)
        assert result["beta"] == pytest.approx(2)
        assert result["alpha"] == pytest.approx(
            0.01 * (argument.get("periods") or factor)
        )
        assert result.attrs["frequency"] == cadence
        assert result.attrs["observations"] == 4
    assert stats.r_squared(strategy, benchmark, **kwargs) == pytest.approx(1)
    assert stats.r2(strategy, benchmark, **kwargs) == pytest.approx(1)
    active = np.array([0.02, -0.01, 0.04, 0.025])
    assert stats.information_ratio(strategy, benchmark, **kwargs) == pytest.approx(
        active.mean() / active.std(ddof=1)
    )


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
def test_holidays_require_frequency_before_preparation(metric, monkeypatch):
    dates = pd.date_range("2026-01-01", periods=4)
    series = pd.Series([0, 0.01, -0.01, 0.02], index=dates)

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Options must be validated before preparation")

    monkeypatch.setattr(utils, "_prepare_returns", unexpected_preparation)
    for kwargs in ({"holidays": []},):
        with pytest.raises(ValueError, match="require an explicit frequency"):
            getattr(stats, metric)(series, series, **kwargs)


def test_short_weekday_sample_uses_observed_daily_rate():
    """A short daily history does not establish a full-year trading calendar."""
    dates = pd.date_range("2026-01-05", periods=4)
    strategy = pd.Series([0, 0.03, -0.03, 0.07], index=dates)
    benchmark = pd.Series([0, 0.01, -0.02, 0.03], index=dates)
    result = stats.greeks(strategy, benchmark, frequency="daily", prepare_returns=False)
    assert result["beta"] == pytest.approx(2)
    # Three return periods cover three calendar days, giving a factor of 365.
    # Each strategy return is twice the benchmark plus .01, so alpha is .01 * 365.
    assert result["alpha"] == pytest.approx(3.65)
    assert result.attrs["periods_per_year"] == pytest.approx(365)


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
@pytest.mark.parametrize("frequency", ["daily"])
def test_public_statistics_reject_long_incompatible_single_anchor(metric, frequency):
    weekly = pd.Series(
        100.0, index=pd.date_range("2021-01-01", "2024-12-31", freq="W-SUN")
    )
    yearly = pd.Series(
        200.0,
        index=pd.date_range("2021-01-01", "2024-12-31", freq=pd.offsets.YearEnd()),
    )
    with pytest.raises(ValueError, match="comparison periods"):
        getattr(stats, metric)(weekly, yearly, frequency=frequency)


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
def test_public_statistics_preserve_undefined_single_anchor_sample(metric):
    """Staggered dates leave no shared return interval to estimate statistics."""
    strategy = pd.Series([0.0, 0.1], index=pd.to_datetime(["2026-09-01", "2026-09-03"]))
    benchmark = pd.Series(
        [0.0, 0.2], index=pd.to_datetime(["2026-09-02", "2026-09-03"])
    )
    result = getattr(stats, metric)(
        strategy, benchmark, prepare_returns=False, frequency="daily"
    )
    if metric == "greeks":
        assert result.isna().all()
        assert result.attrs["observations"] == 0
    else:
        assert np.isnan(result)


@pytest.mark.parametrize(
    "first,last,expected_factor",
    [
        ("2023-01-01", "2024-01-01", 365.0),
        ("2024-01-01", "2025-01-01", 366.0),
        ("2026-01-05", "2026-01-08", 365.0),
        ("2023-12-31", "2024-01-02", 2 / (1 / 365 + 1 / 366)),
    ],
)
def test_explicit_daily_alpha_uses_matched_calendar_coverage(
    first, last, expected_factor
):
    """Annualization accounts for leap years and honors an explicit override."""
    # The first date starts the clock but does not add a return period.
    # The cross-year case covers one day of a 365-day year and one of a 366-day year.
    dates = pd.date_range(first, last)
    b = 0.01 * np.sin(np.arange(len(dates)) * 0.7)
    strategy = pd.Series(2 * b + 0.001, index=dates)
    benchmark = pd.Series(b, index=dates)
    result = stats.greeks(strategy, benchmark, prepare_returns=False, frequency="daily")
    assert result["beta"] == pytest.approx(2)
    assert result["alpha"] == pytest.approx(0.001 * expected_factor)
    assert result.attrs["alpha_periods"] == pytest.approx(expected_factor)
    assert result.attrs["observations"] == len(dates) - 1
    for factor in [1, 123.45]:
        overridden = stats.greeks(
            strategy,
            benchmark,
            prepare_returns=False,
            frequency="daily",
            periods=factor,
        )
        assert overridden["alpha"] == pytest.approx(0.001 * factor)
        assert overridden["beta"] == pytest.approx(2)


def test_daily_rate_counts_consolidated_periods_and_the_starting_anchor():
    # Friday to Monday is three elapsed days but one completed comparison.
    # Monday to Tuesday is one more. First-date returns are deliberately ignored.
    strategy = pd.Series(
        [9, 0.01, 0.02, -0.01, 0.03], index=pd.date_range("2026-01-09", periods=5)
    )
    benchmark = pd.Series(
        [7, 0.005, 0.015],
        index=pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"]),
    )
    s = np.array([1.01 * 1.02 * 0.99 - 1, 0.03])
    b = np.array([0.005, 0.015])
    beta = np.cov(s, b)[0, 1] / b.var(ddof=1)
    result = stats.greeks(strategy, benchmark, prepare_returns=False, frequency="daily")
    assert result.attrs["alpha_periods"] == pytest.approx(2 / (4 / 365))
    assert result["beta"] == pytest.approx(beta)
    assert result["alpha"] == pytest.approx(
        (s.mean() - beta * b.mean()) * (2 / (4 / 365))
    )


@pytest.mark.parametrize("swap", [False, True])
def test_daily_actual_rate_respects_supplied_holiday_returns(swap):
    """Swapping inputs with different row counts must not change annualization."""
    dates = pd.date_range("2024-03-28", "2024-04-02")
    full = pd.Series([0, 0.01, -0.02, 0.03, 0.02, -0.01], index=dates)
    sparse = pd.Series([0, 0.03, -0.02], index=dates[[0, 4, 5]])
    left, right = (sparse, full) if swap else (full, sparse)
    result = stats.greeks(
        left, right, prepare_returns=False, frequency="daily", holidays=["2024-03-29"]
    )
    assert result.attrs["observations"] == 2
    # March 28 to April 2 spans five days in a leap year. Only two return
    # intervals end on dates present in both inputs.
    assert result.attrs["alpha_periods"] == pytest.approx(2 / (5 / 366))


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
def test_none_frequency_retains_default_calls(metric):
    dates = pd.date_range("2026-01-05", periods=6)
    s = pd.Series([0, 0.03, -0.03, 0.07, 0.04, -0.02], index=dates)
    b = pd.Series([0, 0.01, -0.02, 0.03, 0.015, -0.01], index=dates)
    default = getattr(stats, metric)(s, b)
    explicit = getattr(stats, metric)(s, b, frequency=None)
    if metric == "greeks":
        pd.testing.assert_series_equal(default, explicit)
        beta = np.cov(s, b)[0, 1] / b.var(ddof=1)
        assert explicit["alpha"] == pytest.approx((s.mean() - beta * b.mean()) * 252)
    else:
        assert explicit == pytest.approx(default)


@pytest.mark.parametrize("metric", ["greeks", "r_squared", "r2", "information_ratio"])
def test_frequency_is_the_only_matching_control(metric):
    dates = pd.date_range("2026-01-01", periods=5)
    values = pd.Series([0, 0.01, -0.02, 0.03, 0.02], index=dates)
    with pytest.raises(TypeError, match="match_periods"):
        getattr(stats, metric)(values, values, match_periods=True)


@pytest.mark.parametrize(
    "zones", [("America/New_York", "Asia/Tokyo"), ("Asia/Tokyo", "America/New_York")]
)
@pytest.mark.parametrize("clock", [0.5, 23.5])
def test_observed_daily_alpha_uses_local_calendar_days_across_dst(zones, clock):
    # March 8 changes New York's UTC offset. These three daily intervals still
    # cover three calendar days, independent of either input's clock/offset.
    dates = pd.date_range("2026-03-06", periods=4) + pd.Timedelta(hours=clock)
    s = pd.Series([0.001, 0.021, -0.039, 0.061], index=dates.tz_localize(zones[0]))
    b = pd.Series([0, 0.01, -0.02, 0.03], index=dates.tz_localize(zones[1]))
    result = stats.greeks(s, b, prepare_returns=False, frequency="daily")
    assert result["beta"] == pytest.approx(2)
    assert result["alpha"] == pytest.approx(0.365)
    assert result.attrs["alpha_periods"] == pytest.approx(365)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("via_pandas", [False, True])
def test_matched_alpha_scaling_overflow_rejects_without_mutation(
    sign, prepare, via_pandas, monkeypatch
):
    from quantstats import extend_pandas

    dates = pd.date_range("2026-01-05", periods=5)
    strategy = pd.Series(
        sign * np.array([0.0, -0.01, 2.0, 3.0, 4.0]), index=dates, name="Strategy"
    )
    benchmark = pd.Series(
        [0.0, 0.01, -0.02, 0.03, -0.01], index=dates, name="Benchmark"
    )
    # The four completed pairs have beta=-503/59 and intercept=6693/2950
    # before applying sign. A finite 1e308 factor overflows either intercept.
    before = [series.copy(deep=True) for series in (strategy, benchmark)]
    for original, series in zip(before, (strategy, benchmark), strict=True):
        original.index = series.index.copy(deep=True)
    monkeypatch.setattr(utils, "_PREPARE_RETURNS_CACHE", {"sentinel": pd.Series([1.0])})
    if via_pandas:
        extend_pandas()
        calculate, args = strategy.greeks, (benchmark,)
    else:
        calculate, args = stats.greeks, (strategy, benchmark)
    error_mode = np.geterr()
    # Rejection must be a deliberate API error even when the caller requests
    # floating-point exceptions; neither a warning nor infinity may escape.
    with (
        np.errstate(over="raise"),
        pytest.raises(ValueError, match="Scaled alpha must remain finite"),
    ):
        calculate(*args, frequency="daily", prepare_returns=prepare, periods=1e308)
    assert np.geterr() == error_mode
    for actual, original in zip((strategy, benchmark), before, strict=True):
        pd.testing.assert_series_equal(actual, original)
    assert utils._PREPARE_RETURNS_CACHE.keys() == {"sentinel"}
    pd.testing.assert_series_equal(
        utils._PREPARE_RETURNS_CACHE["sentinel"], pd.Series([1.0])
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("factor_kind", ["far-override", "near-limit"])
def test_matched_alpha_scaling_accepts_finite_results(sign, factor_kind):
    dates = pd.date_range("2026-01-05", periods=5)
    strategy = pd.Series(sign * np.array([0.0, -0.01, 2.0, 3.0, 4.0]), index=dates)
    benchmark = pd.Series([0.0, 0.01, -0.02, 0.03, -0.01], index=dates)
    # Independent OLS moments: beta=-503/59, intercept=6693/2950.
    # The near-limit factor yields half the largest representable float.
    intercept = 6693 / 2950
    factor = (
        187 if factor_kind == "far-override" else np.finfo(float).max / (2 * intercept)
    )
    with np.errstate(over="raise"):
        result = stats.greeks(
            strategy,
            benchmark,
            frequency="daily",
            prepare_returns=False,
            periods=factor,
        )
    assert np.isfinite(result["alpha"])
    assert result["beta"] == pytest.approx(sign * (-503 / 59), rel=1e-12)
    assert result["alpha"] == pytest.approx(sign * intercept * factor, rel=1e-12)
    assert result.attrs["alpha_periods"] == factor


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("via_pandas", [False, True])
@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(1, id="unit"),
        pytest.param(187, id="manual"),
        pytest.param(2**63, id="unsigned-boundary"),
        pytest.param(2**64, id="above-uint64"),
        pytest.param(10**20, id="reported"),
        pytest.param(int(np.finfo(float).max), id="largest-finite"),
    ],
)
def test_matched_alpha_accepts_equivalent_integer_and_float_factors(
    factor, prepare, via_pandas
):
    """The same finite annualization factor must work as an integer or float.

    The completed pairs obey s=2*b+1/1024 exactly in binary arithmetic.
    Therefore beta is 2 and alpha is the supplied factor divided by 1024.
    """
    from quantstats import extend_pandas

    dates = pd.date_range("2026-01-05", periods=5)
    benchmark = pd.Series([0.0, -0.25, -0.125, 0.125, 0.25], index=dates)
    strategy = 2 * benchmark + 1 / 1024
    strategy.iloc[0] = 0.0
    before = [series.copy(deep=True) for series in (strategy, benchmark)]
    for snapshot, series in zip(before, (strategy, benchmark), strict=True):
        snapshot.index = series.index.copy(deep=True)
    if via_pandas:
        extend_pandas()
        calculate, args = strategy.greeks, (benchmark,)
    else:
        calculate, args = stats.greeks, (strategy, benchmark)
    error_mode = np.geterr()
    with np.errstate(all="raise"):
        for representation in (factor, float(factor)):
            result = calculate(
                *args,
                frequency="daily",
                prepare_returns=prepare,
                periods=representation,
            )
            assert result["beta"] == 2
            assert result["alpha"] == float(factor) / 1024
            assert result.attrs["alpha_periods"] == representation
            assert all(mode == "raise" for mode in np.geterr().values())
    assert np.geterr() == error_mode
    for actual, snapshot in zip((strategy, benchmark), before, strict=True):
        pd.testing.assert_series_equal(actual, snapshot)


@pytest.mark.parametrize("sample", ["short", "constant-benchmark"])
@pytest.mark.parametrize("factor", [1.0, 1e308])
def test_matched_alpha_scaling_preserves_undefined_estimates(sample, factor):
    dates = pd.date_range("2026-01-05", periods=5)
    strategy = pd.Series([0.0, 0.02, -0.01, 0.03, -0.02], index=dates)
    benchmark = pd.Series([0.0, 0.01, -0.02, 0.02, -0.01], index=dates)
    if sample == "short":
        strategy, benchmark = strategy.iloc[:2], benchmark.iloc[:2]
    else:
        benchmark[:] = 0
    with np.errstate(over="raise", invalid="raise"):
        result = stats.greeks(
            strategy,
            benchmark,
            frequency="daily",
            prepare_returns=False,
            periods=factor,
        )
    assert result.isna().all()
    assert result.attrs["alpha_periods"] == factor

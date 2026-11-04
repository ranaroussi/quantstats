"""
Tests for quantstats.stats module
"""

import numpy as np
import pandas as pd
import pytest

from quantstats import stats


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

    @pytest.mark.parametrize(
        "periods, annualize, compounded, expected",
        [
            (252, True, True, 11.475124869746283),
            (4, True, True, 0.7588770044358986),
            (4, False, True, 0.37676642725870957),
            (252, True, False, 6.148170459575759),
            (4, True, False, 0.7745966692414834),
            (4, False, False, 0.3872983346207417),
        ],
    )
    def test_information_ratio_hand_computed(
        self, periods, annualize, compounded, expected
    ):
        """Compare annual and per-period ratios with independent calculations."""
        dates = pd.date_range("2020-01-01", periods=4)
        returns = pd.Series([0.02, -0.01, 0.03, -0.02], index=dates)
        benchmark = pd.Series([0.01, 0.00, 0.01, -0.02], index=dates)
        # Active returns: [0.01, -0.01, 0.02, 0], mean = 0.005.
        # Sample variance = 0.0005 / 3; terminal wealth is
        # 1.02 * 0.99 * 1.03 * 0.98 = 1.01929212;
        # 1.01 * 1 * 1.01 * 0.98 = 0.999698.
        # Geometric numerator = strategy_wealth**(periods/4)
        # minus benchmark_wealth**(periods/4), using periods=1
        # when not annualized. Divide by sqrt(0.0005/3 * periods).
        result = stats.information_ratio(
            returns, benchmark, periods, annualize, compounded
        )
        assert result == pytest.approx(expected)
        if periods == 252 and annualize and compounded:
            assert stats.information_ratio(returns, benchmark) == pytest.approx(
                expected
            )

    def test_information_ratio_maintainer_example(self):
        returns = pd.Series([-0.0048, 0.0002, 0.0052])
        benchmark = pd.Series([0.0, 0.0, 0.0])
        # Mean = 0.0002, sample standard deviation = 0.005.
        # Per-period IR = 0.04; annual IR = 0.04 * sqrt(252).
        assert stats.information_ratio(
            returns, benchmark, annualize=False, compounded=False
        ) == pytest.approx(0.04)
        assert stats.information_ratio(
            returns, benchmark, compounded=False
        ) == pytest.approx(0.6349803146555018)

    def test_information_ratio_zero_tracking_error(self):
        returns = pd.Series([0.01, -0.01, 0.02])
        assert stats.information_ratio(returns, returns) == 0

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

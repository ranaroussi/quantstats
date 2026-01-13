"""
Tests for quantstats Monte Carlo simulation functionality
"""

import pytest
import pandas as pd
import numpy as np

import quantstats as qs
from quantstats import stats
from quantstats._montecarlo import MonteCarloResult, run_montecarlo


@pytest.fixture
def sample_returns():
    """Generate sample daily returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.randn(252) * 0.02, index=dates, name="Strategy")
    return returns


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""

    def test_stats_property(self, sample_returns):
        """Test stats property returns correct keys."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        stats_dict = mc.stats

        assert "min" in stats_dict
        assert "max" in stats_dict
        assert "mean" in stats_dict
        assert "median" in stats_dict
        assert "std" in stats_dict
        assert "percentile_5" in stats_dict
        assert "percentile_95" in stats_dict

    def test_maxdd_property(self, sample_returns):
        """Test maxdd property returns correct keys."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        maxdd_dict = mc.maxdd

        assert "min" in maxdd_dict
        assert "max" in maxdd_dict
        assert "mean" in maxdd_dict
        assert "median" in maxdd_dict
        # Max drawdown should be negative
        assert maxdd_dict["min"] < 0
        assert maxdd_dict["max"] <= 0

    def test_bust_probability(self, sample_returns):
        """Test bust probability calculation."""
        mc = run_montecarlo(sample_returns, sims=100, bust=-0.05, seed=42)
        
        assert mc.bust_probability is not None
        assert 0 <= mc.bust_probability <= 1

    def test_bust_probability_none_without_threshold(self, sample_returns):
        """Test bust probability is None when no threshold set."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        assert mc.bust_probability is None

    def test_goal_probability(self, sample_returns):
        """Test goal probability calculation."""
        mc = run_montecarlo(sample_returns, sims=100, goal=0.1, seed=42)
        
        assert mc.goal_probability is not None
        assert 0 <= mc.goal_probability <= 1

    def test_goal_probability_none_without_threshold(self, sample_returns):
        """Test goal probability is None when no threshold set."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        assert mc.goal_probability is None

    def test_percentile(self, sample_returns):
        """Test percentile method."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        
        p50 = mc.percentile(50)
        assert isinstance(p50, pd.Series)
        assert len(p50) == len(sample_returns)

    def test_confidence_band(self, sample_returns):
        """Test confidence band method."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        
        lower, upper = mc.confidence_band(0.95)
        assert isinstance(lower, pd.Series)
        assert isinstance(upper, pd.Series)
        # Lower should be less than upper at all points
        assert (lower <= upper).all()


class TestRunMontecarlo:
    """Test run_montecarlo function."""

    def test_basic_simulation(self, sample_returns):
        """Test basic Monte Carlo simulation."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        
        assert isinstance(mc, MonteCarloResult)
        assert mc.data.shape[1] == 100  # 100 simulations
        assert len(mc.original) == len(sample_returns)

    def test_reproducibility_with_seed(self, sample_returns):
        """Test that seed produces reproducible results."""
        mc1 = run_montecarlo(sample_returns, sims=50, seed=123)
        mc2 = run_montecarlo(sample_returns, sims=50, seed=123)
        
        pd.testing.assert_frame_equal(mc1.data, mc2.data)

    def test_different_seeds_produce_different_results(self, sample_returns):
        """Test that different seeds produce different results."""
        mc1 = run_montecarlo(sample_returns, sims=50, seed=123)
        mc2 = run_montecarlo(sample_returns, sims=50, seed=456)
        
        # Results should be different
        assert not mc1.data.equals(mc2.data)

    def test_first_column_is_original(self, sample_returns):
        """Test that first simulation column matches original."""
        mc = run_montecarlo(sample_returns, sims=100, seed=42)
        
        # First column should be the original (unshuffled) returns compounded
        pd.testing.assert_series_equal(
            mc.data.iloc[:, 0], mc.original, check_names=False
        )

    def test_handles_nan_values(self):
        """Test that NaN values are handled properly."""
        returns = pd.Series([0.01, np.nan, -0.02, 0.03, np.nan, 0.01])
        mc = run_montecarlo(returns, sims=50, seed=42)
        
        # Should have 4 periods (NaN dropped)
        assert mc.data.shape[0] == 4


class TestStatsMonteCarlo:
    """Test Monte Carlo functions exposed via stats module."""

    def test_montecarlo_function(self, sample_returns):
        """Test stats.montecarlo function."""
        mc = stats.montecarlo(sample_returns, sims=100, seed=42)
        
        assert isinstance(mc, MonteCarloResult)
        assert mc.data.shape[1] == 100

    def test_montecarlo_with_bust_and_goal(self, sample_returns):
        """Test stats.montecarlo with bust and goal thresholds."""
        mc = stats.montecarlo(
            sample_returns, sims=100, bust=-0.1, goal=0.5, seed=42
        )
        
        assert mc.bust_threshold == -0.1
        assert mc.goal_threshold == 0.5
        assert mc.bust_probability is not None
        assert mc.goal_probability is not None


class TestMonteCarloEdgeCases:
    """Test edge cases for Monte Carlo simulations."""

    def test_single_simulation(self, sample_returns):
        """Test with single simulation."""
        mc = run_montecarlo(sample_returns, sims=1, seed=42)
        
        assert mc.data.shape[1] == 1

    def test_large_number_of_simulations(self, sample_returns):
        """Test with large number of simulations."""
        mc = run_montecarlo(sample_returns, sims=1000, seed=42)
        
        assert mc.data.shape[1] == 1000

    def test_short_return_series(self):
        """Test with short return series."""
        returns = pd.Series([0.01, -0.02, 0.03])
        mc = run_montecarlo(returns, sims=50, seed=42)
        
        assert mc.data.shape[0] == 3

    def test_all_positive_returns(self):
        """Test with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02])
        mc = run_montecarlo(returns, sims=50, seed=42)
        
        # Terminal values should all be positive
        assert (mc.data.iloc[-1] > 0).all()

    def test_all_negative_returns(self):
        """Test with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.02])
        mc = run_montecarlo(returns, sims=50, seed=42)
        
        # Terminal values should all be negative
        assert (mc.data.iloc[-1] < 0).all()

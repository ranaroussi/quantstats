"""
Tests for quantstats.plots module
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quantstats import plots


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


class TestPlotFunctions:
    """Test that plot functions run without errors."""

    def test_snapshot(self, sample_returns):
        """Test snapshot plot generates without error."""
        # snapshot should work with show=False
        fig = plots.snapshot(sample_returns, show=False)
        assert fig is not None

    def test_returns_plot(self, sample_returns, sample_benchmark):
        """Test returns plot."""
        fig = plots.returns(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_log_returns_plot(self, sample_returns):
        """Test log returns plot."""
        fig = plots.log_returns(sample_returns, show=False)
        assert fig is not None

    def test_yearly_returns(self, sample_returns):
        """Test yearly returns bar chart."""
        fig = plots.yearly_returns(sample_returns, show=False)
        assert fig is not None

    @pytest.mark.parametrize("year_count", [10, 11, 15, 19, 21])
    @pytest.mark.parametrize("with_benchmark", [False, True])
    def test_yearly_returns_tick_labels(self, year_count, with_benchmark):
        """Year labels stay on their bar groups when long ranges are thinned."""
        dates = pd.date_range("2010-01-01", f"{2009 + year_count}-06-30", freq="D")
        returns = pd.Series(0.0002, index=dates, name="Strategy")
        benchmark = (
            pd.Series(0.0001, index=dates, name="Benchmark") if with_benchmark else None
        )
        fig = plots.yearly_returns(returns, benchmark=benchmark, show=False)
        try:
            ax = fig.axes[0]
            centers = np.mean(
                [
                    [bar.get_x() + bar.get_width() / 2 for bar in group]
                    for group in ax.containers
                ],
                axis=0,
            )
            np.testing.assert_allclose(ax.get_xticks(), centers)
            step = (year_count + 9) // 10
            assert [tick.get_text() for tick in ax.get_xticklabels()] == [
                str(2010 + i) if i % step == 0 else "" for i in range(year_count)
            ]
        finally:
            plt.close(fig)

    def test_histogram(self, sample_returns, sample_benchmark):
        """Test histogram plot."""
        fig = plots.histogram(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_daily_returns(self, sample_returns, sample_benchmark):
        """Test daily returns scatter plot."""
        fig = plots.daily_returns(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_drawdown(self, sample_returns):
        """Test drawdown plot."""
        fig = plots.drawdown(sample_returns, show=False)
        assert fig is not None

    def test_drawdowns_periods(self, sample_returns):
        """Test drawdown periods plot."""
        fig = plots.drawdowns_periods(sample_returns, show=False)
        assert fig is not None

    def test_rolling_volatility(self, sample_returns):
        """Test rolling volatility plot."""
        fig = plots.rolling_volatility(sample_returns, show=False)
        assert fig is not None

    def test_rolling_sharpe(self, sample_returns):
        """Test rolling Sharpe plot."""
        fig = plots.rolling_sharpe(sample_returns, show=False)
        assert fig is not None

    def test_rolling_sortino(self, sample_returns):
        """Test rolling Sortino plot."""
        fig = plots.rolling_sortino(sample_returns, show=False)
        assert fig is not None

    def test_rolling_beta(self, sample_returns, sample_benchmark):
        """Test rolling beta plot."""
        fig = plots.rolling_beta(sample_returns, sample_benchmark, show=False)
        assert fig is not None

    def test_monthly_heatmap(self, sample_returns):
        """Test monthly heatmap."""
        fig = plots.monthly_heatmap(sample_returns, show=False)
        assert fig is not None

    def test_distribution(self, sample_returns):
        """Test distribution box plot."""
        fig = plots.distribution(sample_returns, show=False)
        assert fig is not None


class TestPlotSaving:
    """Test plot saving functionality."""

    def test_save_to_file(self, sample_returns):
        """Test saving plot to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            plots.returns(sample_returns, savefig=output_path, show=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_save_with_dict_params(self, sample_returns):
        """Test saving plot with dict parameters."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            plots.returns(
                sample_returns,
                savefig={"fname": output_path, "dpi": 100},
                show=False,
            )
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestPlotOptions:
    """Test plot configuration options."""

    def test_grayscale_mode(self, sample_returns):
        """Test grayscale plotting mode."""
        fig = plots.returns(sample_returns, grayscale=True, show=False)
        assert fig is not None

    def test_custom_figsize(self, sample_returns):
        """Test custom figure size."""
        fig = plots.returns(sample_returns, figsize=(12, 8), show=False)
        assert fig is not None

    def test_no_subtitle(self, sample_returns):
        """Test plot without subtitle."""
        fig = plots.returns(sample_returns, subtitle=False, show=False)
        assert fig is not None

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


class TestBenchmarkAlignmentCalendars:
    @pytest.mark.parametrize("metric", ["volatility", "sharpe", "sortino"])
    @pytest.mark.parametrize(
        "calendar", ["shared_calendar", "cross_timezone", "utc", "naive"]
    )
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_rolling_plot_preserves_daily_benchmark_observations(
        self, metric, calendar, as_frame
    ):
        if calendar == "cross_timezone":
            period = pd.date_range(
                "2024-07-01 16:00", periods=7, freq="D", tz="America/New_York"
            )
            benchmark_index = pd.date_range(
                "2024-07-01 15:00", periods=7, freq="D", tz="Asia/Tokyo"
            )
        else:
            tz = {"shared_calendar": "Europe/London", "utc": "UTC", "naive": None}[
                calendar
            ]
            period = pd.date_range("2024-07-01", periods=7, freq="D", tz=tz)
            benchmark_index = period + pd.Timedelta(hours=5)
        returns = pd.Series(
            [0.0, -0.01, 0.03, -0.02, 0.04, -0.01, 0.02],
            index=period,
            name="Strategy",
        )
        benchmark_values = np.array([0.0, -0.02, 0.01, -0.03, 0.02, -0.01, 0.03])
        benchmark = pd.Series(benchmark_values, index=benchmark_index, name="Benchmark")
        if as_frame:
            benchmark = benchmark.to_frame()
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)

        # Each observation belongs to its labeled day in all four calendars.
        # Calculate the three-day windows independently of benchmark preparation
        # and the production rolling statistics; every window has a negative return.
        expected = []
        for end in range(3, len(benchmark_values) + 1):
            window = benchmark_values[end - 3 : end]
            if metric == "volatility":
                value = np.std(window, ddof=1)
            elif metric == "sharpe":
                value = np.mean(window) / np.std(window, ddof=1)
            else:
                downside = np.sqrt(np.sum(np.minimum(window, 0) ** 2) / 3)
                value = np.mean(window) / downside
            expected.append(value * np.sqrt(252))

        fig = getattr(plots, "rolling_" + metric)(
            returns, benchmark, period=3, show=False
        )
        try:
            fig.canvas.draw()
            line = next(
                line for line in fig.axes[0].lines if line.get_label() == "Benchmark"
            )
            np.testing.assert_allclose(line.get_ydata(), expected, atol=1e-12)
            expected_index = period[2:]
            if expected_index.tz is not None:
                expected_index = expected_index.tz_convert("UTC").tz_localize(None)
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(line.get_xdata()), expected_index
            )
            assert fig.axes[0].get_legend_handles_labels()[1][:2] == [
                "Strategy",
                "Benchmark",
            ]
        finally:
            plt.close(fig)
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original_benchmark)
        else:
            pd.testing.assert_series_equal(benchmark, original_benchmark)

    @pytest.mark.parametrize("calendar", ["cross_timezone", "shared_calendar"])
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_log_returns_plot_preserves_daily_benchmark_observations(
        self, calendar, as_frame
    ):
        if calendar == "cross_timezone":
            period = pd.date_range(
                "2024-07-01 16:00", periods=3, freq="D", tz="America/New_York"
            )
            benchmark_index = pd.date_range(
                "2024-07-01 15:00", periods=3, freq="D", tz="Asia/Tokyo"
            )
        else:
            period = pd.date_range(
                "2024-07-01", periods=3, freq="D", tz="Europe/London"
            )
            benchmark_index = period + pd.Timedelta(hours=5)
        returns = pd.Series([0.0, 0.03, 0.04], index=period, name="Strategy")
        benchmark = pd.Series(
            [0.01, 0.02, 0.03], index=benchmark_index, name="Benchmark"
        )
        if as_frame:
            benchmark = benchmark.to_frame()
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)

        fig = plots.log_returns(returns, benchmark, show=False)
        try:
            fig.canvas.draw()
            line = next(
                line for line in fig.axes[0].lines if line.get_label() == "Benchmark"
            )
            # Compound the independently expected daily returns [0, .02, .03].
            np.testing.assert_allclose(
                line.get_ydata(), [0.0, 0.02, 1.02 * 1.03 - 1], atol=1e-12
            )
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(line.get_xdata()),
                period.tz_convert("UTC").tz_localize(None),
            )
        finally:
            plt.close(fig)
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original_benchmark)
        else:
            pd.testing.assert_series_equal(benchmark, original_benchmark)

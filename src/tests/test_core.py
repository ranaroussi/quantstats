from src.quantstats import (
    plot_distribution,
    plot_histogram,
    plot_longest_drawdowns,
    plot_returns_bars,
    plot_rolling_beta,
    plot_rolling_stats,
    plot_timeseries,
)


def test_plot_histogram_1(returns):
    plot_histogram(returns)


def test_plot_distribution(returns):
    plot_distribution(returns)


def test_longest_drawdowns(returns):
    plot_longest_drawdowns(returns["Meta"])


def test_returns_bars(data):
    plot_returns_bars(data.all())


def test_rolling_beta(returns, benchmark):
    plot_rolling_beta(returns, benchmark)


def test_rolling_stats(returns, benchmark):
    plot_rolling_stats(returns, benchmark)


def test_plot_timeseries_1(returns):
    plot_timeseries(returns)


def test_plot_timeseries_2(portfolio):
    plot_timeseries(portfolio.cumsum())

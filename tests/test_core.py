from quantstats._plotting.core import (
    plot_distribution,
    plot_longest_drawdowns,
    plot_returns_bars,
    plot_rolling_beta,
    plot_rolling_stats,
    plot_timeseries,
)


def test_plot_distribution(returns):
    plot_distribution(returns)


# def test_plot_histogram(returns, benchmark):
#    plot_histogram(returns, benchmark)


def test_longest_drawdowns(returns):
    plot_longest_drawdowns(returns)


def test_returns_bars(returns, benchmark):
    plot_returns_bars(returns)
    plot_returns_bars(returns, benchmark)


def test_rolling_beta(returns, benchmark):
    plot_rolling_beta(returns, benchmark)


def test_rolling_stats(returns, benchmark):
    plot_rolling_stats(returns)
    plot_rolling_stats(returns, benchmark)


def test_plot_timeseries(returns, benchmark):
    plot_timeseries(returns)
    plot_timeseries(returns, benchmark)

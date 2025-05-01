import numpy as np
import pandas as pd

from quantstats.data import build_data


def test_names(data):
    assert data.names == ["AAPL", "META"]


def test_benchmark_aligned_frame(data):
    pd.testing.assert_index_equal(data.index, data.returns.index)
    pd.testing.assert_index_equal(data.index, data.benchmark.index)


def test_apply(data, portfolio):
    pd.testing.assert_series_equal(data.apply(np.mean, axis=1), portfolio.loc[data.index].apply(np.mean, axis=1))


def test_all(data):
    frame = data.all()
    assert "Benchmark" in frame.columns


def test_prices(data, portfolio):
    frame = portfolio.loc[data.index]
    print(frame.cumsum())
    x1 = data.prices(compounded=False)[data.names]
    print(x1)
    x2 = data.prices(compounded=True)
    print(x2)


def test_with_excess_series_1(returns):
    rf = pd.Series(index=returns.index, data=0.0010)
    build_data(returns=returns, rf=rf)


def test_with_excess_series_2(returns):
    rf = pd.Series(index=returns.index, data=1.0)
    build_data(returns=returns, rf=rf, nperiods=252)


def test_distribution(returns):
    d = build_data(returns=returns)
    d.distribution()


def test_histogram(returns, benchmark):
    d = build_data(returns=returns, benchmark=benchmark)
    d.histogram()


def test_plot(data):
    # d = build_data(returns=returns, benchmark=benchmark)
    data.plot()


def test_plot_return_bars(data):
    # d = build_data(returns=returns, benchmark=benchmark)
    print(data.all())
    print(data.return_bars())


def test_heatmap(data):
    data.monthly_heatmap()


def test_yearly_returns(data):
    frame = data.yearly_returns()
    print(frame.head())


def test_rolling_sortino(data):
    data.rolling_sortino()


def test_rolling_sharpe(data):
    data.rolling_sharpe()


def test_rolling_volatility(data):
    data.rolling_volatility()


def test_drawdown(data):
    data.drawdown()


def test_drawdowns_periods(data):
    data.drawdowns_periods()


def test_rolling_beta(data):
    data.rolling_beta()


def test_snapshot(data):
    data.snapshot_plotly().show()

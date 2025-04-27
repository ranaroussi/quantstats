from quantstats._plotting.wrappers import (
    daily_returns,
    distribution,
    drawdown,
    drawdowns_periods,
    earnings,
    histogram,
    log_returns,
    monthly_heatmap,
    monthly_returns,
    rolling_beta,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
    snapshot,
    yearly_returns,
)
from quantstats._plotting.wrappers import returns as rrr


def test_daily_returns(returns, benchmark):
    daily_returns(returns, benchmark)


def test_distribution(returns):
    distribution(returns)


def test_drawdown(returns):
    drawdown(returns)


def test_drawdowns_periods(returns):
    drawdowns_periods(returns)


def test_earnings(returns):
    earnings(returns)


def test_histogram(returns):
    histogram(returns)


def test_log_returns(returns):
    log_returns(returns)


def test_monthly_returns(returns):
    monthly_returns(returns)


def test_monthly_heatmap(returns):
    monthly_heatmap(returns)


def test_returns(returns):
    rrr(returns)


def test_rolling_beta(returns, benchmark):
    rolling_beta(returns, benchmark)


def test_rolling_sharpe(returns):
    rolling_sharpe(returns)


def test_rolling_sortino(returns):
    rolling_sortino(returns)


def test_rolling_volatility(returns):
    rolling_volatility(returns)


def test_snapshot(returns, portfolio):
    snapshot(returns)
    snapshot(portfolio)


def test_yearly_returns(returns):
    yearly_returns(returns)

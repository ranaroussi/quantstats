import pytest

import quantstats as qs
from quantstats.stats import calmar, kurtosis, sharpe, skew, smart_sortino

qs.extend_pandas()


@pytest.fixture
def return_series(returns):
    return returns["Meta"].dropna()


def test_sharpe(return_series):
    assert sharpe(return_series) == pytest.approx(0.7147016517340532)
    assert return_series.sharpe() == pytest.approx(0.7147016517340532)


def test_smart_sortino(return_series):
    assert smart_sortino(return_series) == pytest.approx(1.0350120584738784)
    assert return_series.smart_sortino() == pytest.approx(1.0350120584738784)


def test_calmar(return_series):
    assert calmar(return_series) == pytest.approx(0.19875579357361298)
    assert return_series.calmar() == pytest.approx(0.19875579357361298)


def test_kurtosis(return_series):
    assert kurtosis(return_series) == pytest.approx(20.484236053765002)
    assert return_series.kurtosis() == pytest.approx(20.484236053765002)


def test_skewness(return_series):
    assert skew(return_series) == pytest.approx(0.42220532349298445)
    assert return_series.skew() == pytest.approx(0.42220532349298445)

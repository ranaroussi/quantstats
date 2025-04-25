import pandas as pd
import pytest

import quantstats as qs
from quantstats.stats import calmar, kurtosis, sharpe, skew, smart_sortino

qs.extend_pandas()


@pytest.fixture
def returns(resource_dir):
    return pd.read_csv(resource_dir / "meta.csv", parse_dates=True, index_col=0)["Close"].dropna()


def test_sharpe(returns):
    assert sharpe(returns) == pytest.approx(0.7147016517340532)
    assert returns.sharpe() == pytest.approx(0.7147016517340532)


def test_smart_sortino(returns):
    assert smart_sortino(returns) == pytest.approx(1.0350120584738784)
    assert returns.smart_sortino() == pytest.approx(1.0350120584738784)


def test_calmar(returns):
    assert calmar(returns) == pytest.approx(0.19875579357361298)
    assert returns.calmar() == pytest.approx(0.19875579357361298)


def test_kurtosis(returns):
    assert kurtosis(returns) == pytest.approx(20.484236053765002)
    assert returns.kurtosis() == pytest.approx(20.484236053765002)


def test_skewness(returns):
    assert skew(returns) == pytest.approx(0.42220532349298445)
    assert returns.skew() == pytest.approx(0.42220532349298445)

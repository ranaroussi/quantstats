import numpy as np
import pandas as pd

from quantstats.data import build_data


def test_names(portfolio):
    d = build_data(returns=portfolio)
    assert d.name == ["AAPL", "META"]


# def test_excess(portfolio):
#    d = Data(returns=portfolio, rf=0.1)
#    r_excess = d.excess_return(nperiods=252)
#    print(r_excess)
#    print(portfolio)
#    assert r_excess.loc["2025-04-24"]["AAPL"] == pytest.approx(0.018047856680624134)


def test_names_series(returns):
    d = build_data(returns=returns)
    assert d.name == "Strategy"


def test_benchmark_aligned_series(returns, benchmark):
    d = build_data(returns=returns, benchmark=benchmark)
    pd.testing.assert_index_equal(d.benchmark.index, d.returns.index)


def test_benchmark_aligned_frame(portfolio, benchmark):
    d = build_data(returns=portfolio, benchmark=benchmark)
    pd.testing.assert_index_equal(d.benchmark.index, d.returns.index)
    pd.testing.assert_index_equal(d.index, d.benchmark.index)


def test_apply(portfolio):
    d = build_data(portfolio)
    print(d.apply(np.mean, axis=0))


def test_all(returns, benchmark):
    returns = returns.to_frame(name="AAPL")
    d = build_data(returns=returns, benchmark=benchmark)
    d.all()
    print(d.all())

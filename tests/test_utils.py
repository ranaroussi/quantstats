from quantstats.utils import to_excess_returns


def test_excess_returns(returns):
    r = to_excess_returns(returns, rf=0.01)
    print(r)

import quantstats as qs


def test_metrics(returns, benchmark):
    qs.reports.metrics(returns, benchmark=benchmark)


def test_metrics_portfolio(portfolio, benchmark):
    qs.reports.metrics(returns=portfolio, benchmark=benchmark)


def test_plots(returns):
    qs.reports.plots(returns)


def test_plots_portfolio(portfolio, benchmark):
    qs.reports.plots(returns=portfolio, benchmark=benchmark)


def test_full(returns, benchmark):
    qs.reports.full(returns, benchmark=benchmark)


def test_full_portfolio(portfolio, benchmark):
    qs.reports.full(returns=portfolio, benchmark=benchmark)


def test_basic(returns, benchmark):
    qs.reports.basic(returns, benchmark=benchmark)


def test_basic_portfolio(portfolio, benchmark):
    qs.reports.basic(returns=portfolio, benchmark=benchmark)

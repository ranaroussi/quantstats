import quantstats as qs


def test_metrics(returns):
    qs.reports.metrics(returns)


def test_plots(returns):
    qs.reports.plots(returns)


def test_full(returns):
    qs.reports.full(returns)

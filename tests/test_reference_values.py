"""
Reference values for quantstats' estimators.

Each metric is checked against the convention quantstats documents, on five
fixed return series: the 24-month portfolio from Bacon (2008), a seven-point
sample, a Gaussian series, a fat-tailed Student t(3) series and a left-skewed
one. Where R PerformanceAnalytics implements the convention, the expected
value is its output (the call is recorded in tests/data/reference_values.json);
the rest were checked against numpy, scipy or a brute-force definition. The
values come from vetted (https://github.com/WatchTree-19/vetted).

A failure here means an estimator changed. If the change was intended, the
convention recorded for that metric should change with it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantstats import stats

DATA = json.loads(
    (Path(__file__).parent / "data" / "reference_values.json").read_text()
)


def _series(name):
    fixture = DATA["fixtures"][name]
    returns = fixture["returns"]
    freq = "MS" if fixture["periods"] == 12 else "B"
    index = pd.date_range("2000-01-03", periods=len(returns), freq=freq)
    return pd.Series(returns, index=index, dtype=float), fixture["periods"]


METRICS = {
    "volatility_annual": lambda r, p: stats.volatility(r, periods=p, annualize=True),
    "return_annual": lambda r, p: stats.cagr(r, periods=p),
    "sharpe_annual": lambda r, p: stats.sharpe(r, periods=p, annualize=True),
    "sortino_annual": lambda r, p: stats.sortino(r, periods=p, annualize=True),
    "max_drawdown": lambda r, p: stats.max_drawdown(r),
    "calmar": lambda r, p: stats.calmar(r, periods=p),
    "var_95": lambda r, p: stats.value_at_risk(r, confidence=0.95),
    "es_95": lambda r, p: stats.conditional_value_at_risk(r, confidence=0.95),
    "omega_0": lambda r, p: stats.omega(r, periods=p),
    "skewness": lambda r, p: stats.skew(r),
    "kurtosis": lambda r, p: stats.kurtosis(r),
}

CASES = [
    pytest.param(metric, fixture, expected, id=f"{metric}-{fixture}")
    for metric, spec in DATA["metrics"].items()
    for fixture, expected in spec["values"].items()
    if expected is not None
]


def test_every_metric_has_a_check():
    assert set(METRICS) == set(DATA["metrics"])


@pytest.mark.parametrize("metric, fixture, expected", CASES)
def test_reference_value(metric, fixture, expected):
    returns, periods = _series(fixture)
    value = float(np.asarray(METRICS[metric](returns, periods)).ravel()[0])
    convention = DATA["metrics"][metric]["convention"]
    assert value == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"{metric} on {fixture} no longer matches the {convention} convention"
    )

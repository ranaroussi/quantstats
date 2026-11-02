"""Recovery factor must retain the sign of excess total returns."""

import numpy as np
import pandas as pd
import pytest

from quantstats import stats


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("rf", [0.0, 0.02])
def test_losing_strategy_has_negative_recovery_factor(as_frame, rf):
    returns = pd.Series(
        [-0.05, -0.05, 0.02], index=pd.date_range("2024-01-01", periods=3)
    )
    values = returns.to_frame("strategy") if as_frame else returns
    actual = stats.recovery_factor(values, rf=rf)
    actual = actual["strategy"] if as_frame else actual
    # Sum of returns is -0.08; the equity trough is 0.95**2 = 0.9025.
    assert actual == pytest.approx((-0.08 - rf) / 0.0975)
    assert actual < 0


def test_mixed_dataframe_preserves_each_column_sign():
    data = pd.DataFrame(
        {"loss": [-0.05, -0.05, 0.02], "gain": [0.1, -0.05, 0.03]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    actual = stats.recovery_factor(data)
    assert actual["loss"] == pytest.approx(-0.08 / 0.0975)
    assert actual["gain"] == pytest.approx(0.08 / 0.05)


@pytest.mark.parametrize("as_frame", [False, True])
def test_zero_drawdown_still_returns_nan(as_frame):
    values = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
    values = values.to_frame("strategy") if as_frame else values
    actual = stats.recovery_factor(values)
    actual = actual["strategy"] if as_frame else actual
    assert np.isnan(actual)


def _report_returns():
    """A losing strategy and a gaining benchmark over one year of trading days."""
    index = pd.bdate_range("2024-01-01", periods=252)
    rng = np.random.default_rng(578)
    noise = rng.normal(0.0, 0.01, len(index))
    strategy = pd.Series(noise - 0.001, index=index, name="Strategy")
    benchmark = pd.Series(noise[::-1] + 0.001, index=index, name="Benchmark")
    return strategy, benchmark


@pytest.mark.parametrize("with_benchmark", [False, True])
def test_reports_metrics_show_negative_recovery_factor(with_benchmark):
    from quantstats import reports

    strategy, benchmark = _report_returns()
    assert strategy.sum() < 0 < benchmark.sum()
    kw = dict(benchmark=benchmark) if with_benchmark else {}
    table = reports.metrics(strategy, display=False, mode="full", **kw)
    row = table.loc["Recovery Factor"]
    expected = round(stats.recovery_factor(strategy), 2)
    assert expected < 0
    assert row["Strategy"] == pytest.approx(expected)
    if with_benchmark:
        assert row["Benchmark"] == pytest.approx(
            round(stats.recovery_factor(benchmark), 2)
        )
        assert row["Benchmark"] > 0

    # The string table used by the HTML report keeps the minus sign.
    text = reports.metrics(strategy, display=False, mode="full", internal=True, **kw)
    assert text.loc["Recovery Factor", "Strategy"] == str(expected)
    assert text.loc["Recovery Factor", "Strategy"].startswith("-")

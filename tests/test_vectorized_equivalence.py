"""
Regression tests for the vectorized fast paths in group_returns,
drawdown_details and rolling_sortino.

Each test compares the fast path against a straightforward reference
implementation of the original per-group / per-window logic, so any
behavioural drift shows up as a numeric mismatch.
"""

import numpy as np
import pandas as pd
import pytest

from quantstats import stats, utils


def _daily_returns(n=1500, seed=0, nan_every=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    r = pd.Series(rng.normal(0.0003, 0.012, n), index=idx, name="r")
    if nan_every:
        r.iloc[::nan_every] = np.nan
    return r


def _reference_group_returns(returns, groupby):
    return returns.groupby(groupby).apply(lambda x: x.add(1).prod(axis=0) - 1)


def _reference_drawdown_rows(drawdown):
    """Original per-period computation using pandas slicing."""
    no_dd = drawdown == 0
    starts = ~no_dd & no_dd.shift(1)
    starts = list(starts[starts.values].index)
    ends = (no_dd & (~no_dd).shift(1)).shift(-1, fill_value=False)
    ends = list(ends[ends.values].index)
    if not starts:
        return []
    if ends and starts[0] > ends[0]:
        starts.insert(0, drawdown.index[0])
    if not ends or starts[-1] > ends[-1]:
        ends.append(drawdown.index[-1])
    rows = []
    for s, e in zip(starts, ends, strict=False):
        dd = drawdown[s:e]
        clean = -stats.remove_outliers(-dd, 0.99)
        rows.append((dd.idxmin(), dd.min() * 100, clean.min() * 100))
    return rows


class TestGroupReturnsVectorized:
    @pytest.mark.parametrize("nan_every", [None, 7])
    @pytest.mark.parametrize("key", ["year", "month", "quarter", "year_month"])
    def test_series_matches_reference(self, key, nan_every):
        r = _daily_returns(nan_every=nan_every)
        idx = r.index
        groupby = {
            "year": idx.year,
            "month": idx.month,
            "quarter": idx.quarter,
            "year_month": [idx.year, idx.month],
        }[key]
        pd.testing.assert_series_equal(
            utils.group_returns(r, groupby, compounded=True),
            _reference_group_returns(r, groupby),
            rtol=1e-12,
        )

    def test_dataframe_matches_reference(self):
        r = _daily_returns()
        df = pd.DataFrame({"a": r, "b": -r, "c": _daily_returns(seed=1)})
        pd.testing.assert_frame_equal(
            utils.group_returns(df, df.index.year, compounded=True),
            _reference_group_returns(df, df.index.year),
            rtol=1e-12,
        )

    def test_all_nan_group_compounds_to_zero(self):
        r = _daily_returns()
        r.loc["2006"] = np.nan
        out = utils.group_returns(r, r.index.year, compounded=True)
        assert out.loc[2006] == 0.0

    def test_aggregate_returns_yearly(self):
        r = _daily_returns()
        expected = _reference_group_returns(r, r.index.year)
        pd.testing.assert_series_equal(
            utils.aggregate_returns(r, "year"), expected, rtol=1e-12
        )


class TestDrawdownDetailsVectorized:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_reference(self, seed):
        r = _daily_returns(n=2000, seed=seed)
        if seed % 2:
            r.iloc[:5] = -0.01  # series starts in a drawdown
        if seed % 3 == 0:
            r.iloc[-5:] = -0.01  # series ends in a drawdown
        dd = stats.to_drawdown_series(r)
        details = stats.drawdown_details(dd)
        ref = _reference_drawdown_rows(dd)
        assert len(details) == len(ref)
        if not ref:
            return
        np.testing.assert_array_equal(
            details["valley"].to_numpy(),
            [v.strftime("%Y-%m-%d") for v, _, _ in ref],
        )
        np.testing.assert_allclose(details["max drawdown"], [m for _, m, _ in ref])
        np.testing.assert_allclose(
            details["99% max drawdown"], [c for _, _, c in ref], equal_nan=True
        )


class TestRollingSortinoVectorized:
    @pytest.mark.parametrize("window", [20, 126])
    def test_matches_rolling_apply(self, window):
        r = _daily_returns(n=1000)
        r.iloc[200:205] = np.nan

        def downside(x):
            neg = x[x < 0]
            return (neg**2).sum() if len(neg) else 0

        ref_down = r.rolling(window).apply(downside, raw=True) / window
        ref = r.rolling(window).mean() / np.sqrt(ref_down) * np.sqrt(252)
        out = stats.rolling_sortino(r, rolling_period=window, periods_per_year=252)
        pd.testing.assert_series_equal(out, ref, rtol=1e-9, check_names=False)

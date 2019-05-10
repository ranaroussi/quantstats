#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019 Ran Aroussi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ˜
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io as _io
import pandas as _pd
import numpy as _np
import fix_yahoo_finance as _yf
from . import stats as _stats


def multi_shift(df, shift=3):
    """
    get last N rows relative to another row in pandas
    """
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)
    dfs = [df.shift(i) for i in _np.arange(shift)]
    for ix, df in enumerate(dfs[1:]):
        dfs[ix + 1].columns = [str(col) for col in df.columns + str(ix + 1)]
    return _pd.concat(dfs, 1, sort=True)


def to_returns(prices, rf=0.):
    """
    Calculates the simple arithmetic returns of a price series
    """
    return _prepare_returns(prices)


def to_prices(returns, base=1e5):
    """
    Converts returns series to price data
    """
    returns = returns.copy().fillna(0).replace(
        [_np.inf, -_np.inf], float('NaN'))

    return base + base * _stats.compsum(returns)


def log_returns(returns, rf=0., nperiods=None):
    """ shorthand for to_log_returns """
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns, rf=0., nperiods=None):
    """
    Converts returns series to log returns
    """
    returns = _prepare_returns(returns, rf, nperiods)
    try:
        return _np.log(returns+1).replace([_np.inf, -_np.inf], float('NaN'))
    except Exception:
        return 0.


def exponential_stdev(returns, window=30, is_halflife=False):
    """
    Returns series representing exponential volatility of returns
    """
    returns = _prepare_returns(returns)
    halflife = window if is_halflife else None
    return returns.ewm(com=None, span=window,
                       halflife=halflife, min_periods=window).std()


def rebase(prices, base=100.):
    """
    Rebase all series to a given intial base.
    This makes comparing/plotting different series together easier.
    Args:
        * prices: Expects a price series/dataframe
        * base (number): starting value for all series.
    """
    return prices.dropna() / prices.dropna().ix[0] * base


def group_returns(returns, groupby, compounded=False):
    """ summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(_stats.comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """
    Aggregates returns based on date periods
    """

    if period is None or 'day' in period:
        return returns
    index = returns.index
    if 'month' in period:
        returns = group_returns(returns, index.month, compounded=compounded)
    if 'quarter' in period:
        returns = group_returns(returns, index.quarter, compounded=compounded)
    elif 'year' in period or 'eoy' in period or 'yoy' in period or "A" == period:
        returns = group_returns(returns, index.year, compounded=compounded)
    elif 'week' in period:
        returns = group_returns(returns, index.week, compounded=compounded)
    elif 'eow' in period or "W" == period:
        returns = group_returns(returns, [index.year, index.week],
                                compounded=compounded)
    elif 'eom' in period or "M" == period:
        returns = group_returns(returns, [index.year, index.month],
                                compounded=compounded)
    elif 'eoq' in period or "Q" == period:
        returns = group_returns(returns, [index.year, index.quarter],
                                compounded=compounded)
    elif not isinstance(period, str):
        return group_returns(returns, period, compounded)

    return returns


def to_excess_returns(returns, rf, nperiods=None):
    """
    Calculates excess returns by subtracting
    risk-free returns from total returns

    Args:
        * returns (Series, DataFrame): Returns
        * rf (float, Series, DataFrame): Risk-Free rate(s)
        * nperiods (int): Optional. If provided, will convert rf to different
            frequency using deannualize
    Returns:
        * excess_returns (Series, DataFrame): Returns - rf
    """
    if not isinstance(rf, float):
        rf = rf[rf.index.isin(returns.index)]

    if nperiods is not None:
        # deannualize
        rf = _np.power(1 + returns, 1. / nperiods) - 1.

    return returns - rf


def _prepare_prices(data, base=1.):
    """
    Converts return data into prices + cleanup
    """
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col], base)

    elif data.min() <= 0 and data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, _pd.DataFrame) or isinstance(data, _pd.Series):
        data = data.fillna(0).replace(
            [_np.inf, -_np.inf], float('NaN'))

    return data


def _prepare_returns(data, rf=0., nperiods=None):
    """
    Converts price data into returns + cleanup
    """
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() >= 0 or data[col].dropna().max() > 1:
                data[col] = data[col].pct_change()
    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change()

    if isinstance(data, _pd.DataFrame) or isinstance(data, _pd.Series):
        data = data.fillna(0).replace(
            [_np.inf, -_np.inf], float('NaN'))

    if rf > 0:
        return to_excess_returns(data, rf, nperiods)
    return data


def download_returns(ticker, period="max"):
    if isinstance(period, _pd.DatetimeIndex):
        p = {"start": period[0]}
    else:
        p = {"period": period}
    return _yf.Ticker(ticker).history(**p)['Close'].pct_change()


def _prepare_benchmark(benchmark=None, period="max", rf=0.):
    """
    fetch benchmark if ticker is provided, and pass through
    _prepare_returns()

    period can be options or (expected) _pd.DatetimeIndex range
    """
    if benchmark is None:
        return None

    if isinstance(benchmark, str):
        benchmark = download_returns(benchmark)

    elif isinstance(benchmark, _pd.DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()

    if isinstance(period, _pd.DatetimeIndex):
        benchmark = benchmark[benchmark.index.isin(period)]

    return _prepare_returns(benchmark.dropna(), rf=rf)


def _round_to_closest(val, res, decimals=None):
    """ round to closest resolution """
    if decimals is None and "." in str(res):
        decimals = len(str(res).split('.')[1])
    return round(round(val / res) * res, decimals)


def _file_stream():
    """ Returns a file stream """
    return _io.BytesIO()


def _in_notebook():
    """
    Identify enviroment (notebook, terminal, etc)
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # from IPython.core.display import display, HTML, Image
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def _count_consecutive(data):
    """
    Counts consecutive data (like cumsum() with reset on zeroes)
    """

    def _count(data):
        return data * (data.groupby(
            (data != data.shift(1)).cumsum()).cumcount() + 1)

    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def _score_str(val):
    """
    Returns + sign for positive values (used in plots)
    """
    return ("" if "-" in val else "+") + str(val)


def _make_portfolio(returns, start_balance=1e5, round_to=None):
    """
    Calculates compounded value of portfolio
    """
    comp_rev = (start_balance + start_balance *
                returns.shift(1)).fillna(start_balance) * returns
    p1 = start_balance + comp_rev.cumsum()

    # add day before with starting balance
    p0 = _pd.Series(data=start_balance,
                    index=p1.index + _pd.Timedelta(days=-1))[:1]

    portfolio = _pd.concat([p0, p1])

    if isinstance(returns, _pd.DataFrame):
        portfolio.loc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    if round_to:
        portfolio = _np.round(portfolio, round_to)

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """
    Dirty method for flattening multi-index dataframe
    """
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df

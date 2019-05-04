#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019 Ran Aroussi
#
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

from __future__ import division, print_function

from io import BytesIO, StringIO
import pandas as pd
import numpy as np


def compsum(returns):
    """
    Calculates rolling compounded returns
    """
    return returns.add(1).cumprod()


def comp(returns):
    """
    Calculates total compounded returns
    """
    return returns.add(1).prod()


def to_returns(prices, rf=0.):
    """
    Calculates the simple arithmetic returns of a price series
    """
    return _cleanup_returns(prices)


def to_prices(returns, base=1e5):
    """
    Converts returns series to price data
    """
    returns = returns.copy().fillna(0).replace([np.inf, -np.inf], float('NaN'))
    return returns.add(base).cumprod().subtract(base)


def log_returns(returns):
    """
    Converts returns series to log returns
    """
    returns = _cleanup_returns(returns)
    try:
        return np.log(returns+1).replace([np.inf, -np.inf], float('NaN'))
    except Exception:
        return 0.


def exponential_stdev(returns, window=30, is_halflife=False):
    """
    Returns series representing exponential volatility of returns
    """
    returns = _cleanup_returns(returns)
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


def aggregate_returns(returns, period=None, compounded=True):
    """
    Aggregates returns based on date periods
    """

    def _aggregate(returns, groupby, compounded=False):
        """ summarize returns
        _aggregate(df, df.index.year)
        _aggregate(df, [df.index.year, df.index.month])
        """
        if compounded:
            return returns.groupby(groupby).apply(comp)
        return returns.groupby(groupby).sum()

    if period is None or 'day' in period:
        return returns
    index = returns.index
    if 'month' in period:
        returns = _aggregate(returns, index.month, compounded=compounded)
    if 'quarter' in period:
        returns = _aggregate(returns, index.quarter, compounded=compounded)
    elif 'year' in period or 'eoy' in period or 'yoy' in period:
        returns = _aggregate(returns, index.year, compounded=compounded)
    elif 'week' in period:
        returns = _aggregate(returns, index.week, compounded=compounded)
    elif 'eow' in period:
        returns = _aggregate(returns, [index.year, index.week],
                             compounded=compounded)
    elif 'eom' in period:
        returns = _aggregate(returns, [index.year, index.month],
                             compounded=compounded)
    elif 'eoq' in period:
        returns = _aggregate(returns, [index.year, index.quarter],
                             compounded=compounded)
    elif not isinstance(period, str):
        return _aggregate(returns, period, compounded)

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
        rf = np.power(1 + returns, 1. / nperiods) - 1.

    return returns - rf


def _cleanup_prices(data):
    """
    Converts return data into prices + cleanup
    """
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col])

    elif data.dropna().min() <= 0 or data.dropna().max() < 1:
        data = to_prices(data)

    return data.dropna()


def _cleanup_returns(data, rf=0., nperiods=None):
    """
    Converts price data into returns + cleanup
    """
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() >= 0 or data[col].dropna().max() > 1:
                data[col] = data[col].pct_change()
    elif data.dropna().min() >= 0 or data.dropna().max() > 1:
        data = data.pct_change()

    data = data.fillna(0).replace([np.inf, -np.inf], float('NaN'))

    if rf > 0:
        return to_excess_returns(data, rf, nperiods)
    return data


def _file_stream():
    """ Returns a file stream """
    return BytesIO()


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

    if isinstance(data, pd.DataFrame):
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
    p0 = pd.Series(data=start_balance,
                   index=p1.index + pd.Timedelta(days=-1))[:1]

    portfolio = pd.concat([p0, p1])

    if isinstance(returns, pd.DataFrame):
        portfolio.loc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    if round_to:
        portfolio = np.round(portfolio, round_to)

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """
    Dirty method for flattening multi-index dataframe
    """
    s_buf = StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df

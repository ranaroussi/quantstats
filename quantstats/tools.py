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


def file_stream():
    return BytesIO()


def in_notebook():
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


def count_consecutive(data):
    return data * (data.groupby(
        (data != data.shift(1)).cumsum()).cumcount() + 1)


def score_str(i):
    return ("" if "-" in i else "+") + str(i)


def returns(prices, rf=0.):
    """
    Calculates the simple arithmetic returns of a price series.
    Formula is: (t1 / t0) - 1
    Args:
        * prices: Expects a price series/dataframe
    """
    return cleanup_returns(prices)


def to_price_series(returns):
    returns = returns.copy().fillna(0).replace([np.inf, -np.inf], float('NaN'))
    return returns.add(1).cumprod().subtract(1)


def log_returns(returns):
    """ Returns pandas Series representing period log returns """
    returns = cleanup_returns(returns)
    try:
        return np.log(returns+1).replace([np.inf, -np.inf], float('NaN'))
    except Exception:
        return 0.


def exponential_stdev(returns, window=30, is_halflife=False):
    """ Returns pandas Series representing volatility of prices """
    returns = cleanup_returns(returns)
    halflife = window if is_halflife else None
    return returns.ewm(com=None, span=window,
                       halflife=halflife, min_periods=window).std()


def rebase(prices, value=100):
    """
    Rebase all series to a given intial value.
    This makes comparing/plotting different series
    together easier.
    Args:
        * prices: Expects a price series/dataframe
        * value (number): starting value for all series.
    """
    return prices.dropna() / prices.dropna().ix[0] * value


def cleanup_returns(data, rf=0., nperiods=None):
    # data is prices? convert to returns
    if data.dropna().min() >= 0 or data.dropna().max() > 1:
        data = data.pct_change()

    data = data.fillna(0).replace([np.inf, -np.inf], float('NaN'))

    if rf > 0:
        return to_excess_returns(data, rf, nperiods)
    return data


def aggregate_returns(returns, period=None, compounded=True):
    """ shortcut to aggregate """
    index = returns.index
    if 'oem' in period or 'month' in period:
        returns = aggregate(returns, index.month, compounded=compounded)
    elif 'eoy' in period or 'yoy' in period:
        returns = aggregate(returns, index.year, compounded=compounded)
    elif 'year' in period:
        returns = aggregate(returns, [index.year, index.month],
                            compounded=compounded)
    return returns


def aggregate(returns, groupby, compounded=False):
    """ summarize returns
    aggregate(df, df.index.year)
    aggregate(df, [df.index.year, df.index.month])
    """
    def returns_prod(data):
        return (data + 1).prod() - 1
    if compounded:
        return returns.groupby(groupby).apply(returns_prod)
    return returns.groupby(groupby).sum()


def to_excess_returns(returns, rf, nperiods=None):
    """
    Given a series of returns, it will return the excess returns over rf.

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


def _inspect_portfolio(returns, start_balance=1):

    stats = pd.DataFrame(returns)
    stats.columns = ['abs_ret']
    stats['sum_ret'] = stats['abs_ret'].cumsum()
    stats['comp_ret'] = stats['abs_ret'].compsum()

    stats['naive_rev'] = start_balance * stats['abs_ret']
    stats['total_naive_rev'] = start_balance + stats['naive_rev'].cumsum()

    # invest compounded amount
    stats['comp_rev'] = (start_balance + start_balance *
                         stats['abs_ret'].shift(1)
                         ).fillna(start_balance) * stats['abs_ret']
    stats['total_comp_rev'] = start_balance + stats['comp_rev'].cumsum()

    # invest same amount (of less after down days)
    stats['down_comp_rev'] = np.where(stats['abs_ret'].shift(
        1) < 0, stats['comp_rev'], stats['naive_rev'])
    stats['total_down_comp_rev'] = start_balance + \
        stats['down_comp_rev'].cumsum()

    stats['adj_ret'] = stats['total_comp_rev'].pct_change().fillna(
        stats['sum_ret'])
    stats['adj_down_ret'] = stats['total_down_comp_rev'].pct_change().fillna(
        stats['sum_ret'])

    stats['adj_comp_ret'] = stats['adj_ret'].cumsum().fillna(stats['sum_ret'])
    stats['adj_down_comp_ret'] = stats['adj_down_ret'].cumsum().fillna(
        stats['sum_ret'])

    stats['pnl'] = stats['total_comp_rev'] - start_balance
    return stats["abs_ret comp_rev total_comp_rev adj_ret pnl".split()]


def _make_portfolio(returns, start_balance=1e5):
    # invest compounded amount
    comp_rev = (start_balance + start_balance *
                returns.shift(1)).fillna(start_balance) * returns
    port = np.round(start_balance + comp_rev.cumsum(), 2)
    port2 = pd.Series(
        index=(port.index + pd.Timedelta(days=-1)),
        data=start_balance)[:1]
    return pd.concat([port2, port])


def flatten_dataframe(df, set_index=None):
    """ flatten multi-index dataframe """
    s_buf = StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df

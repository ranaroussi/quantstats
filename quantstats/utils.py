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
import datetime as _dt
import pandas as _pd
import numpy as _np
import yfinance as _yf
from . import stats as _stats


def _mtd(df):
    return df[df.index >= _dt.datetime.now(
    ).strftime('%Y-%m-01')]


def _qtd(df):
    date = _dt.datetime.now()
    for q in [1, 4, 7, 10]:
        if date.month <= q:
            return df[df.index >= _dt.datetime(
                date.year, q, 1).strftime('%Y-%m-01')]
    return df[df.index >= date.strftime('%Y-%m-01')]


def _ytd(df):
    return df[df.index >= _dt.datetime.now(
    ).strftime('%Y-01-01')]


def _pandas_date(df, dates):
    if not isinstance(dates, list):
        dates = [dates]
    return df[df.index.isin(dates)]


def _pandas_current_month(df):
    n = _dt.datetime.now()
    daterange = _pd.date_range(_dt.date(n.year, n.month, 1), n)
    return df[df.index.isin(daterange)]


def multi_shift(df, shift=3):
    """ get last N rows relative to another row in pandas """
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)

    dfs = [df.shift(i) for i in _np.arange(shift)]
    for ix, dfi in enumerate(dfs[1:]):
        dfs[ix + 1].columns = [str(col) for col in dfi.columns + str(ix + 1)]
    return _pd.concat(dfs, 1, sort=True)


def to_returns(prices, rf=0.):
    """ Calculates the simple arithmetic returns of a price series """
    return _prepare_returns(prices, rf)


def to_prices(returns, base=1e5):
    """ Converts returns series to price data """
    returns = returns.copy().fillna(0).replace(
        [_np.inf, -_np.inf], float('NaN'))

    return base + base * _stats.compsum(returns)


def log_returns(returns, rf=0., nperiods=None):
    """ shorthand for to_log_returns """
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns, rf=0., nperiods=None):
    """ Converts returns series to log returns """
    returns = _prepare_returns(returns, rf, nperiods)
    try:
        return _np.log(returns+1).replace([_np.inf, -_np.inf], float('NaN'))
    except Exception:
        return 0.


def exponential_stdev(returns, window=30, is_halflife=False):
    """ Returns series representing exponential volatility of returns """
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
    return prices.dropna() / prices.dropna().iloc[0] * base


def group_returns(returns, groupby, compounded=False):
    """ summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(_stats.comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """ Aggregates returns based on date periods """

    if period is None or 'day' in period:
        return returns
    index = returns.index

    if 'month' in period:
        return group_returns(returns, index.month, compounded=compounded)

    if 'quarter' in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    if period == "A" or any(x in period for x in ['year', 'eoy', 'yoy']):
        return group_returns(returns, index.year, compounded=compounded)

    if 'week' in period:
        return group_returns(returns, index.week, compounded=compounded)

    if 'eow' in period or period == "W":
        return group_returns(returns, [index.year, index.week],
                             compounded=compounded)

    if 'eom' in period or period == "M":
        return group_returns(returns, [index.year, index.month],
                             compounded=compounded)

    if 'eoq' in period or period == "Q":
        return group_returns(returns, [index.year, index.quarter],
                             compounded=compounded)

    if not isinstance(period, str):
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
    if isinstance(rf, int):
        rf = float(rf)

    if not isinstance(rf, float):
        rf = rf[rf.index.isin(returns.index)]

    if nperiods is not None:
        # deannualize
        rf = _np.power(1 + rf, 1. / nperiods) - 1.

    return returns - rf


def _prepare_prices(data, base=1.):
    """ Converts return data into prices + cleanup """
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col], base)

    # is it returns?
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace(
            [_np.inf, -_np.inf], float('NaN'))

    return data


def _prepare_returns(data, rf=0., nperiods=None):
    """ Converts price data into returns + cleanup """
    data = data.copy()

    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() >= 0 and data[col].dropna().max() > 1:
                data[col] = data[col].pct_change()
    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change()

    # cleanup data
    data = data.replace([_np.inf, -_np.inf], float('NaN'))

    if isinstance(data, (_pd.DataFrame, _pd.Series)):
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


def _in_notebook(matplotlib_inline=False):
    """ Identify enviroment (notebook, terminal, etc) """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            if matplotlib_inline:
                get_ipython().magic("matplotlib inline")
            return True
        if shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        # Other type (?)
        return False
    except NameError:
        # Probably standard Python interpreter
        return False


def _count_consecutive(data):
    """ Counts consecutive data (like cumsum() with reset on zeroes) """
    def _count(data):
        return data * (data.groupby(
            (data != data.shift(1)).cumsum()).cumcount() + 1)

    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def _score_str(val):
    """ Returns + sign for positive values (used in plots) """
    return ("" if "-" in val else "+") + str(val)


def make_index(ticker_weights, rebalance=None, period="max"):
    """ Makes an index out of the given tickers and weights. """
    # Declare a returns variable
    index = None

    # Iterate over weights
    for ticker, ticker_weight in ticker_weights.items():
        # Download the returns for this ticker, e.g. GOOG
        ticker_returns = download_returns(ticker, period)
        if index is None:
            # Set the returns to this return series if it's empty
            index = ticker_returns * ticker_weight
        else:
            # Otherwise, add weighted return
            index += ticker_returns * ticker_weight
    # Return total index
    return index


def make_portfolio(returns, start_balance=1e5,
                   mode="comp", round_to=None):
    """ Calculates compounded value of portfolio """
    returns = _prepare_returns(returns)

    if mode.lower() in ["cumsum", "sum"]:
        p1 = start_balance + start_balance * returns.cumsum()
    elif mode.lower() in ["compsum", "comp"]:
        p1 = to_prices(returns, start_balance)
    else:
        # fixed amount every day
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
    """ Dirty method for flattening multi-index dataframe """
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df

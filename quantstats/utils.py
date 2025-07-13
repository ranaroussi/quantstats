#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2024 Ran Aroussi
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
from ._compat import safe_yfinance_download
from . import stats as _stats
from ._compat import safe_concat, safe_resample
import inspect
from functools import lru_cache
import hashlib


# Custom exception classes for QuantStats
class QuantStatsError(Exception):
    """Base exception class for QuantStats"""
    pass


class DataValidationError(QuantStatsError):
    """Raised when input data validation fails"""
    pass


class CalculationError(QuantStatsError):
    """Raised when a calculation fails"""
    pass


class PlottingError(QuantStatsError):
    """Raised when plotting operations fail"""
    pass


class BenchmarkError(QuantStatsError):
    """Raised when benchmark data issues occur"""
    pass


def validate_input(data, allow_empty=False):
    """
    Validate input data for QuantStats functions
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data to validate
    allow_empty : bool, default False
        Whether to allow empty datasets
        
    Raises
    ------
    DataValidationError
        If data validation fails
    """
    if data is None:
        raise DataValidationError("Input data cannot be None")
    
    if not isinstance(data, (_pd.Series, _pd.DataFrame)):
        raise DataValidationError(f"Input data must be pandas Series or DataFrame, got {type(data)}")
    
    if not allow_empty and len(data) == 0:
        raise DataValidationError("Input data cannot be empty")
    
    if not allow_empty and data.dropna().empty:
        raise DataValidationError("Input data contains only NaN values")
    
    # Check for valid date index
    if not isinstance(data.index, (_pd.DatetimeIndex, _pd.RangeIndex)):
        try:
            data.index = _pd.to_datetime(data.index)
        except Exception:
            raise DataValidationError("Input data must have a valid datetime index")
    
    return True


# Cache for _prepare_returns function
_PREPARE_RETURNS_CACHE = {}
_CACHE_MAX_SIZE = 100


def _generate_cache_key(data, rf, nperiods):
    """Generate a cache key for the _prepare_returns function"""
    try:
        # Create a hash from the data
        if isinstance(data, _pd.Series):
            data_hash = _pd.util.hash_pandas_object(data).sum()
        elif isinstance(data, _pd.DataFrame):
            data_hash = _pd.util.hash_pandas_object(data).sum()
        else:
            data_hash = hash(str(data))
        
        # Include parameters in the key
        key = f"{data_hash}_{rf}_{nperiods}"
        return key
    except (ValueError, TypeError, AttributeError, MemoryError):
        # If hashing fails, return None to skip caching
        return None


def _clear_cache_if_full():
    """Clear cache if it exceeds maximum size"""
    if len(_PREPARE_RETURNS_CACHE) >= _CACHE_MAX_SIZE:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_PREPARE_RETURNS_CACHE.keys())[:-(_CACHE_MAX_SIZE//2)]
        for key in keys_to_remove:
            del _PREPARE_RETURNS_CACHE[key]


def _mtd(df):
    return df[df.index >= _dt.datetime.now().strftime("%Y-%m-01")]


def _qtd(df):
    date = _dt.datetime.now()
    for q in [1, 4, 7, 10]:
        if date.month <= q:
            return df[df.index >= _dt.datetime(date.year, q, 1).strftime("%Y-%m-01")]
    return df[df.index >= date.strftime("%Y-%m-01")]


def _ytd(df):
    return df[df.index >= _dt.datetime.now().strftime("%Y-01-01")]


def _pandas_date(df, dates):
    if not isinstance(dates, list):
        dates = [dates]
    return df[df.index.isin(dates)]


def _pandas_current_month(df):
    n = _dt.datetime.now()
    daterange = _pd.date_range(_dt.date(n.year, n.month, 1), n)
    return df[df.index.isin(daterange)]


def multi_shift(df, shift=3):
    """Get last N rows relative to another row in pandas - optimized for memory usage"""
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)

    # More memory-efficient approach using dictionary comprehension
    # and direct column assignment
    result = df.copy()
    
    for i in range(1, shift):
        shifted = df.shift(i)
        # Rename columns to avoid conflicts
        shifted.columns = [f"{col}{i}" for col in shifted.columns]
        result = safe_concat([result, shifted], axis=1, sort=True)
    
    return result


def to_returns(prices, rf=0.0):
    """Calculates the simple arithmetic returns of a price series"""
    return _prepare_returns(prices, rf)


def to_prices(returns, base=1e5):
    """Converts returns series to price data"""
    returns = returns.copy().fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    return base + base * _stats.compsum(returns)


def log_returns(returns, rf=0.0, nperiods=None):
    """Shorthand for to_log_returns"""
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns, rf=0.0, nperiods=None):
    """Converts returns series to log returns"""
    returns = _prepare_returns(returns, rf, nperiods)
    try:
        return _np.log(returns + 1).replace([_np.inf, -_np.inf], float("NaN"))
    except (ValueError, TypeError, AttributeError, OverflowError):
        return 0.0


def exponential_stdev(returns, window=30, is_halflife=False):
    """Returns series representing exponential volatility of returns"""
    returns = _prepare_returns(returns)
    halflife = window if is_halflife else None
    return returns.ewm(
        com=None, span=window, halflife=halflife, min_periods=window
    ).std()


def rebase(prices, base=100.0):
    """
    Rebase all series to a given intial base.
    This makes comparing/plotting different series together easier.
    Args:
        * prices: Expects a price series/dataframe
        * base (number): starting value for all series.
    """
    return prices.dropna() / prices.dropna().iloc[0] * base


def group_returns(returns, groupby, compounded=False):
    """Summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(_stats.comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """Aggregates returns based on date periods"""
    if period is None or "day" in period:
        return returns
    index = returns.index

    if "month" in period:
        return group_returns(returns, index.month, compounded=compounded)

    if "quarter" in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    if period == "YE" or any(x in period for x in ["year", "eoy", "yoy"]):
        return group_returns(returns, index.year, compounded=compounded)

    if "week" in period:
        return group_returns(returns, index.week, compounded=compounded)

    if "eow" in period or period == "W":
        return group_returns(returns, [index.year, index.week], compounded=compounded)

    if "eom" in period or period == "ME":
        return group_returns(returns, [index.year, index.month], compounded=compounded)

    if "eoq" in period or period == "QE":
        return group_returns(
            returns, [index.year, index.quarter], compounded=compounded
        )

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
        rf = _np.power(1 + rf, 1.0 / nperiods) - 1.0

    df = returns - rf
    df = df.tz_localize(None)
    return df


def _prepare_prices(data, base=1.0):
    """Converts return data into prices + cleanup"""
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            # Cache dropna operation to avoid repeated computation
            col_clean = data[col].dropna()
            if col_clean.min() <= 0 or col_clean.max() < 1:
                data[col] = to_prices(data[col], base)

    # is it returns?
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    data = data.tz_localize(None)
    return data


def _prepare_returns(data, rf=0.0, nperiods=None):
    """Converts price data into returns + cleanup"""
    # Try to get from cache first
    cache_key = _generate_cache_key(data, rf, nperiods)
    if cache_key and cache_key in _PREPARE_RETURNS_CACHE:
        return _PREPARE_RETURNS_CACHE[cache_key].copy()
    
    data = data.copy()
    function = inspect.stack()[1][3]
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            # Cache dropna operation to avoid repeated computation
            col_clean = data[col].dropna()
            if col_clean.min() >= 0 and col_clean.max() > 1:
                data[col] = data[col].pct_change()
    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change()

    # cleanup data
    data = data.replace([_np.inf, -_np.inf], float("NaN"))

    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))
    unnecessary_function_calls = [
        "_prepare_benchmark",
        "cagr",
        "gain_to_pain_ratio",
        "rolling_volatility",
    ]

    if function not in unnecessary_function_calls:
        if rf > 0:
            result = to_excess_returns(data, rf, nperiods)
            # Cache the result
            if cache_key:
                _clear_cache_if_full()
                _PREPARE_RETURNS_CACHE[cache_key] = result.copy()
            return result

    data = data.tz_localize(None)
    
    # Cache the result
    if cache_key:
        _clear_cache_if_full()
        _PREPARE_RETURNS_CACHE[cache_key] = data.copy()
    
    return data


def download_returns(ticker, period="max", proxy=None):
    params = {
        "tickers": ticker,
        "auto_adjust": True,
        "multi_level_index": False,
        "progress": False,
    }
    if isinstance(period, _pd.DatetimeIndex):
        params["start"] = period[0]
    else:
        params["period"] = period
    
    df = safe_yfinance_download(proxy=proxy, **params)["Close"].pct_change()
    df = df.tz_localize(None)
    return df


def _prepare_benchmark(benchmark=None, period="max", rf=0.0, prepare_returns=True):
    """
    Fetch benchmark if ticker is provided, and pass through
    _prepare_returns()

    period can be options or (expected) _pd.DatetimeIndex range
    """
    if benchmark is None:
        return None

    if isinstance(benchmark, str):
        benchmark = download_returns(benchmark)

    elif isinstance(benchmark, _pd.DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()

    if isinstance(period, _pd.DatetimeIndex) and set(period) != set(benchmark.index):

        # Adjust Benchmark to Strategy frequency
        benchmark_prices = to_prices(benchmark, base=1)
        new_index = _pd.date_range(start=period[0], end=period[-1], freq="D")
        benchmark = (
            benchmark_prices.reindex(new_index, method="bfill")
            .reindex(period)
            .pct_change()
            .fillna(0)
        )
        benchmark = benchmark[benchmark.index.isin(period)]

    benchmark = benchmark.tz_localize(None)

    if prepare_returns:
        return _prepare_returns(benchmark.dropna(), rf=rf)
    return benchmark.dropna()


def _round_to_closest(val, res, decimals=None):
    """Round to closest resolution"""
    if decimals is None and "." in str(res):
        decimals = len(str(res).split(".")[1])
    return round(round(val / res) * res, decimals)


def _file_stream():
    """Returns a file stream"""
    return _io.BytesIO()


def _in_notebook(matplotlib_inline=False):
    """Identify enviroment (notebook, terminal, etc)"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            if matplotlib_inline:
                get_ipython().magic("matplotlib inline")
            return True
        if shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        # Other type (?)
        return False
    except NameError:
        # Probably standard Python interpreter
        return False


def _count_consecutive(data):
    """Counts consecutive data (like cumsum() with reset on zeroes)"""

    def _count(data):
        return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def _score_str(val):
    """Returns + sign for positive values (used in plots)"""
    return ("" if "-" in val else "+") + str(val)


def make_index(
    ticker_weights, rebalance="1M", period="max", returns=None, match_dates=False
):
    """
    Makes an index out of the given tickers and weights.
    Optionally you can pass a dataframe with the returns.
    If returns is not given it try to download them with yfinance

    Args:
        * ticker_weights (Dict): A python dict with tickers as keys
            and weights as values
        * rebalance: Pandas resample interval or None for never
        * period: time period of the returns to be downloaded
        * returns (Series, DataFrame): Optional. Returns If provided,
            it will fist check if returns for the given ticker are in
            this dataframe, if not it will try to download them with
            yfinance
    Returns:
        * index_returns (Series, DataFrame): Returns for the index
    """
    # Declare a returns variable
    index = None
    portfolio = {}

    # Iterate over weights
    for ticker in ticker_weights.keys():
        if (returns is None) or (ticker not in returns.columns):
            # Download the returns for this ticker, e.g. GOOG
            ticker_returns = download_returns(ticker, period)
        else:
            ticker_returns = returns[ticker]

        portfolio[ticker] = ticker_returns

    # index members time-series
    index = _pd.DataFrame(portfolio).dropna()

    if match_dates:
        index = index[max(index.ne(0).idxmax()) :]

    # no rebalance?
    if rebalance is None:
        for ticker, weight in ticker_weights.items():
            index[ticker] = weight * index[ticker]
        return index.sum(axis=1)

    last_day = index.index[-1]

    # rebalance marker
    rbdf = safe_resample(index, rebalance, "first")
    rbdf["break"] = rbdf.index.strftime("%s")

    # index returns with rebalance markers
    index = safe_concat([index, rbdf["break"]], axis=1)

    # mark first day day
    index["first_day"] = _pd.isna(index["break"]) & ~_pd.isna(index["break"].shift(1))
    index.loc[index.index[0], "first_day"] = True

    # multiply first day of each rebalance period by the weight
    for ticker, weight in ticker_weights.items():
        index[ticker] = _np.where(
            index["first_day"], weight * index[ticker], index[ticker]
        )

    # drop first marker
    index = index.drop(columns=["first_day"])

    # drop when all are NaN
    index = index.dropna(how="all")
    return index[index.index <= last_day].sum(axis=1)


def make_portfolio(returns, start_balance=1e5, mode="comp", round_to=None):
    """Calculates compounded value of portfolio"""
    returns = _prepare_returns(returns)

    if mode.lower() in ["cumsum", "sum"]:
        p1 = start_balance + start_balance * returns.cumsum()
    elif mode.lower() in ["compsum", "comp"]:
        p1 = to_prices(returns, start_balance)
    else:
        # fixed amount every day
        comp_rev = (start_balance + start_balance * returns.shift(1)).fillna(
            start_balance
        ) * returns
        p1 = start_balance + comp_rev.cumsum()

    # add day before with starting balance
    p0 = _pd.Series(data=start_balance, index=p1.index + _pd.Timedelta(days=-1))[:1]

    portfolio = safe_concat([p0, p1])

    if isinstance(returns, _pd.DataFrame):
        portfolio.iloc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    if round_to:
        portfolio = _np.round(portfolio, round_to)

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """Dirty method for flattening multi-index dataframe"""
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df

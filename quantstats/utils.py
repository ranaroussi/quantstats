#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2025 Ran Aroussi
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
import threading


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
        raise DataValidationError(
            f"Input data must be pandas Series or DataFrame, got {type(data)}"
        )

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


# Cache for _prepare_returns function with thread safety
_PREPARE_RETURNS_CACHE = {}
_CACHE_MAX_SIZE = 100
_cache_lock = threading.Lock()


def _generate_cache_key(data, rf, nperiods):
    """
    Generate a cache key for the _prepare_returns function

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data to generate hash from
    rf : float
        Risk-free rate parameter
    nperiods : int
        Number of periods parameter

    Returns
    -------
    str or None
        Cache key string or None if hashing fails
    """
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
    """
    Clear cache if it exceeds maximum size

    Uses a simple FIFO strategy, keeping the most recent half of entries
    when cache size limit is exceeded.
    """
    with _cache_lock:
        if len(_PREPARE_RETURNS_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entries (simple FIFO) - keep the most recent half
            keys_to_remove = list(_PREPARE_RETURNS_CACHE.keys())[:-(_CACHE_MAX_SIZE // 2)]
            for key in keys_to_remove:
                del _PREPARE_RETURNS_CACHE[key]


def _mtd(df):
    """
    Filter dataframe to month-to-date data

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Input data with datetime index

    Returns
    -------
    pd.DataFrame or pd.Series
        Filtered data from start of current month
    """
    # Get first day of current month as string
    return df[df.index >= _dt.datetime.now().strftime("%Y-%m-01")]


def _qtd(df):
    """
    Filter dataframe to quarter-to-date data

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Input data with datetime index

    Returns
    -------
    pd.DataFrame or pd.Series
        Filtered data from start of current quarter
    """
    date = _dt.datetime.now()
    # Check which quarter we're in (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
    for q in [1, 4, 7, 10]:  # First month of each quarter
        if date.month <= q:
            return df[df.index >= _dt.datetime(date.year, q, 1).strftime("%Y-%m-01")]
    # Default to current month if no quarter match
    return df[df.index >= date.strftime("%Y-%m-01")]


def _ytd(df):
    """
    Filter dataframe to year-to-date data

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Input data with datetime index

    Returns
    -------
    pd.DataFrame or pd.Series
        Filtered data from start of current year
    """
    # Get first day of current year as string
    return df[df.index >= _dt.datetime.now().strftime("%Y-01-01")]


def _pandas_date(df, dates):
    """
    Filter dataframe to specific dates

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Input data with datetime index
    dates : list or single date
        Date(s) to filter by

    Returns
    -------
    pd.DataFrame or pd.Series
        Filtered data for specified dates
    """
    # Ensure dates is a list for consistent processing
    if not isinstance(dates, list):
        dates = [dates]
    return df[df.index.isin(dates)]


def _pandas_current_month(df):
    """
    Filter dataframe to current month's data

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Input data with datetime index

    Returns
    -------
    pd.DataFrame or pd.Series
        Filtered data for current month
    """
    n = _dt.datetime.now()
    # Create date range from first day of current month to now
    daterange = _pd.date_range(_dt.date(n.year, n.month, 1), n)
    return df[df.index.isin(daterange)]


def multi_shift(df, shift=3):
    """Get last N rows relative to another row in pandas - optimized for memory usage"""
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)

    # More memory-efficient approach using dictionary comprehension
    # and direct column assignment
    result = df.copy()

    # Create lagged versions of the data
    for i in range(1, shift):
        shifted = df.shift(i)
        # Rename columns to avoid conflicts
        shifted.columns = [f"{col}{i}" for col in shifted.columns]
        result = safe_concat([result, shifted], axis=1, sort=True)

    return result


def to_returns(prices, rf=0.0):
    """
    Calculate simple arithmetic returns from price series

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price data
    rf : float, default 0.0
        Risk-free rate

    Returns
    -------
    pd.Series or pd.DataFrame
        Simple arithmetic returns
    """
    return _prepare_returns(prices, rf)


def to_prices(returns, base=1e5):
    """
    Convert returns series to price data

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    base : float, default 1e5
        Starting base value for price series

    Returns
    -------
    pd.Series or pd.DataFrame
        Price data calculated from returns
    """
    # Clean returns data by filling NaN and replacing infinite values
    returns = returns.copy().fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # Convert returns to prices using compounded sum
    return base + base * _stats.compsum(returns)


def log_returns(returns, rf=0.0, nperiods=None):
    """
    Shorthand for to_log_returns function

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    rf : float, default 0.0
        Risk-free rate
    nperiods : int, optional
        Number of periods for risk-free rate conversion

    Returns
    -------
    pd.Series or pd.DataFrame
        Log returns
    """
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns, rf=0.0, nperiods=None):
    """
    Convert returns series to log returns

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    rf : float, default 0.0
        Risk-free rate
    nperiods : int, optional
        Number of periods for risk-free rate conversion

    Returns
    -------
    pd.Series or pd.DataFrame
        Log returns calculated as ln(1 + returns)
    """
    returns = _prepare_returns(returns, rf, nperiods)
    try:
        # Calculate log returns: ln(1 + returns)
        return _np.log(returns + 1).replace([_np.inf, -_np.inf], float("NaN"))  # type: ignore
    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        from warnings import warn
        warn(f"Error converting to log returns: {type(e).__name__}: {e}, returning 0.0")
        return 0.0


def exponential_stdev(returns, window=30, is_halflife=False):
    """
    Calculate exponential weighted standard deviation (volatility) of returns

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    window : int, default 30
        Window size for exponential weighting
    is_halflife : bool, default False
        Whether window parameter represents halflife

    Returns
    -------
    pd.Series or pd.DataFrame
        Exponential weighted standard deviation
    """
    returns = _prepare_returns(returns)
    # Set halflife parameter based on is_halflife flag
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
    # Normalize prices to start at the base value
    return prices.dropna() / prices.dropna().iloc[0] * base


def group_returns(returns, groupby, compounded=False):
    """
    Summarize returns by grouping criteria

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    groupby : grouper object
        Pandas groupby object or criteria
    compounded : bool, default False
        Whether to compound returns or use simple sum

    Returns
    -------
    pd.Series or pd.DataFrame
        Grouped returns

    Examples
    --------
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        # Use compounded returns calculation
        return returns.groupby(groupby).apply(_stats.comp)
    # Use simple sum for non-compounded returns
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """
    Aggregate returns based on specified time periods

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    period : str, optional
        Time period for aggregation ('month', 'quarter', 'year', etc.)
    compounded : bool, default True
        Whether to compound returns

    Returns
    -------
    pd.Series or pd.DataFrame
        Aggregated returns for specified period
    """
    # Return original data if no period specified or daily period
    if period is None or "day" in period:
        return returns

    index = returns.index

    # Group by month
    if "month" in period:
        return group_returns(returns, index.month, compounded=compounded)

    # Group by quarter
    if "quarter" in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    # Group by year (multiple possible period strings)
    if period == "YE" or any(x in period for x in ["year", "eoy", "yoy"]):
        return group_returns(returns, index.year, compounded=compounded)

    # Group by week
    if "week" in period:
        return group_returns(returns, index.week, compounded=compounded)

    # End of week grouping
    if "eow" in period or period == "W":
        return group_returns(returns, [index.year, index.week], compounded=compounded)

    # End of month grouping
    if "eom" in period or period == "ME":
        return group_returns(returns, [index.year, index.month], compounded=compounded)

    # End of quarter grouping
    if "eoq" in period or period == "QE":
        return group_returns(
            returns, [index.year, index.quarter], compounded=compounded
        )

    # Custom period grouping (non-string)
    if not isinstance(period, str):
        return group_returns(returns, period, compounded)

    # Default: return original data
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
    # Convert integer rf to float for consistency
    if isinstance(rf, int):
        rf = float(rf)

    # Align rf with returns index if rf is a series/dataframe
    if not isinstance(rf, float):
        rf = rf[rf.index.isin(returns.index)]  # type: ignore

    # Deannualize rf if nperiods is provided
    if nperiods is not None:
        # deannualize
        rf = _np.power(1 + rf, 1.0 / nperiods) - 1.0

    # Calculate excess returns
    df = returns - rf
    df = df.tz_localize(None)
    return df


def _prepare_prices(data, base=1.0):
    """
    Convert return data into prices and perform cleanup

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data (returns or prices)
    base : float, default 1.0
        Base value for price conversion

    Returns
    -------
    pd.Series or pd.DataFrame
        Cleaned price data
    """
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            # Cache dropna operation to avoid repeated computation
            col_clean = data[col].dropna()
            # Check if data looks like returns (negative values or values < 1)
            if col_clean.min() <= 0 or col_clean.max() < 1:
                data[col] = to_prices(data[col], base)

    # Check if series looks like returns data
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    # Clean data by filling NaN and replacing infinite values
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # Remove timezone information for consistency
    data = data.tz_localize(None)
    return data


def _prepare_returns(data, rf=0.0, nperiods=None):
    """
    Convert price data into returns and perform cleanup

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data (prices or returns)
    rf : float, default 0.0
        Risk-free rate
    nperiods : int, optional
        Number of periods for risk-free rate conversion

    Returns
    -------
    pd.Series or pd.DataFrame
        Cleaned returns data
    """
    # Try to get from cache first
    cache_key = _generate_cache_key(data, rf, nperiods)
    if cache_key:
        with _cache_lock:
            if cache_key in _PREPARE_RETURNS_CACHE:
                return _PREPARE_RETURNS_CACHE[cache_key].copy()

    data = data.copy()
    # Get calling function name for conditional processing
    function = inspect.stack()[1][3]

    # Process DataFrame columns
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            # Cache dropna operation to avoid repeated computation
            col_clean = data[col].dropna()
            # Check if data looks like prices (positive values > 1)
            if col_clean.min() >= 0 and col_clean.max() > 1:
                data[col] = data[col].pct_change()
    # Process Series data
    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change()

    # cleanup data - replace infinite values with NaN
    data = data.replace([_np.inf, -_np.inf], float("NaN"))

    # Fill NaN values with 0 and replace infinite values
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # Functions that don't need excess returns calculation
    unnecessary_function_calls = [
        "_prepare_benchmark",
        "cagr",
        "gain_to_pain_ratio",
        "rolling_volatility",
    ]

    # Calculate excess returns if rf > 0 and function needs it
    if function not in unnecessary_function_calls:
        if rf > 0:
            result = to_excess_returns(data, rf, nperiods)
            # Cache the result
            if cache_key:
                _clear_cache_if_full()
                with _cache_lock:
                    _PREPARE_RETURNS_CACHE[cache_key] = result.copy()
            return result

    # Remove timezone information for consistency
    data = data.tz_localize(None)

    # Cache the result
    if cache_key:
        _clear_cache_if_full()
        with _cache_lock:
            _PREPARE_RETURNS_CACHE[cache_key] = data.copy()

    return data


def download_returns(ticker, period="max", proxy=None):
    """
    Download returns data for a given ticker using yfinance

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    period : str or pd.DatetimeIndex, default "max"
        Time period for data download
    proxy : str, optional
        Proxy server for download

    Returns
    -------
    pd.Series
        Daily returns data for the ticker
    """
    # Set up parameters for yfinance download
    params = {
        "tickers": ticker,
        "auto_adjust": True,
        "multi_level_index": False,
        "progress": False,
    }

    # Handle different period types
    if isinstance(period, _pd.DatetimeIndex):
        params["start"] = period[0]
    else:
        params["period"] = period

    # Download data and calculate returns
    df = safe_yfinance_download(proxy=proxy, **params)["Close"].pct_change()  # type: ignore
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

    # Download benchmark data if ticker string provided
    if isinstance(benchmark, str):
        benchmark = download_returns(benchmark)

    # Extract first column if DataFrame provided
    elif isinstance(benchmark, _pd.DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()

    # Align benchmark with strategy period if needed
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

    # Remove timezone information
    benchmark = benchmark.tz_localize(None)

    # Prepare returns or return raw data
    if prepare_returns:
        return _prepare_returns(benchmark.dropna(), rf=rf)
    return benchmark.dropna()


def _round_to_closest(val, res, decimals=None):
    """
    Round value to closest resolution

    Parameters
    ----------
    val : float
        Value to round
    res : float
        Resolution to round to
    decimals : int, optional
        Number of decimal places

    Returns
    -------
    float
        Rounded value
    """
    # Auto-detect decimals from resolution if not provided
    if decimals is None and "." in str(res):
        decimals = len(str(res).split(".")[1])
    return round(round(val / res) * res, decimals)


def _file_stream():
    """
    Create and return a file stream object

    Returns
    -------
    io.BytesIO
        File stream object for handling bytes
    """
    return _io.BytesIO()


def _in_notebook(matplotlib_inline=False):
    """
    Identify current environment (notebook, terminal, etc.)

    Parameters
    ----------
    matplotlib_inline : bool, default False
        Whether to enable matplotlib inline mode

    Returns
    -------
    bool
        True if running in Jupyter notebook, False otherwise
    """
    try:
        # Get IPython shell class name
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            if matplotlib_inline:
                get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore
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
    """
    Count consecutive occurrences in data (like cumsum() with reset on zeroes)

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data to count consecutive occurrences

    Returns
    -------
    pd.Series or pd.DataFrame
        Data with consecutive counts
    """

    def _count(data):
        # Group by consecutive values and count occurrences
        return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

    # Handle DataFrame by processing each column
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def _score_str(val):
    """
    Format value string with appropriate sign (used in plots)

    Parameters
    ----------
    val : str or numeric
        Value to format

    Returns
    -------
    str
        Formatted string with + or - sign
    """
    # Add + sign for positive values, - is already included for negative
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

    # Iterate over weights and get returns for each ticker
    for ticker in ticker_weights.keys():
        if (returns is None) or (ticker not in returns.columns):
            # Download the returns for this ticker, e.g. GOOG
            ticker_returns = download_returns(ticker, period)
        else:
            ticker_returns = returns[ticker]

        portfolio[ticker] = ticker_returns

    # Create index members time-series
    index = _pd.DataFrame(portfolio).dropna()

    # Match dates to start from first non-zero date
    if match_dates:
        index = index[max(index.ne(0).idxmax()):]

    # Handle case with no rebalancing
    if rebalance is None:
        # Apply weights directly to returns
        for ticker, weight in ticker_weights.items():
            index[ticker] = weight * index[ticker]
        return index.sum(axis=1)

    last_day = index.index[-1]

    # Create rebalance markers
    rbdf = safe_resample(index, rebalance, "first")
    rbdf["break"] = rbdf.index.strftime("%s")

    # Add rebalance markers to index returns
    index = safe_concat([index, rbdf["break"]], axis=1)

    # Mark first day of each rebalance period
    index["first_day"] = _pd.isna(index["break"]) & ~_pd.isna(index["break"].shift(1))
    index.loc[index.index[0], "first_day"] = True

    # Apply weights on first day of each rebalance period
    for ticker, weight in ticker_weights.items():
        index[ticker] = _np.where(
            index["first_day"], weight * index[ticker], index[ticker]
        )

    # Clean up temporary columns
    index = index.drop(columns=["first_day"])

    # Remove rows where all values are NaN
    index = index.dropna(how="all")
    return index[index.index <= last_day].sum(axis=1)


def make_portfolio(returns, start_balance=1e5, mode="comp", round_to=None):
    """
    Calculate compounded value of portfolio from returns

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    start_balance : float, default 1e5
        Starting portfolio balance
    mode : str, default "comp"
        Calculation mode ("comp", "cumsum", "sum", or other)
    round_to : int, optional
        Number of decimal places to round to

    Returns
    -------
    pd.Series or pd.DataFrame
        Portfolio values over time
    """
    returns = _prepare_returns(returns)

    # Calculate portfolio values based on mode
    if mode.lower() in ["cumsum", "sum"]:
        # Simple cumulative sum approach
        p1 = start_balance + start_balance * returns.cumsum()
    elif mode.lower() in ["compsum", "comp"]:
        # Compounded returns approach
        p1 = to_prices(returns, start_balance)
    else:
        # Fixed amount every day approach
        comp_rev = (start_balance + start_balance * returns.shift(1)).fillna(
            start_balance
        ) * returns
        p1 = start_balance + comp_rev.cumsum()

    # Add day before with starting balance
    p0 = _pd.Series(data=start_balance, index=p1.index + _pd.Timedelta(days=-1))[:1]

    # Combine starting balance with portfolio values
    portfolio = safe_concat([p0, p1])

    # Handle DataFrame case
    if isinstance(returns, _pd.DataFrame):
        portfolio.iloc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    # Round if requested
    if round_to:
        portfolio = _np.round(portfolio, round_to)

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """
    Flatten multi-index dataframe using CSV conversion method

    Parameters
    ----------
    df : pd.DataFrame
        Multi-index dataframe to flatten
    set_index : str, optional
        Column to use as index after flattening

    Returns
    -------
    pd.DataFrame
        Flattened dataframe
    """
    # Use string buffer to convert to CSV and back to flatten structure
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    # Read back from CSV to get flattened structure
    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df

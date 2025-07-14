#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compatibility layer for pandas/numpy versions
Handles version differences and deprecated functionality

This module provides a unified interface for working with different versions of pandas
and numpy, ensuring that quantstats functions work consistently across various
dependency versions. It handles deprecated functionality and version-specific changes.
"""

import pandas as pd
import numpy as np
import warnings
from packaging import version
import yfinance as yf
from typing import Union, Optional, List, Callable

# Version detection - Parse version strings to enable version comparisons
PANDAS_VERSION = version.parse(pd.__version__)
NUMPY_VERSION = version.parse(np.__version__)

# Frequency alias mapping for pandas compatibility
# Starting from pandas 2.2.0, frequency aliases changed to be more explicit
# M -> ME (Month End), Q -> QE (Quarter End), A/Y -> YE (Year End)
FREQUENCY_ALIASES = {
    "M": "ME" if PANDAS_VERSION >= version.parse("2.2.0") else "M",
    "Q": "QE" if PANDAS_VERSION >= version.parse("2.2.0") else "Q",
    "A": "YE" if PANDAS_VERSION >= version.parse("2.2.0") else "A",
    "Y": "YE" if PANDAS_VERSION >= version.parse("2.2.0") else "Y",
}


def get_frequency_alias(freq: str) -> str:
    """
    Get the correct frequency alias for current pandas version.

    This function maps old frequency strings to their new equivalents in
    pandas 2.2.0+, ensuring backward compatibility across pandas versions.

    Parameters
    ----------
    freq : str
        The frequency string (e.g., 'M', 'Q', 'A', 'Y')

    Returns
    -------
    str
        The appropriate frequency alias for the current pandas version

    Examples
    --------
    >>> get_frequency_alias('M')  # Returns 'ME' in pandas 2.2.0+, 'M' in older versions
    >>> get_frequency_alias('D')  # Returns 'D' (unchanged)
    """
    # Look up the frequency in our mapping, return original if not found
    return FREQUENCY_ALIASES.get(freq, freq)


def safe_resample(data: Union[pd.Series, pd.DataFrame],
                  freq: str,
                  func_name: Optional[Union[str, Callable]] = None,
                  **kwargs):
    """
    Safe resample operation that works with all pandas versions.

    This function handles the resampling of time series data using the correct
    frequency aliases and aggregation methods that are compatible across
    different pandas versions.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The time series data to resample
    freq : str
        The frequency to resample to (e.g., 'M', 'Q', 'A', 'D')
    func_name : str or callable, optional
        The aggregation function to apply. Can be a string name like 'sum',
        'mean', 'std', etc., or a callable function
    **kwargs
        Additional arguments passed to the aggregation function

    Returns
    -------
    pd.Series or pd.DataFrame
        The resampled data with the specified frequency and aggregation

    Examples
    --------
    >>> safe_resample(data, 'M', 'sum')  # Monthly sum aggregation
    >>> safe_resample(data, 'Q', 'mean')  # Quarterly mean aggregation
    """
    # Convert frequency to the appropriate alias for current pandas version
    freq_alias = get_frequency_alias(freq)

    # Create the resampler object using the correct frequency
    resampler = data.resample(freq_alias)

    # If no aggregation function specified, return the resampler object
    if func_name is None:
        return resampler

    # Handle string function names with explicit method calls
    # This approach avoids deprecation warnings and ensures compatibility
    if isinstance(func_name, str):
        # Map common aggregation functions to their pandas methods
        if func_name == "sum":
            return resampler.sum(**kwargs)
        elif func_name == "mean":
            return resampler.mean(**kwargs)
        elif func_name == "std":
            return resampler.std(**kwargs)
        elif func_name == "count":
            return resampler.count(**kwargs)
        elif func_name == "min":
            return resampler.min(**kwargs)
        elif func_name == "max":
            return resampler.max(**kwargs)
        elif func_name == "first":
            return resampler.first(**kwargs)
        elif func_name == "last":
            return resampler.last(**kwargs)
        else:
            # Try to find the method on the resampler object
            if hasattr(resampler, func_name):
                return getattr(resampler, func_name)(**kwargs)
            else:
                # Fallback to apply for custom string functions
                return resampler.apply(func_name, **kwargs)
    else:
        # For callable functions, use apply method
        return resampler.apply(func_name, **kwargs)


def safe_concat(objs: List[Union[pd.Series, pd.DataFrame]],
                axis: int = 0,
                ignore_index: bool = False,
                sort: bool = False,
                **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """
    Safe concatenation that handles pandas version differences.

    This function provides a wrapper around pd.concat that handles changes
    in parameter support across different pandas versions, particularly
    the 'sort' parameter which was added in pandas 1.0.0.

    Parameters
    ----------
    objs : list of pd.Series or pd.DataFrame
        Objects to concatenate along the specified axis
    axis : int, default 0
        Axis to concatenate along. 0 for rows, 1 for columns
    ignore_index : bool, default False
        Whether to ignore the index and create a new default integer index
    sort : bool, default False
        Whether to sort the result. Only supported in pandas 1.0.0+
    **kwargs
        Additional arguments passed to pd.concat

    Returns
    -------
    pd.Series or pd.DataFrame
        The concatenated result

    Examples
    --------
    >>> safe_concat([df1, df2])  # Concatenate along rows
    >>> safe_concat([df1, df2], axis=1)  # Concatenate along columns
    """
    # Handle sort parameter for older pandas versions
    # The sort parameter was introduced in pandas 1.0.0
    if PANDAS_VERSION < version.parse("1.0.0"):
        # Remove sort parameter if it exists in kwargs for compatibility
        kwargs.pop("sort", None)
    else:
        # Set sort parameter for newer pandas versions
        kwargs["sort"] = sort

    # Perform the concatenation with appropriate parameters
    return pd.concat(objs, axis=axis, ignore_index=ignore_index, **kwargs)  # type: ignore[arg-type]


def safe_append(df: pd.DataFrame,
                other: Union[pd.DataFrame, pd.Series],
                ignore_index: bool = False,
                sort: bool = False) -> pd.DataFrame:
    """
    Safe append operation that works with all pandas versions.

    DataFrame.append() was deprecated in pandas 1.4.0 and removed in 2.0.0.
    This function provides a unified interface that uses the appropriate
    method based on the pandas version.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to append to (base DataFrame)
    other : pd.DataFrame or pd.Series
        The data to append to the base DataFrame
    ignore_index : bool, default False
        Whether to ignore the index and create a new default integer index
    sort : bool, default False
        Whether to sort the result by columns

    Returns
    -------
    pd.DataFrame
        The result of the append operation

    Examples
    --------
    >>> safe_append(df, new_row)  # Append a new row
    >>> safe_append(df, other_df, ignore_index=True)  # Append and reset index
    """
    # Check pandas version to determine which method to use
    if PANDAS_VERSION >= version.parse("1.4.0"):
        # Use concat for newer pandas versions (recommended approach)
        result = safe_concat([df, other], ignore_index=ignore_index, sort=sort)
        # Ensure we return a DataFrame (concat can return Series in some cases)
        if isinstance(result, pd.DataFrame):
            return result
        else:
            # Convert Series to DataFrame - handle the case where result is a Series
            if isinstance(result, pd.Series):
                return pd.DataFrame([result])
            else:
                return pd.DataFrame(result)
    else:
        # Use the deprecated append method for older pandas versions
        return df.append(other, ignore_index=ignore_index, sort=sort)


def safe_frequency_conversion(data: Union[pd.Series, pd.DataFrame],
                              freq: str) -> Union[pd.Series, pd.DataFrame]:
    """
    Safe frequency conversion for time series data.

    This function converts time series data to a specified frequency using
    the most appropriate method available in the current pandas version.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time series data with a datetime index
    freq : str
        Target frequency (e.g., 'D', 'M', 'Q', 'A')

    Returns
    -------
    pd.Series or pd.DataFrame
        Data with converted frequency

    Examples
    --------
    >>> safe_frequency_conversion(data, 'M')  # Convert to monthly frequency
    >>> safe_frequency_conversion(data, 'D')  # Convert to daily frequency
    """
    # Get the appropriate frequency alias for current pandas version
    freq_alias = get_frequency_alias(freq)

    # Handle different methods for frequency conversion
    if hasattr(data, "asfreq"):
        # Use asfreq if available (most direct method)
        return data.asfreq(freq_alias)
    else:
        # Fallback to resampling with 'last' aggregation
        # This preserves the last value in each period
        return safe_resample(data, freq_alias, "last")


def handle_pandas_warnings():
    """
    Context manager to handle pandas warnings appropriately.

    This function returns a context manager that can be used to suppress
    or handle pandas warnings in a controlled manner. Useful for managing
    deprecation warnings when working with multiple pandas versions.

    Returns
    -------
    warnings.catch_warnings
        A context manager for handling warnings

    Examples
    --------
    >>> with handle_pandas_warnings():
    ...     # Code that might generate pandas warnings
    ...     pass
    """
    # Return the warnings context manager for flexible warning handling
    return warnings.catch_warnings()


# Pandas accessor compatibility functions
def get_datetime_accessor(series: pd.Series):
    """
    Get datetime accessor for pandas Series.

    This function provides a consistent interface for accessing datetime
    properties of a pandas Series across different versions.

    Parameters
    ----------
    series : pd.Series
        The series with datetime data to get the accessor for

    Returns
    -------
    pd.Series.dt
        The datetime accessor for the series

    Examples
    --------
    >>> dt_accessor = get_datetime_accessor(date_series)
    >>> dt_accessor.year  # Access year component
    >>> dt_accessor.month  # Access month component
    """
    # Return the datetime accessor - consistent across pandas versions
    return series.dt


def get_string_accessor(series: pd.Series):
    """
    Get string accessor for pandas Series.

    This function provides a consistent interface for accessing string
    methods of a pandas Series across different versions.

    Parameters
    ----------
    series : pd.Series
        The series with string data to get the accessor for

    Returns
    -------
    pd.Series.str
        The string accessor for the series

    Examples
    --------
    >>> str_accessor = get_string_accessor(string_series)
    >>> str_accessor.lower()  # Convert to lowercase
    >>> str_accessor.contains('pattern')  # Check for pattern
    """
    # Return the string accessor - consistent across pandas versions
    return series.str


def safe_yfinance_download(tickers: Union[str, List[str]],
                           proxy: Optional[str] = None,
                           **kwargs) -> pd.DataFrame:
    """
    Safe yfinance download that handles proxy configuration properly.

    This function provides a wrapper around yfinance.download that handles
    proxy configuration differences between yfinance versions. It ensures
    compatibility with both old and new yfinance proxy configuration methods.

    Parameters
    ----------
    tickers : str or list
        Ticker symbols to download data for. Can be a single ticker string
        or a list of ticker symbols
    proxy : str, optional
        Proxy configuration string (e.g., 'http://proxy.server:port')
        Handled automatically based on yfinance version
    **kwargs
        Additional arguments passed to yfinance.download such as:
        - start: Start date for data download
        - end: End date for data download
        - period: Period to download (e.g., '1y', '6mo')
        - interval: Data interval (e.g., '1d', '1h')

    Returns
    -------
    pd.DataFrame
        Downloaded financial data with columns like Open, High, Low, Close, Volume

    Examples
    --------
    >>> data = safe_yfinance_download('AAPL', start='2020-01-01', end='2021-01-01')
    >>> data = safe_yfinance_download(['AAPL', 'MSFT'], period='1y')
    """
    # Handle proxy configuration based on yfinance version
    if proxy is not None:
        # Check if the new configuration method exists in yfinance
        if hasattr(yf, "set_config"):
            # New method: use set_config for global proxy configuration
            # This approach is preferred in newer yfinance versions
            yf.set_config(proxy=proxy)
            # Remove proxy from kwargs to avoid duplicate parameter error
            kwargs.pop("proxy", None)
        else:
            # Old method: pass proxy directly to download function
            # This is for backward compatibility with older yfinance versions
            kwargs["proxy"] = proxy

    # Suppress yfinance warnings about deprecation and future changes
    # This keeps the output clean while maintaining functionality
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
        # Download the data using yfinance with all provided parameters
        result = yf.download(tickers, **kwargs)

        # Handle case where yfinance returns None (network issues, invalid ticker, etc.)
        if result is None:
            # Return empty DataFrame with standard yfinance columns
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

        return result

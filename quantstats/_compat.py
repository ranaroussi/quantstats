#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compatibility layer for pandas/numpy versions
Handles version differences and deprecated functionality
"""

import pandas as pd
import numpy as np
import warnings
from packaging import version
import yfinance as yf

# Version detection
PANDAS_VERSION = version.parse(pd.__version__)
NUMPY_VERSION = version.parse(np.__version__)

# Frequency alias mapping for pandas compatibility
FREQUENCY_ALIASES = {
    "M": "ME" if PANDAS_VERSION >= version.parse("2.2.0") else "M",
    "Q": "QE" if PANDAS_VERSION >= version.parse("2.2.0") else "Q",
    "A": "YE" if PANDAS_VERSION >= version.parse("2.2.0") else "A",
    "Y": "YE" if PANDAS_VERSION >= version.parse("2.2.0") else "Y",
}

def get_frequency_alias(freq):
    """
    Get the correct frequency alias for current pandas version
    
    Parameters
    ----------
    freq : str
        The frequency string (e.g., 'M', 'Q', 'A', 'Y')
        
    Returns
    -------
    str
        The appropriate frequency alias for the current pandas version
    """
    return FREQUENCY_ALIASES.get(freq, freq)

def safe_resample(data, freq, func_name=None, **kwargs):
    """
    Safe resample operation that works with all pandas versions
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The data to resample
    freq : str
        The frequency to resample to
    func_name : str or callable, optional
        The aggregation function to apply
    **kwargs
        Additional arguments passed to the aggregation function
        
    Returns
    -------
    pd.Series or pd.DataFrame
        The resampled data
    """
    freq_alias = get_frequency_alias(freq)
    resampler = data.resample(freq_alias)
    
    if func_name is None:
        return resampler
    
    # Use proper aggregation methods instead of numpy functions
    if isinstance(func_name, str):
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
            # For string function names, try to get the method
            if hasattr(resampler, func_name):
                return getattr(resampler, func_name)(**kwargs)
            else:
                # Fallback to apply
                return resampler.apply(func_name, **kwargs)
    else:
        # For callable functions, use apply
        return resampler.apply(func_name, **kwargs)

def safe_concat(objs, axis=0, ignore_index=False, sort=False, **kwargs):
    """
    Safe concatenation that handles pandas version differences
    
    Parameters
    ----------
    objs : list of pd.Series or pd.DataFrame
        Objects to concatenate
    axis : int, default 0
        Axis to concatenate along
    ignore_index : bool, default False
        Whether to ignore the index
    sort : bool, default False
        Whether to sort the result
    **kwargs
        Additional arguments passed to pd.concat
        
    Returns
    -------
    pd.Series or pd.DataFrame
        The concatenated result
    """
    # Handle sort parameter for older pandas versions
    if PANDAS_VERSION < version.parse("1.0.0"):
        kwargs.pop('sort', None)
    else:
        kwargs['sort'] = sort
    
    return pd.concat(objs, axis=axis, ignore_index=ignore_index, **kwargs)

def safe_append(df, other, ignore_index=False, sort=False):
    """
    Safe append operation that works with all pandas versions
    DataFrame.append() was deprecated in pandas 1.4.0
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to append to
    other : pd.DataFrame or pd.Series
        The data to append
    ignore_index : bool, default False
        Whether to ignore the index
    sort : bool, default False
        Whether to sort the result
        
    Returns
    -------
    pd.DataFrame
        The result of the append operation
    """
    if PANDAS_VERSION >= version.parse("1.4.0"):
        # Use concat for newer pandas versions
        return safe_concat([df, other], ignore_index=ignore_index, sort=sort)
    else:
        # Use append for older pandas versions
        return df.append(other, ignore_index=ignore_index, sort=sort)

def safe_frequency_conversion(data, freq):
    """
    Safe frequency conversion for time series data
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time series data
    freq : str
        Target frequency
        
    Returns
    -------
    pd.Series or pd.DataFrame
        Data with converted frequency
    """
    freq_alias = get_frequency_alias(freq)
    
    # Handle different methods for frequency conversion
    if hasattr(data, 'asfreq'):
        return data.asfreq(freq_alias)
    else:
        # Fallback to resampling
        return safe_resample(data, freq_alias, 'last')

def handle_pandas_warnings():
    """
    Context manager to handle pandas warnings appropriately
    """
    return warnings.catch_warnings()

# Pandas accessor compatibility
def get_datetime_accessor(series):
    """
    Get datetime accessor for pandas Series
    
    Parameters
    ----------
    series : pd.Series
        The series to get the accessor for
        
    Returns
    -------
    pd.Series.dt
        The datetime accessor
    """
    return series.dt

def get_string_accessor(series):
    """
    Get string accessor for pandas Series
    
    Parameters
    ----------
    series : pd.Series
        The series to get the accessor for
        
    Returns
    -------
    pd.Series.str
        The string accessor
    """
    return series.str

def safe_yfinance_download(tickers, proxy=None, **kwargs):
    """
    Safe yfinance download that handles proxy configuration properly
    
    Parameters
    ----------
    tickers : str or list
        Ticker symbols to download
    proxy : str, optional
        Proxy configuration (handled for compatibility)
    **kwargs
        Additional arguments passed to yfinance.download
        
    Returns
    -------
    pd.DataFrame
        Downloaded data
    """
    # Handle proxy configuration based on yfinance version
    if proxy is not None:
        # Check if the new configuration method exists
        if hasattr(yf, 'set_config'):
            # New method: use set_config for proxy configuration
            yf.set_config(proxy=proxy)
            # Don't pass proxy to download function
            kwargs.pop('proxy', None)
        else:
            # Old method: pass proxy directly to download
            kwargs['proxy'] = proxy
    
    # Suppress yfinance warnings about deprecation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
        return yf.download(tickers, **kwargs)
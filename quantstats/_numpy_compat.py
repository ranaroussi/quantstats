#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numpy compatibility layer
Handles numpy version differences and deprecated functionality

This module provides a unified interface for working with different versions of numpy,
ensuring that quantstats functions work consistently across various numpy versions.
It handles deprecated functionality, version-specific changes, and provides fallback
mechanisms for functions that may not exist in older versions.
"""

import numpy as np
import warnings
from packaging import version
from typing import Union, Optional, Any

# Version detection - Parse numpy version string to enable version comparisons
NUMPY_VERSION = version.parse(np.__version__)

# Handle deprecated numpy functions
# In numpy 1.25.0+, np.product was deprecated in favor of np.prod
if NUMPY_VERSION >= version.parse("1.25.0"):
    # Use np.prod instead of deprecated np.product for newer versions
    product = np.prod
else:
    # Use np.product if available, otherwise fallback to np.prod for older versions
    product = getattr(np, "product", np.prod)


def safe_numpy_operation(data, operation: str):
    """
    Safe numpy operations with deprecation handling.

    This function provides a unified interface for numpy operations that may
    have been deprecated or changed behavior across different numpy versions.
    It handles the transition from deprecated functions to their replacements.

    Parameters
    ----------
    data : array_like
        Input data to perform the operation on. Can be any array-like structure
        that numpy can process (list, tuple, numpy array, etc.)
    operation : str
        The numpy operation to perform. Supported operations include 'product',
        'prod', and any other numpy function name that exists in the current version

    Returns
    -------
    ndarray
        Result of the numpy operation applied to the input data

    Examples
    --------
    >>> safe_numpy_operation([1, 2, 3, 4], 'product')  # Returns 24
    >>> safe_numpy_operation([1, 2, 3, 4], 'sum')      # Returns 10
    """
    # Handle the deprecated 'product' operation specifically
    if operation == "product":
        # Use our version-aware product function
        return product(data)
    elif operation == "prod":
        # Use np.prod directly for 'prod' operation
        return np.prod(data)
    else:
        # For all other operations, dynamically get the function from numpy
        # This allows for flexible operation support across numpy versions
        return getattr(np, operation)(data)


def safe_array_function(func_name: str, *args, **kwargs) -> Any:
    """
    Safe wrapper for numpy array functions.

    This function provides a safe way to call numpy functions that may not exist
    in all numpy versions or have been deprecated. It handles version-specific
    function availability and provides meaningful error messages.

    Parameters
    ----------
    func_name : str
        Name of the numpy function to call (e.g., 'mean', 'std', 'product')
    *args
        Positional arguments to pass to the numpy function
    **kwargs
        Keyword arguments to pass to the numpy function

    Returns
    -------
    Any
        Result of the numpy function call. Return type depends on the specific
        function being called and the input data.

    Raises
    ------
    AttributeError
        If the requested function doesn't exist in the current numpy version

    Examples
    --------
    >>> safe_array_function('mean', [1, 2, 3, 4])      # Returns 2.5
    >>> safe_array_function('product', [1, 2, 3, 4])   # Returns 24
    """
    # Handle deprecated functions with special cases
    if func_name == "product":
        # Use our version-aware product function
        return product(*args, **kwargs)

    # Handle functions that might not exist in older numpy versions
    if hasattr(np, func_name):
        # Function exists in current numpy version, call it normally
        return getattr(np, func_name)(*args, **kwargs)
    else:
        # Function doesn't exist in current numpy version, raise informative error
        raise AttributeError(
            f"numpy has no attribute '{func_name}' in version {NUMPY_VERSION}"
        )


def handle_numpy_warnings():
    """
    Context manager to handle numpy warnings appropriately.

    This function returns a context manager that can be used to suppress
    or handle numpy warnings in a controlled manner. Useful for managing
    deprecation warnings and other numpy-specific warnings when working
    with multiple numpy versions.

    Returns
    -------
    warnings.catch_warnings
        A context manager for handling numpy warnings

    Examples
    --------
    >>> with handle_numpy_warnings():
    ...     # Code that might generate numpy warnings
    ...     result = np.some_deprecated_function()
    """
    # Return the warnings context manager for flexible warning handling
    return warnings.catch_warnings()


def safe_percentile(data, percentile: Union[float, list], **kwargs):
    """
    Safe percentile calculation that handles numpy version differences.

    This function provides a wrapper around np.percentile that handles
    parameter changes across numpy versions. The 'method' parameter was
    introduced in numpy 1.22.0, so this function ensures compatibility
    with older versions by removing unsupported parameters.

    Parameters
    ----------
    data : array_like
        Input data to calculate percentiles from. Can be any array-like
        structure that numpy can process
    percentile : float or array_like
        Percentile(s) to compute. Values should be between 0 and 100.
        Can be a single value or an array of values
    **kwargs
        Additional arguments passed to np.percentile. The 'method' parameter
        is handled automatically based on numpy version

    Returns
    -------
    float or ndarray
        The computed percentile(s). Returns float for single percentile,
        ndarray for multiple percentiles

    Examples
    --------
    >>> safe_percentile([1, 2, 3, 4, 5], 50)    # Returns 3.0 (median)
    >>> safe_percentile([1, 2, 3, 4, 5], [25, 75])  # Returns [2.0, 4.0]
    """
    # Handle method parameter for newer numpy versions
    if NUMPY_VERSION >= version.parse("1.22.0"):
        # Numpy 1.22.0+ supports the 'method' parameter for percentile calculation
        return np.percentile(data, percentile, **kwargs)
    else:
        # Remove method parameter for older versions to avoid TypeError
        kwargs.pop("method", None)
        return np.percentile(data, percentile, **kwargs)


def safe_nanpercentile(data, percentile: Union[float, list], **kwargs):
    """
    Safe nanpercentile calculation that handles numpy version differences.

    This function provides a wrapper around np.nanpercentile that handles
    parameter changes across numpy versions. Similar to safe_percentile,
    but ignores NaN values in the calculation. The 'method' parameter was
    introduced in numpy 1.22.0.

    Parameters
    ----------
    data : array_like
        Input data to calculate percentiles from. NaN values are ignored.
        Can be any array-like structure that numpy can process
    percentile : float or array_like
        Percentile(s) to compute. Values should be between 0 and 100.
        Can be a single value or an array of values
    **kwargs
        Additional arguments passed to np.nanpercentile. The 'method' parameter
        is handled automatically based on numpy version

    Returns
    -------
    float or ndarray
        The computed percentile(s) ignoring NaN values. Returns float for
        single percentile, ndarray for multiple percentiles

    Examples
    --------
    >>> safe_nanpercentile([1, 2, np.nan, 4, 5], 50)    # Returns 3.0
    >>> safe_nanpercentile([1, np.nan, 3, 4, 5], [25, 75])  # Returns [2.0, 4.5]
    """
    # Handle method parameter for newer numpy versions
    if NUMPY_VERSION >= version.parse("1.22.0"):
        # Numpy 1.22.0+ supports the 'method' parameter for nanpercentile calculation
        return np.nanpercentile(data, percentile, **kwargs)
    else:
        # Remove method parameter for older versions to avoid TypeError
        kwargs.pop("method", None)
        return np.nanpercentile(data, percentile, **kwargs)


def safe_quantile(data, quantile: Union[float, list], **kwargs):
    """
    Safe quantile calculation that handles numpy version differences.

    This function provides a wrapper around np.quantile that handles
    parameter changes across numpy versions. Quantiles are similar to
    percentiles but use values between 0 and 1 instead of 0 and 100.
    The 'method' parameter was introduced in numpy 1.22.0.

    Parameters
    ----------
    data : array_like
        Input data to calculate quantiles from. Can be any array-like
        structure that numpy can process
    quantile : float or array_like
        Quantile(s) to compute. Values should be between 0 and 1.
        Can be a single value or an array of values
    **kwargs
        Additional arguments passed to np.quantile. The 'method' parameter
        is handled automatically based on numpy version

    Returns
    -------
    float or ndarray
        The computed quantile(s). Returns float for single quantile,
        ndarray for multiple quantiles

    Examples
    --------
    >>> safe_quantile([1, 2, 3, 4, 5], 0.5)    # Returns 3.0 (median)
    >>> safe_quantile([1, 2, 3, 4, 5], [0.25, 0.75])  # Returns [2.0, 4.0]
    """
    # Handle method parameter for newer numpy versions
    if NUMPY_VERSION >= version.parse("1.22.0"):
        # Numpy 1.22.0+ supports the 'method' parameter for quantile calculation
        return np.quantile(data, quantile, **kwargs)
    else:
        # Remove method parameter for older versions to avoid TypeError
        kwargs.pop("method", None)
        return np.quantile(data, quantile, **kwargs)


def safe_random_seed(seed: Optional[int]):
    """
    Safe random seed setting for numpy.

    This function provides a unified interface for setting random seeds
    across different numpy versions. Numpy 1.17.0 introduced a new random
    number generator system, so this function handles both the old and new
    approaches to ensure consistent random number generation.

    Parameters
    ----------
    seed : int or None
        Random seed value to set. If None, the random state is not modified.
        Setting the same seed ensures reproducible random number sequences

    Examples
    --------
    >>> safe_random_seed(42)  # Sets seed for reproducible results
    >>> safe_random_seed(None)  # No seed set, random behavior continues
    """
    # Check if seed is provided before attempting to set it
    if seed is not None:
        if NUMPY_VERSION >= version.parse("1.17.0"):
            # Use the new random number generator for numpy 1.17.0+
            # This creates a new Generator instance with the specified seed
            np.random.default_rng(seed)
        else:
            # Use the legacy random seed method for older numpy versions
            # This sets the global random state
            np.random.seed(seed)


def safe_datetime64_unit(dt, unit: str):
    """
    Safe datetime64 unit conversion.

    This function provides a safe way to convert numpy datetime64 objects
    to different time units. It handles differences in string formatting
    between numpy versions to ensure consistent behavior.

    Parameters
    ----------
    dt : np.datetime64
        Input datetime object to convert
    unit : str
        Target time unit (e.g., 'D' for days, 'H' for hours, 'M' for minutes,
        'S' for seconds, 'ms' for milliseconds)

    Returns
    -------
    np.datetime64
        Converted datetime object with the specified unit

    Examples
    --------
    >>> dt = np.datetime64('2023-01-01T12:00:00')
    >>> safe_datetime64_unit(dt, 'D')  # Convert to day precision
    >>> safe_datetime64_unit(dt, 'H')  # Convert to hour precision
    """
    # Handle string formatting differences between numpy versions
    if NUMPY_VERSION >= version.parse("1.21.0"):
        # Use f-string formatting for newer numpy versions
        return dt.astype(f"datetime64[{unit}]")
    else:
        # Use older string formatting method for compatibility
        return dt.astype("datetime64[{0}]".format(unit))

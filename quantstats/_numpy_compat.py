#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numpy compatibility layer
Handles numpy version differences and deprecated functionality
"""

import numpy as np
import warnings
from packaging import version

# Version detection
NUMPY_VERSION = version.parse(np.__version__)

# Handle deprecated numpy functions
if NUMPY_VERSION >= version.parse("1.25.0"):
    # Use np.prod instead of deprecated np.product
    product = np.prod
else:
    product = getattr(np, "product", np.prod)


def safe_numpy_operation(data, operation):
    """
    Safe numpy operations with deprecation handling

    Parameters
    ----------
    data : array_like
        Input data
    operation : str
        The numpy operation to perform

    Returns
    -------
    ndarray
        Result of the operation
    """
    if operation == "product":
        return product(data)
    elif operation == "prod":
        return np.prod(data)
    else:
        return getattr(np, operation)(data)


def safe_array_function(func_name, *args, **kwargs):
    """
    Safe wrapper for numpy array functions

    Parameters
    ----------
    func_name : str
        Name of the numpy function
    *args
        Arguments to pass to the function
    **kwargs
        Keyword arguments to pass to the function

    Returns
    -------
    Any
        Result of the function call
    """
    # Handle deprecated functions
    if func_name == "product":
        return product(*args, **kwargs)

    # Handle functions that might not exist in older versions
    if hasattr(np, func_name):
        return getattr(np, func_name)(*args, **kwargs)
    else:
        raise AttributeError(
            f"numpy has no attribute '{func_name}' in version {NUMPY_VERSION}"
        )


def handle_numpy_warnings():
    """
    Context manager to handle numpy warnings appropriately
    """
    return warnings.catch_warnings()


def safe_percentile(data, percentile, **kwargs):
    """
    Safe percentile calculation that handles numpy version differences

    Parameters
    ----------
    data : array_like
        Input data
    percentile : float or array_like
        Percentile(s) to compute
    **kwargs
        Additional arguments

    Returns
    -------
    float or ndarray
        The percentile(s)
    """
    # Handle method parameter for newer numpy versions
    if NUMPY_VERSION >= version.parse("1.22.0"):
        return np.percentile(data, percentile, **kwargs)
    else:
        # Remove method parameter for older versions
        kwargs.pop("method", None)
        return np.percentile(data, percentile, **kwargs)


def safe_nanpercentile(data, percentile, **kwargs):
    """
    Safe nanpercentile calculation that handles numpy version differences

    Parameters
    ----------
    data : array_like
        Input data
    percentile : float or array_like
        Percentile(s) to compute
    **kwargs
        Additional arguments

    Returns
    -------
    float or ndarray
        The percentile(s)
    """
    # Handle method parameter for newer numpy versions
    if NUMPY_VERSION >= version.parse("1.22.0"):
        return np.nanpercentile(data, percentile, **kwargs)
    else:
        # Remove method parameter for older versions
        kwargs.pop("method", None)
        return np.nanpercentile(data, percentile, **kwargs)


def safe_quantile(data, quantile, **kwargs):
    """
    Safe quantile calculation that handles numpy version differences

    Parameters
    ----------
    data : array_like
        Input data
    quantile : float or array_like
        Quantile(s) to compute
    **kwargs
        Additional arguments

    Returns
    -------
    float or ndarray
        The quantile(s)
    """
    # Handle method parameter for newer numpy versions
    if NUMPY_VERSION >= version.parse("1.22.0"):
        return np.quantile(data, quantile, **kwargs)
    else:
        # Remove method parameter for older versions
        kwargs.pop("method", None)
        return np.quantile(data, quantile, **kwargs)


def safe_random_seed(seed):
    """
    Safe random seed setting for numpy

    Parameters
    ----------
    seed : int or None
        Random seed value
    """
    if NUMPY_VERSION >= version.parse("1.17.0"):
        # Use the new random number generator
        np.random.default_rng(seed)
    else:
        # Use the legacy random seed
        np.random.seed(seed)


def safe_datetime64_unit(dt, unit):
    """
    Safe datetime64 unit conversion

    Parameters
    ----------
    dt : np.datetime64
        Input datetime
    unit : str
        Target unit

    Returns
    -------
    np.datetime64
        Converted datetime
    """
    if NUMPY_VERSION >= version.parse("1.21.0"):
        return dt.astype(f"datetime64[{unit}]")
    else:
        return dt.astype("datetime64[{0}]".format(unit))

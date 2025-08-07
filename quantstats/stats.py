#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Portfolio Statistics Module

This module provides comprehensive statistical analysis functions for portfolio
performance evaluation, risk assessment, and benchmarking. It includes functions
for calculating various return metrics, risk ratios, drawdown analysis, and
comparison with benchmarks.

The module is designed to work with pandas Series and DataFrames containing
return data, price data, or performance metrics.
"""

from warnings import warn
import pandas as _pd
import numpy as _np
from math import ceil as _ceil, sqrt as _sqrt
from scipy.stats import norm as _norm, linregress as _linregress

from . import utils as _utils
from ._compat import safe_concat
from .utils import validate_input


# ======== STATS ========


def pct_rank(prices, window=60):
    """
    Calculate the percentile rank of prices over a rolling window.

    This function computes the percentile rank (0-100) of each price point
    within a rolling window, useful for identifying relative position of
    current prices compared to recent history.

    Args:
        prices (pd.Series): Series of price data
        window (int): Rolling window size for rank calculation (default: 60)

    Returns:
        pd.Series: Percentile ranks (0-100 scale)

    Example:
        >>> prices = pd.Series([100, 105, 110, 95, 120])
        >>> ranks = pct_rank(prices, window=3)
        >>> print(ranks)
    """
    # Create rolling window shifts and transpose for ranking
    rank = _utils.multi_shift(prices, window).T.rank(pct=True).T
    # Extract first column and convert to percentage scale
    return rank.iloc[:, 0] * 100.0


def compsum(returns):
    """
    Calculate rolling compounded returns (cumulative product).

    This function computes the cumulative compounded returns by adding 1
    to each return, taking the cumulative product, and subtracting 1.

    Args:
        returns (pd.Series): Series of returns

    Returns:
        pd.Series: Cumulative compounded returns

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> cumulative = compsum(returns)
        >>> print(cumulative)
    """
    # Add 1 to convert returns to growth factors, then cumulative product
    return returns.add(1).cumprod(axis=0) - 1


def comp(returns):
    """
    Calculate total compounded returns (final cumulative return).

    This function computes the total compounded return over the entire period
    by converting returns to growth factors and taking their product.

    Args:
        returns (pd.Series): Series of returns

    Returns:
        float: Total compounded return

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> total_return = comp(returns)
        >>> print(total_return)
    """
    # Convert returns to growth factors, take product, subtract 1
    return returns.add(1).prod(axis=0) - 1


def distribution(returns, compounded=True, prepare_returns=True):
    """
    Analyze return distributions across different time periods.

    This function calculates return distributions (including outliers) for
    daily, weekly, monthly, quarterly, and yearly periods. It identifies
    outliers using the IQR method (1.5 * IQR beyond Q1/Q3).

    Args:
        returns (pd.Series): Return series to analyze
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        dict: Dictionary containing distribution data for each period

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01],
        ...                    index=pd.date_range('2023-01-01', periods=3))
        >>> dist = distribution(returns)
        >>> print(dist['Daily']['values'])
    """
    def get_outliers(data):
        """
        Identify outliers using the IQR method.

        Uses 1.5 * IQR rule: values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
        are considered outliers.
        """
        # https://datascience.stackexchange.com/a/57199
        Q1 = data.quantile(0.25)  # First quartile
        Q3 = data.quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range

        # Create filter for non-outlier values
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)

        return {
            "values": data.loc[filtered].tolist(),
            "outliers": data.loc[~filtered].tolist(),
        }

    # Handle DataFrame input by selecting appropriate column
    if isinstance(returns, _pd.DataFrame):
        warn(
            "Pandas DataFrame was passed (Series expected). "
            "Only first column will be used."
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    # Choose aggregation function based on compounded parameter
    apply_fnc = comp if compounded else _np.sum
    daily = returns.dropna()

    # Prepare returns if requested
    if prepare_returns:
        daily = _utils._prepare_returns(daily)

    # Calculate distributions for different time periods
    return {
        "Daily": get_outliers(daily),
        "Weekly": get_outliers(daily.resample("W-MON").apply(apply_fnc)),
        "Monthly": get_outliers(daily.resample("ME").apply(apply_fnc)),
        "Quarterly": get_outliers(daily.resample("QE").apply(apply_fnc)),
        "Yearly": get_outliers(daily.resample("YE").apply(apply_fnc)),
    }


def expected_return(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the expected return (geometric mean) for a given period.

    This function computes the geometric holding period return, which represents
    the expected return per period based on historical data. It's calculated
    as the nth root of the product of (1 + returns) minus 1.

    Args:
        returns (pd.Series): Return series
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Expected return per period

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> expected = expected_return(returns)
        >>> print(f"Expected return: {expected:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns if period specified
    returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # Calculate geometric mean: (product of (1 + returns))^(1/n) - 1
    return _np.prod(1 + returns, axis=0) ** (1 / len(returns)) - 1


def geometric_mean(returns, aggregate=None, compounded=True):
    """
    Calculate geometric mean of returns.

    This is a shorthand function for expected_return() with the same parameters.

    Args:
        returns (pd.Series): Return series
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)

    Returns:
        float: Geometric mean of returns
    """
    return expected_return(returns, aggregate, compounded)


def ghpr(returns, aggregate=None, compounded=True):
    """
    Calculate Geometric Holding Period Return.

    This is a shorthand function for expected_return() with the same parameters.
    GHPR represents the average rate of return per period.

    Args:
        returns (pd.Series): Return series
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)

    Returns:
        float: Geometric holding period return
    """
    return expected_return(returns, aggregate, compounded)


def outliers(returns, quantile=0.95):
    """
    Identify and return outlier returns above a specified quantile.

    This function filters returns to show only those above the specified
    quantile threshold, helping identify extreme positive performance periods.

    Args:
        returns (pd.Series): Return series to analyze
        quantile (float): Quantile threshold (default: 0.95 for 95th percentile)

    Returns:
        pd.Series: Returns above the quantile threshold

    Example:
        >>> returns = pd.Series([0.01, 0.02, 0.05, -0.01, 0.10])
        >>> outlier_returns = outliers(returns, quantile=0.90)
        >>> print(outlier_returns)
    """
    # Filter returns above the specified quantile and remove NaN values
    return returns[returns > returns.quantile(quantile)].dropna(how="all")


def remove_outliers(returns, quantile=0.95):
    """
    Remove outlier returns above a specified quantile.

    This function filters out extreme returns above the quantile threshold,
    useful for robust statistical analysis by removing extreme values.

    Args:
        returns (pd.Series): Return series to filter
        quantile (float): Quantile threshold (default: 0.95 for 95th percentile)

    Returns:
        pd.Series: Returns below the quantile threshold

    Example:
        >>> returns = pd.Series([0.01, 0.02, 0.05, -0.01, 0.10])
        >>> filtered = remove_outliers(returns, quantile=0.90)
        >>> print(filtered)
    """
    # Keep only returns below the specified quantile threshold
    return returns[returns < returns.quantile(quantile)]


def best(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Find the best (highest) return for a given period.

    This function identifies the maximum return over the specified aggregation
    period, helping identify the best performing period in the dataset.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Best (maximum) return for the period

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> best_return = best(returns)
        >>> print(f"Best return: {best_return:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns and find maximum
    return _utils.aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Find the worst (lowest) return for a given period.

    This function identifies the minimum return over the specified aggregation
    period, helping identify the worst performing period in the dataset.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Worst (minimum) return for the period

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> worst_return = worst(returns)
        >>> print(f"Worst return: {worst_return:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns and find minimum
    return _utils.aggregate_returns(returns, aggregate, compounded).min()


def consecutive_wins(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the maximum number of consecutive winning periods.

    This function identifies the longest streak of positive returns, which
    helps assess the consistency of positive performance.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        int: Maximum number of consecutive winning periods

    Example:
        >>> returns = pd.Series([0.01, 0.02, 0.03, -0.01, 0.02])
        >>> max_wins = consecutive_wins(returns)
        >>> print(f"Max consecutive wins: {max_wins}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns and convert to boolean (positive = True)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) > 0

    # Count consecutive True values and return maximum
    return _utils._count_consecutive(returns).max()


def consecutive_losses(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the maximum number of consecutive losing periods.

    This function identifies the longest streak of negative returns, which
    helps assess the potential for extended drawdown periods.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        int: Maximum number of consecutive losing periods

    Example:
        >>> returns = pd.Series([0.01, -0.02, -0.01, -0.01, 0.02])
        >>> max_losses = consecutive_losses(returns)
        >>> print(f"Max consecutive losses: {max_losses}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns and convert to boolean (negative = True)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0

    # Count consecutive True values and return maximum
    return _utils._count_consecutive(returns).max()


def exposure(returns, prepare_returns=True):
    """
    Calculate market exposure time as percentage of periods with non-zero returns.

    This function measures how often the strategy was actually invested
    (had non-zero returns) versus being in cash or having zero positions.

    Args:
        returns (pd.Series or pd.DataFrame): Return series or DataFrame
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float or pd.Series: Exposure percentage (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, 0.00, 0.02, 0.00, 0.03])
        >>> exp = exposure(returns)
        >>> print(f"Market exposure: {exp:.2%}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    def _exposure(ret):
        """
        Calculate exposure for a single return series.

        Counts non-NaN, non-zero returns and divides by total periods.
        Rounds up to nearest percent to avoid zero exposure from rounding.
        """
        # Count non-NaN and non-zero returns
        ex = len(ret[(~_np.isnan(ret)) & (ret != 0)]) / len(ret)
        # Round up to nearest percent
        return _ceil(ex * 100) / 100

    # Handle DataFrame input by calculating exposure for each column
    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return _pd.Series(_df)

    return _exposure(returns)


def win_rate(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the win rate (percentage of profitable periods).

    This function computes the ratio of positive returns to total non-zero
    returns, providing a measure of how often the strategy generates profits.

    Args:
        returns (pd.Series or pd.DataFrame): Return series or DataFrame
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float or pd.Series: Win rate as decimal (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> wr = win_rate(returns)
        >>> print(f"Win rate: {wr:.2%}")
    """
    def _win_rate(series):
        """
        Calculate win rate for a single return series.

        Handles edge cases like no non-zero returns and provides
        error handling for calculation issues.
        """
        try:
            # Filter out zero returns (periods with no trading)
            non_zero_returns = series[series != 0]
            if len(non_zero_returns) == 0:
                warn("No non-zero returns found for win rate calculation, returning 0.0")
                return 0.0

            # Calculate ratio of positive returns to non-zero returns
            return len(series[series > 0]) / len(non_zero_returns)
        except (ValueError, TypeError) as e:
            warn(f"Error calculating win rate: {e}, returning 0.0")
            return 0.0

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns if period specified
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # Handle DataFrame input by calculating win rate for each column
    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])
        return _pd.Series(_df)

    return _win_rate(returns)


def avg_return(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the average return per period (excluding zero returns).

    This function computes the mean of non-zero returns, providing insight
    into the typical magnitude of returns when the strategy is active.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Average return per period

    Example:
        >>> returns = pd.Series([0.01, 0.00, 0.02, -0.01, 0.03])
        >>> avg_ret = avg_return(returns)
        >>> print(f"Average return: {avg_ret:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns if period specified
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # Calculate mean of non-zero returns
    return returns[returns != 0].dropna().mean()


def avg_win(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the average winning return (mean of positive returns).

    This function computes the mean of positive returns only, showing
    the typical magnitude of profitable periods.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Average winning return

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> avg_win_ret = avg_win(returns)
        >>> print(f"Average win: {avg_win_ret:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns if period specified
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # Calculate mean of positive returns only
    return returns[returns > 0].dropna().mean()


def avg_loss(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculate the average losing return (mean of negative returns).

    This function computes the mean of negative returns only, showing
    the typical magnitude of losing periods.

    Args:
        returns (pd.Series): Return series to analyze
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Average losing return (negative value)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> avg_loss_ret = avg_loss(returns)
        >>> print(f"Average loss: {avg_loss_ret:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns if period specified
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    # Calculate mean of negative returns only
    return returns[returns < 0].dropna().mean()


def volatility(returns, periods=252, annualize=True, prepare_returns=True):
    """
    Calculate volatility (standard deviation) of returns.

    This function computes the volatility of returns, which measures the
    degree of variation in returns over time. Higher volatility indicates
    more uncertainty and risk.

    Args:
        returns (pd.Series): Return series to analyze
        periods (int): Number of periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the volatility (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Volatility (annualized if annualize=True)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> vol = volatility(returns)
        >>> print(f"Annualized volatility: {vol:.4f}")
    """
    validate_input(returns)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate standard deviation of returns
    std = returns.std()

    # Annualize by multiplying by square root of periods per year
    if annualize:
        return std * _np.sqrt(periods)

    return std


def rolling_volatility(
    returns, rolling_period=126, periods_per_year=252, prepare_returns=True
):
    """
    Calculate rolling volatility over a specified window.

    This function computes volatility using a rolling window, providing
    a time-varying measure of risk that adapts to changing market conditions.

    Args:
        returns (pd.Series): Return series to analyze
        rolling_period (int): Rolling window size (default: 126, ~6 months)
        periods_per_year (int): Periods per year for annualization (default: 252)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        pd.Series: Rolling volatility series (annualized)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_vol = rolling_volatility(returns, rolling_period=3)
        >>> print(rolling_vol)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns, rolling_period)

    # Calculate rolling standard deviation and annualize
    return returns.rolling(rolling_period).std() * _np.sqrt(periods_per_year)


def implied_volatility(returns, periods=252, annualize=True):
    """
    Calculate implied volatility using log returns.

    This function computes volatility using log returns instead of simple
    returns, which is mathematically more appropriate for continuous compounding.

    Args:
        returns (pd.Series): Return series to analyze
        periods (int): Number of periods for rolling calculation (default: 252)
        annualize (bool): Whether to annualize the volatility (default: True)

    Returns:
        float or pd.Series: Implied volatility

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> impl_vol = implied_volatility(returns)
        >>> print(f"Implied volatility: {impl_vol:.4f}")
    """
    # Convert to log returns for continuous compounding
    logret = _utils.log_returns(returns)

    if annualize:
        # Calculate rolling volatility and annualize
        return logret.rolling(periods).std() * _np.sqrt(periods)

    # Return simple standard deviation
    return logret.std()


def autocorr_penalty(returns, prepare_returns=False):
    """
    Calculate autocorrelation penalty for risk-adjusted metrics.

    This function computes a penalty factor that accounts for autocorrelation
    in returns, which can inflate risk-adjusted ratios. Used to adjust
    Sharpe and Sortino ratios for more realistic risk assessment.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: False)

    Returns:
        float: Autocorrelation penalty factor (>= 1)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> penalty = autocorr_penalty(returns)
        >>> print(f"Autocorrelation penalty: {penalty:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Handle DataFrame input by selecting first column
    if isinstance(returns, _pd.DataFrame):
        returns = returns[returns.columns[0]]

    # returns.to_csv('/Users/ran/Desktop/test.csv')
    num = len(returns)

    # Calculate autocorrelation coefficient between consecutive returns
    coef = _np.abs(_np.corrcoef(returns[:-1], returns[1:])[0, 1])

    # Vectorized calculation instead of list comprehension
    x = _np.arange(1, num)
    # Calculate weighted correlation effects over time
    corr = ((num - x) / num) * (coef**x)

    # Return penalty factor (square root of 1 + 2 * sum of correlations)
    return _np.sqrt(1 + 2 * _np.sum(corr))


# ======= METRICS =======


def sharpe(returns, rf=0.0, periods=252, annualize=True, smart=False):
    """
    Calculate the Sharpe ratio of excess returns.

    The Sharpe ratio measures risk-adjusted returns by dividing excess returns
    (returns - risk-free rate) by the standard deviation of returns.
    Higher values indicate better risk-adjusted performance.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized if periods specified, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the ratio (default: True)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Sharpe ratio

    Raises:
        ValueError: If rf is non-zero but periods is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sharpe_ratio = sharpe(returns, rf=0.02)
        >>> print(f"Sharpe ratio: {sharpe_ratio:.4f}")
    """
    validate_input(returns)

    # Validate parameters for risk-free rate handling
    if rf != 0 and periods is None:
        raise ValueError("periods parameter is required when risk-free rate (rf) is non-zero. "
                         "This is needed to properly annualize the risk-free rate.")

    # Prepare returns (subtract risk-free rate if applicable)
    returns = _utils._prepare_returns(returns, rf, periods)

    # Calculate standard deviation as denominator
    divisor = returns.std(ddof=1)

    # Apply autocorrelation penalty if smart mode enabled
    if smart:
        # penalize sharpe with auto correlation
        divisor = divisor * autocorr_penalty(returns)

    # Calculate base Sharpe ratio
    res = returns.mean() / divisor

    # Annualize if requested
    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sharpe(returns, rf=0.0, periods=252, annualize=True):
    """
    Calculate the Smart Sharpe ratio (Sharpe with autocorrelation penalty).

    This is a wrapper for the sharpe() function with smart=True, which
    applies an autocorrelation penalty to provide more realistic risk-adjusted
    returns for strategies with autocorrelated returns.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the ratio (default: True)

    Returns:
        float: Smart Sharpe ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> smart_sharpe_ratio = smart_sharpe(returns)
        >>> print(f"Smart Sharpe ratio: {smart_sharpe_ratio:.4f}")
    """
    return sharpe(returns, rf, periods, annualize, True)


def rolling_sharpe(
    returns,
    rf=0.0,
    rolling_period=126,
    annualize=True,
    periods_per_year=252,
    prepare_returns=True,
):
    """
    Calculate rolling Sharpe ratio over a specified window.

    This function computes the Sharpe ratio using a rolling window, providing
    a time-varying measure of risk-adjusted performance that adapts to
    changing market conditions.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        rolling_period (int): Rolling window size (default: 126, ~6 months)
        annualize (bool): Whether to annualize the ratio (default: True)
        periods_per_year (int): Periods per year for annualization (default: 252)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        pd.Series: Rolling Sharpe ratio series

    Raises:
        Exception: If rf != 0 and rolling_period is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_sharpe_ratio = rolling_sharpe(returns, rolling_period=3)
        >>> print(rolling_sharpe_ratio)
    """
    # Validate parameters for risk-free rate handling
    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    if prepare_returns:
        returns = _utils._prepare_returns(returns, rf, rolling_period)

    # Calculate rolling mean and standard deviation
    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()

    # Annualize if requested
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


def sortino(returns, rf=0, periods=252, annualize=True, smart=False):
    """
    Calculate the Sortino ratio of excess returns.

    The Sortino ratio is similar to the Sharpe ratio but uses downside deviation
    instead of total volatility, focusing only on harmful volatility.
    This provides a more accurate measure of risk-adjusted returns.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the ratio (default: True)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Sortino ratio

    Raises:
        ValueError: If rf is non-zero but periods is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sortino_ratio = sortino(returns, rf=0.02)
        >>> print(f"Sortino ratio: {sortino_ratio:.4f}")

    Note:
        Calculation is based on this paper by Red Rock Capital:
        http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    validate_input(returns)

    # Validate parameters for risk-free rate handling
    if rf != 0 and periods is None:
        raise ValueError("periods parameter is required when risk-free rate (rf) is non-zero. "
                         "This is needed to properly annualize the risk-free rate.")

    # Prepare returns (subtract risk-free rate if applicable)
    returns = _utils._prepare_returns(returns, rf, periods)

    # Calculate downside deviation (only negative returns)
    downside = _np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    # Apply autocorrelation penalty if smart mode enabled
    if smart:
        # penalize sortino with auto correlation
        downside = downside * autocorr_penalty(returns)

    # Calculate base Sortino ratio
    res = returns.mean() / downside

    # Annualize if requested
    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sortino(returns, rf=0, periods=252, annualize=True):
    """
    Calculate the Smart Sortino ratio (Sortino with autocorrelation penalty).

    This is a wrapper for the sortino() function with smart=True, which
    applies an autocorrelation penalty to provide more realistic risk-adjusted
    returns for strategies with autocorrelated returns.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the ratio (default: True)

    Returns:
        float: Smart Sortino ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> smart_sortino_ratio = smart_sortino(returns)
        >>> print(f"Smart Sortino ratio: {smart_sortino_ratio:.4f}")
    """
    return sortino(returns, rf, periods, annualize, True)


def rolling_sortino(
    returns, rf=0, rolling_period=126, annualize=True, periods_per_year=252, **kwargs
):
    """
    Calculate rolling Sortino ratio over a specified window.

    This function computes the Sortino ratio using a rolling window, providing
    a time-varying measure of downside risk-adjusted performance.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        rolling_period (int): Rolling window size (default: 126, ~6 months)
        annualize (bool): Whether to annualize the ratio (default: True)
        periods_per_year (int): Periods per year for annualization (default: 252)
        **kwargs: Additional keyword arguments (e.g., prepare_returns)

    Returns:
        pd.Series: Rolling Sortino ratio series

    Raises:
        Exception: If rf != 0 and rolling_period is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_sortino_ratio = rolling_sortino(returns, rolling_period=3)
        >>> print(rolling_sortino_ratio)
    """
    # Validate parameters for risk-free rate handling
    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    if kwargs.get("prepare_returns", True):
        returns = _utils._prepare_returns(returns, rf, rolling_period)

    # Optimized downside calculation using vectorized operations
    def calc_downside(x):
        """
        Calculate downside variance more efficiently.

        This function computes the sum of squared negative returns,
        which is used to calculate downside deviation.
        """
        negative_returns = x[x < 0]
        return (negative_returns**2).sum() if len(negative_returns) > 0 else 0

    # Calculate rolling downside deviation
    downside = (
        returns.rolling(rolling_period).apply(calc_downside, raw=True) / rolling_period
    )

    # Calculate rolling Sortino ratio
    res = returns.rolling(rolling_period).mean() / _np.sqrt(downside)

    # Annualize if requested
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


def adjusted_sortino(returns, rf=0, periods=252, annualize=True, smart=False):
    """
    Calculate Jack Schwager's adjusted Sortino ratio.

    This version of the Sortino ratio is adjusted by dividing by sqrt(2)
    to allow for direct comparisons with the Sharpe ratio. This adjustment
    accounts for the difference in calculation methods.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the ratio (default: True)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Adjusted Sortino ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> adj_sortino = adjusted_sortino(returns)
        >>> print(f"Adjusted Sortino ratio: {adj_sortino:.4f}")

    Note:
        See here for more info: https://archive.is/wip/2rwFW
    """
    # Calculate standard Sortino ratio
    data = sortino(returns, rf, periods=periods, annualize=annualize, smart=smart)

    # Apply Schwager's adjustment factor
    return data / _sqrt(2)


def probabilistic_ratio(
    series, rf=0.0, base="sharpe", periods=252, annualize=False, smart=False
):
    """
    Calculate the probabilistic ratio for a given base metric.

    This function computes the probabilistic version of risk-adjusted ratios,
    which accounts for the statistical uncertainty in the ratio estimation.
    It considers skewness and kurtosis to provide more robust estimates.

    Args:
        series (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        base (str): Base metric ('sharpe', 'sortino', 'adjusted_sortino')
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the result (default: False)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Probabilistic ratio (0-1 scale representing probability)

    Raises:
        ValueError: If invalid base metric is provided

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> prob_ratio = probabilistic_ratio(returns, base="sharpe")
        >>> print(f"Probabilistic Sharpe ratio: {prob_ratio:.4f}")
    """
    # Calculate the base ratio depending on the selected metric
    if base.lower() == "sharpe":
        base = sharpe(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "sortino":
        base = sortino(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "adjusted_sortino":
        base = adjusted_sortino(series, periods=periods, annualize=False, smart=smart)
    else:
        raise ValueError(
            f"Invalid metric '{base}'. Must be one of: 'sharpe', 'sortino', or 'adjusted_sortino'"
        )

    # Calculate higher moments for adjustment
    skew_no = skew(series, prepare_returns=False)
    kurtosis_no = kurtosis(series, prepare_returns=False)

    n = len(series)

    # Calculate standard error of the ratio incorporating higher moments
    # Formula accounts for skewness and kurtosis effects on ratio distribution
    sigma_sr = _np.sqrt(
        (1 + (0.5 * base**2) - (skew_no * base) + (((kurtosis_no - 3) / 4) * base**2))
        / (n - 1)
    )

    # Calculate standardized ratio and convert to probability
    ratio = (base - rf) / sigma_sr
    psr = _norm.cdf(ratio)

    # Annualize if requested
    if annualize:
        return psr * (252**0.5)

    return psr


def probabilistic_sharpe_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    This function computes the PSR, which represents the probability that
    the observed Sharpe ratio is statistically greater than a benchmark.
    It accounts for higher moments to provide more robust estimates.

    Args:
        series (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the result (default: False)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Probabilistic Sharpe ratio (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> psr = probabilistic_sharpe_ratio(returns)
        >>> print(f"Probabilistic Sharpe ratio: {psr:.4f}")
    """
    return probabilistic_ratio(
        series, rf, base="sharpe", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_sortino_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    """
    Calculate the Probabilistic Sortino Ratio.

    This function computes the probabilistic version of the Sortino ratio,
    which accounts for statistical uncertainty in the ratio estimation.

    Args:
        series (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the result (default: False)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Probabilistic Sortino ratio (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> psr = probabilistic_sortino_ratio(returns)
        >>> print(f"Probabilistic Sortino ratio: {psr:.4f}")
    """
    return probabilistic_ratio(
        series, rf, base="sortino", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_adjusted_sortino_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    """
    Calculate the Probabilistic Adjusted Sortino Ratio.

    This function computes the probabilistic version of the adjusted Sortino
    ratio, accounting for statistical uncertainty in the ratio estimation.

    Args:
        series (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Whether to annualize the result (default: False)
        smart (bool): Whether to apply autocorrelation penalty (default: False)

    Returns:
        float: Probabilistic adjusted Sortino ratio (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> psr = probabilistic_adjusted_sortino_ratio(returns)
        >>> print(f"Probabilistic adjusted Sortino ratio: {psr:.4f}")
    """
    return probabilistic_ratio(
        series,
        rf,
        base="adjusted_sortino",
        periods=periods,
        annualize=annualize,
        smart=smart,
    )


def treynor_ratio(returns, benchmark, periods=252.0, rf=0.0):
    """
    Calculate the Treynor ratio.

    The Treynor ratio measures risk-adjusted returns relative to systematic risk
    (beta) rather than total risk (volatility). It's calculated as excess return
    divided by beta, useful for comparing portfolios with different market exposure.

    Args:
        returns (pd.Series): Return series to analyze
        benchmark (pd.Series): Benchmark return series for beta calculation
        periods (float): Periods per year for annualization (default: 252.0)
        rf (float): Risk-free rate (annualized, default: 0.0)

    Returns:
        float: Treynor ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> treynor = treynor_ratio(returns, benchmark)
        >>> print(f"Treynor ratio: {treynor:.4f}")
    """
    # Handle DataFrame input by selecting first column
    if isinstance(returns, _pd.DataFrame):
        returns = returns[returns.columns[0]]

    # Calculate beta from the Greeks (alpha, beta analysis)
    beta = greeks(returns, benchmark, periods=periods).to_dict().get("beta", 0)

    # Prevent division by zero
    if beta == 0:
        warn("Beta is zero, cannot calculate Treynor ratio, returning 0")
        return 0

    # Calculate excess return over risk-free rate divided by beta
    return (comp(returns) - rf) / beta


def omega(returns, rf=0.0, required_return=0.0, periods=252):
    """
    Calculate the Omega ratio of a strategy.

    The Omega ratio measures the probability-weighted ratio of gains to losses
    above and below a threshold return. It provides a comprehensive view of
    the return distribution's characteristics.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        required_return (float): Required return threshold (default: 0.0)
        periods (int): Periods per year for annualization (default: 252)

    Returns:
        float: Omega ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> omega_ratio = omega(returns, required_return=0.01)
        >>> print(f"Omega ratio: {omega_ratio:.4f}")

    Note:
        See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """
    validate_input(returns)

    # Validate minimum data requirements
    if len(returns) < 2:
        warn("Insufficient data for omega ratio calculation (need at least 2 returns), returning NaN")
        return _np.nan

    # Validate required return parameter
    if required_return <= -1:
        warn(f"Invalid required_return ({required_return}) for omega ratio, must be > -1, returning NaN")
        return _np.nan

    # Prepare returns (subtract risk-free rate if applicable)
    returns = _utils._prepare_returns(returns, rf, periods)

    # Convert annualized required return to per-period if needed
    if periods == 1:
        return_threshold = required_return
    else:
        return_threshold = (1 + required_return) ** (1.0 / periods) - 1

    # Calculate deviations from threshold
    returns_less_thresh = returns - return_threshold

    # Sum of positive deviations (gains above threshold)
    numer = returns_less_thresh[returns_less_thresh > 0.0].sum()

    # Sum of negative deviations (losses below threshold)
    denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum()

    # Handle both Series and scalar cases
    if isinstance(denom, _pd.Series):
        result = numer / denom
        # Return NaN where denominator is zero
        result = result.where(denom > 0.0, _np.nan)
        return result
    else:
        if denom > 0.0:
            return numer / denom
        return _np.nan


def gain_to_pain_ratio(returns, rf=0, resolution="D"):
    """
    Calculate Jack Schwager's Gain-to-Pain Ratio (GPR).

    This ratio measures the total gains divided by the total losses,
    providing a simple measure of how much profit is generated per
    unit of loss. Higher values indicate better performance.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (default: 0)
        resolution (str): Resampling frequency ('D', 'W', 'M', etc.)

    Returns:
        float: Gain-to-Pain ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> gpr = gain_to_pain_ratio(returns)
        >>> print(f"Gain-to-Pain ratio: {gpr:.4f}")

    Note:
        See here for more info: https://archive.is/wip/2rwFW
    """
    # Prepare returns and resample to specified frequency
    returns = _utils._prepare_returns(returns, rf).resample(resolution).sum()

    # Calculate absolute sum of negative returns (pain)
    downside = abs(returns[returns < 0].sum())

    # Return ratio of total gains to total pain
    return returns.sum() / downside


def cagr(returns, rf=0.0, compounded=True, periods=252):
    """
    Calculate the Compound Annual Growth Rate (CAGR) of excess returns.

    CAGR represents the geometric mean annual growth rate, providing a
    smoothed annualized return that accounts for compounding effects.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        compounded (bool): Whether to compound returns (default: True)
        periods (int): Periods per year for annualization (default: 252)

    Returns:
        float or pd.Series: CAGR percentage

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02],
        ...                    index=pd.date_range('2023-01-01', periods=5))
        >>> cagr_value = cagr(returns)
        >>> print(f"CAGR: {cagr_value:.4f}")
    """
    validate_input(returns)

    # Prepare returns (subtract risk-free rate if applicable)
    total = _utils._prepare_returns(returns, rf)

    # Calculate total return
    if compounded:
        total = comp(total)
    else:
        total = _np.sum(total, axis=0)

    # Calculate time period in years using trading periods
    # This is consistent with how Sharpe, Sortino, and other metrics
    # handle annualization in quantstats
    years = len(returns) / periods

    # Calculate CAGR using geometric mean formula
    res = abs(total + 1.0) ** (1.0 / years) - 1

    # Handle DataFrame input
    if isinstance(returns, _pd.DataFrame):
        res = _pd.Series(res)
        res.index = returns.columns

    return res


def rar(returns, rf=0.0):
    """
    Calculate the Risk-Adjusted Return (RAR).

    RAR is calculated as CAGR divided by exposure, taking into account
    the time the strategy was actually invested. This provides a more
    accurate measure of returns adjusted for actual market participation.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)

    Returns:
        float: Risk-adjusted return

    Example:
        >>> returns = pd.Series([0.01, 0.00, 0.03, 0.00, 0.02])
        >>> rar_value = rar(returns)
        >>> print(f"Risk-adjusted return: {rar_value:.4f}")
    """
    # Prepare returns (subtract risk-free rate if applicable)
    returns = _utils._prepare_returns(returns, rf)

    # Calculate CAGR and divide by exposure time
    return cagr(returns) / exposure(returns)


def skew(returns, prepare_returns=True):
    """
    Calculate returns' skewness.

    Skewness measures the degree of asymmetry of a distribution around its mean.
    Positive skewness indicates a longer tail on the positive side,
    while negative skewness indicates a longer tail on the negative side.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Skewness value

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> skewness = skew(returns)
        >>> print(f"Skewness: {skewness:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate skewness using pandas built-in method
    return returns.skew()


def kurtosis(returns, prepare_returns=True):
    """
    Calculate returns' kurtosis.

    Kurtosis measures the degree to which a distribution is peaked compared
    to a normal distribution. Higher kurtosis indicates more extreme returns
    (fat tails), while lower kurtosis indicates fewer extreme returns.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Kurtosis value (excess kurtosis, normal distribution = 0)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> kurt = kurtosis(returns)
        >>> print(f"Kurtosis: {kurt:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate kurtosis using pandas built-in method (excess kurtosis)
    return returns.kurtosis()


def calmar(returns, prepare_returns=True, periods=252):
    """
    Calculate the Calmar ratio (CAGR / Maximum Drawdown).

    The Calmar ratio measures risk-adjusted returns by dividing the CAGR
    by the absolute value of the maximum drawdown. It provides insight
    into returns relative to the worst-case scenario.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)
        periods (int): Periods per year for annualization (default: 252)

    Returns:
        float: Calmar ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> calmar_ratio = calmar(returns)
        >>> print(f"Calmar ratio: {calmar_ratio:.4f}")
    """
    validate_input(returns)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate CAGR and maximum drawdown
    cagr_ratio = cagr(returns, periods=periods)
    max_dd = max_drawdown(returns)

    # Return ratio of CAGR to absolute maximum drawdown
    return cagr_ratio / abs(max_dd)


def ulcer_index(returns):
    """
    Calculate the Ulcer Index (downside risk measurement).

    The Ulcer Index measures the depth and duration of drawdowns,
    providing a comprehensive measure of downside risk. It's calculated
    as the square root of the mean of squared drawdowns.

    Args:
        returns (pd.Series): Return series to analyze

    Returns:
        float: Ulcer Index value

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> ulcer = ulcer_index(returns)
        >>> print(f"Ulcer Index: {ulcer:.4f}")
    """
    # Convert returns to drawdown series
    dd = to_drawdown_series(returns)

    # Calculate root mean square of drawdowns
    return _np.sqrt(_np.divide((dd**2).sum(), returns.shape[0] - 1))


def ulcer_performance_index(returns, rf=0):
    """
    Calculate the Ulcer Performance Index (UPI).

    The UPI measures risk-adjusted returns using the Ulcer Index as the
    risk measure instead of standard deviation. It provides a better
    measure for strategies with significant drawdowns.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (default: 0)

    Returns:
        float: Ulcer Performance Index

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> upi_value = ulcer_performance_index(returns)
        >>> print(f"Ulcer Performance Index: {upi_value:.4f}")
    """
    # Calculate excess return divided by Ulcer Index
    return (comp(returns) - rf) / ulcer_index(returns)


def upi(returns, rf=0):
    """
    Calculate the Ulcer Performance Index (UPI).

    This is a shorthand function for ulcer_performance_index() with
    the same parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (default: 0)

    Returns:
        float: Ulcer Performance Index
    """
    return ulcer_performance_index(returns, rf)


def serenity_index(returns, rf=0):
    """
    Calculate the Serenity Index.

    The Serenity Index is a comprehensive risk-adjusted return measure
    that combines the Ulcer Index with downside risk considerations.
    It provides a more holistic view of strategy performance.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (default: 0)

    Returns:
        float: Serenity Index

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> serenity = serenity_index(returns)
        >>> print(f"Serenity Index: {serenity:.4f}")

    Note:
        Based on KeyQuant whitepaper:
        https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf
    """
    # Convert returns to drawdown series
    dd = to_drawdown_series(returns)

    # Calculate pitfall measure using conditional value at risk of drawdowns
    pitfall = -cvar(dd) / returns.std()

    # Calculate serenity index incorporating both ulcer index and pitfall
    return (returns.sum() - rf) / (ulcer_index(returns) * pitfall)


def risk_of_ruin(returns, prepare_returns=True):
    """
    Calculate the risk of ruin (probability of losing all capital).

    This function estimates the likelihood of losing all investment capital
    based on the win rate and the number of trades/periods. It's useful
    for position sizing and risk management.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Risk of ruin probability (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> ror_value = risk_of_ruin(returns)
        >>> print(f"Risk of ruin: {ror_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate win rate
    wins = win_rate(returns)

    # Calculate risk of ruin using gambler's ruin formula
    return ((1 - wins) / (1 + wins)) ** len(returns)


def ror(returns):
    """
    Calculate the risk of ruin (probability of losing all capital).

    This is a shorthand function for risk_of_ruin() with the same
    parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze

    Returns:
        float: Risk of ruin probability (0-1 scale)
    """
    return risk_of_ruin(returns)


def value_at_risk(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Calculate the daily Value at Risk (VaR).

    VaR estimates the maximum expected loss over a given time horizon
    at a specified confidence level, using the variance-covariance method.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Value at Risk (negative value representing loss)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> var_value = value_at_risk(returns, confidence=0.95)
        >>> print(f"95% VaR: {var_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate mean and adjust volatility
    mu = returns.mean()
    sigma *= returns.std()

    # Convert percentage confidence to decimal if needed
    if confidence > 1:
        confidence = confidence / 100

    # Calculate VaR using normal distribution inverse CDF
    return _norm.ppf(1 - confidence, mu, sigma)


def var(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Calculate the daily Value at Risk (VaR).

    This is a shorthand function for value_at_risk() with the same
    parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Value at Risk (negative value representing loss)
    """
    return value_at_risk(returns, sigma, confidence, prepare_returns)


def conditional_value_at_risk(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Calculate the Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    CVaR measures the expected loss given that a loss exceeds the VaR threshold.
    It quantifies the amount of tail risk an investment faces, providing a more
    comprehensive risk measure than VaR alone.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Conditional Value at Risk (expected loss beyond VaR)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> cvar_value = conditional_value_at_risk(returns, confidence=0.95)
        >>> print(f"95% CVaR: {cvar_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate VaR threshold
    var = value_at_risk(returns, sigma, confidence)

    # Calculate mean of returns below VaR threshold
    c_var = returns[returns < var].values.mean()

    # Return CVaR if valid, otherwise return VaR
    return c_var if ~_np.isnan(c_var) else var


def cvar(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Calculate the Conditional Value at Risk (CVaR).

    This is a shorthand function for conditional_value_at_risk() with
    the same parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Conditional Value at Risk
    """
    return conditional_value_at_risk(returns, sigma, confidence, prepare_returns)


def expected_shortfall(returns, sigma=1, confidence=0.95):
    """
    Calculate the Expected Shortfall (ES), also known as CVaR.

    This is a shorthand function for conditional_value_at_risk() with
    the same parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)

    Returns:
        float: Expected Shortfall
    """
    return conditional_value_at_risk(returns, sigma, confidence)


def tail_ratio(returns, cutoff=0.95, prepare_returns=True):
    """
    Calculate the tail ratio between right and left tails.

    This function measures the ratio between the right (95%) and left (5%) tails
    of the return distribution, providing insight into the asymmetry of extreme
    returns. Higher values indicate more favorable tail characteristics.

    Args:
        returns (pd.Series): Return series to analyze
        cutoff (float): Percentile cutoff for tail analysis (default: 0.95)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Tail ratio (right tail / left tail)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> tail_r = tail_ratio(returns)
        >>> print(f"Tail ratio: {tail_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate ratio of right tail to left tail
    return abs(returns.quantile(cutoff) / returns.quantile(1 - cutoff))


def payoff_ratio(returns, prepare_returns=True):
    """
    Calculate the payoff ratio (average win / average loss).

    This function measures the ratio of average winning returns to average
    losing returns, providing insight into the reward-to-risk profile
    of individual trades or periods.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Payoff ratio (average win / absolute average loss)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> payoff_r = payoff_ratio(returns)
        >>> print(f"Payoff ratio: {payoff_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate ratio of average win to absolute average loss
    return avg_win(returns) / abs(avg_loss(returns))


def win_loss_ratio(returns, prepare_returns=True):
    """
    Calculate the win-loss ratio (average win / average loss).

    This is a shorthand function for payoff_ratio() with the same
    parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Win-loss ratio
    """
    return payoff_ratio(returns, prepare_returns)


def profit_ratio(returns, prepare_returns=True):
    """
    Calculate the profit ratio (win ratio / loss ratio).

    This function measures the ratio of win frequency to loss frequency,
    providing insight into the consistency of profitable periods.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Profit ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> profit_r = profit_ratio(returns)
        >>> print(f"Profit ratio: {profit_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Separate wins and losses
    wins = returns[returns >= 0]
    loss = returns[returns < 0]

    # Handle edge cases
    if wins.count() == 0:
        warn("No winning returns found for profit ratio calculation")
        return 0.0
    if loss.count() == 0:
        warn("No losing returns found for profit ratio calculation, returning infinity")
        return float('inf')

    # Calculate win and loss ratios
    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())

    try:
        if loss_ratio == 0:
            warn("Loss ratio is zero, returning infinity for profit ratio")
            return float('inf')
        return win_ratio / loss_ratio
    except (ValueError, TypeError) as e:
        warn(f"Error calculating profit ratio: {e}, returning 0.0")
        return 0.0


def profit_factor(returns, prepare_returns=True):
    """
    Calculate the profit factor (total wins / total losses).

    This function measures the ratio of total winning returns to total
    losing returns, providing insight into the overall profitability
    of the strategy.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Profit factor (total wins / total losses)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> pf = profit_factor(returns)
        >>> print(f"Profit factor: {pf:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate total wins and losses
    wins_sum = returns[returns >= 0].sum()
    losses_sum = abs(returns[returns < 0].sum())

    # Handle both Series and scalar cases
    if isinstance(losses_sum, _pd.Series):
        result = wins_sum / losses_sum
        # Replace infinite values with 0
        result = result.replace([_np.inf, -_np.inf], 0)
        return result
    else:
        # Handle division by zero case
        if losses_sum == 0:
            return 0.0 if wins_sum == 0 else float('inf')
        return wins_sum / losses_sum


def cpc_index(returns, prepare_returns=True):
    """
    Calculate the CPC Index (Profit Factor * Win Rate * Win-Loss Ratio).

    The CPC Index is a comprehensive performance measure that combines
    profit factor, win rate, and win-loss ratio to provide a single
    metric for strategy evaluation.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: CPC Index

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> cpc = cpc_index(returns)
        >>> print(f"CPC Index: {cpc:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate composite metric
    return profit_factor(returns) * win_rate(returns) * win_loss_ratio(returns)


def common_sense_ratio(returns, prepare_returns=True):
    """
    Calculate the Common Sense Ratio (Profit Factor * Tail Ratio).

    This ratio combines profit factor with tail ratio to provide a
    measure that considers both profitability and tail risk characteristics.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Common Sense Ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> csr = common_sense_ratio(returns)
        >>> print(f"Common Sense Ratio: {csr:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate composite metric
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=0.99, prepare_returns=True):
    """
    Calculate the outlier winners ratio.

    This function computes the ratio of the 99th percentile of returns
    to the mean positive return, showing how much outlier wins contribute
    to overall performance.

    Args:
        returns (pd.Series): Return series to analyze
        quantile (float): Quantile for outlier threshold (default: 0.99)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Outlier win ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> outlier_win_r = outlier_win_ratio(returns)
        >>> print(f"Outlier win ratio: {outlier_win_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate ratio of high quantile to mean positive return
    return returns.quantile(quantile).mean() / returns[returns >= 0].mean()


def outlier_loss_ratio(returns, quantile=0.01, prepare_returns=True):
    """
    Calculate the outlier losers ratio.

    This function computes the ratio of the 1st percentile of returns
    to the mean negative return, showing how much outlier losses contribute
    to overall risk.

    Args:
        returns (pd.Series): Return series to analyze
        quantile (float): Quantile for outlier threshold (default: 0.01)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Outlier loss ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> outlier_loss_r = outlier_loss_ratio(returns)
        >>> print(f"Outlier loss ratio: {outlier_loss_r:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate ratio of low quantile to mean negative return
    return returns.quantile(quantile).mean() / returns[returns < 0].mean()


def recovery_factor(returns, rf=0.0, prepare_returns=True):
    """
    Calculate the recovery factor (total returns / maximum drawdown).

    This function measures how fast the strategy recovers from drawdowns
    by comparing total returns to the maximum drawdown experienced.
    Higher values indicate better recovery characteristics.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (default: 0.0)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Recovery factor

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rf_value = recovery_factor(returns)
        >>> print(f"Recovery factor: {rf_value:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate total excess returns
    total_returns = returns.sum() - rf

    # Calculate maximum drawdown
    max_dd = max_drawdown(returns)

    # Return ratio of total returns to absolute maximum drawdown
    return abs(total_returns) / abs(max_dd)


def risk_return_ratio(returns, prepare_returns=True):
    """
    Calculate the risk-return ratio (mean return / standard deviation).

    This function calculates the Sharpe ratio without factoring in the
    risk-free rate, providing a simple measure of return per unit of risk.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Risk-return ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rrr = risk_return_ratio(returns)
        >>> print(f"Risk-return ratio: {rrr:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate mean return divided by standard deviation
    return returns.mean() / returns.std()


def _get_baseline_value(prices):
    """
    Determine the appropriate baseline value for drawdown calculations.

    This function analyzes the price series to determine the correct baseline
    value that should represent "no drawdown" (i.e., the starting equity).

    Args:
        prices (pd.Series): Price series

    Returns:
        float: Baseline value for drawdown calculations
    """
    if len(prices) == 0:
        return 1.0

    # Handle both Series and DataFrame cases
    if isinstance(prices, _pd.DataFrame):
        # If prices is a DataFrame, ensure it has at least one column
        if prices.shape[1] == 0:
            return 1.0  # Default baseline for empty DataFrame with no columns
        # Get the first value of the first column
        first_price = prices.iat[0, 0]
    else:
        # If prices is a Series, get the first value directly
        first_price = prices.iloc[0]

    # If the first price is much larger than 1, it's likely from to_prices conversion
    # The to_prices function uses base * (1 + compsum), so we determine the appropriate baseline
    if first_price > 1000:
        # This suggests it came from to_prices with a large base (default 1e5)
        # However, we should use a more reasonable baseline for drawdown calculations
        # We'll use the same scale as the prices but represent the "no loss" baseline
        return 1e5
    elif first_price > 10:
        # Smaller base value scale
        return 100.0
    else:
        # Normal price scale, use 1.0 as baseline
        return 1.0


def max_drawdown(prices):
    """
    Calculate the maximum drawdown from peak to trough.

    This function calculates the maximum observed loss from a peak to a
    subsequent trough, expressed as a percentage. It handles the edge case
    where the first return is negative by establishing a proper baseline.

    Args:
        prices (pd.Series): Price series or cumulative returns

    Returns:
        float: Maximum drawdown (negative value)

    Example:
        >>> prices = pd.Series([100, 110, 105, 120, 115])
        >>> max_dd = max_drawdown(prices)
        >>> print(f"Maximum drawdown: {max_dd:.4f}")
    """
    validate_input(prices)

    # Prepare prices (convert from returns if needed)
    prices = _utils._prepare_prices(prices)

    if len(prices) == 0:
        return 0.0

    # Handle edge case: if first value represents a loss from baseline
    # Add a phantom baseline value to ensure proper drawdown calculation
    try:
        time_delta = prices.index.freq or _pd.Timedelta(days=1)
    except Exception:
        time_delta = _pd.Timedelta(days=1)

    phantom_date = prices.index[0] - time_delta

    # Determine appropriate baseline value
    baseline_value = _get_baseline_value(prices)

    # Create extended series with phantom baseline
    extended_prices = prices.copy()
    extended_prices.loc[phantom_date] = baseline_value
    extended_prices = extended_prices.sort_index()

    # Calculate drawdown with phantom baseline
    return (extended_prices / extended_prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(returns):
    """
    Convert returns series to drawdown series.

    This function converts a return series to a drawdown series showing
    the decline from peak equity at each point in time. It handles the
    edge case where the first return is negative by establishing a proper baseline.

    Args:
        returns (pd.Series): Return series to convert

    Returns:
        pd.Series: Drawdown series (negative values showing decline from peak)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> dd_series = to_drawdown_series(returns)
        >>> print(dd_series)
    """
    validate_input(returns)

    # Convert returns to prices
    prices = _utils._prepare_prices(returns)

    if len(prices) == 0:
        return _pd.Series([], dtype=float, index=returns.index)

    # Handle edge case: if first value represents a loss from baseline
    # Add a phantom baseline value to ensure proper drawdown calculation
    try:
        time_delta = prices.index.freq or _pd.Timedelta(days=1)
    except Exception:
        time_delta = _pd.Timedelta(days=1)

    phantom_date = prices.index[0] - time_delta

    # Determine appropriate baseline value
    baseline_value = _get_baseline_value(prices)

    # Create extended series with phantom baseline
    extended_prices = prices.copy()
    extended_prices.loc[phantom_date] = baseline_value
    extended_prices = extended_prices.sort_index()

    # Calculate drawdown series with phantom baseline
    dd = extended_prices / _np.maximum.accumulate(extended_prices) - 1.0

    # Remove phantom point and return original time series
    dd = dd.drop(phantom_date)

    # Clean up infinite and zero values
    return dd.replace([_np.inf, -_np.inf, -0], 0)  # type: ignore[attr-defined]


def kelly_criterion(returns, prepare_returns=True):
    """
    Calculates the recommended maximum amount of capital that
    should be allocated to the given strategy, based on the
    Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


# ==== VS. BENCHMARK ====


def r_squared(returns, benchmark, prepare_returns=True):
    """
    Calculate the R-squared (coefficient of determination) versus benchmark.

    R-squared measures how well the returns fit a straight line relationship
    with the benchmark. Values closer to 1 indicate higher correlation with
    the benchmark, while values closer to 0 indicate more independent movement.

    Args:
        returns (pd.Series): Return series to analyze
        benchmark (pd.Series): Benchmark return series for comparison
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: R-squared value (0-1 scale)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> r_sq = r_squared(returns, benchmark)
        >>> print(f"R-squared: {r_sq:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Prepare benchmark to match returns index
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # Perform linear regression and extract correlation coefficient
    _, _, r_val, _, _ = _linregress(
        returns, _utils._prepare_benchmark(benchmark, returns.index)
    )

    # Square the correlation coefficient to get R-squared
    return r_val**2


def r2(returns, benchmark):
    """
    Calculate the R-squared (coefficient of determination) versus benchmark.

    This is a shorthand function for r_squared() with the same parameters
    and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        benchmark (pd.Series): Benchmark return series for comparison

    Returns:
        float: R-squared value (0-1 scale)
    """
    return r_squared(returns, benchmark)


def information_ratio(returns, benchmark, prepare_returns=True):
    """
    Calculate the Information Ratio.

    The Information Ratio measures the risk-adjusted excess return of a
    portfolio relative to a benchmark. It's calculated as the active return
    (return - benchmark) divided by the tracking error (standard deviation
    of active returns).

    Args:
        returns (pd.Series): Return series to analyze
        benchmark (pd.Series): Benchmark return series for comparison
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float: Information Ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> info_ratio = information_ratio(returns, benchmark)
        >>> print(f"Information Ratio: {info_ratio:.4f}")
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Prepare benchmark to match returns index
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # Calculate active returns (returns - benchmark)
    diff_rets = returns - _utils._prepare_benchmark(benchmark, returns.index)

    # Calculate tracking error (standard deviation of active returns)
    std = diff_rets.std()

    # Return Information Ratio (active return / tracking error)
    if std != 0:
        return diff_rets.mean() / diff_rets.std()
    return 0


def greeks(returns, benchmark, periods=252.0, prepare_returns=True):
    """
    Calculate portfolio Greeks (alpha and beta) relative to benchmark.

    This function calculates the key portfolio metrics for benchmark comparison:
    - Alpha: Excess return after adjusting for systematic risk (beta)
    - Beta: Sensitivity to benchmark movements (systematic risk)

    Args:
        returns (pd.Series): Return series to analyze
        benchmark (pd.Series): Benchmark return series for comparison
        periods (float): Periods per year for alpha annualization (default: 252.0)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        pd.Series: Series containing 'alpha' and 'beta' values

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> portfolio_greeks = greeks(returns, benchmark)
        >>> print(f"Alpha: {portfolio_greeks['alpha']:.4f}")
        >>> print(f"Beta: {portfolio_greeks['beta']:.4f}")
    """
    # Data preparation
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)
    # ----------------------------

    # Calculate covariance matrix between returns and benchmark
    matrix = _np.cov(returns, benchmark)

    # Calculate beta (sensitivity to benchmark movements)
    beta = matrix[0, 1] / matrix[1, 1]

    # Calculate alpha (excess return after adjusting for beta)
    alpha = returns.mean() - beta * benchmark.mean()

    # Annualize alpha
    alpha = alpha * periods

    # Return results as Series
    return _pd.Series(
        {
            "beta": beta,
            "alpha": alpha,
            # "vol": _np.sqrt(matrix[0, 0]) * _np.sqrt(periods)
        }
    ).fillna(0)


def rolling_greeks(returns, benchmark, periods=252, prepare_returns=True):
    """
    Calculate rolling Greeks (alpha and beta) over time.

    This function calculates time-varying alpha and beta using a rolling
    window, showing how portfolio sensitivity to the benchmark changes
    over time. Useful for analyzing strategy stability and regime changes.

    Args:
        returns (pd.Series): Return series to analyze
        benchmark (pd.Series): Benchmark return series for comparison
        periods (int): Rolling window size (default: 252, ~1 year)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        pd.DataFrame: DataFrame with 'alpha' and 'beta' columns over time

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> rolling_greeks_df = rolling_greeks(returns, benchmark, periods=3)
        >>> print(rolling_greeks_df)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Create combined DataFrame for rolling calculations
    df = _pd.DataFrame(
        data={
            "returns": returns,
            "benchmark": _utils._prepare_benchmark(benchmark, returns.index),
        }
    )

    # Fill NaN values with 0 for calculation stability
    df = df.fillna(0)

    # Calculate rolling correlation and standard deviations
    corr = df.rolling(int(periods)).corr().unstack()["returns"]["benchmark"]
    std = df.rolling(int(periods)).std()

    # Calculate rolling beta
    beta = corr * std["returns"] / std["benchmark"]

    # Calculate rolling alpha (not annualized for rolling version)
    alpha = df["returns"].mean() - beta * df["benchmark"].mean()

    # Return DataFrame with rolling Greeks
    return _pd.DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})


def compare(
    returns,
    benchmark,
    aggregate=None,
    compounded=True,
    round_vals=None,
    prepare_returns=True,
):
    """
    Compare returns to benchmark across different time periods.

    This function provides a comprehensive comparison of portfolio returns
    versus benchmark performance across various aggregation periods
    (daily, weekly, monthly, quarterly, yearly).

    Args:
        returns (pd.Series or pd.DataFrame): Return series to analyze
        benchmark (pd.Series): Benchmark return series for comparison
        aggregate (str): Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        compounded (bool): Whether to compound returns (default: True)
        round_vals (int): Number of decimal places to round (default: None)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        pd.DataFrame: Comparison DataFrame with columns:
            - Benchmark: Benchmark returns for each period
            - Returns: Portfolio returns for each period
            - Multiplier: Portfolio return / Benchmark return
            - Won: '+' if portfolio outperformed, '-' if underperformed

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])
        >>> comparison = compare(returns, benchmark)
        >>> print(comparison)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Store original benchmark for proper aggregation
    # This preserves returns that may fall on non-trading days
    if isinstance(benchmark, str):
        benchmark_original = _utils.download_returns(benchmark)
    elif isinstance(benchmark, _pd.DataFrame):
        benchmark_original = benchmark[benchmark.columns[0]].copy()
    else:
        benchmark_original = benchmark.copy() if benchmark is not None else None
    
    # Prepare benchmark to match returns index for other calculations
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # Handle Series input
    if isinstance(returns, _pd.Series):
        # Aggregate returns and use original benchmark for aggregation
        # This ensures we don't lose benchmark returns on non-trading days
        if benchmark_original is not None:
            benchmark_agg = _utils.aggregate_returns(benchmark_original, aggregate, compounded) * 100
        else:
            benchmark_agg = _utils.aggregate_returns(benchmark, aggregate, compounded) * 100
        returns_agg = _utils.aggregate_returns(returns, aggregate, compounded) * 100

        # Create comparison DataFrame
        data = _pd.DataFrame(
            data={
                "Benchmark": benchmark_agg,
                "Returns": returns_agg,
            }
        )

        # Calculate performance multiplier and win/loss indicator
        data["Multiplier"] = data["Returns"] / data["Benchmark"]
        data["Won"] = _np.where(data["Returns"] >= data["Benchmark"], "+", "-")

    # Handle DataFrame input (multiple strategies)
    elif isinstance(returns, _pd.DataFrame):
        # Aggregate benchmark using original data to preserve non-trading day returns
        if benchmark_original is not None:
            bench = {
                "Benchmark": _utils.aggregate_returns(benchmark_original, aggregate, compounded) * 100
            }
        else:
            bench = {
                "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded) * 100
            }

        # Aggregate each strategy column
        strategy = {
            "Returns_" + str(i): _utils.aggregate_returns(returns[col], aggregate, compounded) * 100
            for i, col in enumerate(returns.columns)
        }

        # Combine into single DataFrame
        data = _pd.DataFrame(data={**bench, **strategy})

    # Apply rounding if specified
    if round_vals is not None:
        return _np.round(data, round_vals)

    return data


def monthly_returns(returns, eoy=True, compounded=True, prepare_returns=True):
    """
    Calculate monthly returns in a pivot table format.

    This function creates a matrix showing returns for each month across
    different years, making it easy to identify seasonal patterns and
    compare performance across time periods.

    Args:
        returns (pd.Series or pd.DataFrame): Return series to analyze
        eoy (bool): Whether to include end-of-year totals (default: True)
        compounded (bool): Whether to compound returns (default: True)
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        pd.DataFrame: Monthly returns matrix with years as rows and months
                     as columns. If eoy=True, includes 'EOY' column with
                     annual returns.

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02],
        ...                    index=pd.date_range('2023-01-01', periods=5, freq='M'))
        >>> monthly_rets = monthly_returns(returns)
        >>> print(monthly_rets)
    """
    # Handle DataFrame input by selecting appropriate column
    if isinstance(returns, _pd.DataFrame):
        warn(
            "Pandas DataFrame was passed (Series expected). "
            "Only first column will be used."
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Store original returns for end-of-year calculations
    original_returns = returns.copy()

    # Group returns by month-year and aggregate
    returns = _pd.DataFrame(
        _utils.group_returns(returns, returns.index.strftime("%Y-%m-01"), compounded)
    )

    # Set up DataFrame structure
    returns.columns = ["Returns"]
    returns.index = _pd.to_datetime(returns.index)

    # Extract year and month for pivot table
    returns["Year"] = returns.index.strftime("%Y")
    returns["Month"] = returns.index.strftime("%b")

    # Create pivot table with years as rows and months as columns
    returns = returns.pivot(index="Year", columns="Month", values="Returns").fillna(0)

    # Ensure all months are present in the DataFrame
    for month in [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # Order columns by calendar month
    returns = returns[
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    ]

    # Add end-of-year totals if requested
    if eoy:
        returns["eoy"] = _utils.group_returns(
            original_returns, original_returns.index.year, compounded=compounded  # type: ignore
        ).values

    # Format column names to uppercase
    returns.columns = map(lambda x: str(x).upper(), returns.columns)  # type: ignore
    returns.index.name = None

    return returns


def drawdown_details(drawdown):
    """
    Calculate detailed drawdown statistics for each drawdown period.

    This function analyzes a drawdown series to provide comprehensive statistics
    for each individual drawdown period, including start/end dates, duration,
    maximum drawdown, and 99th percentile drawdown.

    Args:
        drawdown (pd.Series or pd.DataFrame): Drawdown series to analyze

    Returns:
        pd.DataFrame: Detailed drawdown statistics with columns:
            - start: Start date of drawdown period
            - valley: Date of maximum drawdown
            - end: End date of drawdown period
            - days: Duration in days
            - max drawdown: Maximum drawdown percentage
            - 99% max drawdown: 99th percentile drawdown (excludes outliers)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> dd_series = to_drawdown_series(returns)
        >>> dd_details = drawdown_details(dd_series)
        >>> print(dd_details)
    """

    def _drawdown_details(drawdown):
        """
        Calculate drawdown details for a single drawdown series.

        This internal function processes a single drawdown series to extract
        detailed statistics about each drawdown period.
        """
        # Mark periods with no drawdown (drawdown = 0)
        no_dd = drawdown == 0

        # Extract drawdown start dates (first date of each drawdown period)
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts.values].index)

        # Extract drawdown end dates (last date of each drawdown period)
        ends = no_dd & (~no_dd).shift(1)
        ends = ends.shift(-1, fill_value=False)
        ends = list(ends[ends.values].index)

        # Return empty DataFrame if no drawdowns found
        if not starts:
            return _pd.DataFrame(
                index=[],
                columns=(
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ),
            )

        # Handle edge case: drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # Handle edge case: series ends in a drawdown
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # Build detailed statistics for each drawdown period
        data = []
        for i, _ in enumerate(starts):
            # Extract drawdown for this period
            dd = drawdown[starts[i]:ends[i]]

            # Calculate 99% drawdown (excluding outliers)
            clean_dd = -remove_outliers(-dd, 0.99)

            # Compile statistics for this drawdown period
            data.append(
                (
                    starts[i],                          # Start date
                    dd.idxmin(),                       # Valley date (max drawdown)
                    ends[i],                           # End date
                    (ends[i] - starts[i]).days + 1,   # Duration in days
                    dd.min() * 100,                    # Max drawdown %
                    clean_dd.min() * 100,              # 99% max drawdown %
                )
            )

        # Create DataFrame with results
        df = _pd.DataFrame(
            data=data,
            columns=(
                "start",
                "valley",
                "end",
                "days",
                "max drawdown",
                "99% max drawdown",
            ),
        )

        # Format data types
        df["days"] = df["days"].astype(int)
        df["max drawdown"] = df["max drawdown"].astype(float)
        df["99% max drawdown"] = df["99% max drawdown"].astype(float)

        # Format dates as strings
        df["start"] = df["start"].dt.strftime("%Y-%m-%d")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d")
        df["valley"] = df["valley"].dt.strftime("%Y-%m-%d")

        return df

    # Handle DataFrame input by processing each column separately
    if isinstance(drawdown, _pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return safe_concat(_dfs, axis=1)

    return _drawdown_details(drawdown)

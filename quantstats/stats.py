#!/usr/bin/env python
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

from math import ceil as _ceil
from math import sqrt as _sqrt
from warnings import warn

import numpy as _np
import pandas as _pd
from scipy.stats import linregress as _linregress
from scipy.stats import norm as _norm

from . import utils as _utils
from ._compat import safe_concat, safe_resample
from .utils import validate_input

# Type aliases for common types (Python 3.10+ syntax)
Returns = _pd.Series | _pd.DataFrame
"""Type alias for returns data: can be a pandas Series or DataFrame."""

# ======== STATS ========


def pct_rank(prices: _pd.Series, window: int = 60) -> _pd.Series:
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


def compsum(returns: Returns) -> Returns:
    """
    Calculate rolling compounded returns (cumulative product).

    This function computes the cumulative compounded returns by adding 1
    to each return, taking the cumulative product, and subtracting 1.

    Args:
        returns: Series or DataFrame of returns

    Returns:
        Cumulative compounded returns (same type as input)

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> cumulative = compsum(returns)
        >>> print(cumulative)
    """
    # Add 1 to convert returns to growth factors, then cumulative product.
    # Gaps are treated as flat here: an equity curve has to carry a value
    # through a missing observation, even though the statistics built on the
    # raw returns exclude it.
    return returns.fillna(0).add(1).cumprod(axis=0) - 1


def _paired_observations(returns, benchmark):
    """
    Restrict a strategy and its benchmark to the dates both were observed.

    Joint estimators (covariance, regression) need matched pairs. numpy and
    scipy propagate NaN through the whole calculation, so one missing day on
    either side is enough to make beta, alpha or R-squared undefined.

    Args:
        returns: Strategy return series
        benchmark: Benchmark return series

    Returns:
        tuple: (returns, benchmark) covering only the shared observations
    """
    paired = _pd.DataFrame({"returns": returns, "benchmark": benchmark}).dropna()
    return paired["returns"], paired["benchmark"]


def comp(returns: Returns) -> _pd.Series | float:
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


def distribution(
    returns: Returns,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> dict:
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
        # safe_resample() translates the frequency alias; "ME"/"QE"/"YE" only
        # exist in pandas 2.2+, so resampling on them directly breaks older
        # supported versions.
        "Weekly": get_outliers(safe_resample(daily, "W-MON", apply_fnc)),
        "Monthly": get_outliers(safe_resample(daily, "ME", apply_fnc)),
        "Quarterly": get_outliers(safe_resample(daily, "QE", apply_fnc)),
        "Yearly": get_outliers(safe_resample(daily, "YE", apply_fnc)),
    }


def expected_return(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
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
    # Missing observations are excluded rather than counted as 1.0 growth:
    # np.prod would return NaN for the whole series, and len() would stretch
    # the exponent over periods that were never observed.
    observed = returns.count()
    return _np.nanprod(1 + returns, axis=0) ** (1 / observed) - 1


def geometric_mean(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float:
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


def ghpr(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float:
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


def outliers(returns: Returns, quantile: float = 0.95) -> Returns:
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    # Filter returns above the specified quantile and remove NaN values
    return returns[returns > returns.quantile(quantile)].dropna(how="all")


def remove_outliers(returns: Returns, quantile: float = 0.95) -> Returns:
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


def best(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
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


def worst(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
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


def consecutive_wins(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> int:
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns and convert to boolean (positive = True)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) > 0

    # Count consecutive True values and return maximum
    return _utils._count_consecutive(returns).max()


def consecutive_losses(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> int:
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Aggregate returns and convert to boolean (negative = True)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0

    # Count consecutive True values and return maximum
    return _utils._count_consecutive(returns).max()


def exposure(
    returns: Returns,
    prepare_returns: bool = True,
) -> float | _pd.Series:
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
        observed = int((~_np.isnan(ret)).sum())
        if observed == 0:
            return 0.0
        ex = len(ret[(~_np.isnan(ret)) & (ret != 0)]) / observed
        # Round up to nearest percent
        return _ceil(ex * 100) / 100

    # Handle DataFrame input by calculating exposure for each column
    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return _pd.Series(_df)

    return _exposure(returns)


def win_rate(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float | _pd.Series:
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """

    def _win_rate(series):
        """
        Calculate win rate for a single return series.

        Handles edge cases like no non-zero returns and provides
        error handling for calculation issues.
        """
        try:
            # Drop gaps first: NaN != 0 evaluates True, so missing
            # observations would otherwise sit in the denominator.
            series = series.dropna()
            # Filter out zero returns (periods with no trading)
            non_zero_returns = series[series != 0]
            if len(non_zero_returns) == 0:
                warn(
                    "No non-zero returns found for win rate calculation, returning 0.0"
                )
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


def avg_return(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
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


def avg_win(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
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


def avg_loss(
    returns: Returns,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
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


def volatility(
    returns: Returns,
    periods: int = 252,
    annualize: bool = True,
    prepare_returns: bool = True,
) -> float | _pd.Series:
    """
    Calculate volatility (standard deviation) of returns.

    This function computes the volatility of returns, which measures the
    degree of variation in returns over time. Higher volatility indicates
    more uncertainty and risk.

    Args:
        returns: Return series or DataFrame to analyze
        periods: Number of periods per year for annualization (default: 252)
        annualize: Whether to annualize the volatility (default: True)
        prepare_returns: Whether to prepare returns first (default: True)

    Returns:
        Volatility (annualized if annualize=True)

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
    returns: Returns,
    rolling_period: int = 126,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> _pd.Series:
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
        # This used to pass `rolling_period` positionally into the `rf` slot.
        # It was harmless only because the old inspect.stack() exclusion list
        # stopped rf being applied for this function; spelled out, there is no
        # risk-free adjustment here at all.
        returns = _utils._prepare_returns(returns)

    # Calculate rolling standard deviation and annualize
    return returns.rolling(rolling_period).std() * _np.sqrt(periods_per_year)


def implied_volatility(
    returns: Returns,
    periods: int = 252,
    annualize: bool = True,
) -> float | _pd.Series:
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


def autocorr_penalty(
    returns: Returns,
    prepare_returns: bool = False,
) -> float:
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

    # Gaps carry no autocorrelation information, and corrcoef propagates
    # any NaN straight through.
    returns = returns.dropna()

    num = len(returns)

    # corrcoef needs at least two points to be defined; below that it returns
    # NaN with a RuntimeWarning, so fall back to the neutral penalty.
    if num < 2:
        return 1.0

    # Calculate autocorrelation coefficient between consecutive returns.
    # Constant (zero-variance) returns divide by a zero standard deviation
    # here, so suppress the warning and handle the resulting NaN below.
    with _np.errstate(invalid="ignore", divide="ignore"):
        coef = _np.abs(_np.corrcoef(returns[:-1], returns[1:])[0, 1])

    # A NaN coefficient would otherwise propagate silently through the sum.
    if _np.isnan(coef):
        return 1.0

    # Vectorized calculation instead of list comprehension
    x = _np.arange(1, num)
    # Calculate weighted correlation effects over time
    corr = ((num - x) / num) * (coef**x)

    # Return penalty factor (square root of 1 + 2 * sum of correlations)
    return _np.sqrt(1 + 2 * _np.sum(corr))


# ======= METRICS =======


def sharpe(
    returns: Returns,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | _pd.Series:
    """
    Calculate the Sharpe ratio of excess returns.

    The Sharpe ratio measures risk-adjusted returns by dividing excess returns
    (returns - risk-free rate) by the standard deviation of returns.
    Higher values indicate better risk-adjusted performance.

    Args:
        returns: Return series or DataFrame to analyze
        rf: Risk-free rate (annualized if periods specified, default: 0.0)
        periods: Periods per year for annualization (default: 252)
        annualize: Whether to annualize the ratio (default: True)
        smart: Whether to apply autocorrelation penalty (default: False)

    Returns:
        Sharpe ratio (float for Series input, Series for DataFrame input)

    Raises:
        ValueError: If rf is non-zero but periods is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sharpe_ratio = sharpe(returns, rf=0.02)
        >>> print(f"Sharpe ratio: {sharpe_ratio:.4f}")
    """
    validate_input(returns)

    # Validate parameters for risk-free rate handling
    if _utils._rf_is_nonzero(rf) and periods is None:
        raise ValueError(
            "periods parameter is required when risk-free rate (rf) is non-zero. "
            "This is needed to properly annualize the risk-free rate."
        )

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


def smart_sharpe(
    returns: Returns,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = True,
) -> float | _pd.Series:
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
    returns: Returns,
    rf: float = 0.0,
    rolling_period: int = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> _pd.Series:
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
        Exception: If rf != 0 and periods_per_year is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_sharpe_ratio = rolling_sharpe(returns, rolling_period=3)
        >>> print(rolling_sharpe_ratio)
    """
    # Validate parameters for risk-free rate handling
    if _utils._rf_is_nonzero(rf) and periods_per_year is None:
        raise Exception("Must provide periods_per_year if rf != 0")

    if prepare_returns:
        # The third argument is the number of periods per year used to
        # de-annualize rf, not the window length. Passing rolling_period here
        # subtracted a rate de-annualized over the window instead of the year,
        # which at the defaults overcharged rf by exactly 2x.
        returns = _utils._prepare_returns(returns, rf, periods_per_year)

    # Calculate rolling mean and standard deviation
    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()

    # Annualize if requested
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


def sortino(
    returns: Returns,
    rf: float = 0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | _pd.Series:
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
    if _utils._rf_is_nonzero(rf) and periods is None:
        raise ValueError(
            "periods parameter is required when risk-free rate (rf) is non-zero. "
            "This is needed to properly annualize the risk-free rate."
        )

    # Prepare returns (subtract risk-free rate if applicable)
    returns = _utils._prepare_returns(returns, rf, periods)

    # Calculate downside deviation (only negative returns)
    downside = _np.sqrt((returns[returns < 0] ** 2).sum() / returns.count())

    # Apply autocorrelation penalty if smart mode enabled
    if smart:
        # penalize sortino with auto correlation
        downside = downside * autocorr_penalty(returns)

    # Calculate base Sortino ratio
    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(downside, _pd.Series):
        res = returns.mean() / downside.replace(0, _np.nan)
    else:
        if downside == 0:
            res = _np.nan
        else:
            res = returns.mean() / downside

    # Annualize if requested
    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sortino(
    returns: Returns,
    rf: float = 0,
    periods: int = 252,
    annualize: bool = True,
) -> float | _pd.Series:
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
    returns: Returns,
    rf: float = 0,
    rolling_period: int = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    **kwargs,
) -> _pd.Series:
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
        Exception: If rf != 0 and periods_per_year is None

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> rolling_sortino_ratio = rolling_sortino(returns, rolling_period=3)
        >>> print(rolling_sortino_ratio)
    """
    # Validate parameters for risk-free rate handling
    if _utils._rf_is_nonzero(rf) and periods_per_year is None:
        raise Exception("Must provide periods_per_year if rf != 0")

    if kwargs.get("prepare_returns", True):
        # The third argument is the number of periods per year used to
        # de-annualize rf, not the window length. Passing rolling_period here
        # subtracted a rate de-annualized over the window instead of the year,
        # which at the defaults overcharged rf by exactly 2x.
        returns = _utils._prepare_returns(returns, rf, periods_per_year)

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


def adjusted_sortino(
    returns: Returns,
    rf: float = 0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | _pd.Series:
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
    series: Returns,
    rf: float = 0.0,
    base: str = "sharpe",
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float:
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
        annualize (bool): Deprecated and ignored. A probability has no
            annualized form (default: False)
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
    # Calculate the base ratio on excess returns, the same way sharpe() and
    # sortino() handle rf. rf used to be ignored here and then subtracted from
    # the finished ratio, which mixed an annualized rate into a per-period
    # statistic.
    if base.lower() == "sharpe":
        base = sharpe(series, rf=rf, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "sortino":
        base = sortino(series, rf=rf, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "adjusted_sortino":
        base = adjusted_sortino(
            series, rf=rf, periods=periods, annualize=False, smart=smart
        )
    else:
        raise ValueError(
            f"Invalid metric '{base}'. Must be one of: 'sharpe', 'sortino', or 'adjusted_sortino'"
        )

    # Calculate higher moments for adjustment. kurtosis() returns *excess*
    # kurtosis; the estimator below is defined on raw kurtosis (3 under
    # normality), so convert rather than subtracting 3 a second time.
    skew_no = skew(series, prepare_returns=False)
    kurtosis_no = kurtosis(series, prepare_returns=False) + 3

    n = len(series)

    # Standard error of the ratio (Bailey & Lopez de Prado, 2012). Reduces to
    # Lo's (1 + SR^2 / 2) / (n - 1) for normally distributed returns.
    sigma_sr = _np.sqrt(
        (1 - (skew_no * base) + (((kurtosis_no - 1) / 4) * base**2)) / (n - 1)
    )

    # Probability that the true ratio is greater than zero
    psr = _norm.cdf(base / sigma_sr)

    return psr


def probabilistic_sharpe_ratio(
    series: Returns,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    This function computes the PSR, which represents the probability that
    the observed Sharpe ratio is statistically greater than a benchmark.
    It accounts for higher moments to provide more robust estimates.

    Args:
        series (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year for annualization (default: 252)
        annualize (bool): Deprecated and ignored. A probability has no
            annualized form (default: False)
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
        annualize (bool): Deprecated and ignored. A probability has no
            annualized form (default: False)
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
        annualize (bool): Deprecated and ignored. A probability has no
            annualized form (default: False)
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


def omega(
    returns: Returns,
    rf: float = 0.0,
    required_return: float = 0.0,
    periods: int = 252,
) -> float:
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
        warn(
            "Insufficient data for omega ratio calculation (need at least 2 returns), returning NaN"
        )
        return _np.nan

    # Validate required return parameter
    if required_return <= -1:
        warn(
            f"Invalid required_return ({required_return}) for omega ratio, must be > -1, returning NaN"
        )
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    # Prepare returns and resample to specified frequency. `rf` is accepted
    # for API compatibility but is deliberately not subtracted here, matching
    # long-standing behaviour.
    returns = safe_resample(
        _utils._prepare_returns(returns, rf, apply_rf=False), resolution, "sum"
    )

    # Calculate absolute sum of negative returns (pain)
    downside = abs(returns[returns < 0].sum())

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(downside, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        return returns.sum() / downside.replace(0, _np.nan)
    else:
        # Series input - scalar division
        if downside == 0:
            return _np.nan
        return returns.sum() / downside


def cagr(
    returns: Returns,
    rf: float = 0.0,
    compounded: bool = True,
    periods: int = 252,
) -> float | _pd.Series:
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

    # `rf` is accepted for API compatibility but is not subtracted here,
    # matching long-standing behaviour.
    total = _utils._prepare_returns(returns, rf, apply_rf=False)

    # Calculate total return
    if compounded:
        total = comp(total)
    else:
        total = _np.sum(total, axis=0)

    # Calculate time period in years using trading periods
    # This is consistent with how Sharpe, Sortino, and other metrics
    # handle annualization in quantstats
    years = returns.count() / periods

    # Geometric growth rate. Terminal wealth below zero is reachable with
    # compounded=False once summed returns pass -100%; it has no real-valued
    # growth rate, so report NaN. Taking abs() here used to turn a total
    # wipeout into a positive CAGR.
    wealth = _np.asarray(total + 1.0, dtype=float)
    with _np.errstate(invalid="ignore"):
        res = _np.where(wealth < 0, _np.nan, _np.abs(wealth) ** (1.0 / years) - 1)
    if res.ndim == 0:
        res = float(res)

    # Handle DataFrame input
    if isinstance(returns, _pd.DataFrame):
        res = _pd.Series(res)
        res.index = returns.columns

    return res


def rar(returns, rf=0.0, periods=252, compounded=True):
    """
    Calculate the Risk-Adjusted Return (RAR).

    RAR is calculated as CAGR divided by exposure, taking into account
    the time the strategy was actually invested. This provides a more
    accurate measure of returns adjusted for actual market participation.

    Args:
        returns (pd.Series): Return series to analyze
        rf (float): Risk-free rate (annualized, default: 0.0)
        periods (int): Periods per year, used to de-annualize `rf`
            (default: 252)
        compounded (bool): Whether to compound returns (default: True).
            Set to False for intraday or other non-compounded return streams.

    Returns:
        float: Risk-adjusted return

    Example:
        >>> returns = pd.Series([0.01, 0.00, 0.03, 0.00, 0.02])
        >>> rar_value = rar(returns)
        >>> print(f"Risk-adjusted return: {rar_value:.4f}")
    """
    # Prepare returns (subtract risk-free rate if applicable).
    #
    # `periods` has to reach _prepare_returns: without it the *annual* rf is
    # subtracted from every single period, so at a daily frequency a 5% rate
    # removes 5% per day and the series is wiped out. That reported a
    # risk-adjusted return of -100% for any call with a non-zero rf.
    returns = _utils._prepare_returns(returns, rf, periods)

    # Calculate CAGR and divide by exposure time
    return cagr(returns, compounded=compounded) / exposure(returns)


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


def calmar(
    returns: Returns,
    prepare_returns: bool = True,
    compounded: bool = True,
    periods: int = 252,
) -> float:
    """
    Calculate the Calmar ratio (CAGR / Maximum Drawdown).

    The Calmar ratio measures risk-adjusted returns by dividing the CAGR
    by the absolute value of the maximum drawdown. It provides insight
    into returns relative to the worst-case scenario.

    Args:
        returns (pd.Series): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)
        compounded (bool): Whether to compound returns (default: True).
            Set to False for intraday or other non-compounded return streams.
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
    cagr_ratio = cagr(returns, compounded=compounded, periods=periods)
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
    ulcer = ulcer_index(returns)

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(ulcer, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        return (comp(returns) - rf) / ulcer.replace(0, _np.nan)
    else:
        # Series input - scalar division
        if ulcer == 0:
            return _np.nan
        return (comp(returns) - rf) / ulcer


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
    std_returns = returns.std()

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(std_returns, _pd.Series):
        # DataFrame input - element-wise operations
        pitfall = -cvar(dd) / std_returns.replace(0, _np.nan)
        denominator = ulcer_index(returns) * pitfall
        return (returns.sum() - rf) / denominator.replace(0, _np.nan)
    else:
        # Series input - scalar operations
        if std_returns == 0:
            return _np.nan

        cvar_val = cvar(dd)
        ulcer_val = ulcer_index(returns)

        # Handle cases where these might return Series/array
        if hasattr(cvar_val, "__len__") and len(cvar_val) == 1:
            cvar_val = float(
                cvar_val.iloc[0] if hasattr(cvar_val, "iloc") else cvar_val[0]
            )
        if hasattr(ulcer_val, "__len__") and len(ulcer_val) == 1:
            ulcer_val = float(
                ulcer_val.iloc[0] if hasattr(ulcer_val, "iloc") else ulcer_val[0]
            )

        pitfall = -cvar_val / std_returns
        denominator = ulcer_val * pitfall

        if denominator == 0:
            return _np.nan
        return (returns.sum() - rf) / denominator


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
    return ((1 - wins) / (1 + wins)) ** returns.count()


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


def value_at_risk(
    returns: Returns,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | _pd.Series:
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


def _gaussian_expected_shortfall(mu: float, sd: float, alpha: float) -> float:
    """
    Closed-form expected shortfall of a normal distribution.

    Returns E[X | X <= VaR_alpha] for X ~ N(mu, sd), which is the estimator
    that belongs with value_at_risk()'s variance-covariance method.
    """
    if sd == 0 or _np.isnan(sd):
        return mu
    return mu - sd * _norm.pdf(_norm.ppf(alpha)) / alpha


def conditional_value_at_risk(
    returns: Returns,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
    method: str = "parametric",
) -> float | _pd.Series:
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
        method (str): "parametric" (default) matches value_at_risk() and uses
            the closed-form normal expected shortfall. "historical" averages
            the observations at or below the empirical quantile, which
            captures fat tails but needs enough data to be meaningful.

    Returns:
        float: Conditional Value at Risk (expected loss beyond VaR)

    Note:
        Before 0.0.83 this took the threshold from the *parametric* VaR and
        then averaged the observations below it, mixing two estimators. When
        no observation fell below the threshold it returned the VaR itself,
        which overstates CVaR, since CVaR is by definition at least as severe
        as VaR. Both modes here are internally consistent, and an undefined
        tail now returns NaN rather than the VaR.

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> cvar_value = conditional_value_at_risk(returns, confidence=0.95)
        >>> print(f"95% CVaR: {cvar_value:.4f}")
    """
    if method not in ("parametric", "historical"):
        raise ValueError(f"method must be 'parametric' or 'historical', got {method!r}")

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Accept a confidence given as a percentage, matching value_at_risk()
    if confidence > 1:
        confidence = confidence / 100
    alpha = 1 - confidence

    def _cvar_of(series):
        series = series.dropna()
        if len(series) == 0:
            return _np.nan
        if method == "historical":
            tail = series[series <= series.quantile(alpha)]
            return tail.mean() if len(tail) > 0 else _np.nan
        return _gaussian_expected_shortfall(series.mean(), sigma * series.std(), alpha)

    if isinstance(returns, _pd.DataFrame):
        return _pd.Series({col: _cvar_of(returns[col]) for col in returns.columns})
    return _cvar_of(returns)


def cvar(returns, sigma=1, confidence=0.95, prepare_returns=True, method="parametric"):
    """
    Calculate the Conditional Value at Risk (CVaR).

    This is a shorthand function for conditional_value_at_risk() with
    the same parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)
        prepare_returns (bool): Whether to prepare returns first (default: True)
        method (str): "parametric" (default) or "historical"

    Returns:
        float: Conditional Value at Risk
    """
    return conditional_value_at_risk(
        returns, sigma, confidence, prepare_returns, method
    )


def expected_shortfall(returns, sigma=1, confidence=0.95, method="parametric"):
    """
    Calculate the Expected Shortfall (ES), also known as CVaR.

    This is a shorthand function for conditional_value_at_risk() with
    the same parameters and functionality.

    Args:
        returns (pd.Series): Return series to analyze
        sigma (float): Volatility multiplier (default: 1)
        confidence (float): Confidence level (0.95 = 95%, default: 0.95)
        method (str): "parametric" (default) or "historical"

    Returns:
        float: Expected Shortfall
    """
    return conditional_value_at_risk(returns, sigma, confidence, method=method)


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
    upper_quantile = returns.quantile(cutoff)
    lower_quantile = returns.quantile(1 - cutoff)

    # Handle edge cases: NaN values or zero denominator
    # Check if result is a Series (DataFrame input) or scalar (Series input)
    if isinstance(upper_quantile, _pd.Series):
        # Handle DataFrame input - apply element-wise
        result = _pd.Series(index=upper_quantile.index, dtype=float)
        for col in upper_quantile.index:
            if (
                _pd.isna(upper_quantile[col])
                or _pd.isna(lower_quantile[col])
                or lower_quantile[col] == 0
            ):
                result[col] = _np.nan
            else:
                result[col] = abs(upper_quantile[col] / lower_quantile[col])
        return result
    else:
        # Handle Series input - scalar values
        if _pd.isna(upper_quantile) or _pd.isna(lower_quantile) or lower_quantile == 0:
            return _np.nan
        return abs(upper_quantile / lower_quantile)


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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Calculate ratio of average win to absolute average loss
    avg_loss_val = avg_loss(returns)
    avg_win_val = avg_win(returns)

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(avg_loss_val, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        # Use abs() before replace to handle negative values properly
        return avg_win_val / abs(avg_loss_val).replace(0, _np.nan)
    else:
        # Series input - scalar division
        if avg_loss_val == 0:
            return _np.nan
        return avg_win_val / abs(avg_loss_val)


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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    return payoff_ratio(returns, prepare_returns)


def profit_ratio(returns, prepare_returns=True):
    """
    Calculate the profit ratio (win ratio / loss ratio).

    This function measures the ratio of win frequency to loss frequency,
    providing insight into the consistency of profitable periods.

    Args:
        returns (pd.Series or pd.DataFrame): Return series to analyze
        prepare_returns (bool): Whether to prepare returns first (default: True)

    Returns:
        float or pd.Series: Profit ratio

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> profit_r = profit_ratio(returns)
        >>> print(f"Profit ratio: {profit_r:.4f}")
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    def _profit_ratio(ret):
        # Separate wins and losses
        wins = ret[ret >= 0]
        loss = ret[ret < 0]

        # Handle edge cases
        win_count = len(wins)
        loss_count = len(loss)

        if win_count == 0:
            return 0.0
        if loss_count == 0:
            return _np.nan

        # Calculate win and loss ratios
        win_ratio = abs(wins.mean() / win_count) if win_count > 0 else 0
        loss_ratio = abs(loss.mean() / loss_count) if loss_count > 0 else 0

        if loss_ratio == 0:
            return _np.nan
        return win_ratio / loss_ratio

    # Handle DataFrame by applying to each column
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(_profit_ratio)

    return _profit_ratio(returns)


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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
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
            return 0.0 if wins_sum == 0 else float("inf")
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
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
    Note:
        Computed from the return series, not from discrete trades. A single
        multi-day trade spanning three up days and two down days counts as
        three wins and two losses here. This is well defined and useful for
        systematic strategies with regular rebalancing, but it will not match
        trade-level statistics from a discretionary trading journal. See
        "Period-Based vs Trade-Based Metrics" in the README.
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
    positive_mean = returns[returns >= 0].mean()
    quantile_val = returns.quantile(quantile)  # Series for DataFrame, scalar for Series

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(positive_mean, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        return quantile_val / positive_mean.replace(0, _np.nan)
    else:
        # Series input - scalar division
        if _pd.isna(positive_mean) or positive_mean == 0:
            return _np.nan
        return quantile_val / positive_mean


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
    negative_mean = returns[returns < 0].mean()
    quantile_val = returns.quantile(quantile)  # Series for DataFrame, scalar for Series

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(negative_mean, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        return quantile_val / negative_mean.replace(0, _np.nan)
    else:
        # Series input - scalar division
        if _pd.isna(negative_mean) or negative_mean == 0:
            return _np.nan
        return quantile_val / negative_mean


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

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(max_dd, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        return abs(total_returns) / abs(max_dd).replace(0, _np.nan)
    else:
        # Series input - scalar division
        if max_dd == 0:
            return _np.nan
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
    std = returns.std()

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(std, _pd.Series):
        # DataFrame input - element-wise division with zero protection
        return returns.mean() / std.replace(0, _np.nan)
    else:
        # Series input - scalar division
        if std == 0:
            return _np.nan
        return returns.mean() / std


def _get_baseline_value(prices, from_returns):
    """
    Determine the appropriate baseline value for drawdown calculations.

    The baseline is the equity held immediately before the first observation,
    so that a loss in the very first period is not hidden by treating that
    period's close as the running peak.

    Returns rebuilt by _prepare_prices() are priced as base * (1 + compsum)
    with base 1.0, so 1.0 is exactly the equity they started from. A series
    that was already prices carries no such earlier point: its first
    observation *is* the start of the record, and any other baseline invents
    a peak the portfolio never reached. The previous implementation guessed
    from the price level (>1000 -> 1e5, >10 -> 100.0), which reported a 50%
    drawdown for a $50 stock and 95% for a $5000 one.

    Args:
        prices (pd.Series | pd.DataFrame): Price series
        from_returns (bool | pd.Series): Whether the prices were rebuilt from
            returns; per column for DataFrame input

    Returns:
        float | pd.Series: Baseline value(s) for drawdown calculations
    """
    if len(prices) == 0:
        return 1.0

    if isinstance(prices, _pd.DataFrame):
        if prices.shape[1] == 0:
            return 1.0  # Default baseline for empty DataFrame with no columns
        # Baseline per column, since columns may be on different scales
        converted = _pd.Series(from_returns, index=prices.columns, dtype=bool)
        return prices.iloc[0].mask(converted, 1.0)

    # For a Series, the first value is the start of the record
    return 1.0 if bool(from_returns) else prices.iloc[0]


def max_drawdown(prices: Returns) -> float:
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

    # Record whether these were returns *before* the conversion, so the
    # baseline below knows if there is a known starting equity
    from_returns = _utils._looks_like_returns(prices)

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
    baseline_value = _get_baseline_value(prices, from_returns)

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

    # Record whether these were returns *before* the conversion
    from_returns = _utils._looks_like_returns(returns)

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
    baseline_value = _get_baseline_value(prices, from_returns)

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

    Returns the classic fixed-odds fraction f* = p - q/b, where p is the
    win rate, q = 1 - p, and b the payoff ratio (average win / average loss).
    The result is a fraction of capital, in a range a reader can act on.

    Note:
        0.0.82 to 0.0.83 divided this by the average-loss magnitude on the
        argument that a fraction which does not move when the return series
        is rescaled must be wrong. That quantity is the growth-optimal
        *leverage* for a per-period P&L, not the Kelly fraction: on daily
        returns it reaches double and triple digits, and reports showed
        figures like 3716% where this formula reads 21% (issue #552). The
        fixed-odds fraction depends only on the odds by construction, so
        its scale-invariance is a property, not a defect. Reverted in 0.0.84.
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    # Handle both Series (DataFrame input) and scalar (Series input) cases
    if isinstance(win_loss_ratio, _pd.Series):
        # DataFrame input - element-wise operations with zero/nan protection
        # Replace 0 and NaN values with NaN to avoid division issues
        win_loss_ratio_safe = win_loss_ratio.replace(0, _np.nan)
        return ((win_loss_ratio_safe * win_prob) - lose_prob) / win_loss_ratio_safe
    else:
        # Series input - scalar operations
        if win_loss_ratio == 0 or _pd.isna(win_loss_ratio):
            return _np.nan
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

    # Estimate over the dates on which both series were observed; linregress
    # returns NaN for the whole fit if either input contains a gap.
    paired_returns, paired_benchmark = _paired_observations(returns, benchmark)

    # Perform linear regression and extract correlation coefficient
    _, _, r_val, _, _ = _linregress(paired_returns, paired_benchmark)

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

    # Calculate active returns (returns - benchmark). The already-prepared
    # benchmark is used directly; preparing it a second time here re-ran the
    # price/return detection on data that had just been normalized.
    diff_rets = returns - benchmark

    # Calculate tracking error (standard deviation of active returns)
    std = diff_rets.std()

    # Return Information Ratio (active return / tracking error)
    if std != 0:
        return diff_rets.mean() / std
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

    # Estimate over the dates on which both series were observed. np.cov
    # propagates NaN, so a single gap on either side would otherwise reduce
    # beta and alpha to NaN, which the .fillna(0) below turns into a
    # confident-looking zero.
    returns, benchmark = _paired_observations(returns, benchmark)

    # Calculate covariance matrix between returns and benchmark
    matrix = _np.cov(returns, benchmark)

    # Calculate beta (sensitivity to benchmark movements)
    if matrix[1, 1] == 0:
        beta = _np.nan
    else:
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

    # Calculate rolling beta (protect against division by zero)
    beta = corr * std["returns"] / std["benchmark"].replace(0, _np.nan)

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
    # Normalize timezone for returns to ensure consistent comparisons
    # Convert to UTC if timezone-aware, then make naive
    # This must happen before prepare_returns to avoid issues
    if hasattr(returns.index, "tz") and returns.index.tz is not None:
        returns = returns.tz_convert("UTC").tz_localize(None)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Normalize benchmark timezone first if it's not a string
    if benchmark is not None and not isinstance(benchmark, str):
        if hasattr(
            benchmark.index
            if isinstance(benchmark, _pd.Series)
            else benchmark[benchmark.columns[0]].index,
            "tz",
        ):
            if isinstance(benchmark, _pd.Series) and benchmark.index.tz is not None:
                benchmark = benchmark.tz_convert("UTC").tz_localize(None)
            elif (
                isinstance(benchmark, _pd.DataFrame)
                and benchmark[benchmark.columns[0]].index.tz is not None
            ):
                for col in benchmark.columns:
                    benchmark[col] = benchmark[col].tz_convert("UTC").tz_localize(None)

    # Store original benchmark for proper aggregation
    # This preserves returns that may fall on non-trading days
    if isinstance(benchmark, str):
        benchmark_original = _utils.download_returns(benchmark)
    elif isinstance(benchmark, _pd.DataFrame):
        benchmark_original = benchmark[benchmark.columns[0]].copy()
    else:
        benchmark_original = benchmark.copy() if benchmark is not None else None

    # Normalize timezone for benchmark_original as well (in case it was downloaded)
    if (
        benchmark_original is not None
        and hasattr(benchmark_original.index, "tz")
        and benchmark_original.index.tz is not None
    ):
        benchmark_original = benchmark_original.tz_convert("UTC").tz_localize(None)

    # Prepare benchmark to match returns index for other calculations
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # Handle Series input
    if isinstance(returns, _pd.Series):
        # Aggregate returns and use original benchmark for aggregation
        # This ensures we don't lose benchmark returns on non-trading days
        if benchmark_original is not None:
            benchmark_agg = (
                _utils.aggregate_returns(benchmark_original, aggregate, compounded)
                * 100
            )
        else:
            benchmark_agg = (
                _utils.aggregate_returns(benchmark, aggregate, compounded) * 100
            )
        returns_agg = _utils.aggregate_returns(returns, aggregate, compounded) * 100

        # Create comparison DataFrame
        data = _pd.DataFrame(
            data={
                "Benchmark": benchmark_agg,
                "Returns": returns_agg,
            }
        )

        # Calculate performance multiplier and win/loss indicator
        # Protect against division by zero in benchmark
        data["Multiplier"] = data["Returns"] / data["Benchmark"].replace(0, _np.nan)
        data["Won"] = _np.where(data["Returns"] >= data["Benchmark"], "+", "-")

    # Handle DataFrame input (multiple strategies)
    elif isinstance(returns, _pd.DataFrame):
        # Aggregate benchmark using original data to preserve non-trading day returns
        if benchmark_original is not None:
            bench = {
                "Benchmark": _utils.aggregate_returns(
                    benchmark_original, aggregate, compounded
                )
                * 100
            }
        else:
            bench = {
                "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
                * 100
            }

        # Aggregate each strategy column
        strategy = {
            "Returns_" + str(i): _utils.aggregate_returns(
                returns[col], aggregate, compounded
            )
            * 100
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
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # Order columns by calendar month
    returns = returns[
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    ]

    # Add end-of-year totals if requested
    if eoy:
        returns["eoy"] = _utils.group_returns(
            original_returns,
            original_returns.index.year,
            compounded=compounded,  # type: ignore
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
            dd = drawdown[starts[i] : ends[i]]

            # Calculate 99% drawdown (excluding outliers)
            clean_dd = -remove_outliers(-dd, 0.99)

            # Compile statistics for this drawdown period
            data.append(
                (
                    starts[i],  # Start date
                    dd.idxmin(),  # Valley date (max drawdown)
                    ends[i],  # End date
                    (ends[i] - starts[i]).days + 1,  # Duration in days
                    dd.min() * 100,  # Max drawdown %
                    clean_dd.min() * 100,  # 99% max drawdown %
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


# ======== MONTE CARLO ========


def montecarlo(returns, sims=1000, bust=None, goal=None, seed=None):
    """
    Run Monte Carlo simulation by shuffling returns.

    This function creates multiple simulated return paths by randomly
    shuffling the historical returns. This preserves the return distribution
    while breaking any time-series dependencies, allowing probability-based
    risk assessment.

    Args:
        returns (pd.Series or pd.DataFrame): Daily returns (not prices)
        sims (int): Number of simulations to run (default: 1000)
        bust (float, optional): Drawdown threshold for "bust" probability
            (e.g., -0.1 for -10% drawdown)
        goal (float, optional): Return threshold for "goal" probability
            (e.g., 1.0 for +100% return)
        seed (int, optional): Random seed for reproducibility

    Returns:
        MonteCarloResult: Object containing simulation results with:
            - .data: DataFrame of all simulation paths
            - .stats: Terminal value statistics (min, max, mean, median, std)
            - .maxdd: Max drawdown statistics across simulations
            - .bust_probability: Probability of exceeding bust threshold
            - .goal_probability: Probability of reaching goal threshold
            - .plot(): Visualize simulation paths

    Example:
        >>> import quantstats as qs
        >>> returns = qs.utils.download_returns("SPY")
        >>> mc = qs.stats.montecarlo(returns, sims=1000, bust=-0.2, goal=0.5)
        >>> print(mc.stats)
        >>> print(f"Bust probability: {mc.bust_probability:.1%}")
        >>> print(f"Goal probability: {mc.goal_probability:.1%}")
        >>> mc.plot()
    """
    from ._montecarlo import run_montecarlo

    # Validate and prepare returns
    validate_input(returns)
    returns = _utils._prepare_returns(returns)

    # Handle DataFrame by processing first column (or extend for multi-column)
    if isinstance(returns, _pd.DataFrame):
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            # For multi-column DataFrame, run on first column
            # Future: could return dict of MonteCarloResults
            returns = returns.iloc[:, 0]

    return run_montecarlo(returns, sims=sims, bust=bust, goal=goal, seed=seed)


def montecarlo_sharpe(returns, sims=1000, rf=0, periods=252, seed=None):
    """
    Distribution of Sharpe ratios across Monte Carlo simulations.

    This function runs Monte Carlo simulations and calculates the Sharpe
    ratio for each simulated path, providing a distribution of possible
    Sharpe ratio outcomes.

    Args:
        returns (pd.Series): Daily returns
        sims (int): Number of simulations (default: 1000)
        rf (float): Risk-free rate (default: 0)
        periods (int): Periods per year for annualization (default: 252)
        seed (int, optional): Random seed for reproducibility

    Returns:
        dict: Statistics of Sharpe ratio distribution including
            min, max, mean, median, std, percentile_5, percentile_95

    Example:
        >>> sharpe_dist = qs.stats.montecarlo_sharpe(returns, sims=1000)
        >>> print(f"Expected Sharpe: {sharpe_dist['mean']:.2f}")
        >>> print(f"Sharpe range: {sharpe_dist['percentile_5']:.2f} to "
        ...       f"{sharpe_dist['percentile_95']:.2f}")
    """
    from ._montecarlo import run_montecarlo

    validate_input(returns)
    returns = _utils._prepare_returns(returns)

    if isinstance(returns, _pd.DataFrame):
        returns = returns.iloc[:, 0]

    mc = run_montecarlo(returns, sims=sims, seed=seed)

    # Calculate Sharpe for each simulation path
    sharpe_values = []
    for col in mc.data.columns:
        # Convert cumulative returns back to simple returns
        cumret = mc.data[col] + 1
        sim_returns = cumret.pct_change().dropna()
        if len(sim_returns) > 0 and sim_returns.std() > 0:
            excess = sim_returns.mean() - rf / periods
            sharpe_val = excess / sim_returns.std() * _np.sqrt(periods)
            sharpe_values.append(sharpe_val)

    sharpe_series = _pd.Series(sharpe_values)
    return {
        "min": sharpe_series.min(),
        "max": sharpe_series.max(),
        "mean": sharpe_series.mean(),
        "median": sharpe_series.median(),
        "std": sharpe_series.std(),
        "percentile_5": sharpe_series.quantile(0.05),
        "percentile_95": sharpe_series.quantile(0.95),
    }


def montecarlo_drawdown(returns, sims=1000, seed=None):
    """
    Distribution of maximum drawdowns across Monte Carlo simulations.

    This function runs Monte Carlo simulations and returns statistics
    about the distribution of maximum drawdowns across all paths.

    Args:
        returns (pd.Series): Daily returns
        sims (int): Number of simulations (default: 1000)
        seed (int, optional): Random seed for reproducibility

    Returns:
        dict: Statistics of max drawdown distribution including
            min, max, mean, median, std, percentile_5, percentile_95

    Example:
        >>> dd_dist = qs.stats.montecarlo_drawdown(returns, sims=1000)
        >>> print(f"Expected max drawdown: {dd_dist['mean']:.1%}")
        >>> print(f"Worst case (5th pct): {dd_dist['percentile_5']:.1%}")
    """
    mc = montecarlo(returns, sims=sims, seed=seed)
    return mc.maxdd


def montecarlo_cagr(returns, sims=1000, seed=None):
    """
    Distribution of CAGR across Monte Carlo simulations.

    This function runs Monte Carlo simulations and calculates the
    Compound Annual Growth Rate for each simulated path.

    Args:
        returns (pd.Series): Daily returns
        sims (int): Number of simulations (default: 1000)
        seed (int, optional): Random seed for reproducibility

    Returns:
        dict: Statistics of CAGR distribution including
            min, max, mean, median, std, percentile_5, percentile_95

    Example:
        >>> cagr_dist = qs.stats.montecarlo_cagr(returns, sims=1000)
        >>> print(f"Expected CAGR: {cagr_dist['mean']:.1%}")
    """
    from ._montecarlo import run_montecarlo

    validate_input(returns)
    returns = _utils._prepare_returns(returns)

    if isinstance(returns, _pd.DataFrame):
        returns = returns.iloc[:, 0]

    mc = run_montecarlo(returns, sims=sims, seed=seed)

    # Calculate CAGR for each simulation path
    n_periods = len(mc.data)
    years = n_periods / 252  # Assume daily data

    cagr_values = []
    for col in mc.data.columns:
        terminal = mc.data[col].iloc[-1]
        # CAGR = (1 + total_return)^(1/years) - 1
        if terminal > -1:  # Avoid invalid values
            cagr_val = (1 + terminal) ** (1 / years) - 1
            cagr_values.append(cagr_val)

    cagr_series = _pd.Series(cagr_values)
    return {
        "min": cagr_series.min(),
        "max": cagr_series.max(),
        "mean": cagr_series.mean(),
        "median": cagr_series.median(),
        "std": cagr_series.std(),
        "percentile_5": cagr_series.quantile(0.05),
        "percentile_95": cagr_series.quantile(0.95),
    }

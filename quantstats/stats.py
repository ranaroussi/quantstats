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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function

import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
import quantstats.tools as tools


# ======== STATS ========

def outliers(returns, quantile=.95):
    return returns[returns > returns.quantile(quantile)].copy()


def remove_outliers(returns, quantile=.95):
    return returns[returns < returns.quantile(quantile)].copy()


def best(returns, aggregate=None, compounded=True):
    returns = tools.cleanup_returns(returns)
    return tools.aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True):
    returns = tools.cleanup_returns(returns)
    return tools.aggregate_returns(returns, aggregate, compounded).min()


def consecutive_wins(returns, aggregate=None, compounded=True):
    returns = tools.cleanup_returns(returns)
    returns = 0 < tools.aggregate_returns(returns, aggregate, compounded)
    return tools.count_consecutive(returns)


def consecutive_losses(returns, aggregate=None, compounded=True):
    returns = tools.cleanup_returns(returns)
    returns = 0 > tools.aggregate_returns(returns, aggregate, compounded)
    return tools.count_consecutive(returns)


def exposure(returns):
    returns = tools.cleanup_returns(returns)
    return len(returns[(~np.isnan(returns)) & (returns != 0)]) / len(returns)


def win_rate(returns, aggregate=None, compounded=True):
    """
    Calculate the win ratio of a strategy

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        win rate (float)
    """

    returns = tools.cleanup_returns(returns)
    if aggregate:
        returns = tools.aggregate_returns(returns, aggregate, compounded)

    trades = len(returns[returns != 0])
    wins = len(returns[returns > 0])

    try:
        return wins / trades
    except Exception:
        return 0.


def avg_return(returns):
    """
    Calculate the average return/trade of a strategy

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        avg. return (float)
    """
    returns = tools.cleanup_returns(returns)

    try:
        return np.mean(returns[returns != 0])*1.
    except Exception:
        return 0.


def avg_win(returns):
    """
    Calculate the average win/trade of a strategy

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        avg. win (float)
    """
    returns = tools.cleanup_returns(returns)

    try:
        return np.mean(returns[returns > 0])*1.
    except Exception:
        return 0.


def avg_loss(returns):
    """
    Calculate the average loss/trade of a strategy

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        avg. loss (float)
    """
    returns = tools.cleanup_returns(returns)

    try:
        return np.mean(returns[returns < 0])*1.
    except Exception:
        return 0.


def volatility(returns, periods=252, annualize=True):
    std = tools.cleanup_returns(returns).std()

    if annualize:
        return std * np.sqrt(1 if periods is None else periods)

    return std


def implied_volatility(returns, periods=252):
    """ Returns pandas Series representing volatility of log returns """
    try:
        logret = tools.log_returns(returns)
        return logret.rolling(window=periods).std() * np.sqrt(periods)
    except Exception:
        return 0.


# ======= METRICS =======

def sharpe(returns, rf=0., periods=252, annualize=True):
    """
    Calculates the Sharpe ratio.

    If rf is non-zero, you must specify periods. In this case, rf is assumed
    to be expressed in yearly (annualized) terms.

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Frequency of returns (252 for daily, 12 for monthly)
    """

    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')

    returns = tools.cleanup_returns(returns, rf, periods)
    res = returns.mean() / returns.std()

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def sortino(returns, rf=0, periods=252, annualize=True):
    """
    Calculate the sortino ratio for a strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
        returns - A pandas Series representing period % returns
        riskfree - A pandas Series ror fload epresenting period % returns
        periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc
    """
    returns = tools.cleanup_returns(returns, rf)
    returns = returns[returns != 0]
    downside = returns[returns < 0]

    try:
        res = np.sqrt(periods) * returns.mean() / downside.std()
    except Exception:
        return 0.

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def cagr(returns, rf=0.):
    """
    Calculates a strategy CAGR%
    (Communicative) Annualized Growth Return

    formula:
    ((end_value/start_value) ^ (1 / # of years))-1

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        CAGR (float)
    """

    returns = tools.compsum(tools.cleanup_returns(returns, rf))
    years = len(returns) / 252
    return (returns.values[-1] / 1.0) ** (1.0 / years) - 1


def rar(returns, rf=0):
    """ CAGR% by exposure%. takes time into account. """
    returns = tools.cleanup_returns(returns, rf)
    return cagr(returns) / exposure(returns)


def skew(returns):
    return tools.cleanup_returns(returns).skew()


def kurtosis(returns):
    return tools.cleanup_returns(returns).kurtosis()


def calmar(returns):
    returns = tools.cleanup_returns(returns)
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd) if max_dd != 0 else np.nan


def ulcer(returns):
    """ ulcer index score """
    returns = tools.cleanup_returns(returns)
    dd = to_drawdown_series(returns)
    return np.sqrt(np.divide((dd**2).sum(), returns.shape[0] - 1))


def ror(returns):
    """ risk of ruin """
    returns = tools.cleanup_returns(returns)
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)


def value_at_risk(returns, sigma=1, confidence=0.99):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma
    """
    returns = tools.cleanup_returns(returns)
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence/100

    confidence = 1 - confidence
    return norm.ppf(confidence) * sigma - mu


def conditional_value_at_risk(returns, sigma=1, confidence=0.99):
    """
    # cVAR (conditional_value_at_risk) / expected_shortfall
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio
    of value P.
    """
    returns = tools.cleanup_returns(returns)
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence/100

    confidence = 1 - confidence
    return -confidence**-1 * norm.pdf(norm.ppf(confidence)) * sigma - mu


def tail_ratio(returns, cutoff=0.95):
    """Determines the ratio between the right (95%) and left tail (5%).
    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
    Returns
    -------
    tail_ratio : float
    """
    returns = tools.cleanup_returns(returns)
    return abs(returns.quantile(cutoff) / returns.quantile(1-cutoff))


def payoff_ratio(returns):
    returns = tools.cleanup_returns(returns)
    return avg_win(returns) / avg_loss(returns)


def win_loss_ratio(returns):
    return payoff_ratio(returns)


def profit_ratio(returns):
    """
    Calculates a strategy profit ratio
    (a trading system's ability to generate profits over losses)

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        profit ratio (float)
    """
    returns = tools.cleanup_returns(returns)
    wins = returns[returns >= 0]
    loss = returns[returns < 0]

    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return win_ratio/loss_ratio
    except Exception:
        return 0.


def profit_factor(returns):
    """
    Calculates a strategy profit of winners divided by profit of losers

    Input:
        returns - A pandas Series representing period percentage returns
    Returns:
        profit factor (float)
    """
    returns = tools.cleanup_returns(returns)
    return abs(returns.sum() / returns[returns < 0].sum())


def gain_to_pain_ratio(returns):
    return profit_factor(returns)


def cpc_index(returns):
    returns = tools.cleanup_returns(returns)
    return profit_factor(returns) * win_rate(returns) * \
        win_loss_ratio(returns)


def common_sense_ratio(returns):
    returns = tools.cleanup_returns(returns)
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=.99):
    returns = tools.cleanup_returns(returns)
    wins = returns[returns >= 0]
    return wins.quantile(quantile).mean() / wins.mean()


def outlier_loss_ratio(returns, quantile=.99):
    returns = tools.cleanup_returns(returns)
    losses = returns[returns < 0]
    return losses.quantile(quantile).mean() / losses.mean()


def recovery_factor(returns):
    returns = tools.cleanup_returns(returns)
    total_returns = returns.compound()
    max_dd = max_drawdown(returns)
    return total_returns / abs(max_dd) if max_dd != 0 else np.nan


def risk_return_ratio(returns):
    """
    Calculates the return / risk ratio. Basically the
    Sharpe ratio without factoring in the risk-free rate.
    """
    returns = tools.cleanup_returns(returns)
    return returns.mean() / returns.std()


def max_drawdown(prices):
    prices = prices.dropna()
    if prices.min() <= 0 or prices.max() < 1:
        prices = tools.to_price_series(prices)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(prices):
    prices = prices.dropna()
    if prices.min() <= 0 or prices.max() < 1:
        prices = tools.to_price_series(prices)
    return prices.add(1).div(prices.cummax().add(1)).subtract(1)


def drawdown_details(drawdown):
    # mark no drawdown
    no_dd = drawdown == 0

    # extract dd start dates
    starts = ~no_dd & no_dd.shift(1)
    starts = list(starts[starts == True].index)

    # extract end dates
    ends = no_dd & (~no_dd).shift(1)
    ends = list(ends[ends == True].index)

    # no drawdown :)
    if len(starts) == 0:
        return None

    # drawdown series begins in a drawdown
    if starts[0] > ends[0]:
        starts.insert(0, drawdown.index[0])

    # series endss in a drawdown fill with last date
    if len(ends) == 0 or starts[-1] > ends[-1]:
        ends.append(drawdown.index[-1])

    # build dataframe from results
    data = []
    for i in range(len(starts)):
        data.append((starts[i], ends[i], (ends[i] - starts[i]).days,
                     drawdown[starts[i]:ends[i]].min()))

    return pd.DataFrame(columns=('start', 'end', 'days', 'drawdown'),
                        data=data)


def kelly_criterion(returns):
    """
    http://en.wikipedia.org/wiki/Kelly_criterion
    Kelly provides an indication of the maximum amount of
    trading capital that should be applied to the given strategy.
    The result is in the range 0-100 and indicates the maximum
    percentage that should be allocated.

    The Kelly Criterion is calculated with the formula:

    f = ((b*p)-q)/b

    where:
        f  is the maximum fraction of the total capital to invest
        b is the win-loss ratio
        p is the probability of a winning trade
        q is the probability of a losing trade (1-p)
    """
    returns = tools.cleanup_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    if win_loss_ratio == 0.:
        return np.nan

    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


# ==== VS. BENCHMARK ====

def r2(returns, benchmark):
    return r_squared(returns, benchmark)


def r_squared(returns, benchmark):
    """ Return R^2 where returns and benchmark are array-like."""

    # ----------------------------
    # data cleanup
    returns = tools.cleanup_returns(returns)
    benchmark = tools.cleanup_returns(benchmark)
    # ----------------------------

    slope, intercept, r_val, p_val, std_err = linregress(returns, benchmark)
    return r_val**2


def information_ratio(returns, benchmark):
    """
    http://en.wikipedia.org/wiki/Information_ratio
    """
    # ----------------------------
    # data cleanup
    returns = tools.cleanup_returns(returns)
    benchmark = tools.cleanup_returns(benchmark)
    # ----------------------------

    diff_rets = returns - benchmark
    diff_std = diff_rets.std()

    if np.isnan(diff_std) or diff_std == 0:
        return 0.0

    return diff_rets.mean() / diff_std


def greeks(returns, benchmark, periods=252.):
    """
    Calculate alpha, beta and volatility of the portfolio

    Input:
        returns   - A pandas Series representing portfolio's pnl
        benchmark - A pandas Series representing benchmark prices
        rf  - A pandas Series of float representing the period's annual rf rate
    Returns:
        dictionary with alpha, beta and volatility values

    """

    # ----------------------------
    # data cleanup
    returns = tools.cleanup_returns(returns)
    benchmark = tools.cleanup_returns(benchmark)
    # ----------------------------

    # find covariance
    matrix = np.cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]

    # calculate measures now
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods

    return {
        "beta":  beta,
        "alpha": alpha,
        # "vol": np.sqrt(matrix[0, 0]) * np.sqrt(periods)
    }


def rolling_greeks(returns, benchmark, periods=252.):
    df = pd.DataFrame(data={
        "returns": tools.cleanup_returns(returns),
        "benchmark": tools.cleanup_returns(benchmark)
    })
    corr = df.rolling(periods).corr().unstack()['returns']['benchmark']
    std = df.rolling(periods).std()
    beta = corr * std['returns'] / std['benchmark']

    alpha = df['returns'].mean() - beta * df['benchmark'].mean()

    # limit beta to -1/1
    beta = pd.Series(index=returns.index, data=np.where(beta > 1, 1, beta))
    beta = pd.Series(index=returns.index, data=np.where(beta < -1, -1, beta))

    # alpha = alpha * periods
    return pd.DataFrame(index=returns.index, data={
        "beta": beta,
        "alpha": alpha
    })


def compare(returns, benchmark, aggregate=None, compounded=True, round_vals=2):

    benchmark = tools.cleanup_returns(benchmark)
    returns = tools.cleanup_returns(returns)

    data = pd.DataFrame(data={
        'Benchmark': tools.aggregate_returns(benchmark, aggregate)*100,
        'Returns': tools.aggregate_returns(returns, aggregate)*100
    })

    data['Diff'] = data['Returns'] / data['Benchmark']
    data['Won'] = (data['Returns'] >= data['Benchmark'])

    if round_vals is not None:
        return np.round(data, round_vals)
    return data


# =======================
"""
from pandas.core.base import PandasObject
def extend_pandas():
    PandasObject.returns = tools.returns
    PandasObject.compsum = tools.compsum
    PandasObject.log_returns = tools.log_returns
    PandasObject.volatility = volatility
    PandasObject.implied_volatility = implied_volatility
    PandasObject.to_price_series = tools.to_price_series
    PandasObject.win_rate = win_rate
    PandasObject.avg_return = avg_return
    PandasObject.avg_win = avg_win
    PandasObject.avg_loss = avg_loss

extend_pandas()
"""

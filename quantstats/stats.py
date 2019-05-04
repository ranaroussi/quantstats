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

def expected_return(returns, aggregate=None, compounded=True):
    """
    returns the expected return for a given period
    by calculating the geometric holding period return
    """
    returns = tools._cleanup_returns(returns)
    returns = tools.aggregate_returns(returns, aggregate, compounded)
    return np.product(1 + returns) ** (1 / len(returns)) - 1


def geometric_mean(retruns, aggregate=None, compounded=True):
    """ shorthand for expected_return() """
    return expected_return(retruns)


def ghpr(retruns, aggregate=None, compounded=True):
    """ shorthand for expected_return() """
    return expected_return(retruns)


def outliers(returns, quantile=.95):
    """
    returns series of outliers
    """
    return returns[returns > returns.quantile(quantile)].dropna(how='all')


def remove_outliers(returns, quantile=.95):
    """
    returns series of returns without the outliers
    """
    return returns[returns < returns.quantile(quantile)]


def best(returns, aggregate=None, compounded=True):
    """
    returns the best day/month/week/quarter/year's return
    """
    returns = tools._cleanup_returns(returns)
    return tools.aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True):
    """
    returns the worst day/month/week/quarter/year's return
    """
    returns = tools._cleanup_returns(returns)
    return tools.aggregate_returns(returns, aggregate, compounded).min()


def consecutive_wins(returns, aggregate=None, compounded=True):
    """
    returns the maximum consecutive wins by day/month/week/quarter/year
    """
    returns = tools._cleanup_returns(returns)
    returns = 0 < tools.aggregate_returns(returns, aggregate, compounded)
    return tools.count_consecutive(returns).max()


def consecutive_losses(returns, aggregate=None, compounded=True):
    """
    returns the maximum consecutive losses by day/month/week/quarter/year
    """
    returns = tools._cleanup_returns(returns)
    returns = 0 > tools.aggregate_returns(returns, aggregate, compounded)
    return tools.count_consecutive(returns).max()


def exposure(returns):
    """
    returns the market exposure time (returns != 0)
    """
    returns = tools._cleanup_returns(returns)
    return len(returns[(~np.isnan(returns)) & (returns != 0)]) / len(returns)


def win_rate(returns, aggregate=None, compounded=True):
    """
    calculates the win ratio for a period
    """
    def _win_rate(series):
        try:
            return len(series[series > 0]) / len(series[series != 0])
        except Exception:
            return 0.

    returns = tools._cleanup_returns(returns)
    if aggregate:
        returns = tools.aggregate_returns(returns, aggregate, compounded)

    if isinstance(returns, pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])

        return pd.Series(_df)

    return _win_rate(returns)


def avg_return(returns):
    """
    calculates the average return/trade return for a period
    """
    returns = tools._cleanup_returns(returns)
    return returns[returns != 0].fillna(0).mean()


def avg_win(returns):
    """
    calculates the average winning return/trade return for a period
    """
    returns = tools._cleanup_returns(returns)
    return returns[returns >= 0].fillna(0).mean()


def avg_loss(returns):
    """
    calculates the average lowinf return/trade return for a period
    """
    returns = tools._cleanup_returns(returns)
    return returns[returns < 0].fillna(0).mean()


def volatility(returns, periods=252, annualize=True):
    """
    calculates the volatility of returns for a period
    """
    std = tools._cleanup_returns(returns).std()
    if annualize:
        return std * np.sqrt(1 if periods is None else periods)

    return std


def implied_volatility(returns, periods=252, annualize=True):
    """
    calculates the implied volatility of returns for a period
    """
    logret = tools.log_returns(returns)
    if annualize:
        return logret.rolling(periods).std() * np.sqrt(periods)
    return logret.std()


# ======= METRICS =======

def sharpe(returns, rf=0., periods=252, annualize=True):
    """
    calculates the sharpe ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Frequency of returns (252 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
    """

    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')

    returns = tools._cleanup_returns(returns, rf, periods)
    res = returns.mean() / returns.std()

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def sortino(returns, rf=0, periods=252, annualize=True):
    """
    calculates the sortino ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """

    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')

    returns = tools._cleanup_returns(returns, rf, periods)
    res = returns.mean() / returns[returns < 0].std()

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def cagr(returns, rf=0.):
    """
    calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """

    returns = tools.compsum(tools._cleanup_returns(returns, rf))
    years = len(returns) / 252

    res = (returns.values[-1] / 1.0) ** (1.0 / years) - 1

    if isinstance(returns, pd.DataFrame):
        res = pd.Series(res)
        res.index = returns.columns

    return res


def rar(returns, rf=0.):
    """
    calculates the risk-adjusted return of access returns
    (CAGR / exposure. takes time into account.)

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    returns = tools._cleanup_returns(returns, rf)
    return cagr(returns) / exposure(returns)


def skew(returns):
    """
    calculates returns' skewness
    (the degree of asymmetry of a distribution around its mean)
    """
    return tools._cleanup_returns(returns).skew()


def kurtosis(returns):
    """
    calculates returns' kurtosis
    (the degree to which a distribution peak compared to a normal distribution)
    """
    return tools._cleanup_returns(returns).kurtosis()


def calmar(returns):
    """
    calculates the calmar ratio (CAGR% / MaxDD%)
    """
    returns = tools._cleanup_returns(returns)
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd)


def ulcer(returns):
    """
    calculates the ulcer index score (downside risk measurment)
    """
    returns = tools._cleanup_returns(returns)
    dd = max_drawdown(returns) * 100
    return np.sqrt(np.divide((dd**2).sum(), sharpe(returns) - 1))


def risk_of_ruin(returns):
    """
    calculates the risk of ruin
    (the likelihood of losing all one's investment capital)
    """
    returns = tools._cleanup_returns(returns)
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)


def ror(returns):
    """ shorthand for risk_of_ruin() """
    return risk_of_ruin(returns)


def value_at_risk(returns, sigma=1, confidence=0.99):
    """
    calculats the daily value-at-risk
    (variance-covariance calculation with confidence n)
    """
    returns = tools._cleanup_returns(returns)
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence/100

    confidence = 1 - confidence
    return norm.ppf(confidence) * sigma - mu


def var(returns, sigma=1, confidence=0.99):
    """ shorthand for value_at_risk() """
    return value_at_risk(returns, sigma, confidence)


def conditional_value_at_risk(returns, sigma=1, confidence=0.99):
    """
    calculats the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
    """
    returns = tools._cleanup_returns(returns)
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence/100

    confidence = 1 - confidence
    return -confidence**-1 * norm.pdf(norm.ppf(confidence)) * sigma - mu


def cvar(returns, sigma=1, confidence=0.99):
    """ shorthand for conditional_value_at_risk() """
    return conditional_value_at_risk(returns, sigma, confidence)


def expected_shortfall(returns, sigma=1, confidence=0.99):
    """ shorthand for conditional_value_at_risk() """
    return conditional_value_at_risk(returns, sigma, confidence)


def tail_ratio(returns, cutoff=0.95):
    """
    measures the ratio between the right (95%) and left tail (5%).
    """
    returns = tools._cleanup_returns(returns)
    return abs(returns.quantile(cutoff) / returns.quantile(1-cutoff))


def payoff_ratio(returns):
    """
    measures the payoff ratio (average win/average loss)
    """
    returns = tools._cleanup_returns(returns)
    return avg_win(returns) / avg_loss(returns)


def win_loss_ratio(returns):
    """ shorthand for payoff_ratio() """
    return payoff_ratio(returns)


def profit_ratio(returns):
    """
    measures the profit ratio (win ratio / loss ratio)
    """
    returns = tools._cleanup_returns(returns)
    wins = returns[returns >= 0]
    loss = returns[returns < 0]

    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return win_ratio / loss_ratio
    except Exception:
        return 0.


def profit_factor(returns):
    """
    measures the profit ratio (wins/loss)
    """
    returns = tools._cleanup_returns(returns)
    return abs(returns.sum() / returns[returns < 0].sum())


def gain_to_pain_ratio(returns):
    """ shorthand for profit_factor() """
    return profit_factor(returns)


def cpc_index(returns):
    """
    measures the cpc ratio (profit factor * win % * win loss ratio)
    """
    returns = tools._cleanup_returns(returns)
    return profit_factor(returns) * win_rate(returns) * \
        win_loss_ratio(returns)


def common_sense_ratio(returns):
    """
    measures the common sense ratio (profit factor * tail ratio)
    """
    returns = tools._cleanup_returns(returns)
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=.99):
    """
    calculates the outlier winners ratio
    99th percentile of returns / mean positive return
    """
    returns = tools._cleanup_returns(returns)
    return returns.quantile(quantile).mean() / returns[returns >= 0].mean()


def outlier_loss_ratio(returns, quantile=.01):
    """
    calculates the outlier losers ratio
    1st percentile of returns / mean negative return
    """
    returns = tools._cleanup_returns(returns)
    return returns.quantile(quantile).mean() / returns[returns < 0].mean()


def recovery_factor(returns):
    """
    measures how fast the strategy recovers from drawdowns
    """
    returns = tools._cleanup_returns(returns)
    total_returns = returns.compound()
    max_dd = max_drawdown(returns)
    return total_returns / abs(max_dd)


def risk_return_ratio(returns):
    """
    calculates the return / risk ratio
    (sharpe ratio without factoring in the risk-free rate)
    """
    returns = tools._cleanup_returns(returns)
    return returns.mean() / returns.std()


def max_drawdown(prices):
    """
    calculates the maximum drawdown
    """
    prices = tools._cleanup_prices(prices)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(prices):
    """
    convert price series to drawdown series
    """
    prices = tools._cleanup_prices(prices)
    return prices.add(1).div(prices.cummax().add(1)).subtract(1)


def drawdown_details(drawdown):
    """
    calculates drawdown details, including start/end dates,
    duration and max drawdown for every drawdown period
    """
    def _drawdown_details(drawdown):
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

    if isinstance(drawdown, pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return pd.concat(_dfs, axis=1)

    return _drawdown_details(drawdown)


def kelly_criterion(returns):
    """
    calculates the recommended maximum amount of capital that
    should be allocated to the given strategy, based on the
    Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
    """
    returns = tools._cleanup_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


# ==== VS. BENCHMARK ====

def r_squared(returns, benchmark):
    """
    measures the straight line fit of the equity curve
    """
    slope, intercept, r_val, p_val, std_err = linregress(
        tools._cleanup_returns(returns),
        tools._cleanup_returns(benchmark))
    return r_val**2


def r2(returns, benchmark):
    """ shorthand for r_squared() """
    return r_squared(returns, benchmark)


def information_ratio(returns, benchmark):
    """
    calculates the information ratio
    (basically the risk return ratio of the net profits)
    """
    diff_rets = tools._cleanup_returns(returns) - \
        tools._cleanup_returns(benchmark)

    return diff_rets.mean() / diff_rets.std()


def greeks(returns, benchmark, periods=252.):
    """
    calculates alpha and beta of the portfolio
    """

    # ----------------------------
    # data cleanup
    returns = tools._cleanup_returns(returns)
    benchmark = tools._cleanup_returns(benchmark)
    # ----------------------------

    # find covariance
    matrix = np.cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]

    # calculates measures now
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods

    return pd.Series({
        "beta":  beta,
        "alpha": alpha,
        # "vol": np.sqrt(matrix[0, 0]) * np.sqrt(periods)
    })


def rolling_greeks(returns, benchmark, periods=252):
    """
    calculates rolling alpha and beta of the portfolio
    """
    df = pd.DataFrame(data={
        "returns": tools._cleanup_returns(returns),
        "benchmark": tools._cleanup_returns(benchmark)
    })
    corr = df.rolling(int(periods)).corr().unstack()['returns']['benchmark']
    std = df.rolling(int(periods)).std()
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
    """
    compare returns to benchmark on a day/week/month/quarter/year basis
    """
    benchmark = tools._cleanup_returns(benchmark)
    returns = tools._cleanup_returns(returns)

    data = pd.DataFrame(data={
        'Benchmark': tools.aggregate_returns(benchmark, aggregate)*100,
        'Returns': tools.aggregate_returns(returns, aggregate)*100
    })

    data['Diff'] = data['Returns'] / data['Benchmark']
    data['Won'] = (data['Returns'] >= data['Benchmark'])

    if round_vals is not None:
        return np.round(data, round_vals)
    return data

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

from . import version

__version__ = version.version
__author__ = "Ran Aroussi"

from . import stats, utils, plots, reports

__all__ = ['stats', 'plots', 'reports', 'utils', 'extend_pandas']

# try automatic matplotlib inline
utils._in_notebook(matplotlib_inline=True)


def extend_pandas():
    """
    Extends pandas by exposing methods to be used like:
    df.sharpe(), df.best('day'), ...
    """
    from pandas.core.base import PandasObject as po

    po.compsum = stats.compsum
    po.comp = stats.comp
    po.expected_return = stats.expected_return
    po.geometric_mean = stats.geometric_mean
    po.ghpr = stats.ghpr
    po.outliers = stats.outliers
    po.remove_outliers = stats.remove_outliers
    po.best = stats.best
    po.worst = stats.worst
    po.consecutive_wins = stats.consecutive_wins
    po.consecutive_losses = stats.consecutive_losses
    po.exposure = stats.exposure
    po.win_rate = stats.win_rate
    po.avg_return = stats.avg_return
    po.avg_win = stats.avg_win
    po.avg_loss = stats.avg_loss
    po.volatility = stats.volatility
    po.rolling_volatility = stats.rolling_volatility
    po.implied_volatility = stats.implied_volatility
    po.sharpe = stats.sharpe
    po.smart_sharpe = stats.smart_sharpe
    po.rolling_sharpe = stats.rolling_sharpe
    po.sortino = stats.sortino
    po.smart_sortino = stats.smart_sortino
    po.adjusted_sortino = stats.adjusted_sortino
    po.rolling_sortino = stats.rolling_sortino
    po.omega = stats.omega
    po.cagr = stats.cagr
    po.rar = stats.rar
    po.skew = stats.skew
    po.kurtosis = stats.kurtosis
    po.calmar = stats.calmar
    po.ulcer_index = stats.ulcer_index
    po.ulcer_performance_index = stats.ulcer_performance_index
    po.upi = stats.upi
    po.serenity_index = stats.serenity_index
    po.risk_of_ruin = stats.risk_of_ruin
    po.ror = stats.ror
    po.value_at_risk = stats.value_at_risk
    po.var = stats.var
    po.conditional_value_at_risk = stats.conditional_value_at_risk
    po.cvar = stats.cvar
    po.expected_shortfall = stats.expected_shortfall
    po.tail_ratio = stats.tail_ratio
    po.payoff_ratio = stats.payoff_ratio
    po.win_loss_ratio = stats.win_loss_ratio
    po.profit_ratio = stats.profit_ratio
    po.profit_factor = stats.profit_factor
    po.gain_to_pain_ratio = stats.gain_to_pain_ratio
    po.cpc_index = stats.cpc_index
    po.common_sense_ratio = stats.common_sense_ratio
    po.outlier_win_ratio = stats.outlier_win_ratio
    po.outlier_loss_ratio = stats.outlier_loss_ratio
    po.recovery_factor = stats.recovery_factor
    po.risk_return_ratio = stats.risk_return_ratio
    po.max_drawdown = stats.max_drawdown
    po.to_drawdown_series = stats.to_drawdown_series
    po.kelly_criterion = stats.kelly_criterion
    po.monthly_returns = stats.monthly_returns
    po.pct_rank = stats.pct_rank

    po.treynor_ratio = stats.treynor_ratio
    po.probabilistic_sharpe_ratio = stats.probabilistic_sharpe_ratio
    po.probabilistic_sortino_ratio = stats.probabilistic_sortino_ratio
    po.probabilistic_adjusted_sortino_ratio = stats.probabilistic_adjusted_sortino_ratio

    # methods from utils
    po.to_returns = utils.to_returns
    po.to_prices = utils.to_prices
    po.to_log_returns = utils.to_log_returns
    po.log_returns = utils.log_returns
    po.exponential_stdev = utils.exponential_stdev
    po.rebase = utils.rebase
    po.aggregate_returns = utils.aggregate_returns
    po.to_excess_returns = utils.to_excess_returns
    po.multi_shift = utils.multi_shift
    po.curr_month = utils._pandas_current_month
    po.date = utils._pandas_date
    po.mtd = utils._mtd
    po.qtd = utils._qtd
    po.ytd = utils._ytd

    # methods that requires benchmark stats
    po.r_squared = stats.r_squared
    po.r2 = stats.r2
    po.information_ratio = stats.information_ratio
    po.greeks = stats.greeks
    po.rolling_greeks = stats.rolling_greeks
    po.compare = stats.compare

    # plotting methods
    po.plot_snapshot = plots.snapshot
    po.plot_earnings = plots.earnings
    po.plot_daily_returns = plots.daily_returns
    po.plot_distribution = plots.distribution
    po.plot_drawdown = plots.drawdown
    po.plot_drawdowns_periods = plots.drawdowns_periods
    po.plot_histogram = plots.histogram
    po.plot_log_returns = plots.log_returns
    po.plot_returns = plots.returns
    po.plot_rolling_beta = plots.rolling_beta
    po.plot_rolling_sharpe = plots.rolling_sharpe
    po.plot_rolling_sortino = plots.rolling_sortino
    po.plot_rolling_volatility = plots.rolling_volatility
    po.plot_yearly_returns = plots.yearly_returns
    po.plot_monthly_heatmap = plots.monthly_heatmap

    po.metrics = reports.metrics

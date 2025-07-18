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

from . import version

__version__ = version.version
__author__ = "Ran Aroussi"

from . import stats, utils, plots, reports

__all__ = ["stats", "plots", "reports", "utils", "extend_pandas"]

# try automatic matplotlib inline
utils._in_notebook(matplotlib_inline=True)


def extend_pandas():
    """
    Extends pandas by exposing methods to be used like:
    df.sharpe(), df.best('day'), ...
    """
    from pandas.core.base import PandasObject as _po  # type: ignore[import]

    _po.compsum = stats.compsum  # type: ignore[attr-defined]
    _po.comp = stats.comp  # type: ignore[attr-defined]
    _po.expected_return = stats.expected_return  # type: ignore[attr-defined]
    _po.geometric_mean = stats.geometric_mean  # type: ignore[attr-defined]
    _po.ghpr = stats.ghpr  # type: ignore[attr-defined]
    _po.outliers = stats.outliers  # type: ignore[attr-defined]
    _po.remove_outliers = stats.remove_outliers  # type: ignore[attr-defined]
    _po.best = stats.best  # type: ignore[attr-defined]
    _po.worst = stats.worst  # type: ignore[attr-defined]
    _po.consecutive_wins = stats.consecutive_wins  # type: ignore[attr-defined]
    _po.consecutive_losses = stats.consecutive_losses  # type: ignore[attr-defined]
    _po.exposure = stats.exposure  # type: ignore[attr-defined]
    _po.win_rate = stats.win_rate  # type: ignore[attr-defined]
    _po.avg_return = stats.avg_return  # type: ignore[attr-defined]
    _po.avg_win = stats.avg_win  # type: ignore[attr-defined]
    _po.avg_loss = stats.avg_loss  # type: ignore[attr-defined]
    _po.volatility = stats.volatility  # type: ignore[attr-defined]
    _po.rolling_volatility = stats.rolling_volatility  # type: ignore[attr-defined]
    _po.implied_volatility = stats.implied_volatility  # type: ignore[attr-defined]
    _po.sharpe = stats.sharpe  # type: ignore[attr-defined]
    _po.smart_sharpe = stats.smart_sharpe  # type: ignore[attr-defined]
    _po.rolling_sharpe = stats.rolling_sharpe  # type: ignore[attr-defined]
    _po.sortino = stats.sortino  # type: ignore[attr-defined]
    _po.smart_sortino = stats.smart_sortino  # type: ignore[attr-defined]
    _po.adjusted_sortino = stats.adjusted_sortino  # type: ignore[attr-defined]
    _po.rolling_sortino = stats.rolling_sortino  # type: ignore[attr-defined]
    _po.omega = stats.omega  # type: ignore[attr-defined]
    _po.cagr = stats.cagr  # type: ignore[attr-defined]
    _po.rar = stats.rar  # type: ignore[attr-defined]
    _po.skew = stats.skew  # type: ignore[attr-defined]
    _po.kurtosis = stats.kurtosis  # type: ignore[attr-defined]
    _po.calmar = stats.calmar  # type: ignore[attr-defined]
    _po.ulcer_index = stats.ulcer_index  # type: ignore[attr-defined]
    _po.ulcer_performance_index = stats.ulcer_performance_index  # type: ignore[attr-defined]
    _po.upi = stats.upi  # type: ignore[attr-defined]
    _po.serenity_index = stats.serenity_index  # type: ignore[attr-defined]
    _po.risk_of_ruin = stats.risk_of_ruin  # type: ignore[attr-defined]
    _po.ror = stats.ror  # type: ignore[attr-defined]
    _po.value_at_risk = stats.value_at_risk  # type: ignore[attr-defined]
    _po.var = stats.var  # type: ignore[attr-defined]
    _po.conditional_value_at_risk = stats.conditional_value_at_risk  # type: ignore[attr-defined]
    _po.cvar = stats.cvar  # type: ignore[attr-defined]
    _po.expected_shortfall = stats.expected_shortfall  # type: ignore[attr-defined]
    _po.tail_ratio = stats.tail_ratio  # type: ignore[attr-defined]
    _po.payoff_ratio = stats.payoff_ratio  # type: ignore[attr-defined]
    _po.win_loss_ratio = stats.win_loss_ratio  # type: ignore[attr-defined]
    _po.profit_ratio = stats.profit_ratio  # type: ignore[attr-defined]
    _po.profit_factor = stats.profit_factor  # type: ignore[attr-defined]
    _po.gain_to_pain_ratio = stats.gain_to_pain_ratio  # type: ignore[attr-defined]
    _po.cpc_index = stats.cpc_index  # type: ignore[attr-defined]
    _po.common_sense_ratio = stats.common_sense_ratio  # type: ignore[attr-defined]
    _po.outlier_win_ratio = stats.outlier_win_ratio  # type: ignore[attr-defined]
    _po.outlier_loss_ratio = stats.outlier_loss_ratio  # type: ignore[attr-defined]
    _po.recovery_factor = stats.recovery_factor  # type: ignore[attr-defined]
    _po.risk_return_ratio = stats.risk_return_ratio  # type: ignore[attr-defined]
    _po.max_drawdown = stats.max_drawdown  # type: ignore[attr-defined]
    _po.to_drawdown_series = stats.to_drawdown_series  # type: ignore[attr-defined]
    _po.kelly_criterion = stats.kelly_criterion  # type: ignore[attr-defined]
    _po.monthly_returns = stats.monthly_returns  # type: ignore[attr-defined]
    _po.pct_rank = stats.pct_rank  # type: ignore[attr-defined]

    _po.treynor_ratio = stats.treynor_ratio  # type: ignore[attr-defined]
    _po.probabilistic_sharpe_ratio = stats.probabilistic_sharpe_ratio  # type: ignore[attr-defined]
    _po.probabilistic_sortino_ratio = stats.probabilistic_sortino_ratio  # type: ignore[attr-defined]
    _po.probabilistic_adjusted_sortino_ratio = (  # type: ignore[attr-defined]
        stats.probabilistic_adjusted_sortino_ratio
    )

    # methods from utils
    _po.to_returns = utils.to_returns  # type: ignore[attr-defined]
    _po.to_prices = utils.to_prices  # type: ignore[attr-defined]
    _po.to_log_returns = utils.to_log_returns  # type: ignore[attr-defined]
    _po.log_returns = utils.log_returns  # type: ignore[attr-defined]
    _po.exponential_stdev = utils.exponential_stdev  # type: ignore[attr-defined]
    _po.rebase = utils.rebase  # type: ignore[attr-defined]
    _po.aggregate_returns = utils.aggregate_returns  # type: ignore[attr-defined]
    _po.to_excess_returns = utils.to_excess_returns  # type: ignore[attr-defined]
    _po.multi_shift = utils.multi_shift  # type: ignore[attr-defined]
    _po.curr_month = utils._pandas_current_month  # type: ignore[attr-defined]
    _po.date = utils._pandas_date  # type: ignore[attr-defined]
    _po.mtd = utils._mtd  # type: ignore[attr-defined]
    _po.qtd = utils._qtd  # type: ignore[attr-defined]
    _po.ytd = utils._ytd  # type: ignore[attr-defined]

    # methods that requires benchmark stats
    _po.r_squared = stats.r_squared  # type: ignore[attr-defined]
    _po.r2 = stats.r2  # type: ignore[attr-defined]
    _po.information_ratio = stats.information_ratio  # type: ignore[attr-defined]
    _po.greeks = stats.greeks  # type: ignore[attr-defined]
    _po.rolling_greeks = stats.rolling_greeks  # type: ignore[attr-defined]
    _po.compare = stats.compare  # type: ignore[attr-defined]

    # plotting methods
    _po.plot_snapshot = plots.snapshot  # type: ignore[attr-defined]
    _po.plot_earnings = plots.earnings  # type: ignore[attr-defined]
    _po.plot_daily_returns = plots.daily_returns  # type: ignore[attr-defined]
    _po.plot_distribution = plots.distribution  # type: ignore[attr-defined]
    _po.plot_drawdown = plots.drawdown  # type: ignore[attr-defined]
    _po.plot_drawdowns_periods = plots.drawdowns_periods  # type: ignore[attr-defined]
    _po.plot_histogram = plots.histogram  # type: ignore[attr-defined]
    _po.plot_log_returns = plots.log_returns  # type: ignore[attr-defined]
    _po.plot_returns = plots.returns  # type: ignore[attr-defined]
    _po.plot_rolling_beta = plots.rolling_beta  # type: ignore[attr-defined]
    _po.plot_rolling_sharpe = plots.rolling_sharpe  # type: ignore[attr-defined]
    _po.plot_rolling_sortino = plots.rolling_sortino  # type: ignore[attr-defined]
    _po.plot_rolling_volatility = plots.rolling_volatility  # type: ignore[attr-defined]
    _po.plot_yearly_returns = plots.yearly_returns  # type: ignore[attr-defined]
    _po.plot_monthly_heatmap = plots.monthly_heatmap  # type: ignore[attr-defined]

    _po.metrics = reports.metrics  # type: ignore[attr-defined]


# extend_pandas()

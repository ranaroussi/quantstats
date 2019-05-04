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

__version__ = "0.0.01"
__author__ = "Ran Aroussi"

from pandas.core.base import PandasObject
from . import stats, tools, optimize

__all__ = ['stats', 'tools', 'optimize', 'extend_pandas']


def extend_pandas():
    """
    extends pandas by exposing methods to be used like:
    df.sharpe(), df.best('day'), ...
    """
    PandasObject.expected_return = stats.expected_return
    PandasObject.geometric_mean = stats.geometric_mean
    PandasObject.ghpr = stats.ghpr
    PandasObject.outliers = stats.outliers
    PandasObject.remove_outliers = stats.remove_outliers
    PandasObject.best = stats.best
    PandasObject.worst = stats.worst
    PandasObject.consecutive_wins = stats.consecutive_wins
    PandasObject.consecutive_losses = stats.consecutive_losses
    PandasObject.exposure = stats.exposure
    PandasObject.win_rate = stats.win_rate
    PandasObject.avg_return = stats.avg_return
    PandasObject.avg_win = stats.avg_win
    PandasObject.avg_loss = stats.avg_loss
    PandasObject.volatility = stats.volatility
    PandasObject.implied_volatility = stats.implied_volatility
    PandasObject.sharpe = stats.sharpe
    PandasObject.sortino = stats.sortino
    PandasObject.cagr = stats.cagr
    PandasObject.rar = stats.rar
    PandasObject.skew = stats.skew
    PandasObject.kurtosis = stats.kurtosis
    PandasObject.calmar = stats.calmar
    PandasObject.ulcer = stats.ulcer
    PandasObject.risk_of_ruin = stats.risk_of_ruin
    PandasObject.ror = stats.ror
    PandasObject.value_at_risk = stats.value_at_risk
    PandasObject.var = stats.var
    PandasObject.conditional_value_at_risk = stats.conditional_value_at_risk
    PandasObject.cvar = stats.cvar
    PandasObject.expected_shortfall = stats.expected_shortfall
    PandasObject.tail_ratio = stats.tail_ratio
    PandasObject.payoff_ratio = stats.payoff_ratio
    PandasObject.win_loss_ratio = stats.win_loss_ratio
    PandasObject.profit_ratio = stats.profit_ratio
    PandasObject.profit_factor = stats.profit_factor
    PandasObject.gain_to_pain_ratio = stats.gain_to_pain_ratio
    PandasObject.cpc_index = stats.cpc_index
    PandasObject.common_sense_ratio = stats.common_sense_ratio
    PandasObject.outlier_win_ratio = stats.outlier_win_ratio
    PandasObject.outlier_loss_ratio = stats.outlier_loss_ratio
    PandasObject.recovery_factor = stats.recovery_factor
    PandasObject.risk_return_ratio = stats.risk_return_ratio
    PandasObject.max_drawdown = stats.max_drawdown
    PandasObject.to_drawdown_series = stats.to_drawdown_series
    PandasObject.kelly_criterion = stats.kelly_criterion

    # methods from tools
    PandasObject.compsum = tools.compsum
    PandasObject.comp = tools.comp
    PandasObject.to_returns = tools.to_returns
    PandasObject.to_prices = tools.to_prices
    PandasObject.log_returns = tools.log_returns
    PandasObject.exponential_stdev = tools.exponential_stdev
    PandasObject.rebase = tools.rebase
    PandasObject.aggregate_returns = tools.aggregate_returns
    PandasObject.to_excess_returns = tools.to_excess_returns

    # methods that requires benchmark stats
    PandasObject.r_squared = stats.r_squared
    PandasObject.r2 = stats.r2
    PandasObject.information_ratio = stats.information_ratio
    PandasObject.greeks = stats.greeks
    PandasObject.rolling_greeks = stats.rolling_greeks
    PandasObject.compare = stats.compare


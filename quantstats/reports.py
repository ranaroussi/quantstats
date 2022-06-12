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
#
import numpy as np
import pandas as pd
import os
import re

from base64 import b64encode
from dateutil.relativedelta import relativedelta
from math import ceil, sqrt
from tabulate import tabulate
from typing import List, Union, Tuple

try:
    from IPython.core.display import (
        display as iDisplay,
        HTML as iHTML
    )
except ImportError:
    pass


from . import __version__, stats, utils, plots

PTH = os.path.dirname(
    os.path.abspath(__file__)
)


def _get_trading_periods(periods_per_year=252):
    half_year = ceil(periods_per_year/2)
    return periods_per_year, half_year


def _match_dates(returns, benchmark):
    returns = returns.loc[
        max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax()):]
    benchmark = benchmark.loc[
        max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax()):]

    return returns, benchmark


def html(
    returns: pd.Series,
    benchmark: Union[pd.Series, pd.DataFrame, str, None] = None,
    benchmark_is_prices: bool = False,
    theme: Union[str, dict] = 'default',
    rf: float = 0,
    title: str = 'Strategy Tearsheet',
    compounded: bool = True,
    periods_per_year: int = 252,
    html_filename: str = 'quantstats-tearsheet.html',
    figfmt: str = 'svg',
    template_path: Union[str, None] = None,
    match_dates: bool = False,
    **kwargs
) -> None:
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    if template_path is None:
        template_path = os.path.join(PTH, 'report.html')

    with open(template_path) as f:
        tpl = f.read()

    # prepare timeseries
    returns = utils._prepare_returns(returns)

    if benchmark is not None:
        if 'benchmark_title' not in kwargs:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, pd.Series):
                benchmark_title = benchmark.name
            elif isinstance(benchmark, pd.DataFrame):
                benchmark_title = benchmark[benchmark.columns[0]].name
            else:
                benchmark_title = 'BENCHMARK'

        tpl = tpl.replace(
            '{{benchmark_title}}',
            f'Benchmark is {benchmark_title.upper()} | '
        )

        if (
            isinstance(benchmark, (pd.Series, pd.DataFrame)) and
            benchmark_is_prices
        ):
            benchmark = benchmark.pct_change()

        benchmark = (
            utils
            ._prepare_benchmark(
                benchmark=benchmark,
                period=returns.index,
                rf=rf,
            )
        )

        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    date_range = returns.index.strftime('%e %b, %Y')
    tpl = (
        tpl
        .replace(
            '{{date_range}}',
            date_range[0] + ' - ' + date_range[-1],
        )
    )
    tpl = tpl.replace('{{title}}', title)
    tpl = tpl.replace('{{v}}', __version__)

    mtrx = (
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=False,
            mode='full',
            sep=True,
            internal='True',
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
        )
        [2:]
    )

    mtrx.index.name = 'Metric'
    tpl = tpl.replace(
        '{{metrics}}',
        _html_table(mtrx),
    )
    tpl = tpl.replace(
        '<tr><td></td><td></td><td></td></tr>',
        '<tr><td colspan="3"><hr></td></tr>',
    )
    tpl = tpl.replace(
        '<tr><td></td><td></td></tr>',
        '<tr><td colspan="2"><hr></td></tr>',
    )

    if benchmark is not None:
        yoy = stats.compare(
            returns=returns,
            benchmark=benchmark,
            aggregate='A',
            compounded=compounded,
            prepare_returns=False,
        )
        yoy.columns = [
            'Benchmark',
            'Strategy',
            'Multiplier',
            'Won',
        ]
        yoy.index.name = 'Year'
        tpl = tpl.replace(
            '{{eoy_title}}',
            '<h3>EOY Returns vs Benchmark</h3>',
        )
        tpl = tpl.replace(
            '{{eoy_table}}',
            _html_table(yoy),
        )
    else:
        # pct multiplier
        yoy = pd.DataFrame(
            data=(
                utils
                .group_returns(
                    returns,
                    returns.index.year,
                ) * 100
            )
        )
        yoy.columns = ['Return']
        yoy['Cumulative'] = (
            utils
            .group_returns(
                returns,
                returns.index.year,
                True,
            )
        )
        yoy['Return'] = yoy['Return'].round(2).astype(str) + '%'
        yoy['Cumulative'] = (
            (yoy['Cumulative'] * 100)
            .round(2)
            .astype(str) + '%'
        )
        yoy.index.name = 'Year'
        tpl = tpl.replace(
            '{{eoy_title}}',
            '<h3>EOY Returns</h3>',
        )
        tpl = tpl.replace(
            '{{eoy_table}}',
            _html_table(yoy),
        )

    dd = stats.to_drawdown_series(returns)
    dd_info = (
        stats
        .drawdown_details(dd)
        .sort_values(
            by='max drawdown',
            ascending=True
        )
        [:10]
    )

    dd_info = dd_info[['start', 'end', 'max drawdown', 'days']]
    dd_info.columns = ['Started', 'Recovered', 'Drawdown', 'Days']
    tpl = tpl.replace(
        '{{dd_info}}',
        _html_table(dd_info, False),
    )

    # plots
    figfile = utils._file_stream()
    plots.returns(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(8, 5),
        subtitle=False,
        ylabel=False,
        cumulative=compounded,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{returns}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.log_returns(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(8, 4),
        subtitle=False,
        ylabel=False,
        cumulative=compounded,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{log_returns}}',
            _embed_figure(figfile, figfmt),
        )
    )
    if benchmark is not None:
        figfile = utils._file_stream()
        plots.returns(
            returns=returns,
            benchmark=benchmark,
            theme=theme,
            match_volatility=True,
            figsize=(8, 4),
            subtitle=False,
            ylabel=False,
            cumulative=compounded,
            prepare_returns=False,
            show=False,
            export=True,
            image_file_name=figfile,
            format=figfmt,
        )
        tpl = (
            tpl
            .replace(
                '{{vol_returns}}',
                _embed_figure(figfile, figfmt),
            )
        )
    figfile = utils._file_stream()
    plots.yearly_returns(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(8, 4),
        subtitle=False,
        ylabel=False,
        compounded=compounded,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{eoy_returns}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.histogram(
        returns=returns,
        theme=theme,
        figsize=(8, 4),
        subtitle=False,
        ylabel=False,
        compounded=compounded,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{monthly_dist}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.daily_returns(
        returns=returns,
        theme=theme,
        figsize=(8, 3),
        subtitle=False,
        ylabel=False,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{daily_returns}}',
            _embed_figure(figfile, figfmt),
        )
    )
    if benchmark is not None:
        figfile = utils._file_stream()
        plots.rolling_beta(
            returns=returns,
            benchmark=benchmark,
            theme=theme,
            figsize=(8, 3),
            subtitle=False,
            window1=win_half_year,
            window2=win_year,
            ylabel=False,
            prepare_returns=False,
            show=False,
            export=True,
            image_file_name=figfile,
            format=figfmt,
        )
        tpl = (
            tpl
            .replace(
                '{{rolling_beta}}',
                _embed_figure(figfile, figfmt),
            )
        )
    figfile = utils._file_stream()
    plots.rolling_volatility(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(8, 3),
        subtitle=False,
        ylabel=False,
        period=win_half_year,
        periods_per_year=win_year,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{rolling_vol}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.rolling_sharpe(
        returns=returns,
        theme=theme,
        figsize=(8, 3),
        subtitle=False,
        ylabel=False,
        period=win_half_year,
        periods_per_year=win_year,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{rolling_sharpe}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.rolling_sortino(
        returns=returns,
        theme=theme,
        figsize=(8, 3),
        subtitle=False,
        ylabel=False,
        period=win_half_year,
        periods_per_year=win_year,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{rolling_sortino}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.drawdowns_periods(
        returns=returns,
        theme=theme,
        figsize=(8, 4),
        subtitle=False,
        ylabel=False,
        compounded=compounded,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{dd_periods}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.drawdown(
        returns=returns,
        theme=theme,
        figsize=(8, 3),
        subtitle=False,
        ylabel=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{dd_plot}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.monthly_heatmap(
        returns=returns,
        theme=theme,
        figsize=(8, 4),
        cbar=False,
        ylabel=False,
        compounded=compounded,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{monthly_heatmap}}',
            _embed_figure(figfile, figfmt),
        )
    )
    figfile = utils._file_stream()
    plots.distribution(
        returns=returns,
        theme=theme,
        figsize=(8, 4),
        subtitle=False,
        ylabel=False,
        compounded=compounded,
        prepare_returns=False,
        show=False,
        export=True,
        image_file_name=figfile,
        format=figfmt,
    )
    tpl = (
        tpl
        .replace(
            '{{returns_dist}}',
            _embed_figure(figfile, figfmt),
        )
    )
    tpl = re.sub(r'\{\{(.*?)\}\}', '', tpl)
    tpl = tpl.replace('white-space:pre;', '')

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(tpl)


def full(
    returns: pd.Series,
    benchmark: Union[pd.Series, pd.DataFrame, str, None] = None,
    benchmark_is_prices: bool = False,
    theme: Union[str, dict] = 'default',
    rf: float = 0,
    figsize: Tuple[float] = (8, 5),
    display: bool = True,
    compounded: bool = True,
    periods_per_year: int = 252,
    match_dates: bool = False,
) -> None:
    # prepare timeseries
    returns = utils._prepare_returns(returns)

    if benchmark is not None:
        if (
            isinstance(benchmark, (pd.Series, pd.DataFrame)) and
            benchmark_is_prices
        ):
            benchmark = benchmark.pct_change()

        benchmark = (
            utils
            ._prepare_benchmark(
                benchmark,
                returns.index,
                rf,
            )
        )
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    dd = stats.to_drawdown_series(returns)
    col = stats.drawdown_details(dd).columns[4]
    dd_info = (
        stats
        .drawdown_details(dd)
        .sort_values(
            by=col,
            ascending=True,
        )
        [:5]
    )
    if not dd_info.empty:
        dd_info.index = range(1, min(6, len(dd_info)+1))
        dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)

    if utils._in_notebook():
        iDisplay(iHTML('<h4>Performance Metrics</h4>'))
        iDisplay(
            metrics(
                returns=returns,
                benchmark=benchmark,
                rf=rf,
                display=display,
                mode='full',
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False,
            )
        )
        iDisplay(iHTML('<h4>5 Worst Drawdowns</h4>'))
        if dd_info.empty:
            iDisplay(iHTML("<p>(no drawdowns)</p>"))
        else:
            iDisplay(dd_info)

        iDisplay(iHTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode='full',
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
        )
        print('\n\n')
        print('[5 Worst Drawdowns]\n')
        if dd_info.empty:
            print("(no drawdowns)")
        else:
            print(
                (
                    tabulate(
                        dd_info,
                        headers='keys',
                        tablefmt='simple',
                        floatfmt='.2f',
                    )
                )
            )
        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    create_plots(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=figsize,
        mode='full',
        periods_per_year=periods_per_year,
        prepare_returns=False,
    )


def basic(
    returns: pd.Series,
    benchmark: Union[pd.Series, pd.DataFrame, str, None] = None,
    benchmark_is_prices: bool = True,
    theme: Union[str, dict] = 'default',
    rf: float = 0,
    figsize: Tuple[float] = (8, 5),
    display: bool = True,
    compounded: bool = True,
    periods_per_year: int = 252,
    match_dates: bool = False,
) -> None:
    # prepare timeseries
    returns = utils._prepare_returns(returns)

    if benchmark is not None:
        if (
            isinstance(benchmark, (pd.Series, pd.DataFrame)) and
            benchmark_is_prices
        ):
            benchmark = benchmark.pct_change()

        benchmark = utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    if utils._in_notebook():
        iDisplay(iHTML('<h4>Performance Metrics</h4>'))
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode='basic',
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
        )
        iDisplay(iHTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode='basic',
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
        )

        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    create_plots(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=figsize,
        mode='basic',
        periods_per_year=periods_per_year,
        prepare_returns=False,
    )


def metrics(
    returns: pd.Series,
    benchmark: Union[pd.Series, pd.DataFrame, str, None] = None,
    benchmark_is_prices: bool = False,
    rf: float = 0,
    display: bool = True,
    mode: str = 'basic',
    sep: bool = False,
    compounded: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
    match_dates: bool = False,
    **kwargs,
) -> pd.DataFrame:
    win_year, _ = _get_trading_periods(periods_per_year)

    benchmark_col = 'Benchmark'
    if benchmark is not None:
        if (
            isinstance(benchmark, (pd.Series, pd.DataFrame)) and
            benchmark_is_prices
        ):
            benchmark = benchmark.pct_change()

        if isinstance(benchmark, str):
            benchmark_col = f'Benchmark ({benchmark.upper()})'

        elif isinstance(benchmark, pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    blank = ['']

    if isinstance(returns, pd.DataFrame):
        if len(returns.columns) > 1:
            raise ValueError(
                "`returns` needs to be a Pandas Series or "
                "one column DataFrame. multi colums DataFrame "
                "was passed"
            )
        returns = returns[returns.columns[0]]

    if prepare_returns:
        returns = utils._prepare_returns(returns)

    df = pd.DataFrame({"returns": returns})

    if benchmark is not None:
        blank = ['', '']
        benchmark = utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
        df["returns"] = returns
        df["benchmark"] = benchmark

    df = df.fillna(0)

    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # create return df
    dd = _calc_dd(df, display=(display or "internal" in kwargs),
                  as_pct=kwargs.get("as_pct", False))

    metrics = pd.DataFrame()

    s_start = {'returns': df['returns'].index.strftime('%Y-%m-%d')[0]}
    s_end = {'returns': df['returns'].index.strftime('%Y-%m-%d')[-1]}
    s_rf = {'returns': rf}

    if "benchmark" in df:
        s_start['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[0]
        s_end['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[-1]
        s_rf['benchmark'] = rf

    metrics['Start Period'] = pd.Series(s_start)
    metrics['End Period'] = pd.Series(s_end)
    metrics['Risk-Free Rate %'] = pd.Series(s_rf)*100
    metrics['Time in Market %'] = stats.exposure(df, prepare_returns=False) * pct

    metrics['~'] = blank

    if compounded:
        metrics['Cumulative Return %'] = (
            stats.comp(df) * pct).map('{:,.2f}'.format)
    else:
        metrics['Total Return %'] = (df.sum() * pct).map('{:,.2f}'.format)

    metrics['CAGR﹪%'] = stats.cagr(df, rf, compounded) * pct

    metrics['~~~~~~~~~~~~~~'] = blank

    metrics['Sharpe'] = stats.sharpe(df, rf, win_year, True)
    metrics['Prob. Sharpe Ratio %'] = (
        stats
        .probabilistic_sharpe_ratio(
            df,
            rf,
            win_year,
            False,
        ) * pct
    )
    if mode.lower() == 'full':
        metrics['Smart Sharpe'] = stats.smart_sharpe(df, rf, win_year, True)

    metrics['Sortino'] = stats.sortino(df, rf, win_year, True)
    if mode.lower() == 'full':
        metrics['Smart Sortino'] = stats.smart_sortino(df, rf, win_year, True)

    metrics['Sortino/√2'] = metrics['Sortino'] / sqrt(2)
    if mode.lower() == 'full':
        metrics['Smart Sortino/√2'] = metrics['Smart Sortino'] / sqrt(2)
    metrics['Omega'] = stats.omega(df, rf, 0., win_year)

    metrics['~~~~~~~~'] = blank
    metrics['Max Drawdown %'] = blank
    metrics['Longest DD Days'] = blank

    if mode.lower() == 'full':
        ret_vol = stats.volatility(
            df['returns'], win_year, True, prepare_returns=False) * pct
        if "benchmark" in df:
            bench_vol = stats.volatility(
                df['benchmark'], win_year, True, prepare_returns=False) * pct
            metrics['Volatility (ann.) %'] = [ret_vol, bench_vol]
            metrics['R^2'] = stats.r_squared(
                df['returns'], df['benchmark'], prepare_returns=False)
            metrics['Information Ratio'] = stats.information_ratio(
                df['returns'], df['benchmark'], prepare_returns=False)
        else:
            metrics['Volatility (ann.) %'] = [ret_vol]

        metrics['Calmar'] = stats.calmar(df, prepare_returns=False)
        metrics['Skew'] = stats.skew(df, prepare_returns=False)
        metrics['Kurtosis'] = stats.kurtosis(df, prepare_returns=False)

        metrics['~~~~~~~~~~'] = blank

        metrics['Expected Daily %%'] = stats.expected_return(
            df, prepare_returns=False) * pct
        metrics['Expected Monthly %%'] = stats.expected_return(
            df, aggregate='M', prepare_returns=False) * pct
        metrics['Expected Yearly %%'] = stats.expected_return(
            df, aggregate='A', prepare_returns=False) * pct
        metrics['Kelly Criterion %'] = stats.kelly_criterion(
            df, prepare_returns=False) * pct
        metrics['Risk of Ruin %'] = stats.risk_of_ruin(
            df, prepare_returns=False)

        metrics['Daily Value-at-Risk %'] = -abs(stats.var(
            df, prepare_returns=False) * pct)
        metrics['Expected Shortfall (cVaR) %'] = -abs(stats.cvar(
            df, prepare_returns=False) * pct)

    metrics['~~~~~~'] = blank

    if mode.lower() == 'full':
        metrics['Max Consecutive Wins *int'] = stats.consecutive_wins(df)
        metrics['Max Consecutive Losses *int'] = stats.consecutive_losses(df)

    metrics['Gain/Pain Ratio'] = stats.gain_to_pain_ratio(df, rf)
    metrics['Gain/Pain (1M)'] = stats.gain_to_pain_ratio(df, rf, "M")

    metrics['~~~~~~~'] = blank

    metrics['Payoff Ratio'] = (
        stats
        .payoff_ratio(
            df,
            prepare_returns=False,
        )
    )
    metrics['Profit Factor'] = (
        stats
        .profit_factor(
            df,
            prepare_returns=False,
        )
    )
    metrics['Common Sense Ratio'] = (
        stats
        .common_sense_ratio(
            df,
            prepare_returns=False,
        )
    )
    metrics['CPC Index'] = (
        stats
        .cpc_index(
            df,
            prepare_returns=False,
        )
    )
    metrics['Tail Ratio'] = (
        stats
        .tail_ratio(
            df,
            prepare_returns=False,
        )
    )
    metrics['Outlier Win Ratio'] = (
        stats
        .outlier_win_ratio(
            df,
            prepare_returns=False,
        )
    )
    metrics['Outlier Loss Ratio'] = (
        stats
        .outlier_loss_ratio(
            df,
            prepare_returns=False,
        )
    )

    # comput returns
    metrics['~~'] = blank
    comp_func = stats.comp if compounded else np.sum

    today = df.index[-1]
    tidx = pd.Timestamp(f'{today.year}-{today.month}-1')
    metrics['MTD %'] = (
        comp_func(
            df[df.index >= tidx]
        ) * pct
    )
    d = today - relativedelta(months=3)
    metrics['3M %'] = (
        comp_func(
            df[df.index >= d]
        ) * pct
    )
    d = today - relativedelta(months=6)
    metrics['6M %'] = (
        comp_func(
            df[df.index >= d]
        ) * pct
    )
    metrics['YTD %'] = (
        comp_func(
            df[df.index >= tidx]
        ) * pct
    )
    d = today - relativedelta(years=1)
    metrics['1Y %'] = (
        comp_func(
            df[df.index >= d]
        ) * pct
    )
    d = today - relativedelta(months=35)
    metrics['3Y (ann.) %'] = (
        stats
        .cagr(
            df[df.index >= d],
            0.,
            compounded,
        ) * pct
    )
    d = today - relativedelta(months=59)
    metrics['5Y (ann.) %'] = (
        stats
        .cagr(
            df[df.index >= d],
            0.,
            compounded,
        ) * pct
    )
    d = today - relativedelta(years=10)
    metrics['10Y (ann.) %'] = (
        stats
        .cagr(
            df[df.index >= d],
            0.,
            compounded,
        ) * pct
    )
    metrics['All-time (ann.) %'] = (
        stats
        .cagr(
            df,
            0.,
            compounded,
        ) * pct
    )
    # best/worst
    if mode.lower() == 'full':
        metrics['~~~'] = blank
        metrics['Best Day %'] = (
            stats
            .best(
                df,
                prepare_returns=False,
            ) * pct
        )
        metrics['Worst Day %'] = (
            stats
            .worst(
                df,
                prepare_returns=False,
            ) * pct
        )
        metrics['Best Month %'] = (
            stats
            .best(
                df,
                aggregate='M',
                prepare_returns=False,
            ) * pct
        )
        metrics['Worst Month %'] = (
            stats
            .worst(
                df,
                aggregate='M',
                prepare_returns=False,
            ) * pct
        )
        metrics['Best Year %'] = (
            stats
            .best(
                df,
                aggregate='A',
                prepare_returns=False,
            ) * pct
        )
        metrics['Worst Year %'] = (
            stats
            .worst(
                df,
                aggregate='A',
                prepare_returns=False,
            ) * pct
        )

    # dd
    metrics['~~~~'] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics['Recovery Factor'] = stats.recovery_factor(df)
    metrics['Ulcer Index'] = stats.ulcer_index(df)
    metrics['Serenity Index'] = stats.serenity_index(df, rf)

    # win rate
    if mode.lower() == 'full':
        metrics['~~~~~'] = blank
        metrics['Avg. Up Month %'] = (
            stats
            .avg_win(
                df,
                aggregate='M',
                prepare_returns=False,
            ) * pct
        )
        metrics['Avg. Down Month %'] = (
            stats
            .avg_loss(
                df,
                aggregate='M',
                prepare_returns=False,
            ) * pct
        )
        metrics['Win Days %%'] = (
            stats
            .win_rate(
                df,
                prepare_returns=False,
            ) * pct
        )
        metrics['Win Month %%'] = (
            stats
            .win_rate(
                df,
                aggregate='M',
                prepare_returns=False,
            ) * pct
        )
        metrics['Win Quarter %%'] = (
            stats
            .win_rate(
                df,
                aggregate='Q',
                prepare_returns=False,
            ) * pct
        )
        metrics['Win Year %%'] = (
            stats
            .win_rate(
                df,
                aggregate='A',
                prepare_returns=False,
            ) * pct
        )

        if "benchmark" in df:
            metrics['~~~~~~~~~~~~'] = blank
            greeks = (
                stats
                .greeks(
                    df['returns'],
                    df['benchmark'],
                    win_year,
                    prepare_returns=False,
                )
            )
            metrics['Beta'] = [str(round(greeks['beta'], 2)), '-']
            metrics['Alpha'] = [str(round(greeks['alpha'], 2)), '-']
            metrics['Correlation'] = [
                str(round(df['benchmark'].corr(df['returns']) * pct, 2))+'%',
                '-',
            ]
            trat = (
                stats
                .treynor_ratio(
                    df['returns'],
                    df['benchmark'],
                    win_year,
                    rf,
                )
            )
            metrics['Treynor Ratio'] = [
                str(round(trat * pct, 2))+'%',
                '-',
            ]

    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except Exception:
            pass
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace('.0', '', regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + '%'
    try:
        metrics['Longest DD Days'] = pd.to_numeric(
            metrics['Longest DD Days']).astype('int')
        metrics['Avg. Drawdown Days'] = pd.to_numeric(
            metrics['Avg. Drawdown Days']).astype('int')

        if display or "internal" in kwargs:
            metrics['Longest DD Days'] = metrics['Longest DD Days'].astype(str)
            metrics['Avg. Drawdown Days'] = (
                metrics['Avg. Drawdown Days']
                .astype(str)
            )
    except Exception:
        metrics['Longest DD Days'] = '-'
        metrics['Avg. Drawdown Days'] = '-'
        if display or "internal" in kwargs:
            metrics['Longest DD Days'] = '-'
            metrics['Avg. Drawdown Days'] = '-'

    metrics.columns = [
        col if '~' not in col else '' for col in metrics.columns]
    metrics.columns = [
        col[:-1] if '%' in col else col for col in metrics.columns]
    metrics = metrics.T

    if "benchmark" in df:
        metrics.columns = ['Strategy', benchmark_col]
    else:
        metrics.columns = ['Strategy']

    # cleanups
    metrics.replace([-0, '-0'], 0, inplace=True)
    metrics.replace([np.nan, -np.nan, np.inf, -np.inf,
                     '-nan%', 'nan%', '-nan', 'nan',
                    '-inf%', 'inf%', '-inf', 'inf'], '-', inplace=True)

    if display:
        print(tabulate(metrics, headers="keys", tablefmt='simple'))
        return None

    if not sep:
        metrics = metrics[metrics.index != '']

    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(' %', '').replace(' *int', '').strip()
        for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics


def create_plots(
    returns: pd.Series,
    benchmark: Union[pd.Series, str, None] = None,
    benchmark_is_prices: bool = False,
    theme: Union[str, dict] = 'default',
    figsize: Tuple[float] = (8, 5),
    mode: str = 'basic',
    compounded: bool = True,
    periods_per_year: int = 252,
    prepare_returns: int = True,
    match_dates: bool = False,
) -> None:

    win_year, win_half_year = _get_trading_periods(periods_per_year)

    if prepare_returns:
        returns = utils._prepare_returns(returns)

    if mode.lower() != 'full':
        plots.snapshot(
            returns=returns,
            theme=theme,
            figsize=(figsize[0], figsize[0]),
            mode='comp' if compounded else 'sum',
            show=True,
        )

        plots.monthly_heatmap(
            returns=returns,
            theme=theme,
            figsize=(figsize[0], figsize[0]*.5),
            ylabel=False,
            compounded=compounded,
            show=True,
        )

        return

    # prepare timeseries
    if benchmark is not None:
        if (
            isinstance(benchmark, (pd.Series, pd.DataFrame)) and
            benchmark_is_prices
        ):
            benchmark = benchmark.pct_change()

        benchmark = utils._prepare_benchmark(benchmark, returns.index)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    plots.returns(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.6),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )

    plots.log_returns(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.5),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )

    if benchmark is not None:
        plots.returns(
            returns=returns,
            benchmark=benchmark,
            theme=theme,
            match_volatility=True,
            figsize=(figsize[0], figsize[0]*.5),
            ylabel=False,
            prepare_returns=False,
            show=True,
        )

    plots.yearly_returns(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.5),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )

    plots.histogram(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.5),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )

    plots.daily_returns(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.3),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )

    if benchmark is not None:
        plots.rolling_beta(
            returns=returns,
            benchmark=benchmark,
            theme=theme,
            window1=win_half_year,
            window2=win_year,
            figsize=(figsize[0], figsize[0]*.3),
            ylabel=False,
            prepare_returns=False,
            show=True,
        )

    plots.rolling_volatility(
        returns=returns,
        benchmark=benchmark,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.3),
        ylabel=False,
        period=win_half_year,
        show=True,
    )

    plots.rolling_sharpe(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.3),
        ylabel=False,
        period=win_half_year,
        show=True,
    )

    plots.rolling_sortino(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.3),
        ylabel=False,
        period=win_half_year,
        show=True,
    )

    plots.drawdowns_periods(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.5),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )

    plots.drawdown(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.4),
        ylabel=False,
        show=True,
    )

    plots.monthly_heatmap(
        returns=returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.5),
        ylabel=False,
        show=True,
    )

    plots.distribution(
        returns,
        theme=theme,
        figsize=(figsize[0], figsize[0]*.5),
        ylabel=False,
        prepare_returns=False,
        show=True,
    )


def _calc_dd(
    df: pd.DataFrame,
    display: bool = True,
    as_pct: bool = False,
) -> pd.DataFrame:
    dd = stats.to_drawdown_series(df)
    dd_info = stats.drawdown_details(dd)

    if dd_info.empty:
        return pd.DataFrame()

    if 'returns' in dd_info:
        ret_dd = dd_info['returns']
    else:
        ret_dd = dd_info

    dd_stats = {
        'returns': {
            'Max Drawdown %': (
                ret_dd['max drawdown'].min() / 100
            ),
            'Longest DD Days': (
                ret_dd['days'].max().round()
            ),
            'Avg. Drawdown %': ret_dd['max drawdown'].mean() / 100,
            'Avg. Drawdown Days': ret_dd['days'].mean().round(),
        }
    }
    if "benchmark" in df and (dd_info.columns, pd.MultiIndex):
        bench_dd = dd_info['benchmark'].sort_values(by='max drawdown')
        dd_stats['benchmark'] = {
            'Max Drawdown %': (
                bench_dd['max drawdown'].min() / 100
            ),
            'Longest DD Days': (
                bench_dd['days'].max().round()
            ),
            'Avg. Drawdown %': bench_dd['max drawdown'].mean() / 100,
            'Avg. Drawdown Days': bench_dd['days'].mean().round(),
        }

    # pct multiplier
    pct = 100 if display or as_pct else 1

    dd_stats = pd.DataFrame(dd_stats).T
    dd_stats['Max Drawdown %'] = dd_stats['Max Drawdown %'].astype(float) * pct
    dd_stats['Avg. Drawdown %'] = dd_stats['Avg. Drawdown %'].astype(float) * pct

    return dd_stats.T


def _html_table(obj, showindex="default"):
    obj = tabulate(
        obj,
        headers='keys',
        tablefmt='html',
        floatfmt='.2f',
        showindex=showindex,
    )
    obj = obj.replace(' style="text-align: right;"', '')
    obj = obj.replace(' style="text-align: left;"', '')
    obj = obj.replace(' style="text-align: center;"', '')
    obj = re.sub('<td> +', '<td>', obj)
    obj = re.sub(' +</td>', '</td>', obj)
    obj = re.sub('<th> +', '<th>', obj)
    obj = re.sub(' +</th>', '</th>', obj)

    return obj


def _embed_figure(figfile, figfmt):
    figbytes = figfile.getvalue()
    if figfmt == 'svg':
        return figbytes.decode()
    data_uri = b64encode(figbytes).decode()
    return '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)

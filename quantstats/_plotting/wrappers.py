#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
#
# Copyright 2019 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pandas as pd
import sys
import warnings

from matplotlib.figure import Figure
from typing import Union, Tuple


from .. import stats, utils
from .core import PlottingWrappers

try:
    import plotly
except ModuleNotFoundError:
    pass


def to_plotly(fig):
    if 'plotly' in sys.modules:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig = plotly.tools.mpl_to_plotly(fig)

        return plotly.plotly.iplot(
            fig,
            filename='quantstats-plot',
            overwrite=True,
        )

    return fig


def snapshot(
    returns: pd.Series,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 8),
    title: str = 'Portfolio Summary',
    linewidth: float = 1.5,
    mode: str = 'comp',
    log_scale: bool = False,
    subtitle=True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    returns = (
        utils
        .make_portfolio(returns, 1, mode)
        .pct_change()
        .fillna(0)
    )
    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_snapshot(
        returns=returns,
        figsize=figsize,
        title=title,
        linewidth=linewidth,
        mode=mode,
        subtitle=subtitle,
        log_scale=log_scale,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def earnings(
    returns,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 6),
    start_balance: float = 1e5,
    mode: str = 'comp',
    title: str = 'Portfolio Earnings',
    linewidth: float = 1.5,
    subtitle: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    returns = (
        utils
        .make_portfolio(
            returns=returns,
            start_balance=start_balance,
            mode=mode,
        )
    )
    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_earnings(
        returns=returns,
        figsize=figsize,
        title=title,
        linewidth=linewidth,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def returns(
    returns: pd.Series,
    benchmark: Union[str, pd.Series, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 6),
    linewidth: float = 1.5,
    match_volatility=False,
    compound: bool = True,
    cumulative: bool = True,
    resample: bool = None,
    ylabel: str = 'Cumulative Returns',
    subtitle: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> None:
    title = 'Cumulative Returns' if compound else 'Returns'
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += f' vs {benchmark.upper()}'
        else:
            title += ' vs Benchmark'
        if match_volatility:
            title += ' (Volatility Matched)'

    if prepare_returns:
        returns = utils._prepare_returns(returns)

    benchmark = utils._prepare_benchmark(benchmark, returns.index)

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_timeseries(
        returns=returns,
        benchmark=benchmark,
        title=title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=False,
        resample=resample,
        compound=compound,
        cumulative=cumulative,
        linewidth=linewidth,
        figsize=figsize,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def log_returns(
    returns: pd.Series,
    benchmark: Union[str, pd.Series, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 6),
    linewidth: float = 1.5,
    match_volatility=False,
    compound: bool = True,
    cumulative: bool = True,
    resample: bool = None,
    ylabel: str = 'Cumulative Returns',
    subtitle: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    title = 'Cumulative Returns' if compound else 'Returns'
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += f' vs {benchmark.upper()} (Log Scaled'
        else:
            title += ' vs Benchmark (Log Scaled'
        if match_volatility:
            title += ', Volatility Matched'
    else:
        title += ' (Log Scaled'
    title += ')'

    if prepare_returns:
        returns = utils._prepare_returns(returns)

    benchmark = (
        utils
        ._prepare_benchmark(
            benchmark,
            returns.index,
        )
    )

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_timeseries(
        returns=returns,
        benchmark=benchmark,
        title=title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=True,
        resample=resample,
        compound=compound,
        cumulative=cumulative,
        linewidth=linewidth,
        figsize=figsize,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def daily_returns(
    returns: pd.Series,
    benchmark: Union[str, pd.Series, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 4),
    linewidth: float = 0.5,
    log_scale: bool = False,
    ylabel: str = 'Returns',
    subtitle: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    if prepare_returns:
        returns = utils._prepare_returns(returns)

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_timeseries(
        returns=returns,
        benchmark=benchmark,
        title='Daily Returns',
        ylabel=ylabel,
        match_volatility=False,
        log_scale=log_scale,
        resample='D',
        compound=False,
        linewidth=linewidth,
        figsize=figsize,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def yearly_returns(
    returns,
    benchmark: Union[pd.Series, str, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 5),
    hlinewidth=1.5,
    hlinelabel : Union[str, None] = None,
    match_volatility: bool = False,
    log_scale: bool = False,
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    title = 'EOY Returns'
    if benchmark is not None:
        title += '  vs Benchmark'
        benchmark = (
            utils
            ._prepare_benchmark(
                benchmark,
                returns.index,
            )
            .resample('A')
            .apply(stats.comp)
            .resample('A')
            .last()
        )

    if prepare_returns:
        returns = utils._prepare_returns(returns)

    if compounded:
        returns = (
            returns
            .resample('A')
            .apply(stats.comp)
            .resample('A')
            .last()
        )
    else:
        returns = (
            returns
            .resample('A')
            .apply(np.sum)
            .resample('A')
            .last()
        )

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_returns_bars(
        returns=returns,
        benchmark=benchmark,
        hline=returns.mean(),
        hlinewidth=hlinewidth,
        hlinelabel=hlinelabel,
        title=title,
        match_volatility=match_volatility,
        log_scale=log_scale,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def distribution(
    returns: pd.Series,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 6),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    if prepare_returns:
        returns = utils._prepare_returns(returns)

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_distribution(
        returns=returns,
        figsize=figsize,
        ylabel=ylabel,
        compounded=compounded,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def histogram(
    returns: pd.Series,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 5),
    resample: str = 'M',
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    if prepare_returns:
        returns = utils._prepare_returns(returns)

    lookup = {
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly',
        'A': 'Annual',
    }
    title = f'Distribution of {lookup.get(resample)} Returns'

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_histogram(
        returns=returns,
        resample=resample,
        title=title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def drawdown(
    returns: pd.Series,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 5),
    linewidth: float = 1,
    log_scale: bool = False,
    match_volatility: bool = False,
    compound: bool = False,
    ylabel: str = 'Drawdown',
    resample: Union[str, None] = None,
    subtitle: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    dd = stats.to_drawdown_series(returns)

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_timeseries(
        returns=dd,
        title='Underwater Plot',
        compound=compound,
        fill=True,
        returns_label='Drawdown',
        hline=dd.mean(),
        hlinewidth=2,
        hlinelabel='Average',
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample=resample,
        linewidth=linewidth,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def drawdowns_periods(
    returns: pd.Series,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 5),
    periods: int = 5,
    linewidth: float = 1.5,
    log_scale: bool = False,
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    prepare_returns=True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    if prepare_returns:
        returns = utils._prepare_returns(returns)

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_longest_drawdowns(
        returns=returns,
        periods=periods,
        compounded=compounded,
        linewidth=linewidth,
        log_scale=log_scale,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def rolling_beta(
    returns: pd.Series,
    benchmark: Union[pd.DataFrame, str, None],
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 3),
    window1: int = 126,
    window1_label: str = '6-Months',
    window2: int = 252,
    window2_label: str = '12-Months',
    linewidth: float = 1.5,
    ylabel: bool = True,
    subtitle: bool = True,
    prepare_returns: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    if prepare_returns:
        returns = (
            utils
            ._prepare_returns(returns)
        )
    benchmark = (
        utils
        ._prepare_benchmark(
            benchmark,
            returns.index,
        )
    )
    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_rolling_beta(
        returns=returns,
        benchmark=benchmark,
        window1=window1,
        window1_label=window1_label,
        window2=window2,
        window2_label=window2_label,
        title='Rolling Beta to Benchmark',
        figsize=figsize,
        linewidth=linewidth,
        ylabel=ylabel,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def rolling_volatility(
    returns: pd.Series,
    benchmark: Union[pd.DataFrame, str, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 3),
    period: int = 126,
    period_label: str = '6-Months',
    periods_per_year: int = 252,
    linewidth: float = 1.5,
    ylabel: str = 'Volatility',
    subtitle: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    returns = (
        stats
        .rolling_volatility(
            returns,
            period,
            periods_per_year,
        )
    )
    if benchmark is not None:
        benchmark = (
            utils
            ._prepare_benchmark(
                benchmark,
                returns.index,
            )
        )
        benchmark = (
            stats
            .rolling_volatility(
                benchmark,
                period,
                periods_per_year,
                prepare_returns=False,
            )
        )
    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_rollingstats(
        returns=returns,
        benchmark=benchmark,
        title=f'Rolling Volatility ({period_label})',
        linewidth=linewidth,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def rolling_sharpe(
    returns: pd.Series,
    benchmark: Union[pd.DataFrame, str, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 3),
    rf: float = 0,
    period: int = 126,
    period_label: str = '6-Months',
    periods_per_year: int = 252,
    linewidth: float = 1.25,
    ylabel: str = 'Sharpe',
    subtitle: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    returns = (
        stats
        .rolling_sharpe(
            returns,
            rf,
            period,
            True,
            periods_per_year,
        )
    )
    if benchmark is not None:
        benchmark = (
            utils
            ._prepare_benchmark(
                benchmark,
                returns.index,
                rf,
            )
        )
        benchmark = (
            stats
            .rolling_sharpe(
                benchmark,
                rf,
                period,
                True,
                periods_per_year,
                prepare_returns=False
            )
        )

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_rollingstats(
        returns=returns,
        benchmark=benchmark,
        hline=returns.mean(),
        ylabel=ylabel,
        title=f'Rolling Sharpe ({period_label})',
        linewidth=linewidth,
        figsize=figsize,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def rolling_sortino(
    returns: pd.Series,
    benchmark: Union[pd.DataFrame, str, None] = None,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 3),
    rf: float = 0,
    period: int = 126,
    period_label: str = '6-Months',
    periods_per_year: int = 252,
    linewidth: float = 1.25,
    ylabel: str = 'Sortino',
    subtitle: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    returns = stats.rolling_sortino(
        returns,
        rf,
        period,
        True,
        periods_per_year,
    )

    if benchmark is not None:
        benchmark = utils._prepare_benchmark(
            benchmark,
            returns.index,
            rf,
        )
        benchmark = stats.rolling_sortino(
            benchmark,
            rf,
            period,
            True,
            periods_per_year,
            prepare_returns=False,
        )

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_rollingstats(
        returns=returns,
        benchmark=benchmark,
        hline=returns.mean(),
        ylabel=ylabel,
        title=f'Rolling Sortino ({period_label})',
        linewidth=linewidth,
        figsize=figsize,
        subtitle=subtitle,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None


def monthly_heatmap(
    returns: pd.Series,
    theme: Union[str, dict] = 'default',
    fontname: str = 'Arial',
    font_scale: float = 1.1,
    figsize: Tuple[float] = (10, 5),
    annot_size: float = 10,
    cbar: bool = True,
    square: bool = False,
    compounded: bool = True,
    eoy: bool = False,
    ylabel: bool = True,
    show: bool = True,
    export: bool = False,
    image_file_name: Union[str, None] = None,
    **savefig_kwargs,
) -> Union[Figure, None]:
    returns = (
        stats
        .monthly_returns(
            returns=returns,
            eoy=eoy,
            compounded=compounded,
        ) * 100
    )

    fig_width = figsize[0]
    fig_height = max((figsize[1], len(returns) / 3))

    if cbar:
        fig_width *= 1.04

    figsize = (fig_width, fig_height)

    pwrap = PlottingWrappers(
        theme=theme,
        fontname=fontname,
        font_scale=font_scale,
        show=show,
        export=export,
        **savefig_kwargs,
    )
    fig = pwrap.plot_monthly_heatmap(
        returns=returns,
        annot_size=annot_size,
        figsize=figsize,
        cbar=cbar,
        square=square,
        ylabel=ylabel,
        image_file_name=image_file_name,
    )
    if fig:
        return fig

    return None

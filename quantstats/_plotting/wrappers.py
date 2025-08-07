#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
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

import warnings
import matplotlib.pyplot as _plt
from matplotlib.ticker import (
    StrMethodFormatter as _StrMethodFormatter,
    FuncFormatter as _FuncFormatter,
)

import numpy as _np
import pandas as _pd
from .._compat import safe_resample
import seaborn as _sns

from .. import (
    stats as _stats,
    utils as _utils,
)

from . import core as _core


_FLATUI_COLORS = ["#fedd78", "#348dc1", "#af4b64", "#4fa487", "#9b59b6", "#808080"]
_GRAYSCALE_COLORS = (len(_FLATUI_COLORS) * ["black"]) + ["white"]

# Check if plotly is available for optional conversion functionality
_HAS_PLOTLY = False
try:
    import plotly

    _HAS_PLOTLY = True
except ImportError:
    pass


def to_plotly(fig):
    """
    Convert a matplotlib figure to a Plotly interactive plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert to Plotly format.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive Plotly figure if plotly is available, otherwise original figure.

    Notes
    -----
    This function requires the plotly library to be installed. If plotly is not
    available, the original matplotlib figure is returned unchanged.
    """
    # Return original figure if plotly not available
    if not _HAS_PLOTLY:
        return fig

    # Suppress warnings during conversion to avoid noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Convert matplotlib figure to plotly format
        fig = plotly.tools.mpl_to_plotly(fig)
        # Upload and display the interactive plot
        return plotly.plotly.iplot(fig, filename="quantstats-plot", overwrite=True)  # type: ignore


def snapshot(
    returns,
    grayscale=False,
    figsize=(10, 8),
    title="Portfolio Summary",
    fontname="Arial",
    lw=1.5,
    mode="comp",
    subtitle=True,
    savefig=None,
    show=True,
    log_scale=False,
    **kwargs,
):
    """
    Generate a comprehensive portfolio performance snapshot with multiple subplots.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data. If DataFrame with multiple columns, uses mean or specific column.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    title : str, optional
        Main title for the plot (default: "Portfolio Summary").
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 1.5).
    mode : str, optional
        Calculation mode: "comp" for compound returns or "sum" for simple sum.
    subtitle : bool, optional
        Whether to show subtitle with date range and Sharpe ratio (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axes (default: False).
    **kwargs : dict
        Additional keyword arguments, including strategy_col for column selection.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Creates a three-panel plot showing:
    1. Cumulative returns over time
    2. Drawdown periods
    3. Daily returns distribution
    """
    # Extract strategy column name from kwargs
    strategy_colname = kwargs.get("strategy_col", "Strategy")

    # Handle multi-column DataFrame input
    multi_column = False
    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            # Check if specific strategy column exists
            if strategy_colname in returns.columns:
                returns = returns[strategy_colname]
            else:
                # Use mean of all columns if no specific column found
                multi_column = True
                returns = returns.mean(axis=1)
                title = title + " (daily equal-weighted*)"
        returns.columns = strategy_colname

    # Select color scheme based on grayscale preference
    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    # Convert to portfolio format and calculate percentage changes
    returns = _utils.make_portfolio(returns.dropna(), 1, mode).pct_change().fillna(0)

    # Use current figure size if not specified
    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.75)

    # Create figure with three subplots: cumulative returns, drawdown, daily returns
    fig, axes = _plt.subplots(
        3, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # Add footnote for multi-column DataFrame explanation
    if multi_column:
        _plt.figtext(
            0,
            -0.05,
            "            * When a multi-column DataFrame is passed, the mean of all columns will be used as returns.\n"
            "              To change this behavior, use a pandas Series or pass the column name in the "
            "`strategy_col` parameter.",
            ha="left",
            fontsize=11,
            color="black",
            alpha=0.6,
            linespacing=1.5,
        )

    # Remove spines from all axes for cleaner appearance
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Set main title
    fig.suptitle(
        title, fontsize=14, y=0.97, fontname=fontname, fontweight="bold", color="black"
    )

    fig.set_facecolor("white")

    # Add subtitle with date range and Sharpe ratio
    if subtitle:
        if isinstance(returns, _pd.Series):
            axes[0].set_title(
                "%s - %s ;  Sharpe: %.2f                      \n"
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),  # type: ignore
                    returns.index.date[-1:][0].strftime("%e %b '%y"),  # type: ignore
                    _stats.sharpe(returns),
                ),
                fontsize=12,
                color="gray",
            )
        elif isinstance(returns, _pd.DataFrame):
            axes[0].set_title(
                "\n%s - %s ;  "
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),  # type: ignore
                    returns.index.date[-1:][0].strftime("%e %b '%y"),  # type: ignore
                ),
                fontsize=12,
                color="gray",
            )

    # Configure first subplot: Cumulative Returns
    axes[0].set_ylabel(
        "Cumulative Return", fontname=fontname, fontweight="bold", fontsize=12
    )

    # Plot cumulative returns for Series or DataFrame
    if isinstance(returns, _pd.Series):
        # Calculate cumulative returns based on mode
        if mode.lower() in ["cumsum", "sum"]:
            cum_ret = returns.cumsum() * 100
        else:
            cum_ret = _stats.compsum(returns) * 100
        # Plot cumulative returns line
        axes[0].plot(
            cum_ret,
            color=colors[1],
            lw=1 if grayscale else lw,
            zorder=1,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Plot each column separately for DataFrame
        for col in returns.columns:
            if mode.lower() in ["cumsum", "sum"]:
                cum_ret = returns[col].cumsum() * 100
            else:
                cum_ret = _stats.compsum(returns[col]) * 100
            axes[0].plot(
                cum_ret,
                label=col,
                lw=1 if grayscale else lw,
                zorder=1,
            )
    # Add horizontal line at zero
    axes[0].axhline(0, color="silver", lw=1, zorder=0)

    # Set y-axis scale based on log_scale parameter
    axes[0].set_yscale("symlog" if log_scale else "linear")
    # axes[0].legend(fontsize=12)

    # Configure second subplot: Drawdown
    dd = _stats.to_drawdown_series(returns) * 100
    # Calculate appropriate tick spacing for drawdown
    ddmin = _utils._round_to_closest(abs(dd.min()), 5)
    ddmin_ticks = 5
    if ddmin > 50:
        ddmin_ticks = ddmin / 4
    elif ddmin > 20:
        ddmin_ticks = ddmin / 3
    ddmin_ticks = int(_utils._round_to_closest(ddmin_ticks, 5))

    # ddmin_ticks = int(_utils._round_to_closest(ddmin, 5))
    axes[1].set_ylabel("Drawdown", fontname=fontname, fontweight="bold", fontsize=12)
    axes[1].set_yticks(_np.arange(-ddmin, 0, step=ddmin_ticks))

    # Plot drawdown series
    if isinstance(dd, _pd.Series):
        axes[1].plot(dd, color=colors[2], lw=1 if grayscale else lw, zorder=1)
    elif isinstance(dd, _pd.DataFrame):
        for col in dd.columns:
            axes[1].plot(dd[col], label=col, lw=1 if grayscale else lw, zorder=1)
    axes[1].axhline(0, color="silver", lw=1, zorder=0)

    # Add filled area under drawdown curve if not grayscale
    if not grayscale:
        if isinstance(dd, _pd.Series):
            axes[1].fill_between(dd.index, 0, dd, color=colors[2], alpha=0.25)
        elif isinstance(dd, _pd.DataFrame):
            for i, col in enumerate(dd.columns):
                axes[1].fill_between(
                    dd[col].index, 0, dd[col], color=colors[i + 1], alpha=0.25
                )

    axes[1].set_yscale("symlog" if log_scale else "linear")
    # axes[1].legend(fontsize=12)

    # Configure third subplot: Daily Returns
    axes[2].set_ylabel(
        "Daily Return", fontname=fontname, fontweight="bold", fontsize=12
    )

    # Plot daily returns
    if isinstance(returns, _pd.Series):
        axes[2].plot(
            returns * 100, color=colors[0], label=returns.name, lw=0.5, zorder=1
        )
    elif isinstance(returns, _pd.DataFrame):
        for i, col in enumerate(returns.columns):
            axes[2].plot(
                returns[col] * 100, color=colors[i], label=col, lw=0.5, zorder=1
            )
    # Add horizontal lines at zero
    axes[2].axhline(0, color="silver", lw=1, zorder=0)
    axes[2].axhline(0, color=colors[-1], linestyle="--", lw=1, zorder=2)

    axes[2].set_yscale("symlog" if log_scale else "linear")
    # axes[2].legend(fontsize=12)

    # Calculate appropriate tick spacing for daily returns
    retmax = _utils._round_to_closest(returns.max() * 100, 5)
    retmin = _utils._round_to_closest(returns.min() * 100, 5)
    retdiff = retmax - retmin
    steps = 5
    if retdiff > 50:
        steps = retdiff / 5
    elif retdiff > 30:
        steps = retdiff / 4
    steps = _utils._round_to_closest(steps, 5)
    axes[2].set_yticks(_np.arange(retmin, retmax, step=steps))

    # Apply common formatting to all axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_major_formatter(_StrMethodFormatter("{x:,.0f}%"))

    # Adjust layout
    _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    fig.autofmt_xdate()

    # Apply layout adjustments with error handling
    try:
        _plt.subplots_adjust(hspace=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    # Show plot if requested
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def earnings(
    returns,
    start_balance=1e5,
    mode="comp",
    grayscale=False,
    figsize=(10, 6),
    title="Portfolio Earnings",
    fontname="Arial",
    lw=1.5,
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot portfolio earnings over time showing absolute dollar value growth.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    start_balance : float, optional
        Starting portfolio balance in dollars (default: 100000).
    mode : str, optional
        Calculation mode: "comp" for compound returns or "sum" for simple sum.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    title : str, optional
        Main title for the plot (default: "Portfolio Earnings").
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for the earnings line (default: 1.5).
    subtitle : bool, optional
        Whether to show subtitle with date range and P&L (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows portfolio value over time starting from the specified balance.
    Highlights the maximum portfolio value achieved during the period.
    """
    # Select color scheme and transparency based on grayscale preference
    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    alpha = 0.5 if grayscale else 0.8

    # Convert returns to portfolio dollar values
    returns = _utils.make_portfolio(returns, start_balance, mode)

    # Use current figure size if not specified
    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.55)

    # Create single subplot figure
    fig, ax = _plt.subplots(figsize=figsize)

    # Remove spines for cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title
    fig.suptitle(
        f"    {title}",
        fontsize=12,
        y=0.95,
        fontname=fontname,
        fontweight="bold",
        color="black",
    )

    # Add subtitle with date range and P&L information
    if subtitle:
        ax.set_title(
            "\n%s - %s ;  P&L: %s (%s)                "
            % (
                returns.index.date[1:2][0].strftime("%e %b '%y"),  # type: ignore
                returns.index.date[-1:][0].strftime("%e %b '%y"),  # type: ignore
                _utils._score_str(
                    "${:,}".format(round(returns.values[-1] - returns.values[0], 2))
                ),
                _utils._score_str(
                    "{:,}%".format(
                        round((returns.values[-1] / returns.values[0] - 1) * 100, 2)
                    )
                ),
            ),
            fontsize=10,
            color="gray",
        )

    # Find and highlight maximum portfolio value
    mx = returns.max()
    returns_max = returns[returns == mx]
    ix = returns_max[~_np.isnan(returns_max)].index[0]
    returns_max = _np.where(returns.index == ix, mx, _np.nan)

    # Plot maximum value point as a marker
    ax.plot(
        returns.index,
        returns_max,
        marker="o",
        lw=0,
        alpha=alpha,
        markersize=12,
        color=colors[0],
    )

    # Plot main earnings line
    ax.plot(returns.index, returns, color=colors[1], lw=1 if grayscale else lw)

    # Set y-axis label showing starting balance
    ax.set_ylabel(
        "Value of  ${:,.0f}".format(start_balance),
        fontname=fontname,
        fontweight="bold",
        fontsize=11,
    )

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(_FuncFormatter(_core.format_cur_axis))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    _plt.xticks(fontsize=11)
    _plt.yticks(fontsize=11)

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    fig.autofmt_xdate()

    # Apply layout adjustments with error handling
    try:
        _plt.subplots_adjust(hspace=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    # Show plot if requested
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def returns(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(10, 6),
    fontname="Arial",
    lw=1.5,
    match_volatility=False,
    compound=True,
    resample=None,
    ylabel="Cumulative Returns",
    subtitle=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    """
    Plot cumulative returns over time, optionally compared to a benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 1.5).
    match_volatility : bool, optional
        If True, matches volatility between returns and benchmark (default: False).
    compound : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    resample : str, optional
        Resampling frequency (e.g., 'M' for monthly, 'Q' for quarterly).
    ylabel : str, optional
        Y-axis label (default: "Cumulative Returns").
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Creates a time series plot of cumulative returns. If benchmark is provided,
    both series are plotted for comparison.
    """
    # Build title based on parameters
    title = "Cumulative Returns" if compound else "Returns"
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += " vs %s" % benchmark.upper()
        else:
            title += " vs Benchmark"
        if match_volatility:
            title += " (Volatility Matched)"

        # Prepare benchmark data to match returns index
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Use core plotting function for time series
    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=False,
        resample=resample,
        compound=compound,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def log_returns(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(10, 5),
    fontname="Arial",
    lw=1.5,
    match_volatility=False,
    compound=True,
    resample=None,
    ylabel="Cumulative Returns",
    subtitle=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    """
    Plot cumulative returns on a logarithmic scale for better trend visualization.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 1.5).
    match_volatility : bool, optional
        If True, matches volatility between returns and benchmark (default: False).
    compound : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    resample : str, optional
        Resampling frequency (e.g., 'M' for monthly, 'Q' for quarterly).
    ylabel : str, optional
        Y-axis label (default: "Cumulative Returns").
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Similar to returns() but uses logarithmic scale which is better for visualizing
    exponential growth and making percentage changes more comparable across time.
    """
    # Build title with log scale indication
    title = "Cumulative Returns" if compound else "Returns"
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += " vs %s (Log Scaled" % benchmark.upper()
        else:
            title += " vs Benchmark (Log Scaled"
        if match_volatility:
            title += ", Volatility Matched"
    else:
        title += " (Log Scaled"
    title += ")"

    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Prepare benchmark data to match returns index
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)  # type: ignore

    # Use core plotting function with log scale enabled
    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=True,
        resample=resample,
        compound=compound,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def daily_returns(
    returns,
    benchmark,
    grayscale=False,
    figsize=(10, 4),
    fontname="Arial",
    lw=0.5,
    log_scale=False,
    ylabel="Returns",
    subtitle=True,
    savefig=None,
    show=True,
    prepare_returns=True,
    active=False,
):
    """
    Plot daily returns over time, optionally as active returns vs benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str
        Benchmark returns data or ticker symbol.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 4)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for plots (default: 0.5).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    ylabel : str, optional
        Y-axis label (default: "Returns").
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).
    active : bool, optional
        If True, plots active returns (returns - benchmark) (default: False).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows daily return variations over time. If active=True, displays the difference
    between portfolio returns and benchmark returns.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
        # Calculate active returns if requested
        if active and benchmark is not None:
            benchmark = _utils._prepare_returns(benchmark)
            returns = returns - benchmark

    # Set plot title based on active returns setting
    plot_title = "Daily Active Returns" if active else "Daily Returns"
    plot_title += " (Cumulative Sum)"

    # Use core plotting function for daily time series
    fig = _core.plot_timeseries(
        returns,
        None,  # No benchmark for daily returns plot
        plot_title,
        ylabel=ylabel,
        match_volatility=False,
        log_scale=log_scale,
        resample="D",  # Daily resampling
        compound=False,  # No compounding for daily returns
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def yearly_returns(
    returns,
    benchmark=None,
    fontname="Arial",
    grayscale=False,
    hlw=1.5,
    hlcolor="red",
    hllabel="",
    match_volatility=False,
    log_scale=False,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    """
    Plot end-of-year returns as a bar chart, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    hlw : float, optional
        Horizontal line width for mean line (default: 1.5).
    hlcolor : str, optional
        Color for horizontal mean line (default: "red").
    hllabel : str, optional
        Label for horizontal mean line (default: "").
    match_volatility : bool, optional
        If True, matches volatility between returns and benchmark (default: False).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with date range and statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple sum (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Aggregates returns by year and displays as bars. Shows mean return as horizontal line.
    """
    # Set plot title
    title = "EOY Returns"
    if benchmark is not None:
        title += "  vs Benchmark"
        # Prepare and resample benchmark data
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        benchmark = safe_resample(benchmark, "YE", _stats.comp)
        benchmark = safe_resample(benchmark, "YE", "last")

    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Resample returns to year-end based on compounding preference
    if compounded:
        returns = safe_resample(returns, "YE", _stats.comp)
    else:
        returns = safe_resample(returns, "YE", "sum")
    returns = safe_resample(returns, "YE", "last")

    # Use core plotting function for bar chart
    fig = _core.plot_returns_bars(
        returns,
        benchmark,
        fontname=fontname,
        hline=returns.mean(),  # Show mean as horizontal line
        hlw=hlw,
        hllabel=hllabel,
        hlcolor=hlcolor,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample="YE",
        title=title,
        figsize=figsize,
        grayscale=grayscale,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def distribution(
    returns,
    fontname="Arial",
    grayscale=False,
    ylabel=True,
    figsize=(10, 6),
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    title=None,
    prepare_returns=True,
):
    """
    Plot the distribution of returns using histogram and density curves.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    subtitle : bool, optional
        Whether to show subtitle with distribution statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    title : str, optional
        Custom title for the plot (default: None).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows the distribution of returns with histogram bars and overlaid density curve.
    Helpful for understanding return characteristics and identifying outliers.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Use core plotting function for distribution
    fig = _core.plot_distribution(
        returns,
        fontname=fontname,
        grayscale=grayscale,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        title=title,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def histogram(
    returns,
    benchmark=None,
    resample="ME",
    fontname="Arial",
    grayscale=False,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    """
    Plot histogram of returns resampled to specified frequency.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    resample : str, optional
        Resampling frequency: 'W' for weekly, 'ME' for monthly, 'QE' for quarterly,
        'YE' for yearly (default: 'ME').
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with distribution statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Resamples returns to specified frequency and plots distribution histogram.
    Useful for analyzing return patterns at different time horizons.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
        if benchmark is not None:
            benchmark = _utils._prepare_returns(benchmark)

    # Determine title based on resampling frequency
    if resample == "W":
        title = "Weekly "
    elif resample == "ME":
        title = "Monthly "
    elif resample == "QE":
        title = "Quarterly "
    elif resample == "YE":
        title = "Annual "
    else:
        title = ""

    # Use core plotting function for histogram
    return _core.plot_histogram(
        returns,
        benchmark,
        resample=resample,
        grayscale=grayscale,
        fontname=fontname,
        title="Distribution of %sReturns" % title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )


def drawdown(
    returns,
    grayscale=False,
    figsize=(10, 5),
    fontname="Arial",
    lw=1,
    log_scale=False,
    match_volatility=False,
    compound=False,
    ylabel="Drawdown",
    resample=None,
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot drawdown series over time showing periods of loss from peak values.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    lw : float, optional
        Line width for drawdown plot (default: 1).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    match_volatility : bool, optional
        Not used in drawdown plot (default: False).
    compound : bool, optional
        Not used in drawdown plot (default: False).
    ylabel : str, optional
        Y-axis label (default: "Drawdown").
    resample : str, optional
        Resampling frequency for data aggregation.
    subtitle : bool, optional
        Whether to show subtitle with drawdown statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows underwater plot of drawdowns with filled area. Includes average drawdown
    line as reference. Useful for understanding portfolio risk and recovery periods.
    """
    # Convert returns to drawdown series
    dd = _stats.to_drawdown_series(returns)

    # Use core plotting function for drawdown time series
    fig = _core.plot_timeseries(
        dd,
        title="Underwater Plot",
        hline=dd.mean(),  # Show average drawdown as horizontal line
        hlw=2,
        hllabel="Average",
        returns_label="Drawdown",
        compound=compound,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample=resample,
        fill=True,  # Fill area under drawdown curve
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
        raw_data=True,  # Skip cumulative transformation for drawdown data
    )
    if not show:
        return fig


def drawdowns_periods(
    returns,
    periods=5,
    lw=1.5,
    log_scale=False,
    fontname="Arial",
    grayscale=False,
    title=None,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    """
    Plot the longest drawdown periods as separate lines for detailed analysis.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    periods : int, optional
        Number of longest drawdown periods to display (default: 5).
    lw : float, optional
        Line width for drawdown lines (default: 1.5).
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis (default: False).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    title : str, optional
        Custom title for the plot (default: None).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with drawdown statistics (default: True).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Identifies and plots the longest drawdown periods separately. Each period is
    shown as a different colored line for easy comparison of severity and duration.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Use core plotting function for longest drawdown periods
    fig = _core.plot_longest_drawdowns(
        returns,
        periods=periods,
        lw=lw,
        log_scale=log_scale,
        fontname=fontname,
        grayscale=grayscale,
        title=title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_beta(
    returns,
    benchmark,
    window1=126,
    window1_label="6-Months",
    window2=252,
    window2_label="12-Months",
    lw=1.5,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel=True,
    subtitle=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    """
    Plot rolling beta coefficients over time using multiple window sizes.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str
        Benchmark returns data or ticker symbol.
    window1 : int, optional
        First rolling window size in days (default: 126).
    window1_label : str, optional
        Label for first window (default: "6-Months").
    window2 : int, optional
        Second rolling window size in days (default: 252).
    window2_label : str, optional
        Label for second window (default: "12-Months").
    lw : float, optional
        Line width for beta lines (default: 1.5).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    subtitle : bool, optional
        Whether to show subtitle with beta statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    prepare_returns : bool, optional
        Whether to prepare returns data (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows how portfolio beta (systematic risk) changes over time relative to benchmark.
    Uses two different window sizes to show short-term and long-term beta trends.
    """
    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Prepare benchmark data to match returns index
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)  # type: ignore

    # Use core plotting function for rolling beta
    fig = _core.plot_rolling_beta(
        returns,
        benchmark,
        window1=window1,
        window1_label=window1_label,
        window2=window2,
        window2_label=window2_label,
        title="Rolling Beta to Benchmark",
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_volatility(
    returns,
    benchmark=None,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.5,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel="Volatility",
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot rolling volatility over time, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    period : int, optional
        Rolling window size in days (default: 126).
    period_label : str, optional
        Label for the rolling period (default: "6-Months").
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252).
    lw : float, optional
        Line width for volatility lines (default: 1.5).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : str, optional
        Y-axis label (default: "Volatility").
    subtitle : bool, optional
        Whether to show subtitle with volatility statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows rolling volatility (standard deviation) over time. Includes mean volatility
    as horizontal reference line. Useful for understanding risk patterns over time.
    """
    # Calculate rolling volatility for returns
    returns = _stats.rolling_volatility(returns, period, periods_per_year)

    # Calculate rolling volatility for benchmark if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        benchmark = _stats.rolling_volatility(
            benchmark, period, periods_per_year, prepare_returns=False
        )

    # Use core plotting function for rolling statistics
    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),  # Show mean volatility as horizontal line
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Volatility (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sharpe(
    returns,
    benchmark=None,
    rf=0.0,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.25,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel="Sharpe",
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot rolling Sharpe ratio over time, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    rf : float, optional
        Risk-free rate for Sharpe calculation (default: 0.0).
    period : int, optional
        Rolling window size in days (default: 126).
    period_label : str, optional
        Label for the rolling period (default: "6-Months").
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252).
    lw : float, optional
        Line width for Sharpe lines (default: 1.25).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : str, optional
        Y-axis label (default: "Sharpe").
    subtitle : bool, optional
        Whether to show subtitle with Sharpe statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows rolling Sharpe ratio (risk-adjusted returns) over time. Higher values
    indicate better risk-adjusted performance. Includes mean Sharpe as reference.
    """
    # Calculate rolling Sharpe ratio for returns
    returns = _stats.rolling_sharpe(
        returns,
        rf,
        period,
        True,  # prepare_returns
        periods_per_year,
    )

    # Calculate rolling Sharpe ratio for benchmark if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _stats.rolling_sharpe(
            benchmark, rf, period, True, periods_per_year, prepare_returns=False
        )

    # Use core plotting function for rolling statistics
    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),  # Show mean Sharpe as horizontal line
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sharpe (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sortino(
    returns,
    benchmark=None,
    rf=0.0,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.25,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel="Sortino",
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot rolling Sortino ratio over time, optionally compared to benchmark.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    rf : float, optional
        Risk-free rate for Sortino calculation (default: 0.0).
    period : int, optional
        Rolling window size in days (default: 126).
    period_label : str, optional
        Label for the rolling period (default: "6-Months").
    periods_per_year : int, optional
        Number of periods per year for annualization (default: 252).
    lw : float, optional
        Line width for Sortino lines (default: 1.25).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 3)).
    ylabel : str, optional
        Y-axis label (default: "Sortino").
    subtitle : bool, optional
        Whether to show subtitle with Sortino statistics (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Shows rolling Sortino ratio (downside deviation adjusted returns) over time.
    Similar to Sharpe but only considers downside volatility. Higher values indicate
    better downside-adjusted performance.
    """
    # Calculate rolling Sortino ratio for returns
    returns = _stats.rolling_sortino(returns, rf, period, True, periods_per_year)

    # Calculate rolling Sortino ratio for benchmark if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _stats.rolling_sortino(
            benchmark, rf, period, True, periods_per_year, prepare_returns=False
        )

    # Use core plotting function for rolling statistics
    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),  # Show mean Sortino as horizontal line
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sortino (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def monthly_heatmap(
    returns,
    benchmark=None,
    annot_size=10,
    figsize=(8, 5),
    cbar=True,
    square=False,
    returns_label="Strategy",
    compounded=True,
    eoy=False,
    grayscale=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
    active=False,
):
    """
    Create a heatmap of monthly returns showing performance across years and months.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    benchmark : pandas.Series, pandas.DataFrame, or str, optional
        Benchmark returns data or ticker symbol (default: None).
    annot_size : int, optional
        Font size for annotations in heatmap cells (default: 10).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (8, 5)).
    cbar : bool, optional
        Whether to show color bar (default: True).
    square : bool, optional
        Whether to make heatmap cells square (default: False).
    returns_label : str, optional
        Label for the returns series (default: "Strategy").
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    eoy : bool, optional
        Whether to include end-of-year column (default: False).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).
    active : bool, optional
        If True, shows active returns (returns - benchmark) (default: False).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    Creates a color-coded heatmap where each cell represents a month's performance.
    Green indicates positive returns, red indicates negative returns. Useful for
    identifying seasonal patterns and performance consistency.
    """
    # colors, ls, alpha = _core._get_colors(grayscale)
    # Select color map based on grayscale preference
    cmap = "gray" if grayscale else "RdYlGn"

    # Convert to monthly returns and convert to percentage
    returns = _stats.monthly_returns(returns, eoy=eoy, compounded=compounded) * 100

    # Calculate figure height based on number of years
    fig_height = len(returns) / 2.5

    # Use current figure size if not specified
    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    # Adjust figure size based on data and color bar
    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.051, max([fig_height, figsize[1]]))

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)

    # Remove spines for cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # _sns.set(font_scale=.9)
    # Create heatmap for active returns vs benchmark
    if active and benchmark is not None:
        ax.set_title(
            f"{returns_label} - Monthly Active Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )
        # Calculate benchmark monthly returns
        benchmark = (
            _stats.monthly_returns(benchmark, eoy=eoy, compounded=compounded) * 100
        )
        # Calculate active returns (strategy - benchmark)
        active_returns = returns - benchmark

        # Create heatmap with active returns
        ax = _sns.heatmap(
            active_returns,
            ax=ax,
            annot=True,
            center=0,  # Center colormap at zero
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    else:
        # Create standard monthly returns heatmap
        ax.set_title(
            f"{returns_label} - Monthly Returns (%)\n",
            fontsize=12,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )

        # Create heatmap with monthly returns
        ax = _sns.heatmap(
            returns,
            ax=ax,
            annot=True,
            center=0,  # Center colormap at zero
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )

    # Format color bar if present
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=annot_size)

    # Set y-axis label
    if ylabel:
        ax.set_ylabel("Years", fontname=fontname, fontweight="bold", fontsize=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Format tick labels
    ax.tick_params(colors="#808080")
    _plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    _plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    # Apply layout adjustments with error handling
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    # Show plot if requested
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def monthly_returns(
    returns,
    annot_size=9,
    figsize=(10, 5),
    cbar=True,
    square=False,
    compounded=True,
    eoy=False,
    grayscale=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
):
    """
    Create a heatmap of monthly returns (wrapper function for monthly_heatmap).

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Daily returns data.
    annot_size : int, optional
        Font size for annotations in heatmap cells (default: 9).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 5)).
    cbar : bool, optional
        Whether to show color bar (default: True).
    square : bool, optional
        Whether to make heatmap cells square (default: False).
    compounded : bool, optional
        If True, uses compound returns; if False, uses simple returns (default: True).
    eoy : bool, optional
        Whether to include end-of-year column (default: False).
    grayscale : bool, optional
        If True, uses grayscale colors instead of default color scheme (default: False).
    fontname : str, optional
        Font family for text elements (default: "Arial").
    ylabel : bool, optional
        Whether to show y-axis label (default: True).
    savefig : str or dict, optional
        Path to save figure or dict with matplotlib savefig parameters.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Notes
    -----
    This is a convenience wrapper around monthly_heatmap() with commonly used
    default parameters for displaying monthly returns.
    """
    # Call monthly_heatmap with provided parameters
    return monthly_heatmap(
        returns=returns,
        annot_size=annot_size,
        figsize=figsize,
        cbar=cbar,
        square=square,
        compounded=compounded,
        eoy=eoy,
        grayscale=grayscale,
        fontname=fontname,
        ylabel=ylabel,
        savefig=savefig,
        show=show,
    )

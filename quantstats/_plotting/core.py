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

import matplotlib.pyplot as _plt

# Set default font to Arial, fall back gracefully if not available
try:
    _plt.rcParams["font.family"] = "Arial"
except (KeyError, ValueError, OSError):
    pass

import matplotlib.dates as _mdates
from matplotlib.ticker import (
    FormatStrFormatter as _FormatStrFormatter,
    FuncFormatter as _FuncFormatter,
)

import pandas as _pd
import numpy as _np
import seaborn as _sns
from .. import stats as _stats
from .._compat import safe_resample

# Configure seaborn theme with custom styling
_sns.set_theme(
    font_scale=1.1,
    rc={
        "figure.figsize": (10, 6),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "text.color": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
    },
)

# Color palettes for different chart styles
_FLATUI_COLORS = [
    "#FEDD78",
    "#348DC1",
    "#BA516B",
    "#4FA487",
    "#9B59B6",
    "#613F66",
    "#84B082",
    "#DC136C",
    "#559CAD",
    "#4A5899",
]
_GRAYSCALE_COLORS = [
    "#000000",
    "#222222",
    "#555555",
    "#888888",
    "#AAAAAA",
    "#CCCCCC",
    "#EEEEEE",
    "#333333",
    "#666666",
    "#999999",
]


def _get_colors(grayscale):
    """
    Get color palette, line style, and alpha values based on grayscale setting

    Parameters
    ----------
    grayscale : bool
        Whether to use grayscale colors

    Returns
    -------
    tuple
        (colors, line_style, alpha) - Color palette, line style, and alpha value
    """
    colors = _FLATUI_COLORS
    ls = "-"
    alpha = 0.8
    if grayscale:
        colors = _GRAYSCALE_COLORS
        ls = "-"
        alpha = 0.5
    return colors, ls, alpha


def plot_returns_bars(
    returns,
    benchmark=None,
    returns_label="Strategy",
    hline=None,
    hlw=None,
    hlcolor="red",
    hllabel="",
    resample="YE",
    title="Returns",
    match_volatility=False,
    log_scale=False,
    figsize=(10, 6),
    grayscale=False,
    fontname="Arial",
    ylabel=True,
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot returns as a bar chart with optional benchmark comparison

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data to plot
    benchmark : pd.Series, optional
        Benchmark returns for comparison
    returns_label : str, default "Strategy"
        Label for returns data
    hline : float, optional
        Horizontal line value
    hlw : float, optional
        Horizontal line width
    hlcolor : str, default "red"
        Horizontal line color
    hllabel : str, default ""
        Horizontal line label
    resample : str, default "YE"
        Resampling frequency
    title : str, default "Returns"
        Chart title
    match_volatility : bool, default False
        Whether to match volatility with benchmark
    log_scale : bool, default False
        Whether to use log scale for y-axis
    figsize : tuple, default (10, 6)
        Figure size
    grayscale : bool, default False
        Whether to use grayscale colors
    fontname : str, default "Arial"
        Font name for labels
    ylabel : bool, default True
        Whether to show y-axis label
    subtitle : bool, default True
        Whether to show subtitle with date range
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """
    # Validate volatility matching requirements
    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of " "benchmark.")
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    # Prepare data and colors
    colors, _, _ = _get_colors(grayscale)
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame(index=returns.index, data={returns.name: returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            index=returns.index, data={col: returns[col] for col in returns.columns}
        )

    # Add benchmark data if provided
    if isinstance(benchmark, _pd.Series):
        df[benchmark.name] = benchmark[benchmark.index.isin(returns.index)]
        if isinstance(returns, _pd.Series):
            df = df[[benchmark.name, returns.name]]
        elif isinstance(returns, _pd.DataFrame):
            col_names = [benchmark.name, returns.columns]
            df = df[list(_pd.core.common.flatten(col_names))]

    # Clean data and apply resampling
    df = df.dropna()
    if resample is not None:
        df = safe_resample(df, resample, _stats.comp)
        df = safe_resample(df, resample, "last")
    # ---------------

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # use a more precise date string for the x axis locations in the toolbar
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                df.index.date[:1][0].strftime("%Y"),
                df.index.date[-1:][0].strftime("%Y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Adjust colors if no benchmark
    if benchmark is None:
        colors = colors[1:]
    df.plot(kind="bar", ax=ax, color=colors)

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Format x-axis labels
    try:
        ax.set_xticklabels(df.index.year)
        years = sorted(list(set(df.index.year)))
    except AttributeError:
        ax.set_xticklabels(df.index)
        years = sorted(list(set(df.index)))

    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
    # years = sorted(list(set(df.index.year)))

    # Reduce label density for long time series
    if len(years) > 10:
        mod = int(len(years) / 10)
        _plt.xticks(
            _np.arange(len(years)),
            [str(year) if not i % mod else "" for i, year in enumerate(years)],
        )

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Add horizontal line if specified
    if hline is not None:
        if not isinstance(hline, _pd.Series):
            if grayscale:
                hlcolor = "gray"
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    # Add zero line for reference
    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Only show legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11)

    # Set y-axis scale
    _plt.yscale("symlog" if log_scale else "linear")

    # Configure axis labels
    ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(
            "Returns", fontname=fontname, fontweight="bold", fontsize=12, color="black"
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))

    # Remove legend for single series without benchmark
    if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
        try:
            legend = ax.get_legend()
            if legend:
                legend.remove()
        except (ValueError, AttributeError, TypeError, RuntimeError):
            pass

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    try:
        fig.tight_layout()
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_timeseries(
    returns,
    benchmark=None,
    title="Returns",
    compound=False,
    fill=False,
    returns_label="Strategy",
    hline=None,
    hlw=None,
    hlcolor="red",
    hllabel="",
    percent=True,
    match_volatility=False,
    log_scale=False,
    resample=None,
    lw=1.5,
    figsize=(10, 6),
    ylabel="",
    grayscale=False,
    fontname="Arial",
    subtitle=True,
    savefig=None,
    show=True,
    raw_data=False,
):
    """
    Plot returns as a time series line chart

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data to plot
    benchmark : pd.Series, optional
        Benchmark returns for comparison
    title : str, default "Returns"
        Chart title
    compound : bool, default False
        Whether to compound returns
    fill : bool, default False
        Whether to fill area under the line
    returns_label : str, default "Strategy"
        Label for returns data
    hline : float, optional
        Horizontal line value
    hlw : float, optional
        Horizontal line width
    hlcolor : str, default "red"
        Horizontal line color
    hllabel : str, default ""
        Horizontal line label
    percent : bool, default True
        Whether to format y-axis as percentage
    match_volatility : bool, default False
        Whether to match volatility with benchmark
    log_scale : bool, default False
        Whether to use log scale for y-axis
    resample : str, optional
        Resampling frequency
    lw : float, default 1.5
        Line width
    figsize : tuple, default (10, 6)
        Figure size
    ylabel : str, default ""
        Y-axis label
    grayscale : bool, default False
        Whether to use grayscale colors
    fontname : str, default "Arial"
        Font name for labels
    subtitle : bool, default True
        Whether to show subtitle with date range
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    colors, ls, alpha = _get_colors(grayscale)

    # Fill NaN values with zeros
    returns = returns.fillna(0)
    if isinstance(benchmark, _pd.Series):
        benchmark = benchmark.fillna(0)

    # Validate volatility matching requirements
    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of " "benchmark.")
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    # Transform data based on compound setting (skip for raw data like drawdowns)
    if not raw_data:
        if compound:
            returns = _stats.compsum(returns)
            if isinstance(benchmark, _pd.Series):
                benchmark = _stats.compsum(benchmark)
        else:
            returns = returns.cumsum()
            if isinstance(benchmark, _pd.Series):
                benchmark = benchmark.cumsum()

    # Apply resampling if specified
    if resample:
        from .._compat import safe_resample

        returns = safe_resample(
            returns, resample, "last" if compound is True else "sum"
        )
        if isinstance(benchmark, _pd.Series):
            benchmark = safe_resample(
                benchmark, resample, "last" if compound is True else "sum"
            )
    # ---------------

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s            \n"
            % (
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot benchmark first if provided
    if isinstance(benchmark, _pd.Series):
        ax.plot(benchmark, lw=lw, ls=ls, label=benchmark.name, color=colors[0])

    # Adjust alpha for grayscale
    alpha = 0.25 if grayscale else 1

    # Plot returns data
    if isinstance(returns, _pd.Series):
        ax.plot(returns, lw=lw, label=returns.name, color=colors[1], alpha=alpha)
    elif isinstance(returns, _pd.DataFrame):
        # Plot each column in the DataFrame
        for i, col in enumerate(returns.columns):
            ax.plot(returns[col], lw=lw, label=col, alpha=alpha, color=colors[i + 1])

    # Add fill under the line if requested
    if fill:
        if isinstance(returns, _pd.Series):
            ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=0.25)
        elif isinstance(returns, _pd.DataFrame):
            for i, col in enumerate(returns.columns):
                ax.fill_between(
                    returns[col].index, 0, returns[col], color=colors[i + 1], alpha=0.25
                )

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    # Add horizontal line if specified
    if hline is not None:
        if not isinstance(hline, _pd.Series):
            if grayscale:
                hlcolor = "black"
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    # Add reference lines at zero
    ax.axhline(0, ls="-", lw=1, color="gray", zorder=1)
    ax.axhline(0, ls="--", lw=1, color="white" if grayscale else "black", zorder=2)

    # Only show legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11)

    # Set y-axis scale
    _plt.yscale("symlog" if log_scale else "linear")

    # Format y-axis as percentage if requested
    if percent:
        ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
        # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        #     lambda x, loc: "{:,}%".format(int(x*100))))

    # Configure axis labels
    ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(
            ylabel, fontname=fontname, fontweight="bold", fontsize=12, color="black"
        )
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # Remove legend for single series without benchmark
    if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
        try:
            legend = ax.get_legend()
            if legend:
                legend.remove()
        except (ValueError, AttributeError, TypeError, RuntimeError):
            pass

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    try:
        fig.tight_layout()
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_histogram(
    returns,
    benchmark,
    resample="ME",
    bins=20,
    fontname="Arial",
    grayscale=False,
    title="Returns",
    kde=True,
    figsize=(10, 6),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
):
    """
    Plot histogram of returns with optional KDE and benchmark comparison

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data to plot
    benchmark : pd.Series
        Benchmark returns for comparison
    resample : str, default "ME"
        Resampling frequency
    bins : int, default 20
        Number of histogram bins
    fontname : str, default "Arial"
        Font name for labels
    grayscale : bool, default False
        Whether to use grayscale colors
    title : str, default "Returns"
        Chart title
    kde : bool, default True
        Whether to show kernel density estimate
    figsize : tuple, default (10, 6)
        Figure size
    ylabel : bool, default True
        Whether to show y-axis label
    subtitle : bool, default True
        Whether to show subtitle with date range
    compounded : bool, default True
        Whether to use compounded returns
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    # colors = ['#348dc1', '#003366', 'red']
    # if grayscale:
    #     colors = ['silver', 'gray', 'black']

    colors, _, _ = _get_colors(grayscale)

    # Choose aggregation function based on compounded setting
    apply_fnc = _stats.comp if compounded else _np.sum

    # Process benchmark data
    if benchmark is not None:
        benchmark = benchmark.fillna(0)
        benchmark = safe_resample(benchmark, resample, apply_fnc)
        benchmark = safe_resample(benchmark, resample, "last")

    # Process returns data
    returns = returns.fillna(0)
    returns = safe_resample(returns, resample, apply_fnc)
    returns = safe_resample(returns, resample, "last")

    # Adjust figure size slightly
    figsize = (0.995 * figsize[0], figsize[1])
    fig, ax = _plt.subplots(figsize=figsize)

    # Configure axis appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                returns.index.date[:1][0].strftime("%Y"),
                returns.index.date[-1:][0].strftime("%Y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Convert single-column DataFrame to Series for easier handling
    if isinstance(returns, _pd.DataFrame) and len(returns.columns) == 1:
        returns = returns[returns.columns[0]]

    # Set up color palette and alpha based on data structure
    pallete = colors[1:2] if benchmark is None else colors[:2]
    alpha = 0.7
    if isinstance(returns, _pd.DataFrame):
        pallete = (
            colors[1:(len(returns.columns) + 1)]
            if benchmark is None
            else colors[:(len(returns.columns) + 1)]
        )
        if len(returns.columns) > 1:
            alpha = 0.5

    # Plot histogram with benchmark comparison
    if benchmark is not None:
        if isinstance(returns, _pd.Series):
            combined_returns = (
                benchmark.to_frame()
                .join(returns.to_frame())
                .stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
        elif isinstance(returns, _pd.DataFrame):
            combined_returns = (
                benchmark.to_frame()
                .join(returns)
                .stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
        _sns.histplot(
            data=combined_returns,
            x="Returns",
            bins=bins,
            alpha=alpha,
            kde=kde,
            stat="density",
            hue="",
            palette=pallete,
            ax=ax,
        )

    else:
        # Plot histogram without benchmark
        if isinstance(returns, _pd.Series):
            combined_returns = returns.copy()
            if kde:
                _sns.kdeplot(data=combined_returns, color="black", ax=ax)

            _sns.histplot(
                data=combined_returns,
                bins=bins,
                alpha=alpha,
                kde=False,
                stat="density",
                color=colors[1],
                ax=ax,
            )

        elif isinstance(returns, _pd.DataFrame):
            combined_returns = (
                returns.stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
            # _sns.kdeplot(data=combined_returns, color='black', ax=ax)
            _sns.histplot(
                data=combined_returns,
                x="Returns",
                bins=bins,
                alpha=alpha,
                kde=kde,
                stat="density",
                hue="",
                palette=pallete,
                ax=ax,
            )

    # Add average line for single series
    if isinstance(combined_returns, _pd.Series) or len(combined_returns.columns) == 1:
        ax.axvline(
            combined_returns.mean(),
            ls="--",
            lw=1.5,
            zorder=2,
            label="Average",
            color="red",
        )

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(
        _plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x * 100)))
    )

    # Removed static lines for clarity
    # ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    # ax.axvline(0, lw=1, color="#000000", zorder=2)

    # Configure axis labels
    ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(
            "Occurrences",
            fontname=fontname,
            fontweight="bold",
            fontsize=12,
            color="black",
        )
    else:
        ax.set_ylabel("")

    ax.yaxis.set_label_coords(-0.1, 0.5)

    # fig.autofmt_xdate()

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    try:
        fig.tight_layout()
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_rolling_stats(
    returns,
    benchmark=None,
    title="",
    returns_label="Strategy",
    hline=None,
    hlw=None,
    hlcolor="red",
    hllabel="",
    lw=1.5,
    figsize=(10, 6),
    ylabel="",
    grayscale=False,
    fontname="Arial",
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot rolling statistics time series

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data to plot
    benchmark : pd.Series, optional
        Benchmark returns for comparison
    title : str, default ""
        Chart title
    returns_label : str, default "Strategy"
        Label for returns data
    hline : float, optional
        Horizontal line value
    hlw : float, optional
        Horizontal line width
    hlcolor : str, default "red"
        Horizontal line color
    hllabel : str, default ""
        Horizontal line label
    lw : float, default 1.5
        Line width
    figsize : tuple, default (10, 6)
        Figure size
    ylabel : str, default ""
        Y-axis label
    grayscale : bool, default False
        Whether to use grayscale colors
    fontname : str, default "Arial"
        Font name for labels
    subtitle : bool, default True
        Whether to show subtitle with date range
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    colors, _, _ = _get_colors(grayscale)

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Handle DataFrame case for returns_label
    if isinstance(returns, _pd.DataFrame):
        returns_label = list(returns.columns)

    # Prepare data structure
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame(index=returns.index, data={returns_label: returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            index=returns.index, data={col: returns[col] for col in returns.columns}
        )

    # Add benchmark data if provided
    if isinstance(benchmark, _pd.Series):
        df["Benchmark"] = benchmark[benchmark.index.isin(returns.index)]
        if isinstance(returns, _pd.Series):
            df = df[["Benchmark", returns_label]].dropna()
            ax.plot(
                df[returns_label].dropna(), lw=lw, label=returns.name, color=colors[1]
            )
        elif isinstance(returns, _pd.DataFrame):
            col_names = ["Benchmark", returns_label]
            df = df[list(_pd.core.common.flatten(col_names))].dropna()
            for i, col in enumerate(returns_label):
                ax.plot(df[col], lw=lw, label=col, color=colors[i + 1])
        # Plot benchmark line
        ax.plot(
            df["Benchmark"], lw=lw, label=benchmark.name, color=colors[0], alpha=0.8
        )
    else:
        # Plot without benchmark
        if isinstance(returns, _pd.Series):
            df = df[[returns_label]].dropna()
            ax.plot(
                df[returns_label].dropna(), lw=lw, label=returns.name, color=colors[1]
            )
        elif isinstance(returns, _pd.DataFrame):
            df = df[returns_label].dropna()
            for i, col in enumerate(returns_label):
                ax.plot(df[col], lw=lw, label=col, color=colors[i + 1])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')\

    # Set main title
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                df.index.date[:1][0].strftime("%e %b '%y"),
                df.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Add horizontal line if specified
    if hline is not None:
        if not isinstance(hline, _pd.Series):
            if grayscale:
                hlcolor = "black"
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    # Add zero reference line
    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Configure y-axis label
    if ylabel:
        ax.set_ylabel(
            ylabel, fontname=fontname, fontweight="bold", fontsize=12, color="black"
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Format y-axis with fixed decimals
    ax.yaxis.set_major_formatter(_FormatStrFormatter("%.2f"))

    # Only show legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11)

    # Remove legend for single series without benchmark
    if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
        try:
            legend = ax.get_legend()
            if legend:
                legend.remove()
        except (ValueError, AttributeError, TypeError, RuntimeError):
            pass

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    try:
        fig.tight_layout()
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_rolling_beta(
    returns,
    benchmark,
    window1=126,
    window1_label="",
    window2=None,
    window2_label="",
    title="",
    hlcolor="red",
    figsize=(10, 6),
    grayscale=False,
    fontname="Arial",
    lw=1.5,
    ylabel=True,
    subtitle=True,
    savefig=None,
    show=True,
):
    """
    Plot rolling beta calculation over time

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data to calculate beta for
    benchmark : pd.Series
        Benchmark returns for beta calculation
    window1 : int, default 126
        Primary rolling window size
    window1_label : str, default ""
        Label for primary window
    window2 : int, optional
        Secondary rolling window size
    window2_label : str, default ""
        Label for secondary window
    title : str, default ""
        Chart title
    hlcolor : str, default "red"
        Horizontal line color
    figsize : tuple, default (10, 6)
        Figure size
    grayscale : bool, default False
        Whether to use grayscale colors
    fontname : str, default "Arial"
        Font name for labels
    lw : float, default 1.5
        Line width
    ylabel : bool, default True
        Whether to show y-axis label
    subtitle : bool, default True
        Whether to show subtitle with date range
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    colors, _, _ = _get_colors(grayscale)

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Calculate and plot primary beta window
    i = 1
    if isinstance(returns, _pd.Series):
        beta = _stats.rolling_greeks(returns, benchmark, window1)["beta"].fillna(0)
        ax.plot(beta, lw=lw, label=window1_label, color=colors[1])
    elif isinstance(returns, _pd.DataFrame):
        beta = {
            col: _stats.rolling_greeks(returns[col], benchmark, window1)["beta"].fillna(
                0
            )
            for col in returns.columns
        }
        for name, b in beta.items():
            ax.plot(b, lw=lw, label=name + " " + f"({window1_label})", color=colors[i])
            i += 1

    # Calculate and plot secondary beta window if provided
    i = 1
    if window2:
        lw = lw - 0.5  # Thinner line for secondary window
        if isinstance(returns, _pd.Series):
            ax.plot(
                _stats.rolling_greeks(returns, benchmark, window2)["beta"],
                lw=lw,
                label=window2_label,
                color="gray",
                alpha=0.8,
            )
        elif isinstance(returns, _pd.DataFrame):
            betas_w2 = {
                col: _stats.rolling_greeks(returns[col], benchmark, window2)["beta"]
                for col in returns.columns
            }
            for name, beta_w2 in betas_w2.items():
                ax.plot(
                    beta_w2,
                    lw=lw,
                    ls="--",
                    label=name + " " + f"({window2_label})",
                    alpha=0.5,
                    color=colors[i],
                )
                i += 1

    # Calculate beta range for y-axis ticks
    beta_min = (
        beta.min()
        if isinstance(returns, _pd.Series)
        else min([b.min() for b in beta.values()])
    )
    beta_max = (
        beta.max()
        if isinstance(returns, _pd.Series)
        else max([b.max() for b in beta.values()])
    )
    mmin = min([-100, int(beta_min * 100)])
    mmax = max([100, int(beta_max * 100)])
    step = 50 if (mmax - mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    # Add mean line for single series
    if isinstance(returns, _pd.Series):
        hlcolor = "black" if grayscale else hlcolor
        ax.axhline(beta.mean(), ls="--", lw=1.5, color=hlcolor, zorder=2)

    # Add zero reference line
    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Format dates on x-axis
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter("%Y-%m-%d")

    # Configure y-axis label
    if ylabel:
        ax.set_ylabel(
            "Beta", fontname=fontname, fontweight="bold", fontsize=12, color="black"
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Only show legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11)

    # Remove legend for single series without benchmark
    if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
        try:
            legend = ax.get_legend()
            if legend:
                legend.remove()
        except (ValueError, AttributeError, TypeError, RuntimeError):
            pass

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    try:
        fig.tight_layout()
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_longest_drawdowns(
    returns,
    periods=5,
    lw=1.5,
    fontname="Arial",
    grayscale=False,
    title=None,
    log_scale=False,
    figsize=(10, 6),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
):
    """
    Plot cumulative returns with longest drawdown periods highlighted

    Parameters
    ----------
    returns : pd.Series
        Returns data to analyze
    periods : int, default 5
        Number of longest drawdown periods to highlight
    lw : float, default 1.5
        Line width
    fontname : str, default "Arial"
        Font name for labels
    grayscale : bool, default False
        Whether to use grayscale colors
    title : str, optional
        Chart title
    log_scale : bool, default False
        Whether to use log scale for y-axis
    figsize : tuple, default (10, 6)
        Figure size
    ylabel : bool, default True
        Whether to show y-axis label
    subtitle : bool, default True
        Whether to show subtitle with date range
    compounded : bool, default True
        Whether to use compounded returns
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    colors = ["#348dc1", "#003366", "red"]
    if grayscale:
        colors = ["#000000"] * 3

    # Calculate drawdown statistics
    dd = _stats.to_drawdown_series(returns.fillna(0))
    dddf = _stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(by="days", ascending=False, kind="mergesort")[
        :periods
    ]

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title with period count
    fig.suptitle(
        f"{title} - Worst %.0f Drawdown Periods" % periods,
        y=0.94,
        fontweight="bold",
        fontname=fontname,
        fontsize=14,
        color="black",
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Calculate cumulative returns
    series = _stats.compsum(returns) if compounded else returns.cumsum()
    ax.plot(series, lw=lw, label="Backtest", color=colors[0])

    # Highlight drawdown periods
    highlight = "black" if grayscale else "red"
    # Vectorized approach instead of iterrows
    for start, end in zip(longest_dd["start"], longest_dd["end"]):
        ax.axvspan(
            *_mdates.datestr2num([str(start), str(end)]),
            color=highlight,
            alpha=0.1,
        )

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter("%Y-%m-%d")

    # Add zero reference line
    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Set y-axis scale
    _plt.yscale("symlog" if log_scale else "linear")

    # Configure y-axis label
    if ylabel:
        ax.set_ylabel(
            "Cumulative Returns",
            fontname=fontname,
            fontweight="bold",
            fontsize=12,
            color="black",
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
    # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
    #     lambda x, loc: "{:,}%".format(int(x*100))))

    # Format dates on x-axis
    fig.autofmt_xdate()

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    try:
        fig.tight_layout()
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_distribution(
    returns,
    figsize=(10, 6),
    fontname="Arial",
    grayscale=False,
    ylabel=True,
    subtitle=True,
    compounded=True,
    title=None,
    savefig=None,
    show=True,
):
    """
    Plot box plot showing return distribution across different time periods

    Parameters
    ----------
    returns : pd.Series
        Returns data to analyze
    figsize : tuple, default (10, 6)
        Figure size
    fontname : str, default "Arial"
        Font name for labels
    grayscale : bool, default False
        Whether to use grayscale colors
    ylabel : bool, default True
        Whether to show y-axis label
    subtitle : bool, default True
        Whether to show subtitle with date range
    compounded : bool, default True
        Whether to use compounded returns
    title : str, optional
        Chart title
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    colors = _FLATUI_COLORS
    if grayscale:
        colors = ["#f9f9f9", "#dddddd", "#bbbbbb", "#999999", "#808080"]
    # colors, ls, alpha = _get_colors(grayscale)

    # Create portfolio data structure
    port = _pd.DataFrame(returns.fillna(0))
    port.columns = ["Daily"]

    # Calculate returns for different time periods
    if compounded:
        port["Weekly"] = safe_resample(port["Daily"], "W-MON", _stats.comp)
        port["Monthly"] = safe_resample(port["Daily"], "ME", _stats.comp)
        port["Quarterly"] = safe_resample(port["Daily"], "QE", _stats.comp)
        port["Yearly"] = safe_resample(port["Daily"], "YE", _stats.comp)
    else:
        port["Weekly"] = safe_resample(port["Daily"], "W-MON", "sum")
        port["Monthly"] = safe_resample(port["Daily"], "ME", "sum")
        port["Quarterly"] = safe_resample(port["Daily"], "QE", "sum")
        port["Yearly"] = safe_resample(port["Daily"], "YE", "sum")

    # Forward fill missing values
    port["Weekly"] = port["Weekly"].ffill()
    port["Monthly"] = port["Monthly"].ffill()
    port["Quarterly"] = port["Quarterly"].ffill()
    port["Yearly"] = port["Yearly"].ffill()

    # Create figure and axis
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set main title
    if title:
        title = f"{title} - Return Quantiles"
    else:
        title = "Return Quantiles"
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    # Add subtitle with date range if enabled
    if subtitle:
        ax.set_title(
            "%s - %s            \n"
            % (
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    # Set background colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Create box plot with custom color palette
    _sns.boxplot(
        data=port,
        ax=ax,
        palette={
            "Daily": colors[0],
            "Weekly": colors[1],
            "Monthly": colors[2],
            "Quarterly": colors[3],
            "Yearly": colors[4],
        },
    )

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(
        _plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x * 100)))
    )

    # Configure y-axis label
    if ylabel:
        ax.set_ylabel(
            "Returns", fontname=fontname, fontweight="bold", fontsize=12, color="black"
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Format dates on x-axis
    fig.autofmt_xdate()

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_table(
    tbl,
    columns=None,
    title="",
    title_loc="left",
    header=True,
    colWidths=None,
    rowLoc="right",
    colLoc="right",
    colLabels=None,
    edges="horizontal",
    orient="horizontal",
    figsize=(5.5, 6),
    savefig=None,
    show=False,
):
    """
    Plot a data table as a matplotlib figure

    Parameters
    ----------
    tbl : pd.DataFrame
        Data table to plot
    columns : list, optional
        Column names to use
    title : str, default ""
        Table title
    title_loc : str, default "left"
        Title location
    header : bool, default True
        Whether to show header row
    colWidths : list, optional
        Column widths
    rowLoc : str, default "right"
        Row alignment
    colLoc : str, default "right"
        Column alignment
    colLabels : list, optional
        Column labels
    edges : str, default "horizontal"
        Table edge style
    orient : str, default "horizontal"
        Table orientation
    figsize : tuple, default (5.5, 6)
        Figure size
    savefig : str or dict, optional
        Save figure parameters
    show : bool, default False
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None
    """

    # Set column names if provided
    if columns is not None:
        try:
            tbl.columns = columns
        except (ValueError, AttributeError, TypeError, RuntimeError):
            pass

    # Create figure and axis
    fig = _plt.figure(figsize=figsize)
    ax = _plt.subplot(111, frame_on=False)

    # Set title if provided
    if title != "":
        ax.set_title(
            title, fontweight="bold", fontsize=14, color="black", loc=title_loc
        )

    # Create table
    the_table = ax.table(
        cellText=tbl.values,
        colWidths=colWidths,
        rowLoc=rowLoc,
        colLoc=colLoc,
        edges=edges,
        colLabels=(tbl.columns if header else colLabels),
        loc="center",
        zorder=2,
    )

    # Configure table appearance
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1)

    # Style individual cells
    for (row, col), cell in the_table.get_celld().items():
        cell.set_height(0.08)
        cell.set_text_props(color="black")
        cell.set_edgecolor("#dddddd")
        # Header row styling
        if row == 0 and header:
            cell.set_edgecolor("black")
            cell.set_facecolor("black")
            cell.set_linewidth(2)
            cell.set_text_props(weight="bold", color="black")
        # First column styling for vertical orientation
        elif col == 0 and "vertical" in orient:
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(1)
            cell.set_text_props(weight="bold", color="black")
        # Data row styling
        elif row > 1:
            cell.set_linewidth(1)

    # Remove axis elements
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout
    try:
        _plt.subplots_adjust(hspace=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except (ValueError, AttributeError, TypeError, RuntimeError):
        pass

    # Handle saving and displaying
    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def format_cur_axis(x, _):
    """
    Format currency values for axis labels with appropriate units

    Parameters
    ----------
    x : float
        Value to format
    _ : unused
        Matplotlib formatter parameter (not used)

    Returns
    -------
    str
        Formatted currency string with appropriate unit suffix
    """
    # Format large values with appropriate suffixes
    if x >= 1e12:
        res = "$%1.1fT" % (x * 1e-12)
        return res.replace(".0T", "T")
    if x >= 1e9:
        res = "$%1.1fB" % (x * 1e-9)
        return res.replace(".0B", "B")
    if x >= 1e6:
        res = "$%1.1fM" % (x * 1e-6)
        return res.replace(".0M", "ME")
    if x >= 1e3:
        res = "$%1.0fK" % (x * 1e-3)
        return res.replace(".0K", "K")
    # Format small values without suffix
    res = "$%1.0f" % x
    return res.replace(".0", "")


def format_pct_axis(x, _):
    """
    Format percentage values for axis labels with appropriate units

    Parameters
    ----------
    x : float
        Value to format (as decimal, e.g., 0.01 for 1%)
    _ : unused
        Matplotlib formatter parameter (not used)

    Returns
    -------
    str
        Formatted percentage string with appropriate unit suffix
    """
    # Convert to percentage
    x *= 100  # lambda x, loc: "{:,}%".format(int(x * 100))

    # Format large percentage values with appropriate suffixes
    if x >= 1e12:
        res = "%1.1fT%%" % (x * 1e-12)
        return res.replace(".0T%", "T%")
    if x >= 1e9:
        res = "%1.1fB%%" % (x * 1e-9)
        return res.replace(".0B%", "B%")
    if x >= 1e6:
        res = "%1.1fM%%" % (x * 1e-6)
        return res.replace(".0M%", "M%")
    if x >= 1e3:
        res = "%1.1fK%%" % (x * 1e-3)
        return res.replace(".0K%", "K%")
    # Format small percentage values without suffix
    res = "%1.0f%%" % x
    return res.replace(".0%", "%")

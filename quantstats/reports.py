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

import pandas as _pd
import numpy as _np
from math import sqrt as _sqrt, ceil as _ceil
from datetime import datetime as _dt
from base64 import b64encode as _b64encode
import re as _regex
from tabulate import tabulate as _tabulate
from . import __version__, stats as _stats, utils as _utils, plots as _plots
from dateutil.relativedelta import relativedelta
from io import StringIO
from pathlib import Path

try:
    from IPython.core.display import display as iDisplay, HTML as iHTML
except ImportError:
    from IPython.display import display as iDisplay
    from IPython.core.display import HTML as iHTML


def _get_trading_periods(periods_per_year=252):
    """
    Calculate trading periods for different time windows.

    This helper function computes the number of trading periods for full year
    and half year periods, which are commonly used in financial calculations
    for annualization and rolling window analysis.

    Parameters
    ----------
    periods_per_year : int, default 252
        Number of trading periods in a year (e.g., 252 for daily data,
        12 for monthly data)

    Returns
    -------
    tuple
        A tuple containing (periods_per_year, half_year_periods)

    Examples
    --------
    >>> _get_trading_periods(252)  # Daily data
    (252, 126)
    >>> _get_trading_periods(12)   # Monthly data
    (12, 6)
    """
    # Calculate half year periods using ceiling to ensure we get at least half
    half_year = _ceil(periods_per_year / 2)
    return periods_per_year, half_year


def _match_dates(returns, benchmark):
    """
    Align returns and benchmark data to start from the same date.

    This function ensures that both the returns and benchmark series start
    from the same date by finding the latest start date where both series
    have non-zero values. This is crucial for accurate performance comparisons.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data that may be a Series or DataFrame with multiple columns
    benchmark : pd.Series
        Benchmark returns data

    Returns
    -------
    tuple
        A tuple containing (aligned_returns, aligned_benchmark) both starting
        from the same date

    Examples
    --------
    >>> returns_aligned, bench_aligned = _match_dates(returns, benchmark)
    """
    # Handle different types of returns data (Series vs DataFrame)
    if isinstance(returns, _pd.DataFrame):
        # For DataFrame, use the first column to find the start date
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        # For Series, find the maximum of start dates for both series
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())

    # Slice both series to start from the latest common start date
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark


def html(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    title="Strategy Tearsheet",
    output=None,
    compounded=True,
    periods_per_year=252,
    download_filename="quantstats-tearsheet.html",
    figfmt="svg",
    template_path=None,
    match_dates=True,
    **kwargs,
):
    """
    Generate an HTML tearsheet report for portfolio performance analysis.

    This function creates a comprehensive HTML report containing performance
    metrics, visualizations, and analysis of investment returns. The report
    includes comparisons with benchmarks, drawdown analysis, and various
    performance charts.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison. Can be a Series of returns,
        a ticker symbol string, or None for no benchmark
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal, e.g., 0.02 for 2%)
    grayscale : bool, default False
        Whether to generate charts in grayscale instead of color
    title : str, default "Strategy Tearsheet"
        Title to display at the top of the HTML report
    output : str or None, default None
        File path to save the HTML report. If None, downloads in browser
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year for annualization
    download_filename : str, default "quantstats-tearsheet.html"
        Filename for browser download if output is None
    figfmt : str, default "svg"
        Format for embedded charts ('svg', 'png', 'jpg')
    template_path : str or None, default None
        Path to custom HTML template file. Uses default if None
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments for customization:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active_returns: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Generates HTML file either as download or saved to specified path

    Examples
    --------
    >>> html(returns, benchmark='^GSPC', title='My Strategy')
    >>> html(returns, output='report.html', grayscale=True)

    Raises
    ------
    ValueError
        If output is None and not running in notebook environment
    FileNotFoundError
        If custom template_path doesn't exist
    """
    # Check if output parameter is required (not in notebook environment)
    if output is None and not _utils._in_notebook():
        raise ValueError("`output` must be specified")

    # Clean returns data by removing NaN values if date matching is enabled
    if match_dates:
        returns = returns.dropna()

    # Get trading periods for calculations
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    # Secure file path handling for HTML template
    if template_path is None:
        # Use default template path - report.html in same directory
        template_path = Path(__file__).parent / 'report.html'
    else:
        template_path = Path(template_path)

    # Resolve to absolute path and validate template file existence
    template_path = template_path.resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not template_path.is_file():
        raise ValueError(f"Template path is not a file: {template_path}")

    # Read template securely with UTF-8 encoding
    tpl = template_path.read_text(encoding='utf-8')

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data for analysis
    returns = _utils._prepare_returns(returns)

    # Handle strategy title - can be single string or list for multiple columns
    strategy_title = kwargs.get("strategy_title", "Strategy")
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1 and isinstance(strategy_title, str):
            strategy_title = list(returns.columns)

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
        # Auto-determine benchmark title if not provided
        if kwargs.get("benchmark_title") is None:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, _pd.Series):
                benchmark_title = benchmark.name
            elif isinstance(benchmark, _pd.DataFrame):
                benchmark_title = benchmark[benchmark.columns[0]].name

        # Update template with benchmark information
        tpl = tpl.replace(
            "{{benchmark_title}}", f"Benchmark is {benchmark_title.upper()} | "
        )
        # Prepare benchmark data to match returns index and risk-free rate
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        # Align dates between returns and benchmark if requested
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
    else:
        benchmark_title = None

    # Format date range for display in template
    date_range = returns.index.strftime("%e %b, %Y")
    tpl = tpl.replace("{{date_range}}", date_range[0] + " - " + date_range[-1])
    tpl = tpl.replace("{{title}}", title)
    tpl = tpl.replace("{{v}}", __version__)

    # Set names for data series to be used in charts and tables
    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    # Generate comprehensive performance metrics table
    mtrx = metrics(
        returns=returns,
        benchmark=benchmark,
        rf=rf,
        display=False,
        mode="full",
        sep=True,
        internal="True",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
    )[2:]

    # Format metrics table for HTML display
    mtrx.index.name = "Metric"
    tpl = tpl.replace("{{metrics}}", _html_table(mtrx))

    # Handle table formatting for multiple columns
    if isinstance(returns, _pd.DataFrame):
        num_cols = len(returns.columns)
        # Replace empty table rows with horizontal rule separators
        for i in reversed(range(num_cols + 1, num_cols + 3)):
            str_td = "<td></td>" * i
            tpl = tpl.replace(
                f"<tr>{str_td}</tr>", '<tr><td colspan="{}"><hr></td></tr>'.format(i)
            )

    # Clean up table formatting with horizontal rules
    tpl = tpl.replace(
        "<tr><td></td><td></td><td></td></tr>", '<tr><td colspan="3"><hr></td></tr>'
    )
    tpl = tpl.replace(
        "<tr><td></td><td></td></tr>", '<tr><td colspan="2"><hr></td></tr>'
    )

    # Generate end-of-year (EOY) returns comparison table
    if benchmark is not None:
        # Compare returns vs benchmark on yearly basis
        yoy = _stats.compare(
            returns, benchmark, "YE", compounded=compounded, prepare_returns=False
        )
        # Set appropriate column names based on data type
        if isinstance(returns, _pd.Series):
            yoy.columns = [benchmark_title, strategy_title, "Multiplier", "Won"]
        elif isinstance(returns, _pd.DataFrame):
            yoy.columns = list(
                _pd.core.common.flatten([benchmark_title, strategy_title])
            )
        yoy.index.name = "Year"
        tpl = tpl.replace("{{eoy_title}}", "<h3>EOY Returns vs Benchmark</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(yoy))
    else:
        # Generate EOY returns table without benchmark comparison
        # pct multiplier
        yoy = _pd.DataFrame(_utils.group_returns(returns, returns.index.year) * 100)
        if isinstance(returns, _pd.Series):
            yoy.columns = ["Return"]
            yoy["Cumulative"] = _utils.group_returns(returns, returns.index.year, True)
            yoy["Return"] = yoy["Return"].round(2).astype(str) + "%"
            yoy["Cumulative"] = (yoy["Cumulative"] * 100).round(2).astype(str) + "%"
        elif isinstance(returns, _pd.DataFrame):
            # Don't show cumulative for multiple strategy portfolios
            # just show compounded like when we have a benchmark
            yoy.columns = list(_pd.core.common.flatten(strategy_title))

        yoy.index.name = "Year"
        tpl = tpl.replace("{{eoy_title}}", "<h3>EOY Returns</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(yoy))

    # Generate drawdown analysis table
    if isinstance(returns, _pd.Series):
        # Calculate drawdown series and get worst drawdown periods
        dd = _stats.to_drawdown_series(returns)
        dd_info = _stats.drawdown_details(dd).sort_values(
            by="max drawdown", ascending=True
        )[:10]
        dd_info = dd_info[["start", "end", "max drawdown", "days"]]
        dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
        tpl = tpl.replace("{{dd_info}}", _html_table(dd_info, False))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        dd_info_list = []
        for col in returns.columns:
            dd = _stats.to_drawdown_series(returns[col])
            dd_info = _stats.drawdown_details(dd).sort_values(
                by="max drawdown", ascending=True
            )[:10]
            dd_info = dd_info[["start", "end", "max drawdown", "days"]]
            dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
            dd_info_list.append(_html_table(dd_info, False))

        # Combine all drawdown tables with headers
        dd_html_table = ""
        for html_str, col in zip(dd_info_list, returns.columns):
            dd_html_table = (
                dd_html_table + f"<h3>{col}</h3><br>" + StringIO(html_str).read()
            )
        tpl = tpl.replace("{{dd_info}}", dd_html_table)

    # Get active returns setting for plots
    active = kwargs.get("active_returns", False)

    # Generate all the performance plots and embed them in the HTML
    # plots
    figfile = _utils._file_stream()
    _plots.returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 5),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compound=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{returns}}", _embed_figure(figfile, figfmt))

    # Log returns plot for better visualization of performance
    figfile = _utils._file_stream()
    _plots.log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compound=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{log_returns}}", _embed_figure(figfile, figfmt))

    # Volatility-matched returns plot (only if benchmark exists)
    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compound=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{vol_returns}}", _embed_figure(figfile, figfmt))

    # Yearly returns comparison chart
    figfile = _utils._file_stream()
    _plots.yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{eoy_returns}}", _embed_figure(figfile, figfmt))

    # Returns distribution histogram
    figfile = _utils._file_stream()
    _plots.histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(7, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{monthly_dist}}", _embed_figure(figfile, figfmt))

    # Daily returns scatter plot
    figfile = _utils._file_stream()
    _plots.daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        prepare_returns=False,
        active=active,
    )
    tpl = tpl.replace("{{daily_returns}}", _embed_figure(figfile, figfmt))

    # Rolling beta analysis (only if benchmark exists)
    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 3),
            subtitle=False,
            window1=win_half_year,
            window2=win_year,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            prepare_returns=False,
        )
        tpl = tpl.replace("{{rolling_beta}}", _embed_figure(figfile, figfmt))

    # Rolling volatility analysis
    figfile = _utils._file_stream()
    _plots.rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_vol}}", _embed_figure(figfile, figfmt))

    # Rolling Sharpe ratio analysis
    figfile = _utils._file_stream()
    _plots.rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sharpe}}", _embed_figure(figfile, figfmt))

    # Rolling Sortino ratio analysis
    figfile = _utils._file_stream()
    _plots.rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sortino}}", _embed_figure(figfile, figfmt))

    # Drawdown periods analysis
    figfile = _utils._file_stream()
    if isinstance(returns, _pd.Series):
        _plots.drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _plots.drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(embed, figfmt))

    # Underwater (drawdown) plot
    figfile = _utils._file_stream()
    _plots.drawdown(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel="",
    )
    tpl = tpl.replace("{{dd_plot}}", _embed_figure(figfile, figfmt))

    # Monthly returns heatmap
    figfile = _utils._file_stream()
    if isinstance(returns, _pd.Series):
        _plots.monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 4),
            cbar=False,
            returns_label=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            active=active,
        )
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _plots.monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(8, 4),
                cbar=False,
                returns_label=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                active=active,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(embed, figfmt))

    # Returns distribution analysis
    figfile = _utils._file_stream()

    if isinstance(returns, _pd.Series):
        _plots.distribution(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel="",
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        embed = []
        for col in returns.columns:
            _plots.distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel="",
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(embed, figfmt))

    # Clean up any remaining template placeholders
    tpl = _regex.sub(r"\{\{(.*?)\}\}", "", tpl)
    tpl = tpl.replace("white-space:pre;", "")

    # Handle output - either download in browser or save to file
    if output is None:
        # _open_html(tpl)
        _download_html(tpl, download_filename)
        return

    # Write HTML content to specified output file
    with open(output, "w", encoding="utf-8") as f:
        f.write(tpl)


def full(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    """
    Generate a comprehensive performance analysis report.

    This function creates a full performance analysis including metrics,
    worst drawdowns analysis, and complete visualization suite. It's designed
    for detailed portfolio analysis and can handle both single strategies
    and multiple strategy comparisons.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    grayscale : bool, default False
        Whether to generate charts in grayscale
    figsize : tuple, default (8, 5)
        Figure size for plots as (width, height)
    display : bool, default True
        Whether to display results in notebook/console
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active_returns: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Displays comprehensive analysis including metrics, drawdowns, and plots

    Examples
    --------
    >>> full(returns, benchmark='^GSPC', rf=0.02)
    >>> full(returns, figsize=(10, 6), grayscale=True)
    """
    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data
    returns = _utils._prepare_returns(returns)

    # Process benchmark if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Extract title parameters from kwargs
    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    strategy_title = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active_returns", False)

    # Handle multiple strategy columns
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1 and isinstance(strategy_title, str):
            strategy_title = list(returns.columns)

    # Set names for display purposes
    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    # Calculate drawdown analysis for worst periods display
    dd = _stats.to_drawdown_series(returns)

    # Process drawdown details based on data type
    if isinstance(dd, _pd.Series):
        col = _stats.drawdown_details(dd).columns[4]
        dd_info = _stats.drawdown_details(dd).sort_values(by=col, ascending=True)[:5]
        if not dd_info.empty:
            dd_info.index = range(1, min(6, len(dd_info) + 1))
            dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
    elif isinstance(dd, _pd.DataFrame):
        # Handle multiple strategy columns
        col = _stats.drawdown_details(dd).columns.get_level_values(1)[4]
        dd_info_dict = {}
        for ptf in dd.columns:
            dd_info = _stats.drawdown_details(dd[ptf]).sort_values(
                by=col, ascending=True
            )[:5]
            if not dd_info.empty:
                dd_info.index = range(1, min(6, len(dd_info) + 1))
                dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
            dd_info_dict[ptf] = dd_info

    # Display results based on environment (notebook vs console)
    if _utils._in_notebook():
        # Display in Jupyter notebook with HTML formatting
        iDisplay(iHTML("<h4>Performance Metrics</h4>"))
        iDisplay(
            metrics(
                returns=returns,
                benchmark=benchmark,
                rf=rf,
                display=display,
                mode="full",
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False,
                benchmark_title=benchmark_title,
                strategy_title=strategy_title,
            )
        )

        # Display worst drawdowns analysis
        if isinstance(dd, _pd.Series):
            iDisplay(iHTML('<h4 style="margin-bottom:20px">Worst 5 Drawdowns</h4>'))
            if dd_info.empty:
                iDisplay(iHTML("<p>(no drawdowns)</p>"))
            else:
                iDisplay(dd_info)
        elif isinstance(dd, _pd.DataFrame):
            # Display drawdowns for each strategy
            for ptf, dd_info in dd_info_dict.items():
                iDisplay(
                    iHTML(
                        '<h4 style="margin-bottom:20px">%s - Worst 5 Drawdowns</h4>'
                        % ptf
                    )
                )
                if dd_info.empty:
                    iDisplay(iHTML("<p>(no drawdowns)</p>"))
                else:
                    iDisplay(dd_info)

        iDisplay(iHTML("<h4>Strategy Visualization</h4>"))
    else:
        # Display in console/terminal environment
        print("[Performance Metrics]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="full",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        print("\n\n")
        print("[Worst 5 Drawdowns]\n")

        # Display drawdowns in tabular format
        if isinstance(dd, _pd.Series):
            if dd_info.empty:
                print("(no drawdowns)")
            else:
                print(
                    _tabulate(
                        dd_info, headers="keys", tablefmt="simple", floatfmt=".2f"
                    )
                )
        elif isinstance(dd, _pd.DataFrame):
            for ptf, dd_info in dd_info_dict.items():
                if dd_info.empty:
                    print("(no drawdowns)")
                else:
                    print(f"{ptf}\n")
                    print(
                        _tabulate(
                            dd_info, headers="keys", tablefmt="simple", floatfmt=".2f"
                        )
                    )

        print("\n\n")
        print("[Strategy Visualization]\nvia Matplotlib")

    # Generate comprehensive plots
    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="full",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )


def basic(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    """
    Generate a basic performance analysis report.

    This function creates a simplified performance analysis with essential
    metrics and basic visualizations. It's designed for quick portfolio
    analysis when detailed analysis is not needed.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    grayscale : bool, default False
        Whether to generate charts in grayscale
    figsize : tuple, default (8, 5)
        Figure size for plots as (width, height)
    display : bool, default True
        Whether to display results in notebook/console
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active_returns: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Displays basic analysis including essential metrics and plots

    Examples
    --------
    >>> basic(returns, benchmark='^GSPC')
    >>> basic(returns, figsize=(10, 6), display=False)
    """
    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    # Clean and prepare returns data
    returns = _utils._prepare_returns(returns)

    # Process benchmark if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Extract title parameters from kwargs
    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    strategy_title = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active_returns", False)

    # Handle multiple strategy columns
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1 and isinstance(strategy_title, str):
            strategy_title = list(returns.columns)

    # Display results based on environment (notebook vs console)
    if _utils._in_notebook():
        # Display in Jupyter notebook with HTML formatting
        iDisplay(iHTML("<h4>Performance Metrics</h4>"))
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        iDisplay(iHTML("<h4>Strategy Visualization</h4>"))
    else:
        # Display in console/terminal environment
        print("[Performance Metrics]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )

        print("\n\n")
        print("[Strategy Visualization]\nvia Matplotlib")

    # Generate basic plots
    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="basic",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )


def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    Calculate comprehensive performance metrics for portfolio analysis.

    This function computes a wide range of performance metrics including
    returns, risk measures, ratios, and statistical measures. It can handle
    both single strategies and multiple strategy comparisons with optional
    benchmark analysis.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    display : bool, default True
        Whether to display results in formatted table
    mode : str, default "basic"
        Analysis mode - "basic" for essential metrics, "full" for comprehensive
    sep : bool, default False
        Whether to include separator rows in output
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    prepare_returns : bool, default True
        Whether to prepare/clean returns data
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - as_pct: Whether to return percentages
        - internal: Internal calculation flag

    Returns
    -------
    pd.DataFrame or None
        DataFrame with performance metrics if display=False, else None

    Examples
    --------
    >>> metrics_df = metrics(returns, benchmark='^GSPC', display=False)
    >>> metrics(returns, mode="full", rf=0.02)
    """
    # Clean returns data if date matching is enabled
    if match_dates:
        returns = returns.dropna()
    # Remove timezone information from index for consistent processing
    returns.index = returns.index.tz_localize(None)

    # Get trading periods for annualization calculations
    win_year, _ = _get_trading_periods(periods_per_year)

    # Extract column names from kwargs or use defaults
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")

    # Handle benchmark column naming
    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"Benchmark ({benchmark.upper()})"
        elif isinstance(benchmark, _pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    # Handle strategy column naming for multiple strategies
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]

    # if isinstance(returns, _pd.DataFrame):
    #     if len(returns.columns) > 1:
    #         raise ValueError("`returns` needs to be a Pandas Series or one column DataFrame. "
    #                          "multi colums DataFrame was passed")
    #     returns = returns[returns.columns[0]]

    # Prepare returns data if requested
    if prepare_returns:
        df = _utils._prepare_returns(returns)

    # Create main DataFrame for calculations
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame({"returns": returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
        df["benchmark"] = benchmark
        # Update blank list for proper formatting
        if isinstance(returns, _pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, _pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]

    # Calculate start and end dates for each series
    if isinstance(returns, _pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, _pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    # Add benchmark dates if present
    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    # Fill missing values with zeros for calculations
    df = df.fillna(0)

    # Determine percentage multiplier for display
    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # Initialize metrics DataFrame with basic information
    metrics = _pd.DataFrame()
    metrics["Start Period"] = _pd.Series(s_start)
    metrics["End Period"] = _pd.Series(s_end)
    metrics["Risk-Free Rate %"] = _pd.Series(s_rf) * 100
    metrics["Time in Market %"] = _stats.exposure(df, prepare_returns=False) * pct

    # Add separator row
    metrics["~"] = blank

    # Calculate return metrics based on compounding preference
    if compounded:
        metrics["Cumulative Return %"] = (_stats.comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    # Calculate annualized return (CAGR)
    metrics["CAGR﹪%"] = _stats.cagr(df, rf, compounded, win_year) * pct

    # Add separator row
    metrics["~~~~~~~~~~~~~~"] = blank

    # Calculate risk-adjusted return ratios
    metrics["Sharpe"] = _stats.sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _stats.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )

    # Add advanced Sharpe metrics for full mode
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _stats.smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _stats.probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    # Calculate Sortino ratio (downside deviation-based)
    metrics["Sortino"] = _stats.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        # metrics['Prob. Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino"] = _stats.smart_sortino(df, rf, win_year, True)
        # metrics['Prob. Smart Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate adjusted Sortino ratio
    metrics["Sortino/√2"] = metrics["Sortino"] / _sqrt(2)
    if mode.lower() == "full":
        # metrics['Prob. Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False) * pct
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / _sqrt(2)
        # metrics['Prob. Smart Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate Omega ratio (probability-weighted ratio)
    if isinstance(returns, _pd.Series):
        metrics["Omega"] = _stats.omega(df["returns"], rf, 0.0, win_year)
    elif isinstance(returns, _pd.DataFrame):
        omega_values = [
            _stats.omega(df[strategy_col], rf, 0.0, win_year)
            for strategy_col in df_strategy_columns
        ]
        if "benchmark" in df:
            omega_values.append(_stats.omega(df["benchmark"], rf, 0.0, win_year))
        metrics["Omega"] = omega_values

    # Add separator and prepare for drawdown metrics
    metrics["~~~~~~~~"] = blank
    metrics["Max Drawdown %"] = blank
    metrics["Max DD Date"] = blank
    metrics["Max DD Period Start"] = blank
    metrics["Max DD Period End"] = blank
    metrics["Longest DD Days"] = blank

    # Add detailed volatility and risk metrics for full mode
    if mode.lower() == "full":
        # Calculate annualized volatility
        if isinstance(returns, _pd.Series):
            ret_vol = (
                _stats.volatility(df["returns"], win_year, True, prepare_returns=False)
                * pct
            )
        elif isinstance(returns, _pd.DataFrame):
            ret_vol = [
                _stats.volatility(
                    df[strategy_col], win_year, True, prepare_returns=False
                )
                * pct
                for strategy_col in df_strategy_columns
            ]

        # Add benchmark volatility if present
        if "benchmark" in df:
            bench_vol = (
                _stats.volatility(
                    df["benchmark"], win_year, True, prepare_returns=False
                )
                * pct
            )

            vol_ = [ret_vol, bench_vol]
            if isinstance(ret_vol, list):
                metrics["Volatility (ann.) %"] = list(_pd.core.common.flatten(vol_))
            else:
                metrics["Volatility (ann.) %"] = vol_

            # Calculate benchmark-relative metrics
            if isinstance(returns, _pd.Series):
                metrics["R^2"] = _stats.r_squared(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
                metrics["Information Ratio"] = _stats.information_ratio(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
            elif isinstance(returns, _pd.DataFrame):
                metrics["R^2"] = (
                    [
                        _stats.r_squared(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Information Ratio"] = (
                    [
                        _stats.information_ratio(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            # No benchmark case
            if isinstance(returns, _pd.Series):
                metrics["Volatility (ann.) %"] = [ret_vol]
            elif isinstance(returns, _pd.DataFrame):
                metrics["Volatility (ann.) %"] = ret_vol

        # Additional risk and return metrics
        metrics["Calmar"] = _stats.calmar(df, prepare_returns=False, periods=win_year)
        metrics["Skew"] = _stats.skew(df, prepare_returns=False)
        metrics["Kurtosis"] = _stats.kurtosis(df, prepare_returns=False)

        # Add separator
        metrics["~~~~~~~~~~"] = blank

        # Expected returns at different frequencies
        metrics["Expected Daily %%"] = (
            _stats.expected_return(df, compounded=compounded, prepare_returns=False)
            * pct
        )
        metrics["Expected Monthly %%"] = (
            _stats.expected_return(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Expected Yearly %%"] = (
            _stats.expected_return(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Risk management metrics
        metrics["Kelly Criterion %"] = (
            _stats.kelly_criterion(df, prepare_returns=False) * pct
        )
        metrics["Risk of Ruin %"] = _stats.risk_of_ruin(df, prepare_returns=False)

        # Value at Risk metrics
        metrics["Daily Value-at-Risk %"] = -abs(
            _stats.var(df, prepare_returns=False) * pct
        )
        metrics["Expected Shortfall (cVaR) %"] = -abs(
            _stats.cvar(df, prepare_returns=False) * pct
        )

    # Add separator
    metrics["~~~~~~"] = blank

    # Consecutive wins/losses analysis (full mode only)
    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = _stats.consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = _stats.consecutive_losses(df)

    # Pain-based metrics (Gain/Pain ratio)
    metrics["Gain/Pain Ratio"] = _stats.gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = _stats.gain_to_pain_ratio(df, rf, "ME")
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _stats.gain_to_pain_ratio(df, rf, "QE")
    #     metrics['GPR (6M)'] = _stats.gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _stats.gain_to_pain_ratio(df, rf, "YE")

    # Add separator
    metrics["~~~~~~~"] = blank

    # Trading-based performance metrics
    metrics["Payoff Ratio"] = _stats.payoff_ratio(df, prepare_returns=False)
    metrics["Profit Factor"] = _stats.profit_factor(df, prepare_returns=False)
    metrics["Common Sense Ratio"] = _stats.common_sense_ratio(df, prepare_returns=False)
    metrics["CPC Index"] = _stats.cpc_index(df, prepare_returns=False)
    metrics["Tail Ratio"] = _stats.tail_ratio(df, prepare_returns=False)
    metrics["Outlier Win Ratio"] = _stats.outlier_win_ratio(df, prepare_returns=False)
    metrics["Outlier Loss Ratio"] = _stats.outlier_loss_ratio(df, prepare_returns=False)

    # # returns
    metrics["~~"] = blank

    # Time-based return analysis
    today = df.index[-1]  # _dt.today()
    m3 = today - relativedelta(months=3)
    m6 = today - relativedelta(months=6)
    y1 = today - relativedelta(years=1)

    # Calculate period returns based on compounding preference
    if compounded:
        metrics["MTD %"] = (
            _stats.comp(df[df.index >= _dt(today.year, today.month, 1)]) * pct
        )
        metrics["3M %"] = _stats.comp(df[df.index >= m3]) * pct
        metrics["6M %"] = _stats.comp(df[df.index >= m6]) * pct
        metrics["YTD %"] = _stats.comp(df[df.index >= _dt(today.year, 1, 1)]) * pct
        metrics["1Y %"] = _stats.comp(df[df.index >= y1]) * pct
    else:
        metrics["MTD %"] = (
            _np.sum(df[df.index >= _dt(today.year, today.month, 1)], axis=0) * pct
        )
        metrics["3M %"] = _np.sum(df[df.index >= m3], axis=0) * pct
        metrics["6M %"] = _np.sum(df[df.index >= m6], axis=0) * pct
        metrics["YTD %"] = _np.sum(df[df.index >= _dt(today.year, 1, 1)], axis=0) * pct
        metrics["1Y %"] = _np.sum(df[df.index >= y1], axis=0) * pct

    # Multi-year annualized returns
    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    metrics["All-time (ann.) %"] = _stats.cagr(df, 0.0, compounded, win_year) * pct

    # Best/worst period analysis (full mode only)
    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["Best Day %"] = (
            _stats.best(df, compounded=compounded, prepare_returns=False) * pct
        )
        metrics["Worst Day %"] = _stats.worst(df, prepare_returns=False) * pct
        metrics["Best Month %"] = (
            _stats.best(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Month %"] = (
            _stats.worst(df, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            _stats.best(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Year %"] = (
            _stats.worst(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

    # Calculate and integrate drawdown metrics
    # return drawdown (dd) df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    # Add drawdown metrics to main metrics DataFrame
    # drawdown (dd) detail
    metrics["~~~~"] = blank
    # Properly integrate drawdown data into metrics
    for metric_name in dd.index:
        metrics[metric_name] = dd.loc[metric_name].values

    # Additional drawdown-based metrics
    metrics["Recovery Factor"] = _stats.recovery_factor(df)
    metrics["Ulcer Index"] = _stats.ulcer_index(df)
    metrics["Serenity Index"] = _stats.serenity_index(df, rf)

    # Win rate analysis (full mode only)
    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["Avg. Up Month %"] = (
            _stats.avg_win(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Avg. Down Month %"] = (
            _stats.avg_loss(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Days %%"] = _stats.win_rate(df, prepare_returns=False) * pct
        metrics["Win Month %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Quarter %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="QE", prepare_returns=False
            )
            * pct
        )
        metrics["Win Year %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Greek letters and correlation analysis (if benchmark exists)
        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, _pd.Series):
                # Calculate Greek letters (Beta, Alpha) for single strategy
                greeks = _stats.greeks(
                    df["returns"], df["benchmark"], win_year, prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            _stats.treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, _pd.DataFrame):
                # Calculate Greek letters for multiple strategies
                greeks = [
                    _stats.greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                _stats.treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # Format metrics for display
    # prepare for display
    for col in metrics.columns:
        try:
            # Try to convert to float and round
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except (ValueError, TypeError, AttributeError):
            pass
        # Handle integer columns (marked with *int)
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        # Add percentage signs to percentage columns
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    # Format drawdown days as integers
    try:
        metrics["Longest DD Days"] = _pd.to_numeric(metrics["Longest DD Days"]).astype(
            "int"
        )
        metrics["Avg. Drawdown Days"] = _pd.to_numeric(
            metrics["Avg. Drawdown Days"]
        ).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    # Clean up column names (remove separators and percentage signs)
    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    # Set appropriate column names
    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(_pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # Final data cleaning
    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            _np.nan,
            -_np.nan,
            _np.inf,
            -_np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # Reorder columns to put benchmark first if present
    # move benchmark to be the first column always if present
    if "benchmark" in df:
        metrics = metrics[
            [benchmark_colname]
            + [col for col in metrics.columns if col != benchmark_colname]
        ]

    # Handle display vs return
    if display:
        print(_tabulate(metrics, headers="keys", tablefmt="simple"))
        return None

    # Remove separator rows if not requested
    if not sep:
        metrics = metrics[metrics.index != ""]

    # Final formatting for programmatic use
    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics


def plots(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(8, 5),
    mode="basic",
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    Generate comprehensive visualization plots for portfolio performance.

    This function creates a complete set of performance visualization plots
    including returns, drawdowns, distributions, and rolling metrics. It can
    generate either basic plots or a full comprehensive suite.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    grayscale : bool, default False
        Whether to generate charts in grayscale
    figsize : tuple, default (8, 5)
        Figure size for plots as (width, height)
    mode : str, default "basic"
        Plot mode - "basic" for essential plots, "full" for comprehensive suite
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    prepare_returns : bool, default True
        Whether to prepare/clean returns data
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - active: Whether to show active returns vs benchmark

    Returns
    -------
    None
        Displays various performance plots

    Examples
    --------
    >>> plots(returns, benchmark='^GSPC', mode="full")
    >>> plots(returns, grayscale=True, figsize=(10, 6))
    """
    # Extract title parameters from kwargs
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active", False)

    # Handle multiple strategy columns
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)

    # Get trading periods for rolling window calculations
    win_year, win_half_year = _get_trading_periods(periods_per_year)

    # Clean returns data if date matching is enabled
    if match_dates is True:
        returns = returns.dropna()

    # Prepare returns data if requested
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    # Set names for display in plots
    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_colname

    # Generate basic plots (snapshot and heatmap)
    if mode.lower() != "full":
        # Performance snapshot plot
        _plots.snapshot(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0]),
            show=True,
            mode=("comp" if compounded else "sum"),
            benchmark_title=benchmark_colname,
            strategy_title=strategy_colname,
        )

        # Monthly returns heatmap
        if isinstance(returns, _pd.Series):
            _plots.monthly_heatmap(
                returns,
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                compounded=compounded,
                active=active,
            )
        elif isinstance(returns, _pd.DataFrame):
            # Generate heatmap for each strategy column
            for col in returns.columns:
                _plots.monthly_heatmap(
                    returns[col].dropna(),
                    benchmark,
                    grayscale=grayscale,
                    figsize=(figsize[0], figsize[0] * 0.5),
                    show=True,
                    ylabel="",
                    returns_label=col,
                    compounded=compounded,
                    active=active,
                )

        return

    # Ensure returns is DataFrame for full mode processing
    returns = _pd.DataFrame(returns)

    # prepare timeseries
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        benchmark.name = benchmark_colname
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    # Generate comprehensive plot suite
    # Cumulative returns plot
    _plots.returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.6),
        show=True,
        ylabel="",
        prepare_returns=False,
        compound=compounded,
    )

    # Log returns plot for better visualization
    _plots.log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compound=compounded,
    )

    # Volatility-matched returns (if benchmark exists)
    if benchmark is not None:
        _plots.returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel="",
            prepare_returns=False,
            compound=compounded,
        )

    # Yearly returns comparison
    _plots.yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compounded=compounded,
    )

    # Returns distribution histogram
    _plots.histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel="",
        prepare_returns=False,
        compounded=compounded,
    )

    # Calculate figure size for smaller plots
    small_fig_size = (figsize[0], figsize[0] * 0.35)
    if len(returns.columns) > 1:
        small_fig_size = (
            figsize[0],
            figsize[0] * (0.33 * (len(returns.columns) * 0.66)),
        )

    # Daily returns scatter plot
    _plots.daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        prepare_returns=False,
        active=active,
    )

    # Rolling beta analysis (if benchmark exists)
    if benchmark is not None:
        _plots.rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            window1=win_half_year,
            window2=win_year,
            figsize=small_fig_size,
            show=True,
            ylabel="",
            prepare_returns=False,
        )

    # Rolling volatility analysis
    _plots.rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Rolling Sharpe ratio analysis
    _plots.rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Rolling Sortino ratio analysis
    _plots.rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel="",
        period=win_half_year,
    )

    # Drawdown periods analysis
    if isinstance(returns, _pd.Series):
        _plots.drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel="",
            prepare_returns=False,
            compounded=compounded,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _plots.drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                title=col,
                prepare_returns=False,
                compounded=compounded,
            )

    # Underwater (drawdown) plot
    _plots.drawdown(
        returns,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.4),
        show=True,
        ylabel="",
        compound=compounded,
    )

    # Monthly returns heatmap
    if isinstance(returns, _pd.Series):
        _plots.monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            returns_label=returns.name,
            show=True,
            ylabel="",
            compounded=compounded,
            active=active,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _plots.monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel="",
                returns_label=col,
                compounded=compounded,
                active=active,
            )

    # Returns distribution analysis
    if isinstance(returns, _pd.Series):
        _plots.distribution(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            title=returns.name,
            ylabel="",
            prepare_returns=False,
            compounded=compounded,
        )
    elif isinstance(returns, _pd.DataFrame):
        # Handle multiple strategy columns
        for col in returns.columns:
            _plots.distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                title=col,
                ylabel="",
                prepare_returns=False,
                compounded=compounded,
            )


def _calc_dd(df, display=True, as_pct=False):
    """
    Calculate drawdown statistics for performance analysis.

    This helper function computes comprehensive drawdown statistics including
    maximum drawdown, drawdown dates, recovery periods, and average drawdown
    metrics. It handles both single strategy and multiple strategy analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing returns data with columns for strategies
        and optionally benchmark
    display : bool, default True
        Whether the output is for display purposes (affects formatting)
    as_pct : bool, default False
        Whether to return percentages instead of decimals

    Returns
    -------
    pd.DataFrame
        DataFrame with drawdown statistics including:
        - Max Drawdown %: Maximum drawdown percentage
        - Max DD Date: Date of maximum drawdown
        - Max DD Period Start: Start date of worst drawdown period
        - Max DD Period End: End date of worst drawdown period
        - Longest DD Days: Duration of longest drawdown in days
        - Avg. Drawdown %: Average drawdown percentage
        - Avg. Drawdown Days: Average drawdown duration in days

    Examples
    --------
    >>> dd_stats = _calc_dd(returns_df, display=False)
    >>> dd_stats = _calc_dd(returns_df, as_pct=True)
    """
    # Convert returns to drawdown series
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

    # Return empty DataFrame if no drawdowns found
    if dd_info.empty:
        return _pd.DataFrame()

    # Handle different column structures based on data type
    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    # Calculate drawdown statistics based on data structure
    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        # Multiple strategy columns case
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Max DD Date": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["valley"]
                .values[0],
                "Max DD Period Start": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["start"]
                .values[0],
                "Max DD Period End": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["end"]
                .values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        # Single strategy case
        max_dd = ret_dd.sort_values(by="max drawdown", ascending=True)
        dd_stats = {
            "returns": {
                "Max Drawdown %": max_dd["max drawdown"].values[0] / 100,
                "Max DD Date": max_dd["valley"].values[0],
                "Max DD Period Start": max_dd["start"].values[0],
                "Max DD Period End": max_dd["end"].values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd["days"].mean())),
            }
        }

    # Add benchmark drawdown statistics if present
    if "benchmark" in df and (dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Max DD Date": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["valley"].values[0],
            "Max DD Period Start": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["start"].values[0],
            "Max DD Period End": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["end"].values[0],
            "Longest DD Days": str(
                _np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(_np.round(bench_dd["days"].mean())),
        }

    # Apply percentage multiplier based on display settings
    # pct multiplier
    pct = 100 if display or as_pct else 1

    # Convert to DataFrame and apply percentage formatting
    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T


def _html_table(obj, showindex="default"):
    """
    Convert DataFrame to HTML table format for report generation.

    This helper function converts pandas DataFrames to clean HTML table format
    suitable for embedding in HTML reports. It removes default tabulate styling
    and cleans up spacing for better presentation.

    Parameters
    ----------
    obj : pd.DataFrame
        DataFrame to convert to HTML table
    showindex : str or bool, default "default"
        Whether to show the DataFrame index in the HTML table.
        "default" uses tabulate's default behavior

    Returns
    -------
    str
        HTML string containing the formatted table

    Examples
    --------
    >>> html_str = _html_table(metrics_df)
    >>> html_str = _html_table(metrics_df, showindex=False)
    """
    # Convert DataFrame to HTML table using tabulate
    obj = _tabulate(
        obj, headers="keys", tablefmt="html", floatfmt=".2f", showindex=showindex
    )

    # Remove default tabulate styling attributes
    obj = obj.replace(' style="text-align: right;"', "")
    obj = obj.replace(' style="text-align: left;"', "")
    obj = obj.replace(' style="text-align: center;"', "")

    # Clean up spacing in table cells
    obj = _regex.sub("<td> +", "<td>", obj)
    obj = _regex.sub(" +</td>", "</td>", obj)
    obj = _regex.sub("<th> +", "<th>", obj)
    obj = _regex.sub(" +</th>", "</th>", obj)

    return obj


def _download_html(html, filename="quantstats-tearsheet.html"):
    """
    Generate JavaScript code to download HTML content in browser.

    This helper function creates JavaScript code that triggers a download
    of HTML content in the browser. It's used for downloading tearsheet
    reports directly from Jupyter notebooks.

    Parameters
    ----------
    html : str
        HTML content to be downloaded
    filename : str, default "quantstats-tearsheet.html"
        Filename for the downloaded file

    Returns
    -------
    None
        Displays JavaScript code in notebook to trigger download

    Examples
    --------
    >>> _download_html(html_content, "my_report.html")
    """
    # Create JavaScript code for file download
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var bl=new Blob(['{{html}}'],{type:"text/html"});
    var a=document.createElement("a");
    a.href=URL.createObjectURL(bl);
    a.download="{{filename}}";
    a.hidden=true;document.body.appendChild(a);
    a.innerHTML="download report";
    a.click();</script>""".replace(
            "\n", ""
        ),
    )

    # Insert HTML content and clean up formatting
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))

    # Execute JavaScript in notebook if in notebook environment
    if _utils._in_notebook():
        iDisplay(iHTML(jscode.replace("{{filename}}", filename)))


def _open_html(html):
    """
    Generate JavaScript code to open HTML content in new browser window.

    This helper function creates JavaScript code that opens HTML content
    in a new browser window. It's used for displaying tearsheet reports
    directly in the browser from Jupyter notebooks.

    Parameters
    ----------
    html : str
        HTML content to be displayed in new window

    Returns
    -------
    None
        Displays JavaScript code in notebook to open new window

    Examples
    --------
    >>> _open_html(html_content)
    """
    # Create JavaScript code to open new window with HTML content
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var win=window.open();win.document.body.innerHTML='{{html}}';
    </script>""".replace(
            "\n", ""
        ),
    )

    # Insert HTML content and clean up formatting
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))

    # Execute JavaScript in notebook if in notebook environment
    if _utils._in_notebook():
        iDisplay(iHTML(jscode))


def _embed_figure(figfiles, figfmt):
    """
    Embed matplotlib figures in HTML format for reports.

    This helper function converts matplotlib figure objects to embedded
    HTML format suitable for inclusion in HTML reports. It handles both
    SVG and base64-encoded image formats.

    Parameters
    ----------
    figfiles : io.StringIO or list of io.StringIO
        File-like objects containing figure data. Can be single figure
        or list of figures for multiple plots
    figfmt : str
        Format for the figures ('svg', 'png', 'jpg', etc.)

    Returns
    -------
    str
        HTML string with embedded figure(s) ready for inclusion in report

    Examples
    --------
    >>> embed_str = _embed_figure(figfile, 'svg')
    >>> embed_str = _embed_figure([fig1, fig2], 'png')
    """
    # Handle multiple figures
    if isinstance(figfiles, list):
        embed_string = "\n"
        for figfile in figfiles:
            figbytes = figfile.getvalue()
            if figfmt == "svg":
                # SVG can be embedded directly as text
                return figbytes.decode()
            # For other formats, encode as base64 data URI
            data_uri = _b64encode(figbytes).decode()
            embed_string.join(
                '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)
            )
    else:
        # Handle single figure
        figbytes = figfiles.getvalue()
        if figfmt == "svg":
            # SVG can be embedded directly as text
            return figbytes.decode()
        # For other formats, encode as base64 data URI
        data_uri = _b64encode(figbytes).decode()
        embed_string = '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)

    return embed_string

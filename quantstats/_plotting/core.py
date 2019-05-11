#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
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

import matplotlib.pyplot as _plt
import matplotlib.dates as _mdates
from matplotlib.ticker import (
    FormatStrFormatter as _FormatStrFormatter,
    FuncFormatter as _FuncFormatter
)

import pandas as _pd
import numpy as _np
import seaborn as _sns
from .. import stats as _stats

_sns.set(font_scale=1.1, rc={
    'figure.figsize': (10, 6),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666'
})

_FLATUI_COLORS = ["#fedd78", "#348dc1", "#af4b64",
                  "#4fa487", "#9b59b6", "#808080"]
_GRAYSCALE_COLORS = ['silver', '#222222', 'gray'] * 3


def _get_colors(grayscale):
    colors = _FLATUI_COLORS
    ls = '-'
    alpha = .8
    if grayscale:
        colors = _GRAYSCALE_COLORS
        ls = '-'
        alpha = 0.5
    return colors, ls, alpha


def plot_returns_bars(returns, benchmark=None,
                      returns_label="Strategy",
                      hline=None, hlw=None, hlcolor="red", hllabel="",
                      resample="A", title="Returns", match_volatility=False,
                      log_scale=False, figsize=(10, 6),
                      grayscale=False, fontname='Arial',
                      subtitle=True, savefig=None, show=True):

    if match_volatility and benchmark is None:
        raise ValueError('match_volatility requires passing of '
                         'benchmark.')
    elif match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    colors, ls, alpha = _get_colors(grayscale)
    df = _pd.DataFrame(index=returns.index, data={returns_label: returns})
    if isinstance(benchmark, _pd.Series):
        df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
        df = df[['Benchmark', returns_label]]

    df = df.dropna()
    if resample is not None:
        df = df.resample(resample).apply(
            _stats.comp).resample(resample).last()
    # ---------------

    fig, ax = _plt.subplots(figsize=figsize)

    # use a more precise date string for the x axis locations in the toolbar
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            df.index.date[:1][0].strftime('%Y'),
            df.index.date[-1:][0].strftime('%Y')
        ), fontsize=12, color='gray')

    if benchmark is None:
        colors = colors[1:]
    df.plot(kind='bar', ax=ax, color=colors)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_xticklabels(df.index.year)
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
    years = list(set(df.index.year))
    if len(years) > 10:
        mod = int(len(years)/10)
        _plt.xticks(_np.arange(len(years)), [
            str(year) if not i % mod else '' for i, year in enumerate(years)])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    if hline:
        if grayscale:
            hlcolor = 'gray'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    if isinstance(benchmark, _pd.Series) or hline:
        ax.legend(fontsize=12)

    _plt.yscale("symlog" if log_scale else "linear")

    ax.set_xlabel('')
    ax.set_ylabel("Returns", fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_timeseries(returns, benchmark=None,
                    title="Returns", compound=False,
                    fill=False, returns_label="Strategy",
                    hline=None, hlw=None, hlcolor="red", hllabel="",
                    percent=True, match_volatility=False, log_scale=False,
                    resample=None, lw=1.5, figsize=(10, 6), ylabel="",
                    grayscale=False, fontname="Arial",
                    subtitle=True, savefig=None, show=True):

    colors, ls, alpha = _get_colors(grayscale)

    returns.fillna(0, inplace=True)
    if isinstance(benchmark, _pd.Series):
        benchmark.fillna(0, inplace=True)

    if match_volatility and benchmark is None:
        raise ValueError('match_volatility requires passing of '
                         'benchmark.')
    elif match_volatility and benchmark is not None:
        bmark_vol = benchmark.std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    if compound is True:
        returns = _stats.compsum(returns)
        if isinstance(benchmark, _pd.Series):
            benchmark = _stats.compsum(benchmark)

    if resample:
        returns = returns.resample(resample)
        returns = returns.last() if compound is True else returns.sum()
        if isinstance(benchmark, _pd.Series):
            benchmark = benchmark.resample(resample)
            benchmark = benchmark.last(
            ) if compound is True else benchmark.sum()
    # ---------------

    fig, ax = _plt.subplots(figsize=figsize)
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                  " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    if isinstance(benchmark, _pd.Series):
        ax.plot(benchmark, lw=lw, ls=ls, label="Benchmark", color=colors[0])

    alpha = .25 if grayscale else 1
    ax.plot(returns, lw=lw, label=returns_label, color=colors[1], alpha=alpha)

    if fill:
        ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=.25)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="-", lw=1,
               color='gray', zorder=1)
    ax.axhline(0, ls="--", lw=1,
               color='white' if grayscale else 'black', zorder=2)

    if isinstance(benchmark, _pd.Series) or hline:
        ax.legend(fontsize=12)

    _plt.yscale("symlog" if log_scale else "linear")

    if percent:
        ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
        # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        #     lambda x, loc: "{:,}%".format(int(x*100))))

    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_histogram(returns, resample="M", bins=20,
                   fontname='Arial', grayscale=False,
                   title="Returns", kde=True, figsize=(10, 6),
                   subtitle=True, savefig=None, show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['silver', 'gray', 'black']

    returns = returns.fillna(0).resample(resample).apply(
        _stats.comp).resample(resample).last()

    fig, ax = _plt.subplots(figsize=figsize)
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%Y'),
            returns.index.date[-1:][0].strftime('%Y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.axvline(returns.mean(), ls="--", lw=1.5,
               color=colors[2], zorder=2, label="Average")

    _sns.distplot(returns, bins=bins,
                  axlabel="", color=colors[0], hist_kws=dict(alpha=1),
                  kde=kde,
                  # , label="Kernel Estimate"
                  kde_kws=dict(color='black', alpha=.7),
                  ax=ax)

    ax.xaxis.set_major_formatter(_plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    ax.axvline(0, lw=1, color="#000000", zorder=2)

    ax.set_xlabel('')
    ax.set_ylabel("Occurrences", fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)
    ax.legend(fontsize=12)

    # fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_rolling_stats(returns, benchmark=None, title="",
                       returns_label="Strategy",
                       hline=None, hlw=None, hlcolor="red", hllabel="",
                       lw=1.5, figsize=(10, 6), ylabel="",
                       grayscale=False, fontname="Arial", subtitle=True,
                       savefig=None, show=True):

    colors, ls, alpha = _get_colors(grayscale)

    fig, ax = _plt.subplots(figsize=figsize)

    df = _pd.DataFrame(index=returns.index, data={returns_label: returns})
    if isinstance(benchmark, _pd.Series):
        df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
        df = df[['Benchmark', returns_label]].dropna()
        ax.plot(df['Benchmark'], lw=lw, label="Benchmark",
                color=colors[0], alpha=.8)

    ax.plot(df[returns_label].dropna(), lw=lw,
            label=returns_label, color=colors[1])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')\
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            df.index.date[:1][0].strftime('%e %b \'%y'),
            df.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    ax.set_ylabel(ylabel, fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)
    ax.yaxis.set_major_formatter(_FormatStrFormatter('%.2f'))
    ax.legend(fontsize=12)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)
    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_rolling_beta(returns, benchmark,
                      window1=126, window1_label="",
                      window2=None, window2_label="",
                      title="", hlcolor="red", figsize=(10, 6),
                      grayscale=False, fontname="Arial", lw=1.5,
                      subtitle=True, savefig=None, show=True):

    colors, ls, alpha = _get_colors(grayscale)

    fig, ax = _plt.subplots(figsize=figsize)
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    beta = _stats.rolling_greeks(returns, benchmark, window1)['beta']
    ax.plot(beta, lw=lw, label=window1_label, color=colors[1])

    if window2:
        ax.plot(_stats.rolling_greeks(returns, benchmark, window2)['beta'],
                lw=lw, label=window2_label, color="gray", alpha=0.8)
    mmin = min([-100, int(beta.min()*100)])
    mmax = max([100, int(beta.max()*100)])
    step = 50 if (mmax-mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    hlcolor = 'black' if grayscale else hlcolor
    ax.axhline(beta.mean(), ls="--", lw=1.5,
               color=hlcolor, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
    ax.set_ylabel("Beta", fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")

    ax.yaxis.set_label_coords(-.1, .5)
    ax.legend(fontsize=12)
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_longest_drawdowns(returns, periods=5, lw=1.5,
                           fontname='Arial', grayscale=False,
                           log_scale=False, figsize=(10, 6),
                           subtitle=True, savefig=None, show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['#000000'] * 3

    dd = _stats.to_drawdown_series(returns.fillna(0))
    dddf = _stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(
        by='days', ascending=False, kind='mergesort')[:periods]

    fig, ax = _plt.subplots(figsize=figsize)
    fig.suptitle("Top %.0f Drawdown Periods\n" %
                 periods, y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")
    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(_stats.compsum(returns), lw=lw, label="Backtest", color=colors[0])

    highlight = 'black' if grayscale else 'red'
    for idx, row in longest_dd.iterrows():
        ax.axvspan(*_mdates.datestr2num([str(row['start']), str(row['end'])]),
                   color=highlight, alpha=.1)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)
    _plt.yscale("symlog" if log_scale else "linear")
    ax.set_ylabel("Cumulative Returns", fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")

    ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_distribution(returns, figsize=(10, 6),
                      fontname='Arial', grayscale=False,
                      subtitle=True, savefig=None, show=True):

    colors = _FLATUI_COLORS
    if grayscale:
        colors = ['#f9f9f9', '#dddddd', '#bbbbbb', '#999999', '#808080']
    # colors, ls, alpha = _get_colors(grayscale)

    port = _pd.DataFrame(returns.fillna(0))
    port.columns = ['Daily']

    port['Weekly'] = port['Daily'].resample(
        'W-MON').apply(_stats.comp).resample('W-MON').last()
    port['Weekly'].ffill(inplace=True)

    port['Monthly'] = port['Daily'].resample(
        'M').apply(_stats.comp).resample('M').last()
    port['Monthly'].ffill(inplace=True)

    port['Quarterly'] = port['Daily'].resample(
        'Q').apply(_stats.comp).resample('Q').last()
    port['Quarterly'].ffill(inplace=True)

    port['Yearly'] = port['Daily'].resample(
        'A').apply(_stats.comp).resample('A').last()
    port['Yearly'].ffill(inplace=True)

    fig, ax = _plt.subplots(figsize=figsize)
    fig.suptitle("Return Quantiles\n", y=.99,
                 fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    _sns.boxplot(data=port, ax=ax, palette=tuple(colors[:5]))

    ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    ax.set_ylabel('Rerurns', fontname=fontname,
                  fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)

    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def plot_table(tbl, columns=None, title="", title_loc="left",
               header=True,
               colWidths=None,
               rowLoc='right',
               colLoc='right',
               colLabels=[],
               edges='horizontal',
               orient='horizontal',
               figsize=(10, 6),
               savefig=None,
               show=False):

    if columns is not None:
        try:
            tbl.columns = columns
        except Exception:
            pass

    fig = _plt.figure(figsize=(5.5, 6))
    ax = _plt.subplot(111, frame_on=False)

    if title != "":
        ax.set_title(title, fontweight="bold",
                     fontsize=14, color="black", loc=title_loc)

    the_table = ax.table(cellText=tbl.values,
                         colWidths=colWidths,
                         rowLoc=rowLoc,
                         colLoc=colLoc,
                         edges=edges,
                         colLabels=(tbl.columns if header else None),
                         loc='center',
                         zorder=2
                         )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_height(0.08)
        cell.set_text_props(color='black')
        cell.set_edgecolor('#dddddd')
        if row == 0 and header:
            cell.set_edgecolor('black')
            cell.set_facecolor('black')
            cell.set_linewidth(2)
            cell.set_text_props(weight='bold', color='black')
        elif col == 0 and "vertical" in orient:
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(1)
            cell.set_text_props(weight='bold', color='black')
        elif row > 1:
            cell.set_linewidth(1)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(fig)

    _plt.close()

    if not show:
        return fig


def format_cur_axis(x, pos):
    if x >= 1e6:
        return '$%1.1fM' % (x * 1e-6)
    if x >= 1e3:
        return '$%1.0fK' % (x * 1e-3)
    return '$%1.0f' % x


def format_pct_axis(x, pos):
    x *= 100 # lambda x, loc: "{:,}%".format(int(x * 100))
    if x >= 1e6:
        return '%1.1fM%%' % (x * 1e-6)
    if x >= 1e3:
        return '%1.0fK%%' % (x * 1e-3)
    return '%1.0f%%' % x

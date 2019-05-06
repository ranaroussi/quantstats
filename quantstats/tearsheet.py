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

import pandas as _pd
# import numpy as _np
from datetime import (
    datetime as _dt, timedelta as _td
)
try:
    from IPython.core.display import display as iDisplay, HTML
except Exception:
    pass

import re as _regex
from tabulate import tabulate
from . import stats, utils, plot, __version__


def _html_table(obj, showindex="default"):
    obj = tabulate(obj, headers="keys", tablefmt='html',
                   floatfmt=".2f", showindex=showindex)
    obj = obj.replace(' style="text-align: right;"', '')
    obj = obj.replace(' style="text-align: left;"', '')
    obj = obj.replace(' style="text-align: center;"', '')
    obj = _regex.sub('<td> +', '<td>', obj)
    obj = _regex.sub(' +</td>', '</td>', obj)
    obj = _regex.sub('<th> +', '<th>', obj)
    obj = _regex.sub(' +</th>', '</th>', obj)
    return obj


def html(returns, benchmark=None, rf=0.,
         grayscale=False, title='Strategy Tearsheet'):
    f = open(__file__[:-3] + '.html')
    tpl = f.read()
    f.close()

    date_range = returns.index.strftime('%e %b, %Y')
    tpl = tpl.replace('{{date_range}}', date_range[0] + ' - ' + date_range[-1])
    tpl = tpl.replace('{{title}}', title)
    tpl = tpl.replace('{{v}}', __version__)

    mtrx = metrics(returns=returns, benchmark=benchmark,
                   rf=rf, display=False, mode='full',
                   sep=True)[2:]
    mtrx.index.name = 'Metric'
    tpl = tpl.replace('{{metrics}}', _html_table(mtrx))
    tpl = tpl.replace('<tr><td></td><td></td><td></td></tr>',
                      '<tr><td colspan="3"><hr></td></tr>')

    if benchmark is not None:
        yoy = stats.compare(returns, benchmark, "A")
        yoy.columns = ['Benchmark', 'Strategy', 'Diff%', 'Won']
        yoy.index.name = 'Year'
        tpl = tpl.replace('{{eoy_title}}', '<h3>EOY Returns vs Benchmark</h3>')
        tpl = tpl.replace('{{eoy_table}}', _html_table(yoy))

    dd = stats.to_drawdown_series(returns)
    dd_info = stats.drawdown_details(dd).sort_values(
        by='max drawdown', ascending=True)[:10]

    dd_info = dd_info[['start', 'end', 'max drawdown', 'days']]
    dd_info.columns = ['Started', 'Recovered', 'Drawdown', 'Days']
    tpl = tpl.replace('{{dd_info}}', _html_table(dd_info, False))

    # plots
    figfile = utils._file_stream()
    plot.returns(returns, benchmark, grayscale=grayscale,
                 figsize=(8, 5), subtitle=False,
                 savefig={'fname': figfile, 'format': 'svg'},
                 show=False)
    tpl = tpl.replace('{{returns}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.log_returns(returns, benchmark, grayscale=grayscale,
                     figsize=(8, 4), subtitle=False,
                     savefig={'fname': figfile, 'format': 'svg'},
                     show=False)
    tpl = tpl.replace('{{log_returns}}', figfile.getvalue().decode())

    if benchmark is not None:
        figfile = utils._file_stream()
        plot.returns(returns, benchmark, match_volatility=True,
                     grayscale=grayscale, figsize=(8, 4), subtitle=False,
                     savefig={'fname': figfile, 'format': 'svg'},
                     show=False)
        tpl = tpl.replace('{{vol_returns}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.yearly_returns(returns, benchmark, grayscale=grayscale,
                        figsize=(8, 4), subtitle=False,
                        savefig={'fname': figfile, 'format': 'svg'},
                        show=False)
    tpl = tpl.replace('{{eoy_returns}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.histogram(returns, grayscale=grayscale,
                   figsize=(8, 4), subtitle=False,
                   savefig={'fname': figfile, 'format': 'svg'},
                   show=False)
    tpl = tpl.replace('{{monthly_dist}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.daily_returns(returns, grayscale=grayscale,
                       figsize=(8, 3), subtitle=False,
                       savefig={'fname': figfile, 'format': 'svg'},
                       show=False)
    tpl = tpl.replace('{{daily_returns}}', figfile.getvalue().decode())

    if benchmark is not None:
        figfile = utils._file_stream()
        plot.rolling_beta(returns, benchmark, grayscale=grayscale,
                          figsize=(8, 3), subtitle=False,
                          savefig={'fname': figfile, 'format': 'svg'},
                          show=False)
        tpl = tpl.replace('{{rolling_beta}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.rolling_volatility(returns, benchmark, grayscale=grayscale,
                            figsize=(8, 3), subtitle=False,
                            savefig={'fname': figfile, 'format': 'svg'},
                            show=False)
    tpl = tpl.replace('{{rolling_volatility}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.rolling_sharpe(returns, grayscale=grayscale,
                        figsize=(8, 3), subtitle=False,
                        savefig={'fname': figfile, 'format': 'svg'},
                        show=False)
    tpl = tpl.replace('{{rolling_sharpe}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.rolling_sortino(returns, grayscale=grayscale,
                         figsize=(8, 3), subtitle=False,
                         savefig={'fname': figfile, 'format': 'svg'},
                         show=False)
    tpl = tpl.replace('{{rolling_sortino}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.drawdowns_periods(returns, grayscale=grayscale,
                           figsize=(8, 4), subtitle=False,
                           savefig={'fname': figfile, 'format': 'svg'},
                           show=False)
    tpl = tpl.replace('{{dd_periods}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.drawdown(returns, grayscale=grayscale,
                  figsize=(8, 4), subtitle=False,
                  savefig={'fname': figfile, 'format': 'svg'},
                  show=False)
    tpl = tpl.replace('{{dd_plot}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.monthly_heatmap(returns, grayscale=grayscale,
                         figsize=(8, 4), cbar=False,
                         savefig={'fname': figfile, 'format': 'svg'},
                         show=False)
    tpl = tpl.replace('{{monthly_heatmap}}', figfile.getvalue().decode())

    figfile = utils._file_stream()
    plot.distribution(returns, grayscale=grayscale,
                      figsize=(8, 4), subtitle=False,
                      savefig={'fname': figfile, 'format': 'svg'},
                      show=False)
    tpl = tpl.replace('{{returns_dist}}', figfile.getvalue().decode())

    tpl = _regex.sub('\{\{(.*?)\}\} ', '', tpl)

    # iDisplay(HTML(tpl))
    with open('/Users/ran/Desktop/tearsheet.html', 'w') as file:
        file.write(tpl)

    print('done')
    # return tpl
    # yoy = qs.stats.compare(returns, benchmark, "A")
    # print(tabulate(yoy, headers="keys", tablefmt='simple', floatfmt=".2f"))


def full(returns, benchmark=None, rf=0., grayscale=False, figsize=(8, 5)):

    dd = stats.to_drawdown_series(returns)
    dd_info = stats.drawdown_details(dd).sort_values(
        by='max drawdown', ascending=True)[:5]
    dd_info.index = range(1, 6)

    # dd_info['start'] = dd_info['start'].dt.strftime('%Y-%m-%d')
    # dd_info['end'] = dd_info['end'].dt.strftime('%Y-%m-%d')
    # dd_info['valley'] = dd_info['valley'].dt.strftime('%Y-%m-%d')

    dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
    if utils._in_notebook():
        iDisplay(HTML('<h4>Performance Metrics</h4>'))
        iDisplay(metrics(returns=returns, benchmark=benchmark,
                         rf=rf, display=False, mode='full'))
        iDisplay(HTML('<h4>5 Worst Drawdowns</h4>'))
        iDisplay(dd_info)

        iDisplay(HTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        metrics(returns=returns, benchmark=benchmark,
                rf=rf, display=True, mode='full')
        print('\n\n')
        print('[5 Worst Drawdowns]\n')
        print(tabulate(dd_info, headers="keys",
                       tablefmt='simple', floatfmt=".2f"))
        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    plots(returns=returns, benchmark=benchmark,
          grayscale=grayscale, figsize=figsize, mode='full')


def basic(returns, benchmark=None, rf=0., grayscale=False, figsize=(8, 5)):

    if utils._in_notebook():
        df = metrics(returns=returns, benchmark=benchmark,
                     rf=rf, display=False, mode='basic')
        iDisplay(HTML('<h4>Performance Metrics</h4>'))
        iDisplay(df)

        iDisplay(HTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        metrics(returns=returns, benchmark=benchmark,
                rf=rf, display=True, mode='basic')

        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    plots(returns=returns, benchmark=benchmark,
          grayscale=grayscale, figsize=figsize, mode='basic')


def metrics(returns, benchmark=None, rf=0., display=True,
            mode='basic', sep=False):

    blank = ['']
    df = _pd.DataFrame({"returns": utils._prepare_returns(returns, rf)})
    if benchmark is not None:
        blank = ['', '']
        df["benchmark"] = utils._prepare_benchmark(
            benchmark, returns.index, rf)

    df = df.dropna()

    # return df
    dd = _calc_dd(df)

    metrics = _pd.DataFrame()

    s_start = {'returns': df['returns'].index.strftime('%Y-%m-%d')[0]}
    s_end = {'returns': df['returns'].index.strftime('%Y-%m-%d')[-1]}
    s_rf = {'returns': rf}

    if "benchmark" in df:
        s_start['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[0]
        s_end['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[-1]
        s_rf['benchmark'] = rf

    metrics['Start Period'] = _pd.Series(s_start)
    metrics['End Period'] = _pd.Series(s_end)
    metrics['Risk-free rate %'] = _pd.Series(s_rf)
    metrics['Exposure %%'] = stats.exposure(df) * 100

    metrics['~'] = blank

    metrics['Total Return %'] = stats.comp(df) * 100
    metrics['CAGR%%'] = stats.cagr(df, rf) * 100
    metrics['Sharpe'] = stats.sharpe(df, rf)
    metrics['Sortino'] = stats.sortino(df, rf)
    metrics['Max Drawdown %'] = blank
    metrics['Longest DD Days'] = blank

    if mode.lower() == 'full':
        ret_vol = stats.volatility(df['returns']) * 100
        if "benchmark" in df:
            bench_vol = stats.volatility(df['benchmark']) * 100
            metrics['Volatility (ann.) %'] = [ret_vol, bench_vol]
            metrics['R^2'] = stats.r_squared(df['returns'], df['benchmark'])
        else:
            metrics['Volatility (ann.) %'] = [ret_vol]

        metrics['Calmar'] = stats.calmar(df)
        metrics['Skew'] = stats.skew(df)
        metrics['Kurtosis'] = stats.kurtosis(df)

    if mode.lower() == 'full':
        metrics['~~~~~~~~~~'] = blank

        metrics['Expected Daily %%'] = stats.expected_return(df) * 100
        metrics['Expected Monthly %%'] = stats.expected_return(
            df, aggregate='M') * 100
        metrics['Expected Yearly %%'] = stats.expected_return(
            df, aggregate='A') * 100
        metrics['Kelly Criterion %'] = stats.kelly_criterion(df) * 100
        metrics['Risk of Ruin %'] = stats.risk_of_ruin(df)

        metrics['Daily Value-at-Risk %'] = -stats.var(df) * 100
        metrics['Expected Shortfall (cVaR) %'] = -stats.cvar(df) * 100

    metrics['~~~~~~'] = blank

    metrics['Payoff Ratio'] = stats.payoff_ratio(df)
    metrics['Profit Factor'] = stats.profit_factor(df)
    metrics['Common Sense Ratio'] = stats.common_sense_ratio(df)
    metrics['CPC Index'] = stats.cpc_index(df)
    metrics['Tail Ratio'] = stats.tail_ratio(df)
    metrics['Outlier Win Ratio'] = stats.outlier_win_ratio(df)
    metrics['Outlier Loss Ratio'] = stats.outlier_loss_ratio(df)

    # returns
    metrics['~~'] = blank

    today = _dt.today()
    metrics['MTD %'] = stats.comp(
        df[df.index >= _dt(today.year, today.month, 1)]) * 100

    d = today - _td(3*365/12)
    metrics['3M %'] = stats.comp(
        df[df.index >= _dt(d.year, d.month, d.day)]) * 100

    d = today - _td(6*365/12)
    metrics['6M %'] = stats.comp(
        df[df.index >= _dt(d.year, d.month, d.day)]) * 100

    metrics['YTD %'] = stats.comp(df[df.index >= _dt(today.year, 1, 1)]) * 100

    d = today - _td(12*365/12)
    metrics['1Y %'] = stats.comp(
        df[df.index >= _dt(d.year, d.month, d.day)]) * 100
    metrics['3Y (ann.) %'] = stats.cagr(
        df[df.index >= _dt(today.year-3, today.month, today.day)]) * 100
    metrics['5Y (ann.) %'] = stats.cagr(
        df[df.index >= _dt(today.year-5, today.month, today.day)]) * 100
    metrics['10Y (ann.) %'] = stats.cagr(
        df[df.index >= _dt(today.year-10, today.month, today.day)]) * 100
    metrics['All-time (ann.) %'] = stats.cagr(df) * 100

    # best/worst
    if mode.lower() == 'full':
        metrics['~~~'] = blank
        metrics['Best Day %'] = stats.best(df) * 100
        metrics['Worst Day %'] = stats.worst(df) * 100
        metrics['Best Month %'] = stats.best(df, aggregate='M') * 100
        metrics['Worst Month %'] = stats.worst(df, aggregate='M') * 100
        metrics['Best Year %'] = stats.best(df, aggregate='A') * 100
        metrics['Worst Year %'] = stats.worst(df, aggregate='A') * 100

    # dd
    metrics['~~~~'] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics['Recovery Factor'] = stats.recovery_factor(df)

    # win rate
    if mode.lower() == 'full':
        metrics['~~~~~'] = blank
        metrics['Avg. Up Month %'] = stats.avg_win(df, aggregate='M') * 100
        metrics['Avg. Down Month %'] = stats.avg_loss(df, aggregate='M') * 100
        metrics['Win Days %%'] = stats.win_rate(df) * 100
        metrics['Win Month %%'] = stats.win_rate(df, aggregate='M') * 100
        metrics['Win Quarter %%'] = stats.win_rate(df, aggregate='Q') * 100
        metrics['Win Year %%'] = stats.win_rate(df, aggregate='A') * 100

    if mode.lower() == "full" and "benchmark" in df:
        metrics['~~~~~~~'] = blank
        greeks = stats.greeks(df['returns'], df['benchmark'])
        metrics['Beta'] = [str(round(greeks['beta'], 2)), '-']
        metrics['Alpha'] = [str(round(greeks['alpha'], 2)), '-']

    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2).astype(str)
        except Exception:
            pass
        if "%" in col:
            metrics[col] = metrics[col] + '%'

    metrics.columns = [
        col if '~' not in col else '' for col in metrics.columns]
    metrics.columns = [
        col[:-1] if '%' in col else col for col in metrics.columns]

    metrics['Longest DD Days'] = _pd.to_numeric(
        metrics['Longest DD Days']).astype('int').astype(str)
    metrics['Avg. Drawdown Days'] = _pd.to_numeric(
        metrics['Avg. Drawdown Days']).astype('int').astype(str)

    metrics = metrics.T

    if "benchmark" in df:
        metrics.columns = ['Strategy', 'Benchmark']
    else:
        metrics.columns = ['Strategy']

    if display:
        print(tabulate(metrics, headers="keys", tablefmt='simple'))
        return

    if not sep:
        metrics = metrics[metrics.index != '']
    return metrics


def plots(returns, benchmark=None, grayscale=False,
          figsize=(8, 5), mode='basic'):

    if mode.lower() != 'full':
        plot.snapshot(returns, grayscale=grayscale,
                      figsize=(figsize[0], figsize[0]))

        plot.monthly_heatmap(returns, grayscale=grayscale,
                             figsize=(figsize[0], figsize[0]*.5))

        return

    plot.returns(returns, benchmark, grayscale=grayscale,
                 figsize=(figsize[0], figsize[0]*.6))

    plot.log_returns(returns, benchmark, grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.5))

    if benchmark is not None:
        plot.returns(returns, benchmark, match_volatility=True,
                     grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.5))

    plot.yearly_returns(returns, benchmark,
                        grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.5))

    plot.histogram(returns, grayscale=grayscale,
                   figsize=(figsize[0], figsize[0]*.5))

    plot.daily_returns(returns, grayscale=grayscale,
                       figsize=(figsize[0], figsize[0]*.3))

    if benchmark is not None:
        plot.rolling_beta(returns, benchmark, grayscale=grayscale,
                          figsize=(
                              figsize[0], figsize[0]*.3))

    plot.rolling_volatility(
        returns, benchmark, grayscale=grayscale,
        figsize=(figsize[0], figsize[0]*.3))

    plot.rolling_sharpe(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.3))

    plot.rolling_sortino(returns, grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.3))

    plot.drawdowns_periods(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.5))

    plot.drawdown(returns, grayscale=grayscale,
                  figsize=(figsize[0], figsize[0]*.5))

    plot.monthly_heatmap(returns, grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.5))

    plot.distribution(returns, grayscale=grayscale,
                      figsize=(figsize[0], figsize[0]*.5))


def _calc_dd(df):
    dd = stats.to_drawdown_series(df)
    dd_info = stats.drawdown_details(dd)

    if "returns" in dd_info:
        ret_dd = dd_info['returns']
    else:
        ret_dd = dd_info

    dd_stats = {
        'returns': {
            'Max Drawdown %': ret_dd.sort_values(
                by='max drawdown', ascending=True
            )['max drawdown'].values[0],
            'Longest DD Days': str(round(ret_dd.sort_values(
                by='days', ascending=False)['days'].values[0])),
            'Avg. Drawdown %': ret_dd['max drawdown'].mean(),
            'Avg. Drawdown Days': str(round(ret_dd['days'].mean()))
        }
    }
    if "benchmark" in df and (dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info['benchmark'].sort_values(by='max drawdown')
        dd_stats['benchmark'] = {
            'Max Drawdown %': bench_dd.sort_values(
                by='max drawdown', ascending=True
            )['max drawdown'].values[0],
            'Longest DD Days': str(round(bench_dd.sort_values(
                by='days', ascending=False)['days'].values[0])),
            'Avg. Drawdown %': bench_dd['max drawdown'].mean(),
            'Avg. Drawdown Days': str(round(bench_dd['days'].mean()))
        }

    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats['Max Drawdown %'] = dd_stats['Max Drawdown %'].astype(float)
    dd_stats['Avg. Drawdown %'] = dd_stats['Avg. Drawdown %'].astype(float)
    return dd_stats.T

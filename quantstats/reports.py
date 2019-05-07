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
from datetime import (
    datetime as _dt, timedelta as _td
)
import re as _regex
from tabulate import tabulate as _tabulate
from . import (
    __version__, stats as _stats,
    utils as _utils, plots as _plots
)

try:
    from IPython.core.display import (
        display as iDisplay, HTML as iHTML
    )
except ImportError:
    pass


def html(returns, benchmark=None, rf=0.,
         grayscale=False, title='Strategy Tearsheet',
         file=None):

    if file is None and not _utils._in_notebook():
        raise ValueError("`file` must be specified")

    f = open(__file__[:-4] + '.html')
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
        yoy = _stats.compare(returns, benchmark, "A")
        yoy.columns = ['Benchmark', 'Strategy', 'Diff%', 'Won']
        yoy.index.name = 'Year'
        tpl = tpl.replace('{{eoy_title}}', '<h3>EOY Returns vs Benchmark</h3>')
        tpl = tpl.replace('{{eoy_table}}', _html_table(yoy))

    dd = _stats.to_drawdown_series(returns)
    dd_info = _stats.drawdown_details(dd).sort_values(
        by='max drawdown', ascending=True)[:10]

    dd_info = dd_info[['start', 'end', 'max drawdown', 'days']]
    dd_info.columns = ['Started', 'Recovered', 'Drawdown', 'Days']
    tpl = tpl.replace('{{dd_info}}', _html_table(dd_info, False))

    # plots
    figfile = _utils._file_stream()
    _plots.returns(returns, benchmark, grayscale=grayscale,
                   figsize=(8, 5), subtitle=False,
                   savefig={'fname': figfile, 'format': 'svg'},
                   show=False)
    tpl = tpl.replace('{{returns}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.log_returns(returns, benchmark, grayscale=grayscale,
                       figsize=(8, 4), subtitle=False,
                       savefig={'fname': figfile, 'format': 'svg'},
                       show=False)
    tpl = tpl.replace('{{log_returns}}', figfile.getvalue().decode())

    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.returns(returns, benchmark, match_volatility=True,
                       grayscale=grayscale, figsize=(8, 4), subtitle=False,
                       savefig={'fname': figfile, 'format': 'svg'},
                       show=False)
        tpl = tpl.replace('{{vol_returns}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.yearly_returns(returns, benchmark, grayscale=grayscale,
                          figsize=(8, 4), subtitle=False,
                          savefig={'fname': figfile, 'format': 'svg'},
                          show=False)
    tpl = tpl.replace('{{eoy_returns}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.histogram(returns, grayscale=grayscale,
                     figsize=(8, 4), subtitle=False,
                     savefig={'fname': figfile, 'format': 'svg'},
                     show=False)
    tpl = tpl.replace('{{monthly_dist}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.daily_returns(returns, grayscale=grayscale,
                         figsize=(8, 3), subtitle=False,
                         savefig={'fname': figfile, 'format': 'svg'},
                         show=False)
    tpl = tpl.replace('{{daily_returns}}', figfile.getvalue().decode())

    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.rolling_beta(returns, benchmark, grayscale=grayscale,
                            figsize=(8, 3), subtitle=False,
                            savefig={'fname': figfile, 'format': 'svg'},
                            show=False)
        tpl = tpl.replace('{{rolling_beta}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.rolling_volatility(returns, benchmark, grayscale=grayscale,
                              figsize=(8, 3), subtitle=False,
                              savefig={'fname': figfile, 'format': 'svg'},
                              show=False)
    tpl = tpl.replace('{{rolling_vol}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.rolling_sharpe(returns, grayscale=grayscale,
                          figsize=(8, 3), subtitle=False,
                          savefig={'fname': figfile, 'format': 'svg'},
                          show=False)
    tpl = tpl.replace('{{rolling_sharpe}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.rolling_sortino(returns, grayscale=grayscale,
                           figsize=(8, 3), subtitle=False,
                           savefig={'fname': figfile, 'format': 'svg'},
                           show=False)
    tpl = tpl.replace('{{rolling_sortino}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.drawdowns_periods(returns, grayscale=grayscale,
                             figsize=(8, 4), subtitle=False,
                             savefig={'fname': figfile, 'format': 'svg'},
                             show=False)
    tpl = tpl.replace('{{dd_periods}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.drawdown(returns, grayscale=grayscale,
                    figsize=(8, 3), subtitle=False,
                    savefig={'fname': figfile, 'format': 'svg'},
                    show=False)
    tpl = tpl.replace('{{dd_plot}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.monthly_heatmap(returns, grayscale=grayscale,
                           figsize=(8, 4), cbar=False,
                           savefig={'fname': figfile, 'format': 'svg'},
                           show=False)
    tpl = tpl.replace('{{monthly_heatmap}}', figfile.getvalue().decode())

    figfile = _utils._file_stream()
    _plots.distribution(returns, grayscale=grayscale,
                        figsize=(8, 4), subtitle=False,
                        savefig={'fname': figfile, 'format': 'svg'},
                        show=False)
    tpl = tpl.replace('{{returns_dist}}', figfile.getvalue().decode())

    tpl = _regex.sub('\{\{(.*?)\}\} ', '', tpl)

    if file is None:
        # _open_html(tpl)
        _download_html(tpl, 'quantstats-tearsheet.html')
        return

    with open(file, 'w') as file:
        file.write(tpl)


def full(returns, benchmark=None, rf=0., grayscale=False, figsize=(8, 5)):

    dd = _stats.to_drawdown_series(returns)
    dd_info = _stats.drawdown_details(dd).sort_values(
        by='max drawdown', ascending=True)[:5]
    dd_info.index = range(1, 6)

    dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
    if _utils._in_notebook():
        iDisplay(iHTML('<h4>Performance Metrics</h4>'))
        iDisplay(metrics(returns=returns, benchmark=benchmark,
                         rf=rf, display=False, mode='full'))
        iDisplay(iHTML('<h4>5 Worst Drawdowns</h4>'))
        iDisplay(dd_info)

        iDisplay(iHTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        metrics(returns=returns, benchmark=benchmark,
                rf=rf, display=True, mode='full')
        print('\n\n')
        print('[5 Worst Drawdowns]\n')
        print(_tabulate(dd_info, headers="keys",
                        tablefmt='simple', floatfmt=".2f"))
        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    plots(returns=returns, benchmark=benchmark,
          grayscale=grayscale, figsize=figsize, mode='full')


def basic(returns, benchmark=None, rf=0., grayscale=False, figsize=(8, 5)):

    if _utils._in_notebook():
        df = metrics(returns=returns, benchmark=benchmark,
                     rf=rf, display=False, mode='basic')
        iDisplay(iHTML('<h4>Performance Metrics</h4>'))
        iDisplay(df)

        iDisplay(iHTML('<h4>Strategy Visualization</h4>'))
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
    df = _pd.DataFrame({"returns": _utils._prepare_returns(returns, rf)})
    if benchmark is not None:
        blank = ['', '']
        df["benchmark"] = _utils._prepare_benchmark(
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
    metrics['Exposure %%'] = _stats.exposure(df) * 100

    metrics['~'] = blank

    metrics['Total Return %'] = _stats.comp(df) * 100
    metrics['CAGR%%'] = _stats.cagr(df, rf) * 100
    metrics['Sharpe'] = _stats.sharpe(df, rf)
    metrics['Sortino'] = _stats.sortino(df, rf)
    metrics['Max Drawdown %'] = blank
    metrics['Longest DD Days'] = blank

    if mode.lower() == 'full':
        ret_vol = _stats.volatility(df['returns']) * 100
        if "benchmark" in df:
            bench_vol = _stats.volatility(df['benchmark']) * 100
            metrics['Volatility (ann.) %'] = [ret_vol, bench_vol]
            metrics['R^2'] = _stats.r_squared(df['returns'], df['benchmark'])
        else:
            metrics['Volatility (ann.) %'] = [ret_vol]

        metrics['Calmar'] = _stats.calmar(df)
        metrics['Skew'] = _stats.skew(df)
        metrics['Kurtosis'] = _stats.kurtosis(df)

    if mode.lower() == 'full':
        metrics['~~~~~~~~~~'] = blank

        metrics['Expected Daily %%'] = _stats.expected_return(df) * 100
        metrics['Expected Monthly %%'] = _stats.expected_return(
            df, aggregate='M') * 100
        metrics['Expected Yearly %%'] = _stats.expected_return(
            df, aggregate='A') * 100
        metrics['Kelly Criterion %'] = _stats.kelly_criterion(df) * 100
        metrics['Risk of Ruin %'] = _stats.risk_of_ruin(df)

        metrics['Daily Value-at-Risk %'] = -_stats.var(df) * 100
        metrics['Expected Shortfall (cVaR) %'] = -_stats.cvar(df) * 100

    metrics['~~~~~~'] = blank

    metrics['Payoff Ratio'] = _stats.payoff_ratio(df)
    metrics['Profit Factor'] = _stats.profit_factor(df)
    metrics['Common Sense Ratio'] = _stats.common_sense_ratio(df)
    metrics['CPC Index'] = _stats.cpc_index(df)
    metrics['Tail Ratio'] = _stats.tail_ratio(df)
    metrics['Outlier Win Ratio'] = _stats.outlier_win_ratio(df)
    metrics['Outlier Loss Ratio'] = _stats.outlier_loss_ratio(df)

    # returns
    metrics['~~'] = blank

    today = _dt.today()
    metrics['MTD %'] = _stats.comp(
        df[df.index >= _dt(today.year, today.month, 1)]) * 100

    d = today - _td(3*365/12)
    metrics['3M %'] = _stats.comp(
        df[df.index >= _dt(d.year, d.month, d.day)]) * 100

    d = today - _td(6*365/12)
    metrics['6M %'] = _stats.comp(
        df[df.index >= _dt(d.year, d.month, d.day)]) * 100

    metrics['YTD %'] = _stats.comp(df[df.index >= _dt(today.year, 1, 1)]) * 100

    d = today - _td(12*365/12)
    metrics['1Y %'] = _stats.comp(
        df[df.index >= _dt(d.year, d.month, d.day)]) * 100
    metrics['3Y (ann.) %'] = _stats.cagr(
        df[df.index >= _dt(today.year-3, today.month, today.day)]) * 100
    metrics['5Y (ann.) %'] = _stats.cagr(
        df[df.index >= _dt(today.year-5, today.month, today.day)]) * 100
    metrics['10Y (ann.) %'] = _stats.cagr(
        df[df.index >= _dt(today.year-10, today.month, today.day)]) * 100
    metrics['All-time (ann.) %'] = _stats.cagr(df) * 100

    # best/worst
    if mode.lower() == 'full':
        metrics['~~~'] = blank
        metrics['Best Day %'] = _stats.best(df) * 100
        metrics['Worst Day %'] = _stats.worst(df) * 100
        metrics['Best Month %'] = _stats.best(df, aggregate='M') * 100
        metrics['Worst Month %'] = _stats.worst(df, aggregate='M') * 100
        metrics['Best Year %'] = _stats.best(df, aggregate='A') * 100
        metrics['Worst Year %'] = _stats.worst(df, aggregate='A') * 100

    # dd
    metrics['~~~~'] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics['Recovery Factor'] = _stats.recovery_factor(df)

    # win rate
    if mode.lower() == 'full':
        metrics['~~~~~'] = blank
        metrics['Avg. Up Month %'] = _stats.avg_win(df, aggregate='M') * 100
        metrics['Avg. Down Month %'] = _stats.avg_loss(df, aggregate='M') * 100
        metrics['Win Days %%'] = _stats.win_rate(df) * 100
        metrics['Win Month %%'] = _stats.win_rate(df, aggregate='M') * 100
        metrics['Win Quarter %%'] = _stats.win_rate(df, aggregate='Q') * 100
        metrics['Win Year %%'] = _stats.win_rate(df, aggregate='A') * 100

    if mode.lower() == "full" and "benchmark" in df:
        metrics['~~~~~~~'] = blank
        greeks = _stats.greeks(df['returns'], df['benchmark'])
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
        print(_tabulate(metrics, headers="keys", tablefmt='simple'))
        return

    if not sep:
        metrics = metrics[metrics.index != '']
    return metrics


def plots(returns, benchmark=None, grayscale=False,
          figsize=(8, 5), mode='basic'):

    if mode.lower() != 'full':
        _plots.snapshot(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]))

        _plots.monthly_heatmap(returns, grayscale=grayscale,
                               figsize=(figsize[0], figsize[0]*.5))

        return

    _plots.returns(returns, benchmark, grayscale=grayscale,
                   figsize=(figsize[0], figsize[0]*.6))

    _plots.log_returns(returns, benchmark, grayscale=grayscale,
                       figsize=(figsize[0], figsize[0]*.5))

    if benchmark is not None:
        _plots.returns(returns, benchmark, match_volatility=True,
                       grayscale=grayscale,
                       figsize=(figsize[0], figsize[0]*.5))

    _plots.yearly_returns(returns, benchmark,
                          grayscale=grayscale,
                          figsize=(figsize[0], figsize[0]*.5))

    _plots.histogram(returns, grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.5))

    _plots.daily_returns(returns, grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.3))

    if benchmark is not None:
        _plots.rolling_beta(returns, benchmark, grayscale=grayscale,
                            figsize=(
                                figsize[0], figsize[0]*.3))

    _plots.rolling_volatility(
        returns, benchmark, grayscale=grayscale,
        figsize=(figsize[0], figsize[0]*.3))

    _plots.rolling_sharpe(returns, grayscale=grayscale,
                          figsize=(figsize[0], figsize[0]*.3))

    _plots.rolling_sortino(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.3))

    _plots.drawdowns_periods(returns, grayscale=grayscale,
                             figsize=(figsize[0], figsize[0]*.5))

    _plots.drawdown(returns, grayscale=grayscale,
                    figsize=(figsize[0], figsize[0]*.4))

    _plots.monthly_heatmap(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.5))

    _plots.distribution(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.5))


def _calc_dd(df):
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

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


def _html_table(obj, showindex="default"):
    obj = _tabulate(obj, headers="keys", tablefmt='html',
                    floatfmt=".2f", showindex=showindex)
    obj = obj.replace(' style="text-align: right;"', '')
    obj = obj.replace(' style="text-align: left;"', '')
    obj = obj.replace(' style="text-align: center;"', '')
    obj = _regex.sub('<td> +', '<td>', obj)
    obj = _regex.sub(' +</td>', '</td>', obj)
    obj = _regex.sub('<th> +', '<th>', obj)
    obj = _regex.sub(' +</th>', '</th>', obj)
    return obj


def _download_html(html, filename="quantstats-tearsheet.html"):
    jscode = _regex.sub(' +', ' ', """<script>
    var bl=new Blob(['{{html}}'],{type:"text/html"});
    var a=document.createElement("a");
    a.href=URL.createObjectURL(bl);
    a.download="{{filename}}";
    a.hidden=true;document.body.appendChild(a);
    a.innerHTML="download report";
    a.click();</script>""".replace('\n', ''))
    jscode = jscode.replace('{{html}}', _regex.sub(
        ' +', ' ', html.replace('\n', '')))
    if _utils._in_notebook():
        iDisplay(iHTML(jscode.replace('{{filename}}', filename)))


def _open_html(html):
    jscode = _regex.sub(' +', ' ', """<script>
    var win=window.open();win.document.body.innerHTML='{{html}}';
    </script>""".replace('\n', ''))
    jscode = jscode.replace('{{html}}', _regex.sub(
        ' +', ' ', html.replace('\n', '')))
    if _utils._in_notebook():
        iDisplay(iHTML(jscode))

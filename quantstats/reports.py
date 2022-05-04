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
from base64 import b64encode as _b64encode
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


def html(returns, benchmark=None, rf=0., grayscale=False,
         title='Strategy Tearsheet', output=None, compounded=True,
         periods_per_year=252, download_filename='quantstats-tearsheet.html',
         figfmt='svg', template_path=None, match_dates=False, **kwargs):

    if output is None and not _utils._in_notebook():
        raise ValueError("`file` must be specified")

    win_year, win_half_year = _stats.get_trading_periods(periods_per_year)

    tpl = ""
    with open(template_path or __file__[:-4] + '.html') as f:
        tpl = f.read()
        f.close()

    # prepare timeseries
    returns = _utils._prepare_returns(returns)

    if benchmark is not None:
        benchmark_title = kwargs.get('benchmark_title', 'Benchmark')
        if kwargs.get('benchmark_title') is None:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, _pd.Series):
                benchmark_title = benchmark.name
            elif isinstance(benchmark, _pd.DataFrame):
                benchmark_title = benchmark[benchmark.columns[0]].name

        tpl = tpl.replace('{{benchmark_title}}', f"Benchmark is {benchmark_title.upper()} | ")
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _stats.match_dates(returns, benchmark)

    date_range = returns.index.strftime('%e %b, %Y')
    tpl = tpl.replace('{{date_range}}', date_range[0] + ' - ' + date_range[-1])
    tpl = tpl.replace('{{title}}', title)
    tpl = tpl.replace('{{v}}', __version__)

    mtrx = _stats.metrics(returns=returns, benchmark=benchmark,
                   rf=rf, display=False, mode='full',
                   sep=True, internal="True",
                   compounded=compounded,
                   periods_per_year=periods_per_year,
                   prepare_returns=False)[2:]

    mtrx.index.name = 'Metric'
    tpl = tpl.replace('{{metrics}}', _html_table(mtrx))
    tpl = tpl.replace('<tr><td></td><td></td><td></td></tr>',
                      '<tr><td colspan="3"><hr></td></tr>')
    tpl = tpl.replace('<tr><td></td><td></td></tr>',
                      '<tr><td colspan="2"><hr></td></tr>')

    if benchmark is not None:
        yoy = _stats.compare(
            returns, benchmark, "A", compounded=compounded,
            prepare_returns=False)
        yoy.columns = ['Benchmark', 'Strategy', 'Multiplier', 'Won']
        yoy.index.name = 'Year'
        tpl = tpl.replace('{{eoy_title}}', '<h3>EOY Returns vs Benchmark</h3>')
        tpl = tpl.replace('{{eoy_table}}', _html_table(yoy))
    else:
        # pct multiplier
        yoy = _pd.DataFrame(
            _utils.group_returns(returns, returns.index.year) * 100)
        yoy.columns = ['Return']
        yoy['Cumulative'] = _utils.group_returns(
            returns, returns.index.year, True)
        yoy['Return'] = yoy['Return'].round(2).astype(str) + '%'
        yoy['Cumulative'] = (yoy['Cumulative'] *
                             100).round(2).astype(str) + '%'
        yoy.index.name = 'Year'
        tpl = tpl.replace('{{eoy_title}}', '<h3>EOY Returns</h3>')
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
                   savefig={'fname': figfile, 'format': figfmt},
                   show=False, ylabel=False, cumulative=compounded,
                   prepare_returns=False)
    tpl = tpl.replace('{{returns}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.log_returns(returns, benchmark, grayscale=grayscale,
                       figsize=(8, 4), subtitle=False,
                       savefig={'fname': figfile, 'format': figfmt},
                       show=False, ylabel=False, cumulative=compounded,
                       prepare_returns=False)
    tpl = tpl.replace('{{log_returns}}', _embed_figure(figfile, figfmt))

    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.returns(returns, benchmark, match_volatility=True,
                       grayscale=grayscale, figsize=(8, 4), subtitle=False,
                       savefig={'fname': figfile, 'format': figfmt},
                       show=False, ylabel=False, cumulative=compounded,
                       prepare_returns=False)
        tpl = tpl.replace('{{vol_returns}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.yearly_returns(returns, benchmark, grayscale=grayscale,
                          figsize=(8, 4), subtitle=False,
                          savefig={'fname': figfile, 'format': figfmt},
                          show=False, ylabel=False, compounded=compounded,
                          prepare_returns=False)
    tpl = tpl.replace('{{eoy_returns}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.histogram(returns, grayscale=grayscale,
                     figsize=(8, 4), subtitle=False,
                     savefig={'fname': figfile, 'format': figfmt},
                     show=False, ylabel=False, compounded=compounded,
                     prepare_returns=False)
    tpl = tpl.replace('{{monthly_dist}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.daily_returns(returns, grayscale=grayscale,
                         figsize=(8, 3), subtitle=False,
                         savefig={'fname': figfile, 'format': figfmt},
                         show=False, ylabel=False,
                         prepare_returns=False)
    tpl = tpl.replace('{{daily_returns}}', _embed_figure(figfile, figfmt))

    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.rolling_beta(returns, benchmark, grayscale=grayscale,
                            figsize=(8, 3), subtitle=False,
                            window1=win_half_year, window2=win_year,
                            savefig={'fname': figfile, 'format': figfmt},
                            show=False, ylabel=False,
                            prepare_returns=False)
        tpl = tpl.replace('{{rolling_beta}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.rolling_volatility(returns, benchmark, grayscale=grayscale,
                              figsize=(8, 3), subtitle=False,
                              savefig={'fname': figfile, 'format': figfmt},
                              show=False, ylabel=False, period=win_half_year,
                              periods_per_year=win_year)
    tpl = tpl.replace('{{rolling_vol}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.rolling_sharpe(returns, grayscale=grayscale,
                          figsize=(8, 3), subtitle=False,
                          savefig={'fname': figfile, 'format': figfmt},
                          show=False, ylabel=False, period=win_half_year,
                          periods_per_year=win_year)
    tpl = tpl.replace('{{rolling_sharpe}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.rolling_sortino(returns, grayscale=grayscale,
                           figsize=(8, 3), subtitle=False,
                           savefig={'fname': figfile, 'format': figfmt},
                           show=False, ylabel=False, period=win_half_year,
                           periods_per_year=win_year)
    tpl = tpl.replace('{{rolling_sortino}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.drawdowns_periods(returns, grayscale=grayscale,
                             figsize=(8, 4), subtitle=False,
                             savefig={'fname': figfile, 'format': figfmt},
                             show=False, ylabel=False, compounded=compounded,
                             prepare_returns=False)
    tpl = tpl.replace('{{dd_periods}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.drawdown(returns, grayscale=grayscale,
                    figsize=(8, 3), subtitle=False,
                    savefig={'fname': figfile, 'format': figfmt},
                    show=False, ylabel=False)
    tpl = tpl.replace('{{dd_plot}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.monthly_heatmap(returns, grayscale=grayscale,
                           figsize=(8, 4), cbar=False,
                           savefig={'fname': figfile, 'format': figfmt},
                           show=False, ylabel=False, compounded=compounded)
    tpl = tpl.replace('{{monthly_heatmap}}', _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.distribution(returns, grayscale=grayscale,
                        figsize=(8, 4), subtitle=False,
                        savefig={'fname': figfile, 'format': figfmt},
                        show=False, ylabel=False, compounded=compounded,
                        prepare_returns=False)
    tpl = tpl.replace('{{returns_dist}}', _embed_figure(figfile, figfmt))

    tpl = _regex.sub(r'\{\{(.*?)\}\}', '', tpl)
    tpl = tpl.replace('white-space:pre;', '')

    if output is None:
        # _open_html(tpl)
        _download_html(tpl, download_filename)
        return

    with open(download_filename, 'w', encoding='utf-8') as f:
        f.write(tpl)


def full(returns, benchmark=None, rf=0., grayscale=False,
         figsize=(8, 5), display=True, compounded=True,
         periods_per_year=252, match_dates=False):

    # prepare timeseries
    returns = _utils._prepare_returns(returns)
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _stats.match_dates(returns, benchmark)

    dd = _stats.to_drawdown_series(returns)
    col = _stats.drawdown_details(dd).columns[4]
    dd_info = _stats.drawdown_details(dd).sort_values(by = col,
                                                       ascending = True)[:5]

    if not dd_info.empty:
        dd_info.index = range(1, min(6, len(dd_info)+1))
        dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)

    if _utils._in_notebook():
        iDisplay(iHTML('<h4>Performance Metrics</h4>'))
        iDisplay(_stats.metrics(returns=returns, benchmark=benchmark,
                         rf=rf, display=display, mode='full',
                         compounded=compounded,
                         periods_per_year=periods_per_year,
                         prepare_returns=False))
        iDisplay(iHTML('<h4>5 Worst Drawdowns</h4>'))
        if dd_info.empty:
            iDisplay(iHTML("<p>(no drawdowns)</p>"))
        else:
            iDisplay(dd_info)

        iDisplay(iHTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        _stats.metrics(returns=returns, benchmark=benchmark,
                rf=rf, display=display, mode='full',
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False)
        print('\n\n')
        print('[5 Worst Drawdowns]\n')
        if dd_info.empty:
            print("(no drawdowns)")
        else:
            print(_tabulate(dd_info, headers="keys",
                            tablefmt='simple', floatfmt=".2f"))
        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    plots(returns=returns, benchmark=benchmark,
          grayscale=grayscale, figsize=figsize, mode='full',
          periods_per_year=periods_per_year, prepare_returns=False)


def basic(returns, benchmark=None, rf=0., grayscale=False,
          figsize=(8, 5), display=True, compounded=True,
          periods_per_year=252, match_dates=False):

    # prepare timeseries
    returns = _utils._prepare_returns(returns)
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _stats.match_dates(returns, benchmark)

    if _utils._in_notebook():
        iDisplay(iHTML('<h4>Performance Metrics</h4>'))
        _stats.metrics(returns=returns, benchmark=benchmark,
                rf=rf, display=display, mode='basic',
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False)
        iDisplay(iHTML('<h4>Strategy Visualization</h4>'))
    else:
        print('[Performance Metrics]\n')
        _stats.metrics(returns=returns, benchmark=benchmark,
                rf=rf, display=display, mode='basic',
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False)

        print('\n\n')
        print('[Strategy Visualization]\nvia Matplotlib')

    plots(returns=returns, benchmark=benchmark,
          grayscale=grayscale, figsize=figsize, mode='basic',
          periods_per_year=periods_per_year,
          prepare_returns=False)


def plots(returns, benchmark=None, grayscale=False,
          figsize=(8, 5), mode='basic', compounded=True,
          periods_per_year=252, prepare_returns=True, match_dates=False):

    win_year, win_half_year = _stats.get_trading_periods(periods_per_year)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    if mode.lower() != 'full':
        _plots.snapshot(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]),
                        show=True, mode=("comp" if compounded else "sum"))

        _plots.monthly_heatmap(returns, grayscale=grayscale,
                               figsize=(figsize[0], figsize[0]*.5),
                               show=True, ylabel=False,
                               compounded=compounded)

        return

    # prepare timeseries
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        if match_dates is True:
            returns, benchmark = _stats.match_dates(returns, benchmark)

    _plots.returns(returns, benchmark, grayscale=grayscale,
                   figsize=(figsize[0], figsize[0]*.6),
                   show=True, ylabel=False,
                   prepare_returns=False)

    _plots.log_returns(returns, benchmark, grayscale=grayscale,
                       figsize=(figsize[0], figsize[0]*.5),
                       show=True, ylabel=False,
                       prepare_returns=False)

    if benchmark is not None:
        _plots.returns(returns, benchmark, match_volatility=True,
                       grayscale=grayscale,
                       figsize=(figsize[0], figsize[0]*.5),
                       show=True, ylabel=False,
                       prepare_returns=False)

    _plots.yearly_returns(returns, benchmark,
                          grayscale=grayscale,
                          figsize=(figsize[0], figsize[0]*.5),
                          show=True, ylabel=False,
                          prepare_returns=False)

    _plots.histogram(returns, grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.5),
                     show=True, ylabel=False,
                     prepare_returns=False)

    _plots.daily_returns(returns, grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.3),
                         show=True, ylabel=False,
                         prepare_returns=False)

    if benchmark is not None:
        _plots.rolling_beta(returns, benchmark, grayscale=grayscale,
                            window1=win_half_year, window2=win_year,
                            figsize=(figsize[0], figsize[0]*.3),
                            show=True, ylabel=False,
                            prepare_returns=False)

    _plots.rolling_volatility(
        returns, benchmark, grayscale=grayscale,
        figsize=(figsize[0], figsize[0]*.3), show=True, ylabel=False,
        period=win_half_year)

    _plots.rolling_sharpe(returns, grayscale=grayscale,
                          figsize=(figsize[0], figsize[0]*.3),
                          show=True, ylabel=False, period=win_half_year)

    _plots.rolling_sortino(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.3),
                           show=True, ylabel=False, period=win_half_year)

    _plots.drawdowns_periods(returns, grayscale=grayscale,
                             figsize=(figsize[0], figsize[0]*.5),
                             show=True, ylabel=False,
                             prepare_returns=False)

    _plots.drawdown(returns, grayscale=grayscale,
                    figsize=(figsize[0], figsize[0]*.4),
                    show=True, ylabel=False)

    _plots.monthly_heatmap(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.5),
                           show=True, ylabel=False)

    _plots.distribution(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.5),
                        show=True, ylabel=False,
                        prepare_returns=False)


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


def _embed_figure(figfile, figfmt):
    figbytes = figfile.getvalue()
    if figfmt == 'svg':
        return figbytes.decode()
    data_uri = _b64encode(figbytes).decode()
    return '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)


def metrics(*args, **kwargs):
    return _stats.metrics(*args, **kwargs)

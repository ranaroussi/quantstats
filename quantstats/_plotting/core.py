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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from matplotlib.ticker import FormatStrFormatter, FuncFormatter, StrMethodFormatter
from matplotlib.figure import Figure
from typing import List, Union, Tuple

from .. import stats, utils
from .themes import defaults

try:
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    pass


class PlottingTheme:
    def __init__(
        self,
        theme: Union[str, dict] = 'default',
        fontname: str = 'Arial',
        font_scale: float = 1.1,
    ) -> None:
        if isinstance(theme, str):
            if theme in defaults:
                theme = defaults.get(theme)
            else:
                msg = 'specified theme not found, falling back to default'
                warnings.warn(msg)

                theme = defaults.get('default')

        required_keys = (
            'table_cmap',
            'colors',
            'histogram_colors',
            'highlight_color',
            'table_colors',
            'cell_color',
            'hlinecolor',
            'hlinecolor_0',
            'alpha',
            'ts_alpha',
            'title_text_color',
            'subtitle_text_color',
            'facecolor',
            'rc',
        )
        if any(set(required_keys).difference(theme)):
            msg = 'malformed theme specification, falling back to default'
            warnings.warn(msg)

            theme = defaults.get('theme')

        for key, val in theme.items():
            if key == 'rc':
                # use instance to affect global seaborn settings
                sns.set(
                    font_scale=font_scale,
                    rc=val,
                )
            else:
                setattr(self, key, val)

        self.fontname = fontname

    @property
    def cmap(self):
        if isinstance(self.table_cmap, dict):
            return sns.diverging_palette(
                h_neg=self.table_cmap['h_neg'],
                h_pos=self.table_cmap['h_pos'],
                s=self.table_cmap['s'],
                l=self.table_cmap['l'],
                center=self.table_cmap['center'],
                as_cmap=True,
            )

        return self.table_cmap


class PlottingWrappers(PlottingTheme):
    def __init__(
        self,
        theme: Union[str, dict] = 'default',
        fontname: str = 'Arial',
        font_scale: float = 1.1,
        show: bool = False,
        export: bool = False,
        **savefig_kwargs,
    ) -> None:
        super().__init__(
            theme=theme,
            fontname=fontname,
            font_scale=font_scale,
        )
        self.show = show
        self.export = export
        self.savefig_kwargs = savefig_kwargs

    def plot_returns_bars(
        self,
        returns: pd.Series,
        benchmark: Union[pd.Series, None] = None,
        returns_label: str = 'Strategy',
        hline: Union[float, None] = None,
        hlinewidth: float = 1,
        hlinelabel: Union[str, None] = None,
        resample: str = 'A',
        title: str = 'Returns',
        match_volatility: bool = False,
        log_scale: bool = False,
        figsize: Tuple[float] = (10, 6),
        ylabel: bool = True,
        subtitle: bool = True,
        image_file_name: str = 'returns_bars.png',
    ) -> Union[Figure, None]:
        if match_volatility:
            if isinstance(benchmark, pd.Series):
                bmark_vol = benchmark.loc[returns.index].std()
                returns = (returns / returns.std()) * bmark_vol

            else:
                msg = 'match_volatility requires passing of benchmark.'
                raise ValueError(msg)

        df = pd.DataFrame(
            data={returns_label: returns},
            index=returns.index,
        )
        if isinstance(benchmark, pd.Series):
            df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
            df = df[['Benchmark', returns_label]]

        df = df.dropna()

        if resample is not None:
            df = (
                df
                .resample(resample)
                .apply(stats.comp)
                .resample(resample)
                .last()
            )

        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        # use a more precise date string for the x axis locations in the toolbar
        fig.suptitle(
            f'{title}\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )

        if subtitle:
            dmin = df.index.min().strftime('%Y')
            dmax = df.index.max().strftime('%Y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        if 'Benchmark' in df:
            pcolors = [
                self.colors['benchmark'],
                self.colors['returns'],
            ]
        else:
            pcolors = self.colors['returns']

        df.plot(
            kind='bar',
            ax=ax,
            color=pcolors,
            edgecolor='none',
        )

        try:
            ax.set_xticklabels(df.index.year)
            years = list(set(df.index.year))

        except AttributeError:
            ax.set_xticklabels(df.index)
            years = list(set(df.index))

        if len(years) > 10:
            mod = len(years) // 10
            (
                plt
                .xticks(
                    np.arange(len(years)),
                    [f'{year}' if not i % mod else '' for i, year in enumerate(years)],
                )
            )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        if hline:
            (
                ax.axhline(
                    hline,
                    linestyle='--',
                    linewidth=hlinewidth,
                    color=self.hlinecolor,
                    label=hlinelabel,
                    zorder=2,
                )
            )

        ax.axhline(
            0,
            linestyle='--',
            linewidth=1,
            color=self.hlinecolor_0,
            zorder=2,
        )

        if isinstance(benchmark, pd.Series) or hline:
            ax.legend(fontsize=12)

        plt.yscale('symlog' if log_scale else 'linear')

        ax.set_xlabel('')
        if ylabel:
            ax.set_ylabel(
                'Returns',
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout()

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_timeseries(
        self,
        returns: pd.Series,
        benchmark: Union[pd.Series, None] = None,
        title: str = 'Returns',
        compound: bool = False,
        cumulative: bool = True,
        fill: bool = False,
        returns_label: str = 'Strategy',
        hline: Union[float, None] = None,
        hlinewidth: float = 1,
        hlinelabel: Union[str, None] = None,
        percent: bool = True,
        match_volatility: bool = False,
        log_scale: bool = False,
        resample: Union[str, None] = None,
        linewidth: float = 1.5,
        figsize: Tuple[float] = (10, 6),
        ylabel: Union[str, None] = None,
        subtitle: bool = True,
        image_file_name: str = 'timeseries.png',
    ) -> Union[Figure, None]:
        returns = returns.fillna(0)

        if isinstance(benchmark, pd.Series):
            benchmark = benchmark.fillna(0)

        if match_volatility:
            if isinstance(benchmark, pd.Series):
                bmark_vol = benchmark.std()
                returns = (returns / returns.std()) * bmark_vol

            else:
                msg = 'match_volatility requires passing of benchmark.'
                raise ValueError(msg)

        if compound:
            if cumulative:
                returns = stats.compsum(returns)
                if isinstance(benchmark, pd.Series):
                    benchmark = stats.compsum(benchmark)

            else:
                returns = returns.cumsum()
                if isinstance(benchmark, pd.Series):
                    benchmark = benchmark.cumsum()

        if resample:
            returns = returns.resample(resample)
            returns = (
                returns.last() if compound else returns.sum()
            )

            if isinstance(benchmark, pd.Series):
                benchmark = benchmark.resample(resample)
                benchmark = (
                    benchmark.last() if compound else benchmark.sum()
                )

        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        # use a more precise date string for the x axis locations in the toolbar
        fig.suptitle(
            f'{title}\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )

        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        if isinstance(benchmark, pd.Series):
            ax.plot(
                benchmark,
                linewidth=linewidth,
                label='Benchmark',
                color=self.colors['benchmark'],
            )

        ax.plot(
            returns,
            linewidth=linewidth,
            label=returns_label,
            color=self.colors['returns'],
            alpha=self.ts_alpha,
        )

        if fill:
            ax.fill_between(
                returns.index,
                0,
                returns,
                color=self.colors['returns'],
                alpha=.25,
            )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        if hline:
            (
                ax.axhline(
                    hline,
                    linestyle='--',
                    linewidth=hlinewidth,
                    color=self.hlinecolor,
                    label=hlinelabel,
                    zorder=2,
                )
            )

        ax.axhline(
            0,
            linestyle='--',
            linewidth=1,
            color=self.hlinecolor_0,
            zorder=2,
        )

        if isinstance(benchmark, pd.Series) or hline:
            ax.legend(fontsize=12)

        plt.yscale('symlog' if log_scale else 'linear')

        ax.set_xlabel('')
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        if percent:
            ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout()

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_histogram(
        self,
        returns: pd.Series,
        resample: str = 'M',
        bins: int = 20,
        title: str = 'Returns',
        kde: bool = True,
        figsize: Tuple[float] = (10, 6),
        ylabel: bool = True,
        subtitle: bool = True,
        compounded: bool = True,
        image_file_name: str = 'histogram.png',
    ) -> Union[Figure, None]:
        fn = stats.comp if compounded else np.sum
        returns = (
            returns
            .fillna(0)
            .resample(resample)
            .apply(fn)
            .resample(resample)
            .last()
        )

        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        fig.suptitle(
            f'{title}\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )

        if subtitle:
            dmin = returns.index.min().strftime('%Y')
            dmax = returns.index.max().strftime('%Y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )
        ax.axvline(
            returns.mean(),
            linestyle='--',
            linewidth=1.5,
            color=self.histogram_colors['avg_color'],
            zorder=2,
            label='Average',
        )
        sns.histplot(
            returns,
            bins=bins,
            color=self.histogram_colors['bar_color'],
            edgecolor=self.facecolor,
            alpha=1,
            kde=kde,
            stat='density',
            ax=ax,
        )
        sns.kdeplot(
            returns,
            color=self.histogram_colors['kde_color'],
            linewidth=1.5,
        )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, loc: '{:,}%'.format(int(x*100))))

        ax.axvline(
            0,
            linewidth=1,
            color=self.hlinecolor_0,
            zorder=2,
        )
        ax.set_xlabel('')
        if ylabel:
            ax.set_ylabel(
                'Occurrences',
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        ax.legend(fontsize=12)

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout()

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_rollingstats(
        self,
        returns: pd.Series,
        benchmark: Union[pd.Series, None] = None,
        title: str = None,
        returns_label: str = 'Strategy',
        linewidth: float = 1,
        hline: Union[float, None] = None,
        hlinelabel: Union[str, None] = None,
        hlinewidth: float = 1.5,
        figsize: Tuple[float] = (10, 6),
        ylabel: Union[str, None] = None,
        subtitle: bool = True,
        image_file_name: str = 'rollingstats.png',
    ) -> Union[Figure, None]:
        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        df = pd.DataFrame(
            data={returns_label: returns},
            index=returns.index,
        )
        if isinstance(benchmark, pd.Series):
            df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
            df = df[['Benchmark', returns_label]].dropna()
            ax.plot(
                df['Benchmark'],
                linewidth=linewidth,
                label='Benchmark',
                color=self.colors['benchmark'],
                alpha=self.alpha,
            )

        ax.plot(
            df[returns_label].dropna(),
            linewidth=linewidth,
            label=returns_label,
            color=self.colors['returns'],
        )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        fig.suptitle(
            f'{title}\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )

        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        if hline:
            ax.axhline(
                hline,
                linestyle='--',
                linewidth=hlinewidth,
                color=self.hlinecolor,
                label=hlinelabel,
                zorder=2,
            )

        ax.axhline(
            0,
            linestyle='--',
            linewidth=1,
            color=self.hlinecolor_0,
            zorder=2,
        )

        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.legend(fontsize=12)

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout()

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_rolling_beta(
        self,
        returns: pd.Series,
        benchmark: pd.Series,
        window1: int = 126,
        window1_label: Union[str, None] = None,
        window2: Union[int, None] = None,
        window2_label: Union[str, None] = None,
        title: Union[str, None] = None,
        figsize: Tuple[float] = (10, 6),
        linewidth: float = 1.5,
        ylabel: bool = True,
        subtitle: bool = True,
        image_file_name: str = 'rolling_beta.png',
    ) -> Union[Figure, None]:
        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        fig.suptitle(
            f'{title}\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )

        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        beta = (
            stats
            .rolling_greeks(
                returns,
                benchmark,
                window1,
            )['beta']
        )
        ax.plot(
            beta,
            linewidth=linewidth,
            label=window1_label,
            color=self.colors['returns'],
        )

        if window2:
            beta2 = (
                stats
                .rolling_greeks(
                    returns,
                    benchmark,
                    window2,
                )['beta']
            )
            ax.plot(
                beta2,
                linewidth=linewidth,
                label=window2_label,
                color=self.colors['extra_1'],
            )
        mmin = min((-100, int(beta.min() * 100)))
        mmax = max((100, int(beta.max() * 100)))
        step = 50 if (mmax-mmin) >= 200 else 100
        ax.set_yticks([x / 100 for x in range(mmin, mmax, step)])

        ax.axhline(
            beta.mean(),
            linestyle='--',
            linewidth=1.5,
            color=self.hlinecolor,
            zorder=2,
        )

        ax.axhline(
            0,
            linestyle='--',
            linewidth=1,
            color=self.hlinecolor_0,
            zorder=2,
        )

        fig.autofmt_xdate()

        # use a more precise date string for the x axis locations in the toolbar
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        if ylabel:
            ax.set_ylabel(
                'Beta',
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        ax.legend(fontsize=12)

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout()

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_longest_drawdowns(
        self,
        returns: pd.Series,
        periods: int = 5,
        linewidth: float = 1.5,
        log_scale: bool = False,
        figsize: Tuple[float] = (10, 6),
        ylabel: bool = True,
        subtitle: bool = True,
        compounded: bool = True,
        image_file_name: str = 'longest_drawdowns.png',
    ) -> Union[Figure, None]:
        dd = stats.to_drawdown_series(returns.fillna(0))
        dddf = stats.drawdown_details(dd)
        longest_dd = (
            dddf
            .sort_values(
                by='days',
                ascending=False,
                kind='mergesort',
            )[:periods]
        )

        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        fig.suptitle(
            f'Worst {periods:.0f} Drawdown Periods\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )
        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        series = stats.compsum(returns) if compounded else returns.cumsum()
        ax.plot(
            series,
            linewidth=linewidth,
            label='Backtest',
            color=self.colors['returns'],
        )

        for _, row in longest_dd.iterrows():
            (
                ax.axvspan(
                    *mdates.datestr2num(
                        [
                            str(row['start']),
                            str(row['end']),
                        ]
                    ),
                    color=self.highlight_color,
                    alpha=.1,
                )
            )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        # use a more precise date string for the x axis locations in the toolbar
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        ax.axhline(
            0,
            linestyle='--',
            linewidth=1,
            color=self.hlinecolor_0,
            zorder=2,
        )
        plt.yscale('symlog' if log_scale else 'linear')
        if ylabel:
            ax.set_ylabel(
                'Cumulative Returns',
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))
        fig.autofmt_xdate()

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout()

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_distribution(
        self,
        returns: pd.Series,
        figsize: Tuple[float] = (10, 6),
        ylabel: bool = True,
        compounded: bool = True,
        subtitle: bool = True,
        image_file_name: str = 'distribution.png',
    ) -> Union[Figure, None]:
        port = pd.DataFrame(
            returns.fillna(0).values,
            columns=['Daily'],
            index=returns.index,
        )

        fn = stats.comp if compounded else np.sum
        newcols = (
            'Weekly',
            'Monthly',
            'Quarterly',
            'Yearly',
        )
        resamples = (
            'W-MON',
            'M',
            'Q',
            'A',
        )
        for col, rsmp in zip(newcols, resamples):
            port[col] = (
                port['Daily']
                .resample(rsmp)
                .apply(fn)
                .ffill()
            )

        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        fig.suptitle(
            'Return Quantiles\n',
            y=.99,
            fontweight='bold',
            fontname=self.fontname,
            fontsize=14,
            color=self.title_text_color,
        )
        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        box_colors = (
            'benchmark',
            'returns',
            'drawdown',
            'extra_0',
            'extra_1',
        )
        sns.boxplot(
            data=port,
            ax=ax,
            palette=tuple((self.colors[e] for e in box_colors)),
        )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, loc: '{:,}%'.format(int(x*100))))

        if ylabel:
            ax.set_ylabel(
                'Returns',
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        fig.autofmt_xdate()

        try:
            plt.subplots_adjust(
                hspace=0,
            )
        except Exception:
            pass

        try:
            fig.tight_layout(
                w_pad=0,
                h_pad=0,
            )

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_table(
        self,
        tbl: pd.DataFrame,
        columns: Union[List[str], None] = None,
        title: Union[str, None] = None,
        title_loc: str = 'left',
        header: bool = True,
        col_widths=None,
        row_loc: str = 'right',
        col_loc: str = 'right',
        col_labels=None,
        edges: str = 'horizontal',
        orient: str = 'horizontal',
        figsize: Tuple[float] = (5.5, 6),
        image_file_name: str = 'table.png',
    ) -> Union[Figure, None]:
        if isinstance(columns, list):
            try:
                tbl.columns = columns
            except Exception:
                pass

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, frame_on=False)

        if title:
            ax.set_title(
                title,
                fontweight='bold',
                fontsize=14,
                color=self.title_text_color,
                loc=title_loc,
            )

        the_table = ax.table(
            cellText=tbl.values,
            col_widths=col_widths,
            row_loc=row_loc,
            col_loc=col_loc,
            edges=edges,
            col_labels=tbl.columns if header else col_labels,
            loc='center',
            zorder=2,
        )

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.scale(1, 1)

        for (row, col), cell in the_table.get_celld().items():
            cell.set_height(0.08)
            cell.set_text_props(color=self.title_text_color)
            cell.set_edgecolor(self.table_colors['inner_edge'])
            if row == 0 and header:
                cell.set_edgecolor(self.table_colors['outer_edge'])
                cell.set_facecolor(self.table_colors['face'])
                cell.set_linewidth(2)
                cell.set_text_props(
                    weight='bold',
                    color=self.title_text_color,
                )
            elif col == 0 and 'vertical' in orient:
                cell.set_edgecolor(self.table_colors['inner_edge'])
                cell.set_linewidth(1)
                cell.set_text_props(
                    weight='bold',
                    color=self.title_text_color,
                )
            elif row > 1:
                cell.set_linewidth(1)

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        try:
            plt.subplots_adjust(
                hspace=0,
            )
        except Exception:
            pass

        try:
            fig.tight_layout(
                w_pad=0,
                h_pad=0,
            )

        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        annot_size: float = 10,
        figsize: Tuple[float] = (10, 5),
        cbar: bool = True,
        square: bool = False,
        ylabel=True,
        image_file_name: str = 'monthly_heatmap.png'
    ) -> Union[Figure, None]:
        fig, ax = plt.subplots(figsize=figsize)
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        ax.set_title(
            f'{"Monthly Returns (%)":>25}\n',
            fontsize=14,
            y=.995,
            fontname=self.fontname,
            fontweight='bold',
            color=self.title_text_color,
        )
        sns.heatmap(
            returns,
            annot=True,
            center=0,
            annot_kws={'size': annot_size},
            fmt='0.2f',
            linewidths=0.5,
            linecolor=self.cell_color,
            square=square,
            cbar=cbar,
            cmap=self.cmap,
            cbar_kws={'format': '%.0f%%'},
            ax=ax,
        )

        # align plot to match other
        if ylabel:
            ax.set_ylabel(
                'Years',
                fontname=self.fontname,
                fontweight='bold',
                fontsize=12,
                color=self.title_text_color,
            )
            ax.yaxis.set_label_coords(-.1, .5)

        plt.xticks(rotation=0, fontsize=annot_size*1.2)
        plt.yticks(rotation=0, fontsize=annot_size*1.2)

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout(
                w_pad=0,
                h_pad=0,
            )
        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_earnings(
        self,
        returns: pd.Series,
        figsize: Tuple[float] = (10, 6),
        title: str = 'Portfolio Earnings',
        linewidth: float = 1.5,
        subtitle: bool = True,
        image_file_name: str = 'earnings.png',
    ) -> Union[Figure, None]:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=figsize,
        )
        for pos in ('top', 'right', 'bottom', 'left'):
            ax.spines[pos].set_visible(False)

        fig.suptitle(
            title,
            fontsize=14,
            y=.995,
            fontname=self.fontname,
            fontweight='bold',
            color=self.title_text_color,
        )
        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            pnl_c = utils._score_str(
                f'${returns.values[-1] - returns.values[0]:,.2f}',
            )
            pnl_p = utils._score_str(
                f'{(returns.values[-1] / returns.values[0] - 1) * 100:,.2f}%',
            )
            (
                ax
                .set_title(
                    f'\n{dmin} - {dmax}; P&L: {pnl_c} ({pnl_p})',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        returns_max = np.where(
            returns == returns.max(),
            returns.max(),
            np.nan,
        )

        ax.plot(
            returns.index,
            returns_max,
            marker='o',
            linewidth=0,
            alpha=self.alpha,
            markersize=12,
            color=self.colors['benchmark'],
        )
        ax.plot(
            returns.index,
            returns,
            linewidth=linewidth,
            color=self.colors['returns'],
        )
        ax.set_ylabel(
            'Value of  ${start_balance:,.0f}',
            fontname=self.fontname,
            fontweight='bold',
            fontsize=12,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(format_cur_axis))
        ax.yaxis.set_label_coords(-.1, .5)

        fig.autofmt_xdate()

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout(
                w_pad=0,
                h_pad=0,
            )
        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig

    def plot_snapshot(
        self,
        returns: pd.Series,
        figsize: Tuple[float] = (10, 8),
        title: str = 'Portfolio Summary',
        linewidth: float = 1.5,
        mode: str = 'comp',
        subtitle: bool = True,
        log_scale: bool = False,
        image_file_name: str = 'snapshot.png',
    ) -> Union[Figure, None]:
        returns = (
            utils
            .make_portfolio(
                returns,
                1,
                mode,
            )
            .pct_change()
            .fillna(0)
        )

        fig, ax = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={
                'height_ratios': [3, 1, 1],
            },
        )

        fig.suptitle(
            title,
            fontsize=14,
            y=.995,
            fontname=self.fontname,
            fontweight='bold',
            color=self.title_text_color,
        )

        if subtitle:
            dmin = returns.index.min().strftime('%e %b \'%y')
            dmax = returns.index.max().strftime('%e %b \'%y')
            sharpe = stats.sharpe(returns)
            (
                ax[0]
                .set_title(
                    f'\n{dmin} - {dmax}; Sharpe: {sharpe:.2f}',
                    fontsize=12,
                    color=self.subtitle_text_color,
                )
            )

        ax[0].set_ylabel(
            'Cumulative Return',
            fontname=self.fontname,
            fontweight='bold',
            fontsize=12,
            color=self.title_text_color,
        )
        ax[0].plot(
            stats.compsum(returns) * 100,
            color=self.colors['returns'],
            linewidth=linewidth,
            zorder=1,
        )
        ax[0].axhline(
            0,
            color=self.hlinecolor_0,
            linewidth=1,
            zorder=0,
        )
        ax[0].set_yscale('symlog' if log_scale else 'linear')

        dd = stats.to_drawdown_series(returns) * 100
        ddmin = utils._round_to_closest(abs(dd.min()), 5)
        ddmin_ticks = 5
        if ddmin > 50:
            ddmin_ticks = ddmin / 4
        elif ddmin > 20:
            ddmin_ticks = ddmin / 3
        ddmin_ticks = int(utils._round_to_closest(ddmin_ticks, 5))

        ax[1].set_ylabel(
            'Drawdown',
            fontname=self.fontname,
            fontweight='bold',
            fontsize=12,
        )
        ax[1].plot(
            dd,
            color=self.colors['drawdown'],
            linewidth=linewidth,
            zorder=1,
        )
        ax[1].axhline(
            0,
            color=self.hlinecolor_0,
            linewidth=1,
            zorder=0,
        )
        ax[1].fill_between(
            dd.index,
            0,
            dd,
            color=self.colors['drawdown'],
            alpha=.1,
        )
        ax[1].set_yticks(np.arange(-ddmin, 0, step=ddmin_ticks))
        ax[1].set_yscale('symlog' if log_scale else 'linear')

        ax[2].set_ylabel(
            'Daily Return',
            fontname=self.fontname,
            fontweight='bold',
            fontsize=12,
            color=self.title_text_color,
        )
        ax[2].plot(
            returns * 100,
            color=self.colors['benchmark'],
            linewidth=linewidth / 2,
            zorder=1,
        )
        ax[2].axhline(
            0,
            color=self.hlinecolor_0,
            linestyle='--',
            linewidth=1,
            zorder=2,
        )
        ax[2].set_yscale('symlog' if log_scale else 'linear')

        retmax = utils._round_to_closest(returns.max() * 100, 5)
        retmin = utils._round_to_closest(returns.min() * 100, 5)
        retdiff = (retmax - retmin)
        steps = 5
        if retdiff > 50:
            steps = retdiff / 5
        elif retdiff > 30:
            steps = retdiff / 4
        steps = int(utils._round_to_closest(steps, 5))

        ax[2].set_yticks(np.arange(retmin, retmax, step=steps))

        for elem in ax:
            elem.yaxis.set_label_coords(-.1, .5)
            elem.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}%'))
            for pos in ('top', 'right', 'bottom', 'left'):
                elem.spines[pos].set_visible(False)

        fig.autofmt_xdate()

        try:
            plt.subplots_adjust(
                hspace=0,
                bottom=0,
                top=1,
            )
        except Exception:
            pass

        try:
            fig.tight_layout(
                w_pad=0,
                h_pad=0,
            )
        except Exception:
            pass

        if self.export:
            plt.savefig(image_file_name, **self.savefig_kwargs)

        if self.show:
            plt.show(block=False)

            return None

        plt.close()

        return fig


def format_cur_axis(x, _):
    if x >= 1e12:
        res = '$%1.1fT' % (x * 1e-12)
        return res.replace('.0T', 'T')
    if x >= 1e9:
        res = '$%1.1fB' % (x * 1e-9)
        return res.replace('.0B', 'B')
    if x >= 1e6:
        res = '$%1.1fM' % (x * 1e-6)
        return res.replace('.0M', 'M')
    if x >= 1e3:
        res = '$%1.0fK' % (x * 1e-3)
        return res.replace('.0K', 'K')
    res = '$%1.0f' % x
    return res.replace('.0', '')


def format_pct_axis(x, _):
    x *= 100
    if x >= 1e12:
        res = '%1.1fT%%' % (x * 1e-12)
        return res.replace('.0T%', 'T%')
    if x >= 1e9:
        res = '%1.1fB%%' % (x * 1e-9)
        return res.replace('.0B%', 'B%')
    if x >= 1e6:
        res = '%1.1fM%%' % (x * 1e-6)
        return res.replace('.0M%', 'M%')
    if x >= 1e3:
        res = '%1.1fK%%' % (x * 1e-3)
        return res.replace('.0K%', 'K%')
    res = '%1.0f%%' % x
    return res.replace('.0%', '%')

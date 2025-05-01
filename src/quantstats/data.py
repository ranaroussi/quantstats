import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import stats as _stats
from . import utils as _utils
from .core import (
    plot_distribution,
    plot_histogram,
    plot_longest_drawdowns,
    plot_rolling_beta,
    plot_rolling_stats,
    plot_timeseries,
)


@dataclasses.dataclass
class _Data:
    returns: pd.DataFrame
    benchmark: pd.Series | None = None

    def all(self) -> pd.DataFrame:
        r = self.returns.copy()  # Copy to avoid mutating the original returns

        # Only add 'Benchmark' column if benchmark data is available
        if self.benchmark is not None:
            r["Benchmark"] = self.benchmark

        # Return the combined DataFrame
        return r.fillna(0.0)

    @property
    def index(self):
        return self.returns.index

    @property
    def names(self):
        return list(self.returns.columns)

    def apply(self, fct, **kwargs):
        return fct(self.returns, **kwargs)

    def prices(self, compounded=False):
        if compounded:
            return self.all().fillna(0.0).add(1).cumprod(axis=0) - 1
        else:
            return self.all().fillna(0.0).cumsum()

    def distribution(self, fontname="Arial", compounded=True, title=None):
        return plot_distribution(
            self.returns,
            fontname=fontname,
            title=title,
            compounded=compounded,
        )

    def histogram(
        self,
        resample="ME",
        fontname="Arial",
        subtitle=True,
        compounded=True,
    ):
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

        return plot_histogram(
            self.returns,
            resample=resample,
            fontname=fontname,
            title="Distribution of %sReturns" % title,
            subtitle=subtitle,
            compounded=compounded,
        )

    def plot(self, compounded=False):
        return plot_timeseries(self.prices(compounded=compounded), title="Prices")

    def return_bars(self, resample="YE", compounded=False):
        frame = self.all().fillna(0.0)

        if compounded:
            frame = frame.resample(resample).apply(_stats.comp)
        else:
            frame = frame.resample(resample).sum()
        return frame

    def yearly_returns(self, compounded=False):
        return self.return_bars(resample="YE", compounded=compounded)

    def monthly_heatmap(
        self,
        annot_size=13,
        cbar=True,
        returns_label="Strategy",
        compounded=True,
        eoy=False,
        fontname="Arial",
        ylabel=True,
    ):
        cmap = "RdYlGn"

        # Prepare returns: (monthly returns as percentage)
        returns = _stats.monthly_returns(self.returns, eoy=eoy, compounded=compounded) * 100

        zmin = -max(abs(returns.min().min()), abs(returns.max().max()))
        zmax = max(abs(returns.min().min()), abs(returns.max().max()))

        # Make heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=returns.values,
                x=[col for col in returns.columns],  # e.g., Jan, Feb, etc.
                y=returns.index.astype(str),  # Years
                text=np.round(returns.values, 2),
                texttemplate="%{text:.2f}%",  # Annotate inside cells
                colorscale=cmap,
                zmid=0,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title="Return (%)",
                    # titleside="right",
                    ticksuffix="%",
                    tickfont=dict(size=annot_size),
                    # titlefont=dict(size=annot_size + 2),
                )
                if cbar
                else None,
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%",
            )
        )

        # Layout
        fig.update_layout(
            title={
                "text": f"{returns_label} - Monthly Returns (%)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(family=fontname, size=16, color="black"),
            },
            xaxis=dict(
                title="",
                side="top",
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            yaxis=dict(
                title="Years" if ylabel else "",
                autorange="reversed",
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=80, b=0),
        )

        return fig

    def rolling_sortino(
        self,
        period_label="6-Months",
        lw=1.25,
        fontname="Arial",
        ylabel="Sortino",
        subtitle=True,
    ):
        res = _stats.rolling_sortino(self.returns)

        return plot_rolling_stats(
            res, ylabel=ylabel, title="Rolling Sortino (%s)" % period_label, fontname=fontname, lw=lw, subtitle=subtitle
        )

    def rolling_sharpe(
        self,
        period_label="6-Months",
        lw=1.25,
        fontname="Arial",
        ylabel="Sharpe Ratio",
        subtitle=True,
    ):
        res = _stats.rolling_sharpe(self.returns)

        return plot_rolling_stats(
            res, ylabel=ylabel, title="Rolling Sharpe (%s)" % period_label, fontname=fontname, lw=lw, subtitle=subtitle
        )

    def rolling_volatility(
        self,
        period=126,
        period_label="6-Months",
        periods_per_year=252,
        lw=1.5,
        fontname="Arial",
        ylabel="Volatility",
        subtitle=True,
    ):
        res = _stats.rolling_volatility(self.all(), period, periods_per_year)

        return plot_rolling_stats(
            res,
            ylabel=ylabel,
            title="Rolling Volatility (%s)" % period_label,
            fontname=fontname,
            lw=lw,
            subtitle=subtitle,
        )

    def daily_returns(
        self,
        fontname="Arial",
        lw=0.5,
        log_scale=False,
        ylabel="Returns",
        subtitle=True,
    ):
        plot_title = "Daily Returns"

        return plot_timeseries(
            self.returns,
            None,
            plot_title,
            ylabel=ylabel,
            log_scale=log_scale,
            lw=lw,
            fontname=fontname,
            subtitle=subtitle,
        )

    def snapshot_plotly(self, title="Portfolio Summary", mode="comp", log_scale=False):
        returns = _utils.make_portfolio(self.returns.dropna(), 1, mode).pct_change().fillna(0)
        dd = _stats.to_drawdown_series(returns)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.03,
            subplot_titles=["Cumulative Return", "Drawdown", "Daily Return"],
        )

        # Cumulative returns
        if isinstance(returns, pd.Series):
            cum_returns = _stats.compsum(returns) * 100
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index, y=cum_returns, name=returns.name or "Strategy", line=dict(color="#348dc1")
                ),
                row=1,
                col=1,
            )
        elif isinstance(returns, pd.DataFrame):
            for col in returns.columns:
                cum_returns = _stats.compsum(returns[col]) * 100
                fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name=col), row=1, col=1)

        # Drawdowns
        if isinstance(dd, pd.Series):
            fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, name="Drawdown", line=dict(color="#af4b64")), row=2, col=1)
            fig.add_trace(
                go.Scatter(x=dd.index, y=[0] * len(dd), showlegend=False, line=dict(color="lightgray", dash="dash")),
                row=2,
                col=1,
            )
        elif isinstance(dd, pd.DataFrame):
            for col in dd.columns:
                fig.add_trace(go.Scatter(x=dd[col].index, y=dd[col] * 100, name=f"DD: {col}"), row=2, col=1)

        # Daily returns
        if isinstance(returns, pd.Series):
            fig.add_trace(
                go.Scatter(x=returns.index, y=returns * 100, name="Daily Return", line=dict(color="#fedd78", width=1)),
                row=3,
                col=1,
            )
        elif isinstance(returns, pd.DataFrame):
            for i, col in enumerate(returns.columns):
                fig.add_trace(
                    go.Scatter(x=returns[col].index, y=returns[col] * 100, name=f"{col} Return"), row=3, col=1
                )

        fig.update_layout(height=800, title_text=title, showlegend=True, template="plotly_white")

        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", row=2, col=1)
            fig.update_yaxes(type="log", row=3, col=1)

        return fig

    def drawdown(
        self,
        fontname="Arial",
        lw=1,
        log_scale=False,
        ylabel="Drawdown",
        subtitle=True,
    ):
        dd = _stats.to_drawdown_series(self.returns)

        return plot_timeseries(
            dd,
            title="Underwater Plot",
            # hline=dd.mean(),
            # hlw=2,
            # hllabel="Average",
            log_scale=log_scale,
            lw=lw,
            ylabel=ylabel,
            fontname=fontname,
            subtitle=subtitle,
        )

    def drawdowns_periods(
        self,
        periods=5,
        lw=1.5,
        log_scale=False,
        fontname="Arial",
        title=None,
        ylabel=True,
        subtitle=True,
        compounded=True,
    ):
        # if prepare_returns:
        #    returns = _utils._prepare_returns(returns)

        return plot_longest_drawdowns(
            self.returns[self.returns.columns[0]],
            periods=periods,
            lw=lw,
            log_scale=log_scale,
            fontname=fontname,
            title=title,
            ylabel=ylabel,
            subtitle=subtitle,
            compounded=compounded,
        )

    def rolling_beta(
        self,
        window1=126,
        window1_label="6-Months",
        window2=252,
        window2_label="12-Months",
        lw=1.5,
        fontname="Arial",
        ylabel=True,
        subtitle=True,
    ):
        return plot_rolling_beta(
            self.returns,
            self.benchmark,
            window1=window1,
            window1_label=window1_label,
            window2=window2,
            window2_label=window2_label,
            title="Rolling Beta to Benchmark",
            fontname=fontname,
            lw=lw,
            ylabel=ylabel,
            subtitle=subtitle,
        )


def build_data(returns: pd.DataFrame | pd.Series, rf=0.0, benchmark=None, nperiods=None) -> _Data:
    def _excess(x, rf=0.0, nperiods=None):
        if not isinstance(rf, float):
            rf = rf[rf.index.isin(x.index)]

        if nperiods is not None:
            # deannualize
            rf = np.power(1 + rf, 1.0 / nperiods) - 1.0

        df = x - rf
        df = df.tz_localize(None)
        return df

    if isinstance(returns, pd.Series):
        r = returns.copy().to_frame(name="returns")
    else:
        r = returns.copy()

    if benchmark is None:
        return _Data(returns=_excess(r, rf, nperiods=nperiods))

    else:
        common = sorted(list(set(r.index) & set(benchmark.index)))
        return _Data(
            returns=_excess(r.loc[common], rf, nperiods=nperiods),
            benchmark=_excess(benchmark.loc[common], rf, nperiods=nperiods),
        )

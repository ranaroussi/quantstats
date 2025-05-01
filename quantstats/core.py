# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
#
# Copyright 2019-2024 Ran Aroussi
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

from . import stats as _stats

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


# def _compound(x):
#    return (1 + x).cumprod() - 1


def _get_colors():
    colors = _FLATUI_COLORS
    ls = "-"
    alpha = 0.8
    return colors, ls, alpha


def plot_returns_bars(df):
    colors, _, _ = _get_colors()

    fig = go.Figure()

    for idx, col in enumerate(df.columns):
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[col],
                name=col,
                marker_color=colors[idx % len(colors)],
            )
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            tickformat="%Y",
            showgrid=False,
        ),
        yaxis=dict(
            tickformat=".0%",
            showgrid=True,
            gridcolor="lightgray",
        ),
    )

    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

    return fig


def plot_timeseries(
    df,
    title="Returns",
    percent=True,
    log_scale=False,
    lw=1.5,
    ylabel="",
    fontname="Arial",
    subtitle=True,
):
    colors, ls, alpha = _get_colors()

    fig = go.Figure()

    # Plot each series
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(color=colors[i + 1], width=lw),
                opacity=alpha,
                hovertemplate=f"<b>{col}</b><br>%{{y:.2%}}<extra></extra>"
                if percent
                else f"<b>{col}</b><br>%{{y:.2f}}<extra></extra>",
            )
        )

    # Always add 0-line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

    # Layout polish
    date_range = f"{df.index.min():%d %b '%y} - {df.index.max():%d %b '%y}"

    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><sub>{date_range}</sub>" if subtitle else f"<b>{title}</b>",
            "y": 0.93,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(family=fontname, size=14),
        yaxis_title=ylabel if ylabel else None,
        yaxis_tickformat=".0%" if percent else None,
        yaxis_type="log" if log_scale else "linear",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
        hovermode="x unified",
        margin=dict(l=30, r=30, t=60, b=30),
    )

    # Softer gridlines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray", zeroline=False)
    return fig


def plot_histogram(
    returns,
    benchmark=None,
    resample="ME",
    bins=20,
    fontname="Arial",
    title="Returns",
    kde=True,
    subtitle=True,
    compounded=True,
):
    colors, _, _ = _get_colors()

    apply_fnc = _stats.compsum if compounded else np.sum

    # Resample and prepare data
    if benchmark is not None:
        benchmark = benchmark.fillna(0).resample(resample).apply(apply_fnc).resample(resample).last()

    returns = returns.fillna(0).resample(resample).apply(apply_fnc).resample(resample).last()

    if isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
        returns = returns.squeeze()

    fig = go.Figure()

    # Add histogram(s)
    if benchmark is not None:
        data = pd.concat([returns, benchmark.rename("Benchmark")], axis=1).dropna()
        for idx, col in enumerate(data.columns):
            fig.add_trace(
                go.Histogram(
                    x=data[col],
                    name=col,
                    nbinsx=bins,
                    marker_color=colors[idx % len(colors)],
                    opacity=0.7,
                    histnorm="density",  # Density so it matches KDE
                )
            )
    else:
        if isinstance(returns, pd.Series):
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name="Returns",
                    nbinsx=bins,
                    marker_color=colors[1 % len(colors)],
                    opacity=0.7,
                    histnorm="density",
                )
            )
        else:
            for idx, col in enumerate(returns.columns):
                fig.add_trace(
                    go.Histogram(
                        x=returns[col],
                        name=col,
                        nbinsx=bins,
                        marker_color=colors[(idx + 1) % len(colors)],
                        opacity=0.5,
                        histnorm="density",
                    )
                )

    # Add KDE curve(s) if requested
    if kde:
        sample_data = returns if benchmark is None else data
        if isinstance(sample_data, pd.DataFrame):
            for idx, col in enumerate(sample_data.columns):
                kde_x = np.linspace(sample_data[col].min(), sample_data[col].max(), 200)
                kde_y = stats.gaussian_kde(sample_data[col])(kde_x)
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode="lines",
                        name=f"{col} KDE",
                        line=dict(color=colors[idx % len(colors)], width=2, dash="dot"),
                    )
                )
        else:
            kde_x = np.linspace(sample_data.min(), sample_data.max(), 200)
            kde_y = stats.gaussian_kde(sample_data)(kde_x)
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name="KDE",
                    line=dict(color="black", width=2, dash="dot"),
                )
            )

    # Add vertical average line
    avg_val = returns.mean() if isinstance(returns, pd.Series) else returns.stack().mean()
    fig.add_vline(
        x=avg_val,
        line=dict(color="red", dash="dash"),
        annotation_text="Average",
        annotation_position="top right",
    )

    # Create full title
    full_title = title
    if subtitle:
        full_title += f"<br><sub>{returns.index[0].year} - {returns.index[-1].year}</sub>"

    # Layout polish
    fig.update_layout(
        title={
            "text": full_title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(family=fontname, size=20, color="black"),
        },
        bargap=0.1,
        bargroupgap=0.05,
        xaxis_title="",
        yaxis_title="Density",  # <- changed from "Occurrences"
        font=dict(family=fontname, size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(tickformat=".0%")

    return fig


def plot_rolling_stats(
    returns,
    benchmark=None,
    title="",
    returns_label="Strategy",
    lw=1.5,
    ylabel="",
    fontname="Arial",
    subtitle=True,
    # hline=None,
    # hlw=1,
    # hlcolor="red",
    # hllabel="",
):
    colors, _, _ = _get_colors()

    fig = go.Figure()

    if isinstance(returns, pd.DataFrame):
        returns_label = list(returns.columns)

    if isinstance(returns, pd.Series):
        df = pd.DataFrame(index=returns.index, data={returns_label: returns})
    elif isinstance(returns, pd.DataFrame):
        df = pd.DataFrame(index=returns.index, data={col: returns[col] for col in returns.columns})

    if isinstance(benchmark, pd.Series):
        df["Benchmark"] = benchmark[benchmark.index.isin(df.index)]
        if isinstance(returns, pd.Series):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[returns_label],
                    mode="lines",
                    name=returns.name if returns.name else "Strategy",
                    line=dict(color=colors[1], width=lw),
                )
            )
        elif isinstance(returns, pd.DataFrame):
            for i, col in enumerate(returns_label):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode="lines",
                        name=col,
                        line=dict(color=colors[i + 1], width=lw),
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Benchmark"],
                mode="lines",
                name=benchmark.name if benchmark.name else "Benchmark",
                line=dict(color=colors[0], width=lw, dash="dot"),
                opacity=0.8,
            )
        )
    else:
        if isinstance(returns, pd.Series):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[returns_label],
                    mode="lines",
                    name=returns.name if returns.name else "Strategy",
                    line=dict(color=colors[1], width=lw),
                )
            )
        elif isinstance(returns, pd.DataFrame):
            for i, col in enumerate(returns_label):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode="lines",
                        name=col,
                        line=dict(color=colors[i + 1], width=lw),
                    )
                )

    # # Add optional horizontal line
    # if hline is not None:
    #     fig.add_hline(
    #         y=hline,
    #         line_dash="dash",
    #         line_color=hlcolor,
    #         line_width=hlw,
    #         annotation_text=hllabel,
    #         annotation_position="top left",
    #     )

    # Always a zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        line_width=1,
    )

    # Build the title
    base_title = title if title else "Rolling Statistics"
    full_title = base_title
    if subtitle:
        subtitle_text = f"{df.index.date[0].strftime('%e %b %y')} - {df.index.date[-1].strftime('%e %b %y')}"
        full_title += f"<br><sub>{subtitle_text}</sub>"

    fig.update_layout(
        title={
            "text": full_title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(family=fontname, size=20, color="black"),
        },
        xaxis_title="",
        yaxis_title=ylabel,
        font=dict(family=fontname, size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_yaxes(tickformat=".2f")

    # Hide legend if no benchmark and single strategy
    if benchmark is None and isinstance(returns, pd.Series):
        fig.update_layout(showlegend=False)

    return fig


def plot_rolling_beta(
    returns,
    benchmark,
    window1=126,
    window1_label="",
    window2=None,
    window2_label="",
    title="Rolling Beta",
    hlcolor="red",
    fontname="Arial",
    lw=1.5,
    ylabel=True,
    subtitle=True,
):
    colors, _, _ = _get_colors()

    fig = go.Figure()

    if isinstance(returns, pd.Series):
        beta = _stats.rolling_greeks(returns, benchmark, window1)["beta"].fillna(0)
        fig.add_trace(
            go.Scatter(
                x=beta.index,
                y=beta,
                mode="lines",
                name=window1_label if window1_label else f"Beta ({window1})",
                line=dict(color=colors[1], width=lw),
            )
        )
    elif isinstance(returns, pd.DataFrame):
        beta = {
            col: _stats.rolling_greeks(returns[col], benchmark, window1)["beta"].fillna(0) for col in returns.columns
        }
        for i, (name, b) in enumerate(beta.items(), start=1):
            fig.add_trace(
                go.Scatter(
                    x=b.index,
                    y=b,
                    mode="lines",
                    name=f"{name} ({window1_label})" if window1_label else f"{name} (Beta {window1})",
                    line=dict(color=colors[i], width=lw),
                )
            )

    # Second window if provided
    if window2:
        lw2 = max(lw - 0.5, 1)
        if isinstance(returns, pd.Series):
            beta2 = _stats.rolling_greeks(returns, benchmark, window2)["beta"].fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=beta2.index,
                    y=beta2,
                    mode="lines",
                    name=window2_label if window2_label else f"Beta ({window2})",
                    line=dict(color="gray", width=lw2, dash="dot"),
                    opacity=0.8,
                )
            )
        elif isinstance(returns, pd.DataFrame):
            beta2 = {
                col: _stats.rolling_greeks(returns[col], benchmark, window2)["beta"].fillna(0)
                for col in returns.columns
            }
            for i, (name, b2) in enumerate(beta2.items(), start=1):
                fig.add_trace(
                    go.Scatter(
                        x=b2.index,
                        y=b2,
                        mode="lines",
                        name=f"{name} ({window2_label})" if window2_label else f"{name} (Beta {window2})",
                        line=dict(color=colors[i], width=lw2, dash="dot"),
                        opacity=0.5,
                    )
                )

    # Mean line (only if single series)
    if isinstance(returns, pd.Series):
        fig.add_hline(
            y=beta.mean(),
            line_dash="dash",
            line_color=hlcolor,
            line_width=1.5,
            annotation_text="Mean Beta",
            annotation_position="top left",
        )

    # Always draw horizontal 0 line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        line_width=1,
    )

    # Find min/max for smart y-axis ticks
    if isinstance(returns, pd.Series):
        beta_min, beta_max = beta.min(), beta.max()
    else:
        beta_min = min([b.min() for b in beta.values()])
        beta_max = max([b.max() for b in beta.values()])

    mmin = min(-1, int(beta_min * 10) / 10)
    mmax = max(1, int(beta_max * 10) / 10)

    fig.update_yaxes(tickmode="linear", tick0=mmin, dtick=0.5, range=[mmin - 0.1, mmax + 0.1])

    # Title and subtitle
    base_title = title if title else "Rolling Beta"
    full_title = base_title
    if subtitle:
        subtitle_text = f"{returns.index.date[0].strftime('%e %b %y')} - {returns.index.date[-1].strftime('%e %b %y')}"
        full_title += f"<br><sub>{subtitle_text}</sub>"

    fig.update_layout(
        title={
            "text": full_title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(family=fontname, size=20, color="black"),
        },
        xaxis_title="",
        yaxis_title="Beta" if ylabel else "",
        font=dict(family=fontname, size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return fig


def plot_longest_drawdowns(
    returns: pd.Series,
    periods=5,
    lw=1.5,
    fontname="Arial",
    title=None,
    log_scale=False,
    ylabel=True,
    subtitle=True,
    compounded=True,
):
    colors = ["#348dc1", "#003366", "red"]

    if not isinstance(returns, pd.Series):
        raise TypeError("returns is expected to be pd.Series")

    # Calculate drawdowns
    dd = _stats.to_drawdown_series(returns.fillna(0))
    dddf = _stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(by="days", ascending=False, kind="mergesort").head(periods)

    # Calculate cumulative returns
    series = _stats.compsum(returns) if compounded else returns.cumsum()

    fig = go.Figure()

    # Plot cumulative returns line
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Backtest",
            line=dict(color=colors[0], width=lw),
        )
    )

    # Highlight drawdown periods
    for _, row in longest_dd.iterrows():
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor="red",
            opacity=0.1,
            line_width=0,
            layer="below",
        )

    # Title
    base_title = title if title else "Returns"
    full_title = f"{base_title} - Worst {periods:.0f} Drawdown Periods"
    if subtitle:
        subtitle_text = f"{returns.index.date[0].strftime('%e %b %y')} - {returns.index.date[-1].strftime('%e %b %y')}"
        full_title += f"<br><sub>{subtitle_text}</sub>"

    fig.update_layout(
        title={
            "text": full_title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(family=fontname, size=20, color="black"),
        },
        xaxis_title="",
        yaxis_title="Cumulative Returns" if ylabel else "",
        font=dict(family=fontname, size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    if log_scale:
        fig.update_yaxes(type="log")

    return fig


def plot_distribution(returns, fontname="Arial", compounded=True, title=None):
    colors = _FLATUI_COLORS  # your color palette

    # Ensure DataFrame and fill missing values
    port = pd.DataFrame(returns.fillna(0))
    port.columns = ["Daily"]

    apply_fnc = _stats.compsum if compounded else np.sum

    # Resample returns
    port["Weekly"] = port["Daily"].resample("W-MON").apply(apply_fnc).ffill()
    port["Monthly"] = port["Daily"].resample("M").apply(apply_fnc).ffill()
    port["Quarterly"] = port["Daily"].resample("Q").apply(apply_fnc).ffill()
    port["Yearly"] = port["Daily"].resample("Y").apply(apply_fnc).ffill()

    # Create box plots for each frequency
    fig = go.Figure()

    for i, col in enumerate(["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]):
        fig.add_trace(
            go.Box(
                y=port[col],
                name=col,
                marker_color=colors[i],
                boxmean="sd",  # show mean and std
            )
        )

    # Title and layout
    if not title:
        title = "Return Quantiles"
    else:
        title = f"{title} - Return Quantiles"

    date_range = f"{returns.index.min():%d %b '%y} - {returns.index.max():%d %b '%y}"

    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><sub>{date_range}</sub>",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(family=fontname, size=14),
        yaxis_title="Returns (%)",
        yaxis_tickformat=".0%",
        boxmode="group",
        plot_bgcolor="white",
    )

    return fig

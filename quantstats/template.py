import math
from matplotlib.scale import SymmetricalLogTransform
import numpy as np


def process_symlog_data(values, base=10, linthresh=2, linscale=1):
    transform = SymmetricalLogTransform(base, linthresh, linscale)

    return transform.transform(np.array(values))


def get_selected_years(returns, benchmark):
    years_returns = [date.year for date, _ in returns.items()]
    years_benchmark = [date.year for date, _ in benchmark.items()]

    all_years = years_returns + years_benchmark

    first_year = min(all_years)
    last_year = max(all_years)

    selected_years = list(range(first_year, last_year + 1, 2))
    if selected_years[-1] != last_year:
        selected_years.append(last_year + 1)

    return selected_years


def calculate_step_size(min_value, max_value, min_ticks=8):
    range_value = max_value - min_value
    raw_step = range_value / (min_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = math.ceil(raw_step / magnitude) * magnitude

    return step


def get_mix_max_axisY(returns, benchmark):
    min_axisY = math.floor(min(returns + benchmark) / 100) * 100
    max_axisY = math.ceil(max(returns + benchmark) / 100) * 100

    return min_axisY, max_axisY


def to_log_scale(value, linthresh=1.0):
    if value < -linthresh:
        return -math.log(-value)
    elif value > linthresh:
        return math.log(value)
    else:
        return value


def gen_template_for_return_chart(
    chart_id,
    title,
    returns,
    benchmark,
    hoverable=True,
    font_size=14,
    font_name="Arial",
    figsize=(10, 6),
    log_scale=False,
    line_opacity=1,
):
    if log_scale:
        returns_log_scale_transform_values = [
            item.item() * 100
            for item in process_symlog_data([value for _, value in returns.items()])
        ]
        benchmark_log_scale_transform_values = [
            item.item() * 100
            for item in process_symlog_data([value for _, value in benchmark.items()])
        ]
    else:
        returns_log_scale_transform_values = [
            value * 100 for _, value in returns.items()
        ]
        benchmark_log_scale_transform_values = [
            value * 100 for _, value in benchmark.items()
        ]

    returns_log_scale_transform_keys = [str(date) for date, _ in returns.items()]
    benchmark_log_scale_transform_keys = [str(date) for date, _ in benchmark.items()]

    chart_data = {
        "returns": {
            "keys": returns_log_scale_transform_keys,
            "values": returns_log_scale_transform_values,
        },
        "benchmark": {
            "keys": benchmark_log_scale_transform_keys,
            "values": benchmark_log_scale_transform_values,
        },
    }

    min_axisY, max_axisY = get_mix_max_axisY(
        returns_log_scale_transform_values, benchmark_log_scale_transform_values
    )

    step_size_value = calculate_step_size(min_axisY, max_axisY)

    selected_years = get_selected_years(returns, benchmark)

    aspect_ratio = figsize[0] / figsize[1]

    func_custom_color = """
        function(context) {
            if (context.tick.value === 0) {
                return 'black';
            }
            return 'rgba(0, 0, 0, 0.1)';
        }
    """

    format_pct_axisY = """
        function(value) {
            if (value >= 1e12) {
                const res = (value * 1e-12).toFixed(1) + 'T%';
                return res.replace('.0T%', 'T%');
            }
            if (value >= 1e9) {
                const res = (value * 1e-9).toFixed(1) + 'B%';
                return res.replace('.0B%', 'B%');
            }
            if (value >= 1e6) {
                const res = (value * 1e-6).toFixed(1) + 'M%';
                return res.replace('.0M%', 'M%');
            }
            if (value >= 1e3) {
                const res = (value * 1e-3).toFixed(1) + 'K%';
                return res.replace('.0K%', 'K%');
            }
            const res = value.toFixed(0) + '%';
            return res.replace('.0%', '%');
        }
    """

    format_pct_axisX = f"""
        function(_, index) {{
            const current_year = new Date(label_{chart_id}[index]).getFullYear();
            if (selected_years_{chart_id}.includes(current_year) && !arrYearsDisplayed_{chart_id}.includes(current_year)) {{
                arrYearsDisplayed_{chart_id}.push(current_year);
                return current_year;
            }} else {{
                return null;
            }}
        }}
    """

    func_line_width = """
        function(context) {
            return context.tick.value === 0 ? 2 : 1;
        }
    """

    func_custom_legend = f"""
        function(chart) {{
            const chartWidth = chart.width;
            const labelWith = chartWidth * 0.05

            const legend = chart.options.plugins.legend;
            
            if (legend && legend.labels) {{
                if (labelWith !== legend.labels.pointStyleWidth) {{
                    legend.labels.pointStyleWidth = labelWith;
                    arrYearsDisplayed_{chart_id} = [];
                    chart.update();
                }}
            }}
        }}
    """

    func_custom_legend_click = """
        function(e, legendItem, legend) {
            return;
        }
    """

    return f"""
        <script>
            let selected_years_{chart_id} = {selected_years};
            let arrYearsDisplayed_{chart_id} = [];
            let data_{chart_id} = {chart_data};
            let label_{chart_id} = data_{chart_id}.returns.keys;
            let ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
            new Chart(ctx_{chart_id}, {{
                type: 'line',
                data: {{
                    labels: label_{chart_id},
                    datasets: [
                        {{
                            label: `{benchmark.name}`,
                            data: data_{chart_id}.benchmark.values,
                            borderColor: 'rgba(254,221,120,{line_opacity})',
                            borderWidth: 1.2,
                            radius: 0.1,
                            fill: false
                        }},
                        {{
                            label: `{returns.name}`,
                            data: data_{chart_id}.returns.values,
                            borderColor: 'rgba(52,141,193,{line_opacity})',
                            borderWidth: 1.2,
                            radius: 0.1,
                            fill: false
                        }}
                    ]
                }},
                plugins: [
                    {{
                        id: 'customLegendWidth',
                        afterRender: {func_custom_legend}
                    }}
                ],
                options: {{
                    interaction: {{
                        mode: `{"nearest" if hoverable else "null"}`
                    }},
                    aspectRatio: {aspect_ratio},
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: `{title}`,
                            font: {{
                                size: 16,
                                family: `{font_name}`,
                                weight: 'bold'
                            }},
                            color: "#000000",
                        }},
                        legend: {{
                            onClick: {func_custom_legend_click},
                            labels: {{
                                usePointStyle: true,
                                pointStyle: 'line',
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: ''
                            }},
                            grid: {{
                                drawOnChartArea: true,
                            }},
                            ticks: {{
                                minRotation: 25,
                                font: {{
                                    size: {font_size},
                                    family: `{font_name}`
                                }},
                                padding: 20,
                                callback: {format_pct_axisX}
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: ''
                            }},
                            min: {min_axisY},
                            max: {max_axisY},
                            ticks: {{
                                font: {{
                                    size: {font_size},
                                    family: `{font_name}`
                                }},
                                stepSize: {step_size_value},
                                callback: {format_pct_axisY},
                            }},
                            grid: {{
                                color: {func_custom_color},
                                lineWidth: {func_line_width}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    """

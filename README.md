[![Python version](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat)](https://pypi.python.org/pypi/quantstats)
[![PyPi version](https://img.shields.io/pypi/v/quantstats.svg?maxAge=60)](https://pypi.python.org/pypi/quantstats)
[![PyPi status](https://img.shields.io/pypi/status/quantstats.svg?maxAge=60)](https://pypi.python.org/pypi/quantstats)
[![PyPi downloads](https://img.shields.io/pypi/dm/quantstats.svg?maxAge=2592000&label=installs&color=%2327B1FF)](https://pypi.python.org/pypi/quantstats)
[![Star this repo](https://img.shields.io/github/stars/ranaroussi/quantstats.svg?style=social&label=Star&maxAge=60)](https://github.com/ranaroussi/quantstats)
[![Follow me on twitter](https://img.shields.io/twitter/follow/aroussi.svg?style=social&label=Follow&maxAge=60)](https://twitter.com/aroussi)

# QuantStats: Portfolio analytics for quants

**QuantStats** Python library that performs portfolio profiling, allowing quants and portfolio managers to understand their performance better by providing them with in-depth analytics and risk metrics.

[Changelog »](./CHANGELOG.md)

### QuantStats is comprised of 3 main modules:

1. `quantstats.stats` - for calculating various performance metrics, like Sharpe ratio, Win rate, Volatility, etc.
2. `quantstats.plots` - for visualizing performance, drawdowns, rolling statistics, monthly returns, etc.
3. `quantstats.reports` - for generating metrics reports, batch plotting, and creating tear sheets that can be saved as an HTML file.

---

### **NEW! Monte Carlo Simulations**

<img src="https://raw.githubusercontent.com/ranaroussi/pandas-montecarlo/master/demo.png" alt="Monte Carlo Simulation" width="640">

Run probabilistic risk analysis with built-in Monte Carlo simulations:

```python
mc = qs.stats.montecarlo(returns, sims=1000, bust=-0.20, goal=0.50)
print(f"Bust probability: {mc.bust_probability:.1%}")
print(f"Goal probability: {mc.goal_probability:.1%}")
mc.plot()
```

[Full Monte Carlo documentation »](./docs/montecarlo.md)

---

## Quick Start

```python
%matplotlib inline
import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = qs.utils.download_returns('META')

# show sharpe ratio
qs.stats.sharpe(stock)

# or using extend_pandas() :)
stock.sharpe()
```

Output:

```
0.7604779884378278
```

### Visualize stock performance

```python
qs.plots.snapshot(stock, title='Facebook Performance', show=True)

# can also be called via:
# stock.plot_snapshot(title='Facebook Performance', show=True)
```

Output:

![Snapshot plot](https://github.com/ranaroussi/quantstats/blob/main/docs/snapshot.webp?raw=true)

### Creating a report

You can create 7 different report tearsheets:

1. `qs.reports.metrics(mode='basic|full", ...)` - shows basic/full metrics
2. `qs.reports.plots(mode='basic|full", ...)` - shows basic/full plots
3. `qs.reports.basic(...)` - shows basic metrics and plots
4. `qs.reports.full(...)` - shows full metrics and plots
5. `qs.reports.html(...)` - generates a complete report as html

Let's create an html tearsheet:

```python
# benchmark can be a pandas Series or ticker
qs.reports.html(stock, "SPY")
```

Output will generate something like this:

![HTML tearsheet](https://github.com/ranaroussi/quantstats/blob/main/docs/report.webp?raw=true)

[View original html file](https://rawcdn.githack.com/ranaroussi/quantstats/main/docs/tearsheet.html)

### Available methods

To view a complete list of available methods, run:

```python
[f for f in dir(qs.stats) if f[0] != '_']
```

```python
['avg_loss',
 'avg_return',
 'avg_win',
 'best',
 'cagr',
 'calmar',
 'common_sense_ratio',
 'comp',
 'compare',
 'compsum',
 'conditional_value_at_risk',
 'consecutive_losses',
 'consecutive_wins',
 'cpc_index',
 'cvar',
 'drawdown_details',
 'expected_return',
 'expected_shortfall',
 'exposure',
 'gain_to_pain_ratio',
 'geometric_mean',
 'ghpr',
 'greeks',
 'implied_volatility',
 'information_ratio',
 'kelly_criterion',
 'kurtosis',
 'max_drawdown',
 'monthly_returns',
 'montecarlo',
 'montecarlo_cagr',
 'montecarlo_drawdown',
 'montecarlo_sharpe',
 'outlier_loss_ratio',
 'outlier_win_ratio',
 'outliers',
 'payoff_ratio',
 'profit_factor',
 'profit_ratio',
 'r2',
 'r_squared',
 'rar',
 'recovery_factor',
 'remove_outliers',
 'risk_of_ruin',
 'risk_return_ratio',
 'rolling_greeks',
 'ror',
 'sharpe',
 'skew',
 'sortino',
 'adjusted_sortino',
 'tail_ratio',
 'to_drawdown_series',
 'ulcer_index',
 'ulcer_performance_index',
 'upi',
 'value_at_risk',
 'var',
 'volatility',
 'win_loss_ratio',
 'win_rate',
 'worst']
```

```python
[f for f in dir(qs.plots) if f[0] != '_']
```

```python
['daily_returns',
 'distribution',
 'drawdown',
 'drawdowns_periods',
 'earnings',
 'histogram',
 'log_returns',
 'monthly_heatmap',
 'montecarlo',
 'montecarlo_distribution',
 'returns',
 'rolling_beta',
 'rolling_sharpe',
 'rolling_sortino',
 'rolling_volatility',
 'snapshot',
 'yearly_returns']
```

**\*\*\* Full documentation coming soon \*\*\***

In the meantime, you can get insights as to optional parameters for each method, by using Python's `help` method:

```python
help(qs.stats.conditional_value_at_risk)
```

```
Help on function conditional_value_at_risk in module quantstats.stats:

conditional_value_at_risk(returns, sigma=1, confidence=0.99)
    calculates the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
```

## Installation

Install using `pip`:

```bash
$ pip install quantstats --upgrade --no-cache-dir
```

Install using `conda`:

```bash
$ conda install -c ranaroussi quantstats
```

## Requirements

* [Python](https://www.python.org) >= 3.5+
* [pandas](https://github.com/pydata/pandas) (tested to work with >=0.24.0)
* [numpy](http://www.numpy.org) >= 1.15.0
* [scipy](https://www.scipy.org) >= 1.2.0
* [matplotlib](https://matplotlib.org) >= 3.0.0
* [seaborn](https://seaborn.pydata.org) >= 0.9.0
* [tabulate](https://bitbucket.org/astanin/python-tabulate) >= 0.8.0
* [yfinance](https://github.com/ranaroussi/yfinance) >= 0.1.38
* [plotly](https://plot.ly/) >= 3.4.1 (optional, for using `plots.to_plotly()`)

## Questions?

This is a new library... If you find a bug, please
[open an issue](https://github.com/ranaroussi/quantstats/issues).

If you'd like to contribute, a great place to look is the
[issues marked with help-wanted](https://github.com/ranaroussi/quantstats/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Known Issues

For some reason, I couldn't find a way to tell seaborn not to return the
monthly returns heatmap when instructed to save - so even if you save the plot (by passing `savefig={...}`) it will still show the plot.

## Legal Stuff

**QuantStats** is distributed under the **Apache Software License**. See the [LICENSE.txt](./LICENSE.txt) file in the release for details.

## P.S.

Please drop me a note with any feedback you have.

**Ran Aroussi**

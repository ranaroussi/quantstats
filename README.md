[![Python version](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat)](https://pypi.python.org/pypi/quantstats)
[![PyPi version](https://img.shields.io/pypi/v/quantstats.svg?maxAge=60)](https://pypi.python.org/pypi/quantstats)
[![PyPi status](https://img.shields.io/pypi/status/quantstats.svg?maxAge=60)](https://pypi.python.org/pypi/quantstats)
[![PyPi downloads](https://img.shields.io/pypi/dm/quantstats.svg?maxAge=2592000&label=installs&color=%2327B1FF)](https://pypi.python.org/pypi/quantstats)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ranaroussi/quantstats)
[![Star this repo](https://img.shields.io/github/stars/ranaroussi/quantstats.svg?style=social&label=Star&maxAge=60)](https://github.com/ranaroussi/quantstats)
[![Follow me on twitter](https://img.shields.io/twitter/follow/aroussi.svg?style=social&label=Follow&maxAge=60)](https://twitter.com/aroussi)

# QuantStats: Portfolio analytics for quants

**QuantStats** Python library that performs portfolio profiling, allowing quants and portfolio managers to understand their performance better by providing them with in-depth analytics and risk metrics.



## Navigation
- [Quick Start](#quick-start)
- [Understanding Returns](#understanding-returns-important)
- [Documentation](docs/docs_and_api_ref.md)
    - [NEW! Monte Carlo Simulations](docs/montecarlo.md)
- [Installation](#installation)
- [Questions and Contributions](#questions-and-contributions)
- [Changelog »](./CHANGELOG.md)


## Quick Start

### QuantStats is comprised of 3 main modules:

1. `quantstats.stats` - for calculating various performance metrics, like Sharpe ratio, Win rate, Volatility, etc.
2. `quantstats.plots` - for visualizing performance, drawdowns, rolling statistics, monthly returns, etc.
3. `quantstats.reports` - for generating metrics reports, batch plotting, and creating tear sheets that can be saved as an HTML file.

```python
%matplotlib inline
import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
returns = qs.utils.download_returns('META')

# show sharpe ratio
qs.stats.sharpe(returns)

# or using extend_pandas() :)
returns.sharpe()
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

Output (full-size version [here](https://github.com/ranaroussi/quantstats/blob/main/docs/snapshot.webp?raw=true)):

<p align="center">
  <img
    src="https://github.com/ranaroussi/quantstats/blob/main/docs/snapshot.webp?raw=true"
    alt="Snapshot plot"
    width="450"
  />
</p>

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

Output (full-size version [here](https://rawcdn.githack.com/ranaroussi/quantstats/main/docs/tearsheet.html)):

<p align="center">
  <img
    src="https://github.com/ranaroussi/quantstats/blob/main/docs/report.webp?raw=true"
    alt="HTML tearsheet"
    width="450"
  />
</p>

### The Returns object
As seen in previous examples, QuantStat's primary input is a time-series:
```python
returns = qs.utils.download_returns('META')
qs.stats.sharpe(returns)
```
Or a dataframe (several series with a shared index):
```python
returns = DataFrame()
returns["META"] = qs.utils.download_returns('META')
returns["TSLA"] = qs.utils.download_returns('TSLA')
qs.stats.sharpe(returns)
```

Quantstats makes several notable assumptions about this object by default:
1. This is a daily (`datetime64[s]` index) time-series of returns (`float64`). These returns are the percent gain from the previous period (e.g. 10% growth -> 0.1, NOT 1.1).
2. There are 252 periods (rows) in a year (not 365!) corresponding to the number of trading days in the U.S. stock market. Many methods assume this amount of periods by default; if you have substantially more or less data in your time-series, the annualization math may be skewed.
3. For an example of correctly shaped returns, see time-series returned by `qs.utils.download_returns()`


## Additional Notes

### Period-Based vs Trade-Based Metrics

QuantStats analyzes **return series** (daily, weekly, monthly returns), not discrete trade data. This means:

- **Win Rate** = percentage of periods with positive returns
- **Consecutive Wins/Losses** = consecutive positive/negative return periods
- **Payoff Ratio** = average winning period return / average losing period return
- **Profit Factor** = sum of positive returns / sum of negative returns

These metrics are **valid and useful** for:
- Systematic/algorithmic strategies with regular rebalancing
- Analyzing return-series behavior over time
- Comparing strategies on a period-by-period basis

For **discretionary traders** with multi-day trades, these period-based metrics may differ from trade-level statistics. A single 5-day trade might span 3 positive days and 2 negative days - QuantStats would count these as 3 "wins" and 2 "losses" at the daily level.

This is consistent with how all return-based analytics work (Sharpe ratio, Sortino ratio, drawdown analysis, etc.) - they operate on return periods, not discrete trade entries/exits.


### Help on function conditional_value_at_risk in module quantstats.stats:
```
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

### Requirements

* [Python](https://www.python.org) >= 3.10
* [pandas](https://github.com/pydata/pandas) >= 1.5.0
* [numpy](http://www.numpy.org) >= 1.24.0
* [scipy](https://www.scipy.org) >= 1.11.0
* [matplotlib](https://matplotlib.org) >= 3.7.0
* [seaborn](https://seaborn.pydata.org) >= 0.13.0
* [tabulate](https://bitbucket.org/astanin/python-tabulate) >= 0.9.0
* [yfinance](https://github.com/ranaroussi/yfinance) >= 0.2.40
* [plotly](https://plot.ly/) >= 5.0.0 (optional, for using `plots.to_plotly()`)

## Questions and Contributions

This is a new library... If you find a bug, please
[open an issue](https://github.com/ranaroussi/quantstats/issues).

If you'd like to contribute, a great place to look is the
[issues marked with help-wanted](https://github.com/ranaroussi/quantstats/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Legal Stuff

**QuantStats** is distributed under the **Apache Software License**. See the [LICENSE.txt](./LICENSE.txt) file in the release for details.

## P.S.

Please drop me a note with any feedback you have.

**Ran Aroussi**

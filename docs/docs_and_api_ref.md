## Further References 


Full documentation is still in the works. In the meantime...

1. Many methods already contain detailed docstrings in the source code, so refer to those for further information on each method's parameters, return values, and additional information.

2. AI-Generated Documentation can be found [here](https://deepwiki.com/ranaroussi/quantstats/1-overview).

3. You can get insights as to optional parameters for each method, by using Python's `help` method:

```python
help(qs.stats.conditional_value_at_risk)
```

## API Reference

```python
import quantstats as qs

qs.extend_pandas()

df.cagr()
qs.cagr(df)
```

### Stats

Many of the methods below in qs.stats can be extended onto Pandas' Time-series and DataFrame objects. For more information, refer to qs.extend_pandas()

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

### Plots

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

### Reports

```python
[f for f in dir(qs.reports) if f[0] != '_']
```

```python
['Path',
    'StringIO',
    'basic',
    'full', 'html',
    'metrics', 
    'plots',
    'relativedelta',
    'tempfile',
    'webbrowser']
```
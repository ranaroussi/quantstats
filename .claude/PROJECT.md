---
updated: 2026-01-13T14:49:14Z
---

# QuantStats

## Purpose
Portfolio analytics library for quants - calculates 50+ performance metrics (Sharpe, Sortino, max drawdown, VaR, etc.) and generates HTML tearsheet reports comparing strategies to benchmarks.

## Domain Knowledge
- Returns can be simple (arithmetic) or log returns; most metrics use simple
- "Compounded" means geometric growth (1+r1)*(1+r2)... vs arithmetic sum
- Drawdown = peak-to-trough decline; max drawdown is worst historical decline
- Risk-free rate (rf) typically US Treasury; affects Sharpe/Sortino calculations
- Trading days: 252/year (US equities), 365 for crypto
- Monte Carlo: shuffling returns preserves distribution but varies paths
- **Period-based metrics**: win_rate, consecutive_wins, payoff_ratio analyze return periods (not discrete trades)

## Conventions
- `_prepare_returns()` normalizes all input data before calculations
- `_compat.py` handles pandas 1.5+/2.0+ specifics (freq aliases, timezone normalization)
- `_numpy_compat.py` handles numpy 1.24+ specifics (product deprecation)
- Stats functions return scalar for Series input, Series for DataFrame input
- `extend_pandas()` adds methods directly to DataFrame/Series objects
- Monte Carlo functions return `MonteCarloResult` dataclass with properties

## Current Focus
v0.0.78 released - 2026 modernization complete.
Remaining open: #493 (trade-based metrics - documented as period-based, not removing)

## Recently Completed (v0.0.78)
- GitHub release v0.0.78 published
- 12 issues closed (#472, #475, #477, #479, #480, #481, #484, #485, #486, #489, #491, #492)
- 5 PRs closed (superseded by #494)
- PR #495 merged: README updated with period-based metrics clarification
- PR #483 merged: Dependabot - actions/checkout v4 → v6
- Feature #472: Parameters table in HTML/text reports
- Type hints: Python 3.10+ union syntax throughout
- pyproject.toml migration (setup.py removed)
- Monte Carlo integration (stats, plots, pandas extension)
- pandas >=1.5.0 (supports both 1.x and 2.x)
- 104 tests passing

## Project Structure

```
quantstats/
├── __init__.py          # Main exports (stats, plots, reports, utils)
├── stats.py             # Statistical metrics (Sharpe, Sortino, drawdown, etc.)
├── plots.py             # Visualization functions
├── reports.py           # Report generation (HTML, metrics tables)
├── utils.py             # Utility functions (data prep, download)
├── _compat.py           # Pandas version compatibility layer
├── _numpy_compat.py     # NumPy version compatibility layer
├── _montecarlo.py       # Monte Carlo simulation module
├── version.py           # Version string
└── report.html          # HTML template for reports
```

## Development

### Testing
```bash
pytest tests/ -v                          # Run all tests
pytest tests/test_stats.py -v             # Run specific test file
pytest tests/ --cov=quantstats            # Run with coverage
```

### Code Quality
```bash
ruff check quantstats/                    # Lint
pyright quantstats/                       # Type check
```

## Common Tasks

### Adding a New Metric
1. Add function to `stats.py` with full type hints
2. Add tests to `tests/test_stats.py`
3. If shown in reports, update `reports.py` metrics dict

### Updating Dependencies
1. Edit `pyproject.toml` dependencies section
2. Update README Requirements section to match
3. If pandas/numpy behavior changed, update `_compat.py` or `_numpy_compat.py`

### Version Bump
1. Update `quantstats/version.py`
2. Update `CHANGELOG.md`
3. Create GitHub release with tag

## Gotchas

1. **yfinance MultiIndex**: When downloading data, use `.squeeze()` to flatten single-ticker results
2. **Frequency aliases**: Use `_compat.get_frequency_alias()` for pandas 2.2.0+ compatibility
3. **DataFrame vs Series**: Many stats functions handle both - check `isinstance()` and handle appropriately
4. **NaN handling**: Use `dropna()` before calculations, especially for CVaR/VaR

## Notes
- Author: Ran Aroussi (same as yfinance)
- Target: Python 3.10+, pandas >=1.5.0, numpy >=1.24.0
- PR #470 still open (Dependabot setup-python) - requires workflow scope to merge

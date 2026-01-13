---
updated: 2026-01-13T13:45:26Z
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

## Conventions
- `_prepare_returns()` normalizes all input data before calculations
- `_compat.py` handles pandas 2.0+ specifics (freq aliases, timezone normalization)
- `_numpy_compat.py` handles numpy 1.24+ specifics (product deprecation)
- Stats functions return scalar for Series input, Series for DataFrame input
- `extend_pandas()` adds methods directly to DataFrame/Series objects
- Monte Carlo functions return `MonteCarloResult` dataclass with properties

## Current Focus
2026 modernization - ALL COMPLETE:
- Optional: documentation (API docs, examples notebook)

## Completed
- All bugs fixed (#477, #475, #479, #480, #481, #484, #485, #486, #489, #491, #492, #493)
- Feature #472: Parameters table in HTML/text reports
- Feature #473: Mass issue closure communication
- Performance benchmark tests (tests/test_benchmarks.py)
- Type hints: stats.py (20+ functions), utils.py (key functions), plotting wrappers.py
- pyproject.toml migration (setup.py removed)
- GitHub Actions CI with Codecov
- Monte Carlo integration (stats, plots, pandas extension, docs)
- README RST -> Markdown, py.typed marker, legacy headers removed
- Compat layers simplified (removed obsolete pandas/numpy version checks)
- 104 tests passing

## Notes
- Author: Ran Aroussi (same as yfinance)
- Plan doc: `.claude/2026-modernization-plan.md`
- Target: Python 3.10+, pandas 2.0+, numpy 1.24+
- CI: Tests on Python 3.10-3.13, Ubuntu/macOS/Windows

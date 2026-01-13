# QuantStats 2026 Modernization Plan

## Overview

Modernize QuantStats for Python 3.10+ with modern tooling, better typing, and bug fixes.

---

## 1. Python Version & Packaging

### Drop Python 3.8/3.9 Support
- **Minimum**: Python 3.10 (or 3.11 for even cleaner code)
- **Benefits**:
  - Native `X | Y` union types (no `Union[X, Y]`)
  - `match` statements for cleaner branching
  - Better error messages
  - Structural pattern matching
  - Parenthesized context managers

### Migrate setup.py → pyproject.toml
```toml
[project]
name = "quantstats"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    ...
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Tasks:**
- [x] Create `pyproject.toml` with all metadata
- [x] Remove `setup.py`
- [x] Update classifiers for Python 3.10-3.13 only
- [x] Update `README.md` requirements section

---

## 2. Dependency Updates

| Current | Proposed | Why |
|---------|----------|-----|
| `pandas>=1.5.0` | `pandas>=2.0.0` | Better performance, CoW |
| `numpy>=1.21.0` | `numpy>=1.24.0` | NumPy 2.0 compat |
| `seaborn>=0.11.0` | `seaborn>=0.13.0` | Better defaults |
| `matplotlib>=3.3.0` | `matplotlib>=3.7.0` | Style improvements |
| `scipy>=1.7.0` | `scipy>=1.11.0` | Performance |

**Optional modern deps:**
- `polars` (optional) for faster data processing
- `plotly>=5.0` for interactive plots

---

## 3. Type Hints & Code Quality

### Add Comprehensive Type Hints
Currently minimal typing. Add throughout:

```python
# Before
def sharpe(returns, rf=0.0, periods=252, annualize=True, smart=False):

# After
def sharpe(
    returns: pd.Series | pd.DataFrame,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | pd.Series:
```

**Tasks:**
- [x] Add type hints to key public functions in `stats.py` (sharpe, volatility, etc.)
- [ ] Add type hints to remaining functions in `stats.py`
- [ ] Add type hints to all public functions in `utils.py`
- [x] Add type hints to plotting functions
- [x] Add `py.typed` marker for PEP 561
- [x] Configure `pyright` in CI (with continue-on-error)

### Remove Legacy Patterns
- [ ] Remove `_pd`, `_np` import aliases (use `pd`, `np`) - optional, low priority
- [x] Remove `# -*- coding: UTF-8 -*-` (Python 3 default)
- [x] Fix deprecated IPython import

---

## 4. Open Issues to Address

### High Priority Bugs

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 493 | Trade-analysis metrics computed from returns | HIGH | Documented (valid for returns) |
| 491 | Error generating HTML report with benchmark Series | HIGH | FIXED |
| 486 | reports.metrics vs reports.full inconsistency | HIGH | FIXED |
| 485 | Benchmark Omega always same as Strategy | HIGH | FIXED |
| 484 | make_index is incorrect | HIGH | FIXED |

### Medium Priority

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 481 | NaN in EOY Returns vs Benchmark | MEDIUM | FIXED |
| 480 | Inconsistent metrics benchmark vs return-only | MEDIUM | FIXED |
| 479 | EOY Returns vs Benchmark section issues | MEDIUM | FIXED |
| 475 | Double "%" in HTML report | MEDIUM | FIXED |
| 477 | Noisy variance warning messages | MEDIUM | FIXED |

### Low Priority / Features

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 492 | Chrome dark mode styling | LOW | FIXED |
| 489 | Underwater plot average drawdown | LOW | FIXED |
| 472 | Add parameters table to HTML report | FEATURE | FIXED |
| 473 | Mass closure communication | META | Open |

---

## 5. Code Structure Improvements

### Simplify Module Structure
```
quantstats/
├── __init__.py
├── stats.py          # Core statistics
├── plots.py          # Plotting wrappers
├── reports.py        # Report generation
├── utils.py          # Utilities
├── _montecarlo.py    # Monte Carlo (new)
├── _plotting/
│   ├── __init__.py
│   ├── core.py       # Plot implementations
│   └── wrappers.py   # Public API
└── py.typed          # PEP 561 marker
```

### Remove Compat Layers (if dropping old Python)
- [ ] Review `_compat.py` - may be removable
- [ ] Review `_numpy_compat.py` - may be removable

---

## 6. Testing & CI

### Current: 179 tests passing

### Improvements:
- [x] Add GitHub Actions workflow
- [x] Add type checking to CI (`pyright --warnings`)
- [x] Add coverage reporting (Codecov integration)
- [x] Add benchmarking tests for performance regression

### Test Categories:
```yaml
# .github/workflows/test.yml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
```

---

## 7. Documentation

- [ ] Keep README.md (done - converted from RST)
- [ ] Monte Carlo docs (done)
- [ ] Generate API docs with `mkdocs` or `pdoc`
- [ ] Add examples notebook

---

## 8. Implementation Order

### Phase 1: Foundation (Week 1)
1. Create `pyproject.toml`
2. Update Python version requirement
3. Update dependency versions
4. Remove `setup.py`

### Phase 2: Bug Fixes (Week 2-3)
1. Fix #493 (trade metrics)
2. Fix #491 (HTML report benchmark)
3. Fix #486 (metrics inconsistency)
4. Fix #485 (Omega benchmark)
5. Fix #484 (make_index)

### Phase 3: Modernization (Week 4)
1. Add type hints to stats.py
2. Add type hints to utils.py
3. Remove legacy patterns
4. Add `py.typed`

### Phase 4: Polish (Week 5)
1. Fix remaining issues
2. Add GitHub Actions CI
3. Update all docs
4. Release v1.0.0

---

## 9. Breaking Changes

Document these for CHANGELOG:

1. **Python 3.10+ required** (drop 3.8, 3.9)
2. **pandas 2.0+ required** (drop 1.x)
3. Any function signature changes from bug fixes

---

## 10. New Features (Post-Modernization)

Ideas for future:
- [ ] Polars backend support
- [ ] Async data fetching
- [ ] Interactive Plotly reports
- [ ] PDF export
- [ ] Multi-strategy comparison reports

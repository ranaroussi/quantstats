#!/usr/bin/env python3
"""
Quick validation script for CI/CD compatibility testing.

This script runs basic sanity checks to verify QuantStats works
with the current pandas/numpy versions.
"""

import sys
import warnings

# Suppress expected deprecation warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np


def print_versions():
    """Print dependency versions."""
    print(f"Python: {sys.version}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    try:
        import quantstats as qs
        print(f"quantstats: {qs.__version__}")
    except Exception as e:
        print(f"quantstats import failed: {e}")
        sys.exit(1)


def test_basic_import():
    """Test that basic import works."""
    print("\n[1/6] Testing basic import...")
    import quantstats as qs
    assert hasattr(qs, "stats")
    assert hasattr(qs, "reports")
    assert hasattr(qs, "plots")
    assert hasattr(qs, "utils")
    print("  PASSED")


def test_stats_functions():
    """Test core stats functions work."""
    print("\n[2/6] Testing stats functions...")
    import quantstats as qs

    # Generate test data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)

    # Test key functions
    sharpe = qs.stats.sharpe(returns)
    assert isinstance(sharpe, (float, np.floating)), f"sharpe returned {type(sharpe)}"

    sortino = qs.stats.sortino(returns)
    assert isinstance(sortino, (float, np.floating)), f"sortino returned {type(sortino)}"

    max_dd = qs.stats.max_drawdown(returns)
    assert isinstance(max_dd, (float, np.floating)), f"max_drawdown returned {type(max_dd)}"
    assert max_dd <= 0, "max_drawdown should be non-positive"

    vol = qs.stats.volatility(returns)
    assert isinstance(vol, (float, np.floating)), f"volatility returned {type(vol)}"
    assert vol >= 0, "volatility should be non-negative"

    cagr = qs.stats.cagr(returns)
    assert isinstance(cagr, (float, np.floating)), f"cagr returned {type(cagr)}"

    print("  PASSED")


def test_cvar_dataframe():
    """Test CVaR works with DataFrame (issue #467 regression)."""
    print("\n[3/6] Testing CVaR with DataFrame (issue #467)...")
    import quantstats as qs

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    returns_series = pd.Series(np.random.randn(252) * 0.01, index=dates)
    returns_df = pd.DataFrame({"returns": returns_series})

    cvar_series = qs.stats.cvar(returns_series)
    cvar_df = qs.stats.cvar(returns_df)

    assert not np.isnan(cvar_series), "CVaR Series should not be NaN"

    # DataFrame may return array or scalar
    if hasattr(cvar_df, "__len__"):
        assert not np.isnan(cvar_df).any(), "CVaR DataFrame should not contain NaN"
    else:
        assert not np.isnan(cvar_df), "CVaR DataFrame should not be NaN"

    print("  PASSED")


def test_frequency_aliases():
    """Test frequency alias handling for pandas 2.2.0+."""
    print("\n[4/6] Testing frequency alias handling...")
    import quantstats as qs
    from quantstats._compat import get_frequency_alias

    # Test that old aliases get converted
    assert get_frequency_alias("M") in ["M", "ME"], "Monthly alias should work"
    assert get_frequency_alias("Q") in ["Q", "QE"], "Quarterly alias should work"
    assert get_frequency_alias("Y") in ["Y", "YE"], "Yearly alias should work"

    print("  PASSED")


def test_utils_functions():
    """Test utility functions."""
    print("\n[5/6] Testing utils functions...")
    import quantstats as qs

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

    # Test to_prices and back
    prices = qs.utils.to_prices(returns, base=100)
    assert isinstance(prices, pd.Series), "to_prices should return Series"
    # Note: to_prices calculates (1 + returns).cumprod() * base
    # So first price is base * (1 + first_return), not exactly base
    assert abs(prices.iloc[0] - 100) < 2, "to_prices first value should be near base"

    # Test rebase
    rebased = qs.utils.rebase(prices, base=1000)
    assert rebased.iloc[0] == 1000, "rebase should set first value to base"

    print("  PASSED")


def test_reports_metrics():
    """Test reports.metrics function."""
    print("\n[6/6] Testing reports.metrics...")
    import quantstats as qs

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)

    # Test metrics without benchmark
    metrics = qs.reports.metrics(returns, mode="basic", display=False)
    assert isinstance(metrics, pd.DataFrame), "metrics should return DataFrame"
    assert len(metrics) > 0, "metrics should have rows"

    # Test metrics with benchmark
    benchmark = pd.Series(np.random.randn(252) * 0.008, index=dates)
    metrics_bench = qs.reports.metrics(
        returns, benchmark=benchmark, mode="basic", display=False
    )
    assert isinstance(metrics_bench, pd.DataFrame), "metrics with benchmark should return DataFrame"

    print("  PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("QuantStats Compatibility Test Suite")
    print("=" * 60)

    print_versions()

    tests = [
        test_basic_import,
        test_stats_functions,
        test_cvar_dataframe,
        test_frequency_aliases,
        test_utils_functions,
        test_reports_metrics,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("All tests PASSED!")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"{failed} test(s) FAILED!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Regression test to verify that results are identical when first return is not negative.
"""

import pandas as pd
import numpy as np
import quantstats as qs


def create_original_max_drawdown(prices):
    """Original implementation of max_drawdown function (before fix)."""
    # Validate input (assuming validate_input exists)
    if hasattr(qs.stats, 'validate_input'):
        qs.stats.validate_input(prices)
    
    # Prepare prices (convert from returns if needed)
    prices = qs.utils._prepare_prices(prices)
    
    # Original calculation (without phantom baseline)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def create_original_to_drawdown_series(returns):
    """Original implementation of to_drawdown_series function (before fix)."""
    # Validate input (assuming validate_input exists)
    if hasattr(qs.stats, 'validate_input'):
        qs.stats.validate_input(returns)
    
    # Convert returns to prices
    prices = qs.utils._prepare_prices(returns)
    
    # Original calculation (without phantom baseline)
    dd = prices / np.maximum.accumulate(prices) - 1.0
    
    # Clean up infinite and zero values
    return dd.replace([np.inf, -np.inf, -0], 0)


def test_positive_first_return_cases():
    """Test multiple cases where first return is positive."""
    print("=" * 70)
    print("REGRESSION TEST: Positive First Return Cases")
    print("=" * 70)
    
    test_cases = [
        # Case 1: Simple positive start
        [0.05, -0.02, 0.01, -0.01, 0.03],
        
        # Case 2: Large positive start with subsequent decline
        [0.10, -0.05, -0.03, 0.02, 0.01],
        
        # Case 3: Small positive start with mixed returns
        [0.001, 0.002, -0.005, 0.003, -0.001],
        
        # Case 4: All positive returns
        [0.02, 0.01, 0.03, 0.005, 0.015],
        
        # Case 5: Positive start with large drawdown later
        [0.08, 0.02, -0.15, -0.05, 0.10],
        
        # Case 6: Very small positive start
        [0.0001, -0.02, 0.01, -0.005, 0.02]
    ]
    
    all_identical = True
    
    for i, returns_list in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {returns_list}")
        
        # Create pandas Series
        returns = pd.Series(returns_list, 
                          index=pd.date_range('2023-01-01', periods=len(returns_list), freq='D'))
        
        # Test max_drawdown
        original_max_dd = create_original_max_drawdown(returns)
        new_max_dd = qs.stats.max_drawdown(returns)
        
        # Test to_drawdown_series
        original_dd_series = create_original_to_drawdown_series(returns)
        new_dd_series = qs.stats.to_drawdown_series(returns)
        
        # Compare max_drawdown
        max_dd_match = abs(original_max_dd - new_max_dd) < 1e-10
        print(f"  Max Drawdown - Original: {original_max_dd:.8f}, New: {new_max_dd:.8f}, Match: {max_dd_match}")
        
        # Compare drawdown series
        dd_series_match = np.allclose(original_dd_series.values, new_dd_series.values, atol=1e-10)
        print(f"  Drawdown Series Match: {dd_series_match}")
        
        if not dd_series_match:
            print("  Differences in drawdown series:")
            for j, (orig, new) in enumerate(zip(original_dd_series, new_dd_series)):
                if abs(orig - new) > 1e-10:
                    print(f"    Day {j+1}: Original={orig:.8f}, New={new:.8f}, Diff={abs(orig-new):.2e}")
        
        # Overall match for this case
        case_match = max_dd_match and dd_series_match
        print(f"  Overall Match: {'✅' if case_match else '❌'}")
        
        if not case_match:
            all_identical = False
    
    print(f"\n{'='*70}")
    print(f"REGRESSION TEST RESULT: {'✅ ALL IDENTICAL' if all_identical else '❌ DIFFERENCES FOUND'}")
    print(f"{'='*70}")
    
    return all_identical


def test_zero_first_return():
    """Test case where first return is exactly zero."""
    print("\n" + "=" * 70)
    print("SPECIAL CASE: Zero First Return")
    print("=" * 70)
    
    # Case with zero first return
    returns_list = [0.0, -0.02, 0.01, -0.005, 0.03]
    returns = pd.Series(returns_list, 
                      index=pd.date_range('2023-01-01', periods=len(returns_list), freq='D'))
    
    print(f"Returns: {returns_list}")
    
    # Test max_drawdown
    original_max_dd = create_original_max_drawdown(returns)
    new_max_dd = qs.stats.max_drawdown(returns)
    
    # Test to_drawdown_series
    original_dd_series = create_original_to_drawdown_series(returns)
    new_dd_series = qs.stats.to_drawdown_series(returns)
    
    # Compare results
    max_dd_match = abs(original_max_dd - new_max_dd) < 1e-10
    dd_series_match = np.allclose(original_dd_series.values, new_dd_series.values, atol=1e-10)
    
    print(f"Max Drawdown - Original: {original_max_dd:.8f}, New: {new_max_dd:.8f}, Match: {max_dd_match}")
    print(f"Drawdown Series Match: {dd_series_match}")
    
    if not dd_series_match:
        print("Differences in drawdown series:")
        for j, (orig, new) in enumerate(zip(original_dd_series, new_dd_series)):
            if abs(orig - new) > 1e-10:
                print(f"  Day {j+1}: Original={orig:.8f}, New={new:.8f}, Diff={abs(orig-new):.2e}")
    
    overall_match = max_dd_match and dd_series_match
    print(f"Overall Match: {'✅' if overall_match else '❌'}")
    
    return overall_match


def test_boundary_cases():
    """Test boundary cases around zero."""
    print("\n" + "=" * 70)
    print("BOUNDARY CASES: Very Small Positive Returns")
    print("=" * 70)
    
    boundary_cases = [
        [1e-10, -0.01, 0.005],  # Extremely small positive
        [1e-8, -0.02, 0.01],    # Very small positive
        [1e-6, -0.001, 0.002],  # Small positive
        [1e-4, -0.005, 0.003],  # Tiny positive
    ]
    
    all_match = True
    
    for i, returns_list in enumerate(boundary_cases, 1):
        print(f"\nBoundary Case {i}: First return = {returns_list[0]:.2e}")
        
        returns = pd.Series(returns_list, 
                          index=pd.date_range('2023-01-01', periods=len(returns_list), freq='D'))
        
        # Test max_drawdown
        original_max_dd = create_original_max_drawdown(returns)
        new_max_dd = qs.stats.max_drawdown(returns)
        
        # Test to_drawdown_series
        original_dd_series = create_original_to_drawdown_series(returns)
        new_dd_series = qs.stats.to_drawdown_series(returns)
        
        # Compare results
        max_dd_match = abs(original_max_dd - new_max_dd) < 1e-10
        dd_series_match = np.allclose(original_dd_series.values, new_dd_series.values, atol=1e-10)
        
        print(f"  Max Drawdown - Original: {original_max_dd:.8f}, New: {new_max_dd:.8f}, Match: {max_dd_match}")
        print(f"  Drawdown Series Match: {dd_series_match}")
        
        case_match = max_dd_match and dd_series_match
        print(f"  Overall Match: {'✅' if case_match else '❌'}")
        
        if not case_match:
            all_match = False
    
    return all_match


def run_regression_tests():
    """Run all regression tests."""
    print("COMPREHENSIVE REGRESSION TEST")
    print("Testing that results are identical when first return is non-negative")
    print("=" * 70)
    
    results = []
    
    # Test positive first return cases
    results.append(test_positive_first_return_cases())
    
    # Test zero first return
    results.append(test_zero_first_return())
    
    # Test boundary cases
    results.append(test_boundary_cases())
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL REGRESSION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Test sections passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 REGRESSION TEST PASSED!")
        print("✅ Results are identical when first return is non-negative")
        print("✅ The fix preserves all existing functionality")
        print("✅ No breaking changes introduced")
    else:
        print("❌ REGRESSION TEST FAILED!")
        print("❌ Some results differ when first return is non-negative")
        print("❌ The fix may have introduced unintended changes")
    
    return passed == total


if __name__ == "__main__":
    success = run_regression_tests()
    exit(0 if success else 1)
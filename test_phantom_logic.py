#!/usr/bin/env python3
"""
Test to understand exactly when the phantom baseline is added and its effect.
"""

import pandas as pd
import numpy as np
import quantstats as qs


def analyze_phantom_baseline_logic():
    """Analyze when and how the phantom baseline affects calculations."""
    print("=" * 70)
    print("ANALYZING PHANTOM BASELINE LOGIC")
    print("=" * 70)
    
    test_cases = [
        ("Positive first return", [0.05, -0.02, 0.01]),
        ("Zero first return", [0.0, -0.02, 0.01]),
        ("Negative first return", [-0.05, -0.02, 0.01]),
        ("Very small positive", [0.00001, -0.02, 0.01]),
        ("Very small negative", [-0.00001, -0.02, 0.01]),
    ]
    
    for name, returns_list in test_cases:
        print(f"\n{name}: {returns_list}")
        
        returns = pd.Series(returns_list, 
                          index=pd.date_range('2023-01-01', periods=len(returns_list), freq='D'))
        
        # Convert to prices to see the scale
        prices = qs.utils._prepare_prices(returns)
        print(f"  Prices: {prices.tolist()}")
        
        # Get baseline value that would be used
        baseline_value = qs.stats._get_baseline_value(prices)
        print(f"  Baseline value: {baseline_value}")
        
        # Check if first price is less than baseline (indicates phantom baseline would matter)
        first_price = prices.iloc[0]
        phantom_matters = first_price < baseline_value
        print(f"  First price < baseline: {phantom_matters} ({first_price} < {baseline_value})")
        
        # Calculate drawdown
        dd_series = qs.stats.to_drawdown_series(returns)
        print(f"  Drawdown series: {dd_series.tolist()}")
        print(f"  First drawdown: {dd_series.iloc[0]:.8f}")
        
        # Expected first drawdown if phantom baseline is used
        expected_first_dd = (first_price / baseline_value) - 1
        print(f"  Expected first drawdown: {expected_first_dd:.8f}")
        
        # Check if they match
        matches_expected = abs(dd_series.iloc[0] - expected_first_dd) < 1e-10
        print(f"  Matches expected: {matches_expected}")
        
        print(f"  Analysis: {'Phantom baseline active' if phantom_matters else 'No phantom baseline needed'}")


def test_when_phantom_baseline_activates():
    """Test exactly when the phantom baseline logic activates."""
    print("\n" + "=" * 70)
    print("TESTING WHEN PHANTOM BASELINE ACTIVATES")
    print("=" * 70)
    
    # The key insight: phantom baseline should only matter when first return is negative
    # because that's when the first price is below the "no loss" baseline
    
    # Test with returns that result in first prices around 1.0
    test_returns = [
        0.1,     # Price becomes 1.1 (above baseline 1.0)
        0.01,    # Price becomes 1.01 (above baseline 1.0)
        0.001,   # Price becomes 1.001 (above baseline 1.0)
        0.0001,  # Price becomes 1.0001 (above baseline 1.0)
        0.0,     # Price becomes 1.0 (equals baseline 1.0)
        -0.0001, # Price becomes 0.9999 (below baseline 1.0)
        -0.001,  # Price becomes 0.999 (below baseline 1.0)
        -0.01,   # Price becomes 0.99 (below baseline 1.0)
        -0.1,    # Price becomes 0.9 (below baseline 1.0)
    ]
    
    for first_return in test_returns:
        returns = pd.Series([first_return, -0.01, 0.005], 
                          index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        prices = qs.utils._prepare_prices(returns)
        first_price = prices.iloc[0]
        baseline = qs.stats._get_baseline_value(prices)
        
        dd_series = qs.stats.to_drawdown_series(returns)
        first_dd = dd_series.iloc[0]
        
        # When first return is negative, first drawdown should equal first return
        # When first return is non-negative, first drawdown should be 0
        expected_first_dd = first_return if first_return < 0 else 0.0
        
        matches = abs(first_dd - expected_first_dd) < 1e-10
        
        print(f"First return: {first_return:8.4f} -> Price: {first_price:.4f} -> DD: {first_dd:.8f} -> Expected: {expected_first_dd:.8f} -> {'✅' if matches else '❌'}")


if __name__ == "__main__":
    analyze_phantom_baseline_logic()
    test_when_phantom_baseline_activates()
#!/usr/bin/env python3
"""Test script to reproduce Issue #467: CVAR calculation issue"""

import quantstats as qs
import pandas as pd
import numpy as np

print(f"Testing with quantstats version: {qs.__version__}")

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
returns_series = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
returns_df = pd.DataFrame({'returns': returns_series})

print("\n=== Testing with Series ===")
var_series = qs.stats.var(returns_series)
cvar_series = qs.stats.cvar(returns_series)
print(f"VaR (Series): {var_series:.4%}")
print(f"CVaR (Series): {cvar_series:.4%}")

print("\n=== Testing with DataFrame ===")
var_df = qs.stats.var(returns_df)
cvar_df = qs.stats.cvar(returns_df)
print(f"VaR (DataFrame): {var_df:.4%}")
print(f"CVaR (DataFrame): {cvar_df:.4%}")

print("\n=== Testing in metrics report ===")
metrics = qs.reports.metrics(returns_series, mode='full', display=False)
# Print all available metrics to see the keys
print("Available metrics keys:")
for key in metrics.index[:10]:  # Show first 10 keys
    print(f"  - {key}")
if 'Daily Value-at-Risk' in metrics.index:
    print(f"VaR from metrics: {metrics.loc['Daily Value-at-Risk']}")
if 'Expected Shortfall (cVaR)' in metrics.index:
    print(f"CVaR from metrics: {metrics.loc['Expected Shortfall (cVaR)']}")

# Debug: Check what happens inside cvar calculation
print("\n=== Debug: Manual CVaR calculation ===")
var_threshold = qs.stats.value_at_risk(returns_series, sigma=1, confidence=0.95)
print(f"VaR threshold: {var_threshold:.4%}")

# For Series
below_var_series = returns_series[returns_series < var_threshold]
cvar_manual_series = below_var_series.mean()
print(f"Manual CVaR (Series): {cvar_manual_series:.4%}")

# For DataFrame - this is where the issue likely occurs
var_threshold_df = qs.stats.value_at_risk(returns_df, sigma=1, confidence=0.95)
print(f"VaR threshold (DataFrame): {var_threshold_df}")
below_var_df = returns_df[returns_df < var_threshold_df]
print(f"Shape of below_var_df: {below_var_df.shape}")
print(f"Type of below_var_df: {type(below_var_df)}")
print(f"below_var_df.values type: {type(below_var_df.values)}")
print(f"below_var_df.values shape: {below_var_df.values.shape}")
result = below_var_df.values.mean()
print(f"below_var_df.values.mean(): {result}")
print(f"Is result NaN? {np.isnan(result)}")

# The issue is that when filtering a DataFrame, the result has NaN values
print("\n=== The Issue ===")
print("When filtering DataFrame with condition returns_df < var_threshold_df,")
print("rows that don't meet the condition become NaN, not filtered out.")
print("This causes .values.mean() to return NaN.")
print("\nVerification:")
print(f"Number of non-NaN values in below_var_df: {below_var_df.count().values[0]}")
print(f"Correct CVaR calculation should be: {below_var_df.dropna().values.mean():.4%}")

print("\n=== ISSUE #467 CONFIRMED ===")
if np.isnan(result):
    print("The CVaR calculation is broken for DataFrames!")
    print("The issue is that the function uses .values.mean() on a DataFrame with NaN values,")
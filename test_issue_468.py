#!/usr/bin/env python3
"""Test script to reproduce Issue #468: mode.use_inf_as_null error"""

import quantstats as qs
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

print(f"Testing with quantstats version: {qs.__version__}")
print(f"Testing with pandas version: {pd.__version__}")
print(f"Testing with seaborn version: ")
import seaborn as sns
print(f"{sns.__version__}")

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
benchmark = pd.Series(np.random.randn(len(dates)) * 0.008, index=dates)

try:
    # Try to generate HTML report (this is where the error occurs)
    print("\nTrying to generate HTML report...")
    qs.reports.html(returns, benchmark, benchmark_title='SPY', output='/tmp/test_report.html', title='Test Report')
    print("SUCCESS: HTML report generated without error!")
except Exception as e:
    print(f"ERROR encountered: {type(e).__name__}: {e}")
    if "mode.use_inf_as_null" in str(e):
        print("\nISSUE #468 CONFIRMED: The 'mode.use_inf_as_null' error still occurs!")
    import traceback
    traceback.print_exc()
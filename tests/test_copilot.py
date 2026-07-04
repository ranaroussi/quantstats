import quantstats as qs

# Optional: add convenience methods to pandas Series
qs.extend_pandas()

# Example: download daily returns for a stock
stock = qs.utils.download_returns("META")

# Basic metrics
print("Sharpe:", qs.stats.sharpe(stock))
print("Sortino:", qs.stats.sortino(stock))
print("Max drawdown:", qs.stats.max_drawdown(stock))

# Plot performance
qs.plots.snapshot(stock, title="META Performance", show=True)

# Generate an HTML tear sheet
qs.reports.html(stock, "SPY", output="meta_report.html")
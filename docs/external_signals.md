# External Signal Examples

QuantStats works with return series, so external datasets should be transformed
into positions or strategy returns before they are passed into reports, plots, or
stats functions.

This keeps data fetching and signal logic outside QuantStats while preserving the
existing API:

```python
import pandas as pd
import quantstats as qs

asset_returns = qs.utils.download_returns("TSLA")
sentiment_signal = pd.Series(
    data=[0.15, 0.31, -0.08],
    index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
)

aligned_signal = sentiment_signal.reindex(asset_returns.index).ffill().fillna(0)
positions = aligned_signal.gt(0).astype(float)
strategy_returns = positions.shift(1).fillna(0) * asset_returns

qs.reports.metrics(strategy_returns, benchmark="SPY")
qs.reports.html(strategy_returns, "SPY", output="sentiment_strategy.html")
```

The signal source is intentionally external to QuantStats. It can be a local CSV,
a research factor, or an API response.

## Optional Adanos Sentiment Signal

[Adanos Market Sentiment API](https://api.adanos.org/docs/) is one optional
source for stock sentiment and buzz snapshots from Reddit, X/Twitter, and
Polymarket. A typical workflow is:

1. Fetch the sentiment or buzz score for the tickers you are testing.
2. Convert the sentiment into a position rule outside QuantStats.
3. Multiply lagged positions by asset returns.
4. Pass the resulting strategy returns into QuantStats.

Example sketch:

```python
import os
import pandas as pd
import requests
import quantstats as qs

api_key = os.environ["ADANOS_API_KEY"]
ticker = "TSLA"

response = requests.get(
    f"https://api.adanos.org/reddit/stocks/v1/stock/{ticker}",
    headers={"X-API-Key": api_key},
    params={"days": 30},
    timeout=10,
)
response.raise_for_status()
payload = response.json()

sentiment = pd.Series(
    data=[row["sentiment_score"] for row in payload["daily_trend"]],
    index=pd.to_datetime([row["date"] for row in payload["daily_trend"]]),
).dropna()

asset_returns = qs.utils.download_returns(ticker)
signal = sentiment.reindex(asset_returns.index).ffill().fillna(0)
positions = signal.gt(0).astype(float)
strategy_returns = positions.shift(1).fillna(0) * asset_returns

qs.reports.html(strategy_returns, "SPY", output="adanos_sentiment_strategy.html")
```

The API key remains optional and is only needed if you choose Adanos as the
external signal provider.

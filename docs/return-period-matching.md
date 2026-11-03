# Compare returns over matching periods

Comparing a portfolio with a benchmark can give misleading statistics when their
returns cover different periods. One series might include weekends while the
other reports only on trading days, or one might report daily and the other weekly.

Set `frequency` to combine returns into matching periods before comparison.
In this example, Saturday, Sunday and Monday returns combine into one
Friday-to-Monday return:

```python
import pandas as pd
import quantstats as qs

strategy = pd.Series(
    [0.0, 0.01, 0.02, -0.01, 0.03, -0.02],  # Fri, Sat, Sun, Mon, Tue, Wed
    index=pd.date_range("2026-01-09", periods=6),
)
benchmark = pd.Series(
    [0.0, 0.019898, 0.03, -0.02],  # Fri, Mon, Tue, Wed
    index=pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14"]),
)
result = qs.stats.r_squared(
    strategy, benchmark,
    prepare_returns=False,  # Both inputs already contain returns.
    frequency="daily",
)
print(round(result, 6))  # 1.0: identical returns over three compared periods
```

## Choose a frequency

The `frequency` parameter accepts `"daily"`, `"weekly"`, `"monthly"`,
`"quarterly"` or `"yearly"`. You can use this parameter with `greeks`,
`r_squared`, its alias `r2`, and `information_ratio`. It is also available
through their pandas methods.

Choose the frequency you want to compare. For instance, if you have weekly portfolio
returns and daily benchmark returns, you can use `frequency="weekly"`.

Weeks end on Sunday. Months, quarters and years end on their calendar boundaries.

The function raises an error if the data cannot support the frequency you request.

Omit `frequency` to keep existing behavior.

## Provide complete data

Provide the portfolio and benchmark as pandas Series, with dates as the index
and returns as the values. Put the dates in order and provide at most one
return per day.

Each return you provide must cover the entire interval since the preceding
observation. For Friday followed by Monday, Monday's return must include the
change from Friday through Monday. The first date establishes when measurement
begins for the second date. The return recorded on that first date is not included
in the calculations.

If a return is unavailable, mark it as `NaN`. Do not replace it with zero.
Zero means the investment's value did not change. The function raises an error
if a calculation needs that missing return.

You do not need to add rows for weekends without returns. To identify other
non-trading dates, provide a holiday list, such as `holidays=["2024-03-29"]`.
The list applies to both the portfolio and benchmark. Any returns that you do
provide on weekends or holidays, however, will still be used.

## Interpret the results

Beta, R-squared and information ratio are not annualized. Alpha is annualized
automatically, including when the return history is shorter than a year.
Weekly, monthly, quarterly and yearly comparisons use 52, 12, 4 and 1 periods
per year. For daily comparisons, the annualization factor is calculated from
the number of completed return periods and the time they cover, accounting for
leap years.

To choose the number of periods per year yourself, pass, for example,
`periods=252` to `greeks`. Use `periods=1` to leave alpha unannualized.
Insufficient data or an undefined estimate can produce `NaN`.

These options apply to the statistics above, not rolling calculations, reports
or plots.

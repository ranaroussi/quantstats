import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from src.quantstats.utils import (
    _mtd,
    _pandas_current_month,
    _pandas_date,
    _qtd,
    _ytd,
    download_returns,
    exponential_stdev,
    log_returns,
    multi_shift,
    to_excess_returns,
    to_prices,
    to_returns,
)


@pytest.fixture
def returns(resource_dir):
    return pd.read_csv(resource_dir / "meta.csv", parse_dates=True, index_col=0)["Close"]


@pytest.fixture
def today(monkeypatch):
    """Fixture that fakes datetime.datetime.now() to return a fixed 'today'."""
    fake_now = _dt.datetime(2025, 4, 24)

    class FakeDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fake_now

    monkeypatch.setattr(_dt, "datetime", FakeDateTime)

    return fake_now


def test_excess_returns(returns):
    r = to_excess_returns(returns, rf=0.01)
    print(r)


def test_mtd_filters_start_of_month(today):
    # Create a sample DataFrame
    idx = pd.to_datetime(["2025-03-30", "2025-04-01", "2025-04-15", "2025-04-20", "2025-05-01"])
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=idx)

    # Apply _mtd
    result = _mtd(df)

    # Expected: only dates from 2025-04-01 onwards
    expected_idx = pd.to_datetime(["2025-04-01", "2025-04-15", "2025-04-20"])
    expected_df = df.loc[expected_idx]

    pd.testing.assert_frame_equal(result, expected_df)


def test_qtd_filters_start_of_month(today):
    # Create a sample DataFrame
    idx = pd.to_datetime(["2025-03-30", "2025-04-01", "2025-04-15", "2025-04-20", "2025-05-01", "2025-09-01"])
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=idx)

    # Apply _mtd
    result = _qtd(df)

    # Expected: only dates from 2025-04-01 onwards
    expected_idx = pd.to_datetime(["2025-04-01", "2025-04-15", "2025-04-20", "2025-05-01"])
    expected_df = df.loc[expected_idx]

    pd.testing.assert_frame_equal(result, expected_df)


def test_ytd_filters_start_of_month(today):
    # Create a sample DataFrame
    idx = pd.to_datetime(["2024-03-30", "2024-04-01", "2025-04-15", "2025-04-20", "2025-05-01", "2025-09-01"])
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=idx)

    # Apply _mtd
    result = _ytd(df)

    # Expected: only dates from 2025-04-01 onwards
    expected_idx = pd.to_datetime(["2025-04-15", "2025-04-20", "2025-05-01", "2025-09-01"])
    expected_df = df.loc[expected_idx]

    pd.testing.assert_frame_equal(result, expected_df)


def test_to_prices(returns):
    prices = to_prices(returns, 100)
    rrr = to_returns(prices, rf=0)
    pd.testing.assert_series_equal(returns, rrr)


def test_log_returns(returns):
    lll = log_returns(returns)
    # compute exp(x) - 1 in a stable way
    a = np.expm1(lll)
    a[a.index[0]] = np.nan
    pd.testing.assert_series_equal(returns, a)


def test_pandas_current_month(today):
    idx = pd.to_datetime(["2024-03-30", "2024-04-01", "2025-04-15", "2025-04-20", "2025-05-01", "2025-09-01"])
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=idx)

    result = _pandas_current_month(df)

    expected_idx = pd.to_datetime(["2025-04-15", "2025-04-20"])
    expected_df = df.loc[expected_idx]

    pd.testing.assert_frame_equal(result, expected_df)


def test_pandas_date():
    idx = pd.to_datetime(["2024-03-30", "2024-04-01", "2025-04-15", "2025-04-20", "2025-05-01", "2025-09-01"])
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=idx)

    dates = [pd.Timestamp("2024-03-30"), pd.Timestamp("2024-04-01")]
    result = _pandas_date(df, dates=dates)

    expected_idx = pd.to_datetime(["2024-03-30", "2024-04-01"])
    expected_df = df.loc[expected_idx]

    pd.testing.assert_frame_equal(result, expected_df)


def test_exponential_stdev(returns):
    pd.testing.assert_series_equal(
        returns.dropna().ewm(span=20, min_periods=20).std(), exponential_stdev(returns.dropna(), window=20)
    )


def test_multi_shift(returns):
    multi_shift(returns, shift=3)


def test_download_returns(returns):
    download_returns(ticker="SPY")

"""
Tests for quantstats._compat module
"""

import pytest
import pandas as pd
import numpy as np

from quantstats._compat import (
    get_frequency_alias,
    normalize_timezone,
    safe_resample,
    safe_concat,
    safe_append,
)


@pytest.fixture
def sample_series():
    """Generate sample time series data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(np.random.randn(100), index=dates)


@pytest.fixture
def sample_series_tz():
    """Generate sample time series with timezone."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="US/Eastern")
    return pd.Series(np.random.randn(100), index=dates)


class TestFrequencyAlias:
    """Test frequency alias compatibility."""

    def test_monthly_alias(self):
        """Test monthly frequency alias."""
        result = get_frequency_alias("M")
        assert result in ["M", "ME"]  # Depends on pandas version

    def test_quarterly_alias(self):
        """Test quarterly frequency alias."""
        result = get_frequency_alias("Q")
        assert result in ["Q", "QE"]

    def test_yearly_alias(self):
        """Test yearly frequency alias."""
        result = get_frequency_alias("Y")
        assert result in ["Y", "YE"]

    def test_daily_unchanged(self):
        """Test daily frequency unchanged."""
        result = get_frequency_alias("D")
        assert result == "D"


class TestNormalizeTimezone:
    """Test timezone normalization."""

    def test_normalize_tz_aware(self, sample_series_tz):
        """Test normalizing timezone-aware series."""
        result = normalize_timezone(sample_series_tz)
        assert result.index.tz is None

    def test_normalize_tz_naive(self, sample_series):
        """Test that tz-naive series passes through."""
        result = normalize_timezone(sample_series)
        assert result.index.tz is None
        pd.testing.assert_series_equal(result, sample_series)


class TestSafeResample:
    """Test safe resample function."""

    def test_resample_sum(self, sample_series):
        """Test resample with sum aggregation."""
        result = safe_resample(sample_series, "M", "sum")
        assert len(result) < len(sample_series)

    def test_resample_mean(self, sample_series):
        """Test resample with mean aggregation."""
        result = safe_resample(sample_series, "M", "mean")
        assert len(result) < len(sample_series)

    def test_resample_no_func(self, sample_series):
        """Test resample without aggregation function."""
        result = safe_resample(sample_series, "M", None)
        # Should return resampler object
        assert hasattr(result, "sum")


class TestSafeConcat:
    """Test safe concatenation."""

    def test_concat_series(self, sample_series):
        """Test concatenating series."""
        s1 = sample_series[:50]
        s2 = sample_series[50:]
        result = safe_concat([s1, s2])
        assert len(result) == len(sample_series)

    def test_concat_dataframes(self, sample_series):
        """Test concatenating DataFrames."""
        df1 = pd.DataFrame({"A": sample_series[:50]})
        df2 = pd.DataFrame({"A": sample_series[50:]})
        result = safe_concat([df1, df2])
        assert len(result) == len(sample_series)

    def test_concat_axis1(self, sample_series):
        """Test concatenating along axis 1."""
        s1 = sample_series.copy()
        s1.name = "A"
        s2 = sample_series.copy()
        s2.name = "B"
        result = safe_concat([s1, s2], axis=1)
        assert result.shape[1] == 2


class TestSafeAppend:
    """Test safe append function."""

    def test_append_dataframes(self, sample_series):
        """Test appending DataFrames."""
        df1 = pd.DataFrame({"A": sample_series[:50]})
        df2 = pd.DataFrame({"A": sample_series[50:]})
        result = safe_append(df1, df2)
        assert len(result) == len(sample_series)

    def test_append_ignore_index(self, sample_series):
        """Test append with ignore_index."""
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [4, 5, 6]})
        result = safe_append(df1, df2, ignore_index=True)
        assert list(result.index) == [0, 1, 2, 3, 4, 5]

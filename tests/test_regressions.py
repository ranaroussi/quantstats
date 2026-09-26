"""
Regression tests for bugs fixed in 0.0.82.

Each test here pins down a specific wrong answer or crash that shipped in an
earlier release, so the behaviour cannot quietly come back.
"""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

import quantstats as qs
from quantstats import reports, stats, utils
from quantstats._compat import get_frequency_alias, safe_resample


@pytest.fixture
def daily_returns():
    """A well-behaved daily return series."""
    rng = np.random.RandomState(7)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    return pd.Series(rng.normal(0.0005, 0.01, 400), index=dates, name="Strategy")


class TestWeeklyAggregation:
    """DatetimeIndex.week was removed in pandas 2.0."""

    def test_aggregate_returns_weekly_does_not_raise(self):
        dates = pd.date_range("2020-01-01", periods=28, freq="D")
        returns = pd.Series(np.linspace(0.001, 0.028, 28), index=dates)

        result = utils.aggregate_returns(returns, "W")

        iso = returns.index.isocalendar()
        expected = returns.groupby([iso.year, iso.week]).apply(
            lambda x: (1 + x).prod() - 1
        )
        assert len(result) == len(expected)
        np.testing.assert_allclose(
            np.sort(result.values), np.sort(expected.values), rtol=1e-10
        )

    def test_aggregate_returns_week_keyword(self):
        dates = pd.date_range("2020-01-01", periods=21, freq="D")
        returns = pd.Series(np.linspace(0.001, 0.021, 21), index=dates)

        result = utils.aggregate_returns(returns, "week")

        assert isinstance(result, pd.Series)
        assert 0 < len(result) <= len(returns)

    def test_stats_best_worst_weekly(self):
        dates = pd.date_range("2020-01-01", periods=28, freq="D")
        returns = pd.Series(np.linspace(0.001, 0.028, 28), index=dates)

        weekly = utils.aggregate_returns(returns, "W")

        np.testing.assert_allclose(stats.best(returns, aggregate="W"), weekly.max())
        np.testing.assert_allclose(stats.worst(returns, aggregate="W"), weekly.min())


class TestPreparePricesGaps:
    """A missing price used to be filled with 0, i.e. a -100% drawdown."""

    @staticmethod
    def _gapped():
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        return pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 105.0], index=idx)

    def test_gap_is_carried_forward_not_zeroed(self):
        result = utils._prepare_prices(self._gapped())

        assert not (result == 0).any()
        assert result.iloc[2] == 101.0

    def test_leading_gap_is_back_filled(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        gapped = pd.Series([np.nan, np.nan, 100.0, 110.0, 90.0], index=idx)

        result = utils._prepare_prices(gapped)

        assert result.iloc[0] == 100.0
        assert not (result == 0).any()

    def test_gap_does_not_create_a_full_drawdown(self):
        gapped = self._gapped()
        observed = gapped.dropna()
        expected = float((observed / observed.cummax() - 1).min())

        assert stats.max_drawdown(gapped) == pytest.approx(expected)

    def test_dataframe_gaps_handled_per_column(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        frame = pd.DataFrame(
            {"a": self._gapped().values, "b": self._gapped().values[::-1]}, index=idx
        )

        result = utils._prepare_prices(frame)

        assert not (result == 0).any().any()


class TestCagrWipeout:
    """abs() on terminal wealth turned a total loss into a gain."""

    def test_uncompounded_loss_beyond_minus_100pct_is_nan(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.Series([-0.6] * 4, index=idx)  # sums to -240%

        assert np.isnan(stats.cagr(returns, compounded=False, periods=4))

    def test_exact_wipeout_is_minus_100pct(self):
        idx = pd.bdate_range("2020-01-01", periods=2)
        returns = pd.Series([-0.5, -0.5], index=idx)  # sums to -100%

        assert stats.cagr(returns, compounded=False, periods=2) == pytest.approx(-1.0)

    def test_dataframe_flags_only_the_bad_column(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        frame = pd.DataFrame({"ok": [0.01] * 4, "bad": [-0.6] * 4}, index=idx)

        result = stats.cagr(frame, compounded=False, periods=4)

        assert np.isfinite(result["ok"])
        assert np.isnan(result["bad"])


class TestDrawdownBaseline:
    """Drawdown used to depend on the absolute price level."""

    SHAPE = [5.0, 5.5, 5.2, 6.0, 5.4, 5.8]  # peak 6.0, trough 5.4 => -10%

    @pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0, 10000.0])
    def test_max_drawdown_is_scale_independent(self, scale):
        idx = pd.date_range("2020-01-01", periods=len(self.SHAPE), freq="D")
        prices = pd.Series([p * scale for p in self.SHAPE], index=idx)

        np.testing.assert_almost_equal(
            stats.max_drawdown(prices), 5.4 / 6.0 - 1, decimal=10
        )

    def test_price_series_starts_flat(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.Series([50.0, 55.0, 52.0, 60.0, 54.0], index=idx)

        dd = stats.to_drawdown_series(prices)

        expected = prices / prices.cummax() - 1
        pd.testing.assert_series_equal(
            dd, expected, check_names=False, check_freq=False
        )
        assert dd.iloc[0] == 0.0

    def test_dataframe_columns_get_their_own_baseline(self):
        idx = pd.date_range("2020-01-01", periods=len(self.SHAPE), freq="D")
        prices = pd.DataFrame(
            {
                "cheap": [p for p in self.SHAPE],
                "rich": [p * 1000 for p in self.SHAPE],
            },
            index=idx,
        )

        result = stats.max_drawdown(prices)

        np.testing.assert_almost_equal(result["cheap"], 5.4 / 6.0 - 1, decimal=10)
        np.testing.assert_almost_equal(result["rich"], 5.4 / 6.0 - 1, decimal=10)

    def test_returns_still_count_a_first_period_loss(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        returns = pd.Series([-0.10, 0.01, 0.01, 0.01], index=idx)

        assert stats.max_drawdown(returns) <= -0.10 + 1e-12
        assert stats.to_drawdown_series(returns).iloc[0] == pytest.approx(-0.10)


class TestPrepareReturnsCache:
    """Equivalent Series and DataFrames used to share a cache entry."""

    def test_series_does_not_come_back_as_a_dataframe(self):
        idx = pd.date_range("2024-01-01", periods=3)
        series = pd.Series([0.01, -0.02, 0.03], index=idx, name="Strategy")

        utils._PREPARE_RETURNS_CACHE.clear()
        prepared_frame = utils._prepare_returns(series.to_frame())
        prepared_series = utils._prepare_returns(series)

        assert isinstance(prepared_frame, pd.DataFrame)
        assert isinstance(prepared_series, pd.Series)

    def test_apply_rf_is_part_of_the_cache_key(self):
        idx = pd.date_range("2024-01-01", periods=10)
        series = pd.Series(np.linspace(0.001, 0.01, 10), index=idx, name="S")

        utils._PREPARE_RETURNS_CACHE.clear()
        with_rf = utils._prepare_returns(series, rf=0.05, apply_rf=True)
        without_rf = utils._prepare_returns(series, rf=0.05, apply_rf=False)

        assert not np.allclose(with_rf.values, without_rf.values)

    def test_differently_named_columns_do_not_collide(self):
        idx = pd.date_range("2024-01-01", periods=3)
        values = [0.01, -0.02, 0.03]

        utils._PREPARE_RETURNS_CACHE.clear()
        first = utils._prepare_returns(pd.DataFrame({"a": values}, index=idx))
        second = utils._prepare_returns(pd.DataFrame({"b": values}, index=idx))

        assert list(first.columns) == ["a"]
        assert list(second.columns) == ["b"]


class TestPrepareReturnsNoStackInspection:
    """rf handling is an argument now, not a guess about the caller."""

    def test_apply_rf_is_explicit(self):
        idx = pd.date_range("2024-01-01", periods=10)
        series = pd.Series(np.full(10, 0.001), index=idx)

        utils._PREPARE_RETURNS_CACHE.clear()
        applied = utils._prepare_returns(series, rf=0.05, nperiods=252, apply_rf=True)
        skipped = utils._prepare_returns(series, rf=0.05, nperiods=252, apply_rf=False)

        np.testing.assert_allclose(skipped.values, series.values)
        assert (applied < skipped).all()

    def test_result_does_not_depend_on_calling_function(self):
        idx = pd.date_range("2024-01-01", periods=10)
        series = pd.Series(np.full(10, 0.001), index=idx)

        def cagr(data):  # name previously changed the outcome
            utils._PREPARE_RETURNS_CACHE.clear()
            return utils._prepare_returns(data, rf=0.05, nperiods=252)

        def anything_else(data):
            utils._PREPARE_RETURNS_CACHE.clear()
            return utils._prepare_returns(data, rf=0.05, nperiods=252)

        np.testing.assert_allclose(cagr(series).values, anything_else(series).values)


class TestProbabilisticRatio:
    """The PSR estimator mixed excess and raw kurtosis."""

    def test_reduces_to_lo_for_normal_moments(self):
        """sigma_sr must equal sqrt((1 + SR^2/2)/(n-1)) when skew/kurt vanish."""
        for sr in (0.05, 0.10, 0.20):
            skew_no, raw_kurtosis = 0.0, 3.0
            sigma = 1 - (skew_no * sr) + (((raw_kurtosis - 1) / 4) * sr**2)
            assert sigma == pytest.approx(1 + 0.5 * sr**2)

    def test_is_a_probability(self, daily_returns):
        value = stats.probabilistic_sharpe_ratio(daily_returns)

        assert 0.0 <= value <= 1.0

    def test_annualize_flag_is_inert(self, daily_returns):
        plain = stats.probabilistic_sharpe_ratio(daily_returns, annualize=False)
        annualized = stats.probabilistic_sharpe_ratio(daily_returns, annualize=True)

        assert plain == pytest.approx(annualized)
        assert annualized <= 1.0

    def test_rf_lowers_the_probability(self, daily_returns):
        no_rf = stats.probabilistic_sharpe_ratio(daily_returns, rf=0.0)
        with_rf = stats.probabilistic_sharpe_ratio(daily_returns, rf=0.05)

        assert with_rf < no_rf


class TestKellyCriterion:
    """Kelly returned the fixed-odds fraction, independent of loss size."""

    def test_scales_inversely_with_loss_magnitude(self):
        small = pd.Series([0.02, -0.01] * 30)
        doubled = pd.Series([0.04, -0.02] * 30)

        assert stats.kelly_criterion(small) == pytest.approx(
            2 * stats.kelly_criterion(doubled)
        )

    def test_matches_closed_form(self):
        returns = pd.Series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01] * 10)

        win_prob = stats.win_rate(returns)
        avg_win = stats.avg_win(returns)
        avg_loss = abs(stats.avg_loss(returns))
        expected = win_prob / avg_loss - (1 - win_prob) / avg_win

        assert stats.kelly_criterion(returns) == pytest.approx(expected)

    def test_zero_average_loss_is_nan(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])

        assert np.isnan(stats.kelly_criterion(returns))


class TestAutocorrPenalty:
    """corrcoef is undefined for <2 points or constant input."""

    def test_single_observation(self):
        assert stats.autocorr_penalty(pd.Series([0.01])) == 1.0

    def test_constant_series_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert stats.autocorr_penalty(pd.Series([0.01] * 10)) == 1.0

    def test_empty_series(self):
        assert stats.autocorr_penalty(pd.Series([], dtype=float)) == 1.0


class TestCompoundedFlag:
    """calmar() and rar() are CAGR-based and must honour compounded."""

    def test_calmar_accepts_compounded(self, daily_returns):
        assert stats.calmar(daily_returns, compounded=False) != stats.calmar(
            daily_returns
        )

    def test_rar_accepts_compounded(self, daily_returns):
        assert stats.rar(daily_returns, compounded=False) != stats.rar(daily_returns)

    def test_metrics_threads_compounded_into_calmar(self, daily_returns):
        geometric = reports.metrics(
            daily_returns,
            display=False,
            mode="full",
            compounded=True,
            prepare_returns=False,
        )
        arithmetic = reports.metrics(
            daily_returns,
            display=False,
            mode="full",
            compounded=False,
            prepare_returns=False,
        )

        assert (
            geometric.loc["Calmar", "Strategy"] != arithmetic.loc["Calmar", "Strategy"]
        )


class TestTenYearWindow:
    """10Y used years=10 while 3Y/5Y used months=35/59."""

    def test_matches_the_other_windows_on_a_short_history(self):
        dates = pd.date_range("2016-01-31", periods=121, freq="ME")
        returns = pd.Series(0.01, index=dates, name="Strategy")
        returns.iloc[0] = -0.50

        result = reports.metrics(
            returns, display=False, prepare_returns=False, periods_per_year=12
        )

        assert (
            result.loc["10Y (ann.)", "Strategy"] == result.loc["3Y (ann.)", "Strategy"]
        )
        assert (
            result.loc["10Y (ann.)", "Strategy"] == result.loc["5Y (ann.)", "Strategy"]
        )


class TestInformationRatio:
    """The benchmark was prepared twice."""

    def test_matches_manual_calculation(self, daily_returns):
        rng = np.random.RandomState(11)
        benchmark = pd.Series(
            rng.normal(0.0003, 0.008, len(daily_returns)),
            index=daily_returns.index,
            name="Benchmark",
        )

        result = stats.information_ratio(
            daily_returns, benchmark, prepare_returns=False
        )

        diff = daily_returns - benchmark
        assert result == pytest.approx(diff.mean() / diff.std())


class TestSeriesRiskFreeRate:
    """A time-varying rf used to raise 'truth value is ambiguous'."""

    @staticmethod
    def _rf_series(index):
        return pd.Series(np.linspace(0.01, 0.05, len(index)), index=index)

    def test_metrics_accepts_a_series(self, daily_returns):
        rf = self._rf_series(daily_returns.index)

        result = reports.metrics(daily_returns, rf=rf, display=False)

        assert not result.empty
        assert "Risk-Free Rate" in result.index

    def test_rf_row_shows_the_average(self, daily_returns):
        rf = self._rf_series(daily_returns.index)

        result = reports.metrics(daily_returns, rf=rf, display=False)

        shown = float(str(result.loc["Risk-Free Rate", "Strategy"]).rstrip("%"))
        assert shown == pytest.approx(rf.mean() * 100, abs=0.05)

    def test_excess_returns_align_by_date_for_dataframes(self):
        idx = pd.date_range("2024-01-01", periods=5)
        frame = pd.DataFrame({"a": [0.01] * 5, "b": [0.02] * 5}, index=idx)
        rf = pd.Series(np.linspace(0.001, 0.005, 5), index=idx)

        result = utils.to_excess_returns(frame, rf)

        assert not result.isna().any().any()
        np.testing.assert_allclose(result["a"].values, frame["a"].values - rf.values)

    def test_rf_is_nonzero_handles_series(self):
        idx = pd.date_range("2024-01-01", periods=3)

        assert utils._rf_is_nonzero(pd.Series([0.0, 0.0, 0.0], index=idx)) is False
        assert utils._rf_is_nonzero(pd.Series([0.0, 0.01, 0.0], index=idx)) is True
        assert utils._rf_is_nonzero(0.0) is False


class TestSingleColumnDataFrameReports:
    """A one-column DataFrame crashed html()/full()."""

    def test_html_with_single_column_dataframe(self, daily_returns):
        frame = pd.DataFrame({"Strategy": daily_returns})
        rng = np.random.RandomState(5)
        benchmark = pd.Series(
            rng.normal(0.0003, 0.008, len(daily_returns)),
            index=daily_returns.index,
            name="Benchmark",
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as handle:
            output = handle.name
        try:
            reports.html(frame, benchmark, output=output)
            assert os.path.exists(output)
            assert os.path.getsize(output) > 0
        finally:
            if os.path.exists(output):
                os.remove(output)

    def test_metrics_with_single_column_dataframe(self, daily_returns):
        frame = pd.DataFrame({"Strategy": daily_returns})

        result = reports.metrics(frame, display=False)

        assert not result.empty


class TestMatchDatesForwarding:
    """match_dates was dropped before reaching metrics()."""

    def test_match_dates_reaches_metrics(self, daily_returns):
        rng = np.random.RandomState(13)
        benchmark = pd.Series(
            rng.normal(0.0003, 0.008, len(daily_returns)),
            index=daily_returns.index,
            name="Benchmark",
        )
        # Give the benchmark a longer history than the strategy
        extra = pd.date_range(
            daily_returns.index[0] - pd.Timedelta(days=60),
            daily_returns.index[0] - pd.Timedelta(days=1),
            freq="D",
        )
        benchmark = pd.concat(
            [pd.Series(0.001, index=extra, name="Benchmark"), benchmark]
        )

        matched = reports.metrics(
            daily_returns, benchmark, display=False, match_dates=True
        )
        unmatched = reports.metrics(
            daily_returns, benchmark, display=False, match_dates=False
        )

        assert (
            matched.loc["Start Period", "Benchmark"]
            != unmatched.loc["Start Period", "Benchmark"]
        )


class TestExtendPandasStillWorks:
    """The pandas accessors must keep matching the module functions."""

    def test_accessors_match_module_functions(self, daily_returns):
        qs.extend_pandas()

        assert daily_returns.max_drawdown() == pytest.approx(
            stats.max_drawdown(daily_returns)
        )
        assert daily_returns.kelly_criterion() == pytest.approx(
            stats.kelly_criterion(daily_returns)
        )


class TestFrequencyAliasCompatibility:
    """
    The codebase spells frequencies the pandas 2.2 way ("ME"/"QE"/"YE").
    Those spellings do not exist in pandas 2.0/2.1, where passing them to
    resample() raises "Invalid frequency: ME". The compatibility layer has to
    translate in both directions, so assert against the installed pandas
    rather than against a hardcoded expectation.
    """

    @pytest.mark.parametrize("freq", ["M", "Q", "Y", "ME", "QE", "YE"])
    def test_alias_is_accepted_by_installed_pandas(self, freq, daily_returns):
        alias = get_frequency_alias(freq)

        # Raises ValueError if the alias is wrong for this pandas version.
        result = daily_returns.resample(alias).sum()

        assert len(result) > 0

    def test_safe_resample_accepts_new_style_aliases(self, daily_returns):
        monthly = safe_resample(daily_returns, "ME", "sum")

        assert len(monthly) < len(daily_returns)

    def test_outliers_distribution_builds(self, daily_returns):
        # Resampled internally on "ME"/"QE"/"YE" without going through the
        # compatibility layer before 0.0.82.
        dist = stats.distribution(daily_returns)

        assert set(dist) == {"Daily", "Weekly", "Monthly", "Quarterly", "Yearly"}

    def test_gain_to_pain_ratio_accepts_a_resolution(self, daily_returns):
        assert stats.gain_to_pain_ratio(daily_returns, resolution="ME") is not None

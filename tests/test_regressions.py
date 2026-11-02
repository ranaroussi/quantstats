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

    def test_rf_series_differing_mid_sample_do_not_collide(self):
        # The key formatted rf with its repr, which pandas truncates to the
        # first and last rows, so two rate series differing only in between
        # shared an entry and the result depended on call order (#555).
        idx = pd.date_range("2024-01-01", periods=100)
        series = pd.Series(np.full(100, 0.001), index=idx, name="S")
        low = pd.Series(0.01, index=idx)
        high = low.copy()
        high.iloc[10:-10] = 0.05  # same first and last rows as `low`

        assert str(low) == str(high)  # the collision a repr-based key cannot see

        utils._PREPARE_RETURNS_CACHE.clear()
        expected = utils._prepare_returns(series, rf=high, nperiods=252)

        utils._PREPARE_RETURNS_CACHE.clear()
        utils._prepare_returns(series, rf=low, nperiods=252)
        result = utils._prepare_returns(series, rf=high, nperiods=252)

        np.testing.assert_allclose(result.values, expected.values)


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
    """Kelly must return the classic fixed-odds fraction p - q/b.

    0.0.82-0.0.83 divided it by the average-loss magnitude, turning a
    capital fraction into a per-period leverage: reports showed figures like
    3716% where this reads 21% (issue #552). Reverted in 0.0.84.
    """

    def test_scale_invariant_by_construction(self):
        small = pd.Series([0.02, -0.01] * 30)
        doubled = pd.Series([0.04, -0.02] * 30)

        # The fixed-odds fraction depends only on the odds ratio, so
        # doubling the magnitude of every return must not move it.
        assert stats.kelly_criterion(small) == pytest.approx(
            stats.kelly_criterion(doubled)
        )

    def test_matches_closed_form(self):
        returns = pd.Series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01] * 10)

        win_prob = stats.win_rate(returns)
        payoff = stats.payoff_ratio(returns)
        expected = win_prob - (1 - win_prob) / payoff

        assert stats.kelly_criterion(returns) == pytest.approx(expected)

    def test_zero_average_loss_is_nan(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])

        assert np.isnan(stats.kelly_criterion(returns))


class TestBenchmarkGaps:
    """A gap must not destroy joint estimators (#553, PR by WatchTree-19).

    0.0.83 stopped filling gaps with 0, which let NaN propagate through
    np.cov/linregress: a single missing day reduced beta and alpha to 0 and
    R-squared to NaN. Fixed in 0.0.84 by estimating over the dates on which
    both series were observed.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-01", periods=756)
        bench = pd.Series(rng.normal(0.0003, 0.01, 756), index=idx)
        strat = 0.8 * bench + rng.normal(0, 0.006, 756)
        gapped = strat.copy()
        gapped.iloc[[100, 200, 300]] = np.nan
        return strat, gapped, bench

    def test_greeks_survive_gaps(self):
        strat, gapped, bench = self._data()

        beta_gap = stats.greeks(gapped, bench)["beta"]
        beta_full = stats.greeks(strat, bench)["beta"]

        assert beta_gap == pytest.approx(beta_full, abs=5e-3)

    def test_greeks_match_hand_computed_pairwise_estimate(self):
        _, gapped, bench = self._data()

        paired = pd.DataFrame({"r": gapped, "b": bench}).dropna()
        matrix = np.cov(paired["r"], paired["b"])
        expected_beta = matrix[0, 1] / matrix[1, 1]

        result = stats.greeks(gapped, bench)

        assert result["beta"] == pytest.approx(expected_beta)
        assert result["alpha"] == pytest.approx(
            (paired["r"].mean() - expected_beta * paired["b"].mean()) * 252
        )

    def test_nan_gap_and_absent_date_are_different_questions(self):
        # Dropping the dates instead of leaving them NaN makes the strategy
        # look like an irregular-frequency series, so _prepare_benchmark
        # compounds the benchmark across each missing day to match. Leaving
        # them NaN keeps a daily series with three unobserved days, and the
        # benchmark's own move on those days is simply discarded. Both are
        # defensible; they are not the same estimate, and that is deliberate.
        _, gapped, bench = self._data()

        assert stats.greeks(gapped, bench)["beta"] != pytest.approx(
            stats.greeks(gapped.dropna(), bench)["beta"]
        )

    def test_r_squared_survives_gaps(self):
        strat, gapped, bench = self._data()

        r2_gap = stats.r_squared(gapped, bench)

        assert np.isfinite(r2_gap)
        assert r2_gap == pytest.approx(stats.r_squared(strat, bench), abs=0.05)

    def test_r_squared_matches_hand_computed_pairwise_estimate(self):
        _, gapped, bench = self._data()

        paired = pd.DataFrame({"r": gapped, "b": bench}).dropna()
        expected = paired["r"].corr(paired["b"]) ** 2

        assert stats.r_squared(gapped, bench) == pytest.approx(expected)

    def test_treynor_survives_gaps(self):
        _, gapped, bench = self._data()

        treynor_gap = stats.treynor_ratio(gapped, bench)

        assert np.isfinite(treynor_gap)
        # 0.0.83 returned the beta-is-zero fallback here.
        assert treynor_gap != 0.0


class TestCausalBenchmarkAlignment:
    """A benchmark observation must not move to an earlier strategy date."""

    @staticmethod
    def _weekend_gap():
        """Model a daily strategy spanning a business-day benchmark gap."""
        period = pd.date_range("2024-01-05", "2024-01-08", freq="D")
        benchmark = pd.Series(
            [0.0, 0.10],
            index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
            name="Benchmark",
        )
        expected = pd.Series([0.0, 0.0, 0.0, 0.10], index=period, name="Benchmark")
        return period, benchmark, expected

    def test_sparse_return_stays_on_its_observation_date(self):
        period, benchmark, expected = self._weekend_gap()

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        # Fill direction is independent of return sign.
        negative_benchmark = benchmark.copy()
        negative_benchmark.iloc[-1] = -0.10
        negative_expected = expected.copy()
        negative_expected.iloc[-1] = -0.10
        pd.testing.assert_series_equal(
            utils._prepare_benchmark(negative_benchmark, period, prepare_returns=False),
            negative_expected,
            rtol=1e-12,
            atol=1e-12,
        )
        # An already aligned benchmark bypasses calendar reconstruction.
        pd.testing.assert_series_equal(
            utils._prepare_benchmark(benchmark, benchmark.index, prepare_returns=False),
            benchmark,
        )

    def test_leading_dates_stay_flat_until_the_next_observed_return(self):
        period = pd.date_range("2024-01-03", "2024-01-09", freq="D")
        benchmark = pd.Series(
            [0.10, 0.20],
            index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
            name="Benchmark",
        )
        # The initial 10% has no preceding price from which to reconstruct a
        # return; the later 20% must remain on January 8, with trailing dates flat.
        expected = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.0],
            index=period,
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    def test_irregular_period_compounds_intervening_benchmark_returns(self):
        period = pd.to_datetime(["2024-01-01", "2024-01-03"])
        benchmark = pd.Series(
            [0.0, 0.10, 0.20],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
            name="Benchmark",
        )
        # The two intervening returns compound: 1.10 * 1.20 - 1 == 0.32.
        expected = pd.Series([0.0, 0.32], index=period, name="Benchmark")

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    def test_timezone_dataframe_alignment_preserves_the_input(self):
        period, benchmark, expected = self._weekend_gap()
        period = period.tz_localize("US/Eastern")
        benchmark = benchmark.tz_localize("US/Eastern").to_frame()
        original = benchmark.copy(deep=True)
        # Preserve the established contract: Eastern midnight becomes 05:00
        # after conversion to UTC and removal of timezone metadata.
        expected.index = period.tz_convert("UTC").tz_localize(None)
        expected.index.freq = None

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        pd.testing.assert_frame_equal(benchmark, original)

    def test_information_ratio_matches_the_causal_benchmark(self):
        period, benchmark, expected_benchmark = self._weekend_gap()
        returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=period)
        active_returns = returns - expected_benchmark
        expected = active_returns.mean() / active_returns.std()

        result = stats.information_ratio(returns, benchmark, prepare_returns=False)

        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("strategy_hour", "benchmark_hour", "strategy_tz", "benchmark_tz"),
        [
            (0, 5, None, None),
            (5, 0, None, None),
            (0, 0, "UTC", "US/Eastern"),
            (0, 0, "US/Eastern", "UTC"),
        ],
    )
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_daily_offsets_preserve_same_date_returns(
        self, strategy_hour, benchmark_hour, strategy_tz, benchmark_tz, as_frame
    ):
        period = pd.date_range("2024-01-02", periods=4, freq="D", tz=strategy_tz)
        period += pd.Timedelta(hours=strategy_hour)
        benchmark_index = pd.date_range(
            "2024-01-02", periods=4, freq="D", tz=benchmark_tz
        ) + pd.Timedelta(hours=benchmark_hour)
        benchmark = pd.Series(
            [0.01, 0.02, 0.03, 0.04], index=benchmark_index, name="Benchmark"
        )
        if as_frame:
            benchmark = benchmark.to_frame()
        original = benchmark.copy(deep=True)
        output_index = period
        if period.tz is not None:
            output_index = period.tz_convert("UTC").tz_localize(None)
        # Reconstructing from prices retains the initial zero; each later
        # observation belongs to its own day, regardless of the label's hour.
        expected = pd.Series(
            [0.0, 0.02, 0.03, 0.04], index=output_index, name="Benchmark"
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        returns = pd.Series([0.0, 0.03, 0.04, 0.05], index=period)
        original_returns = returns.copy(deep=True)
        # Active returns [0, .01, .01, .01] have mean .0075 and sample
        # standard deviation .005, independently giving an information ratio 1.5.
        assert stats.information_ratio(returns, benchmark) == pytest.approx(1.5)
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original)
        else:
            pd.testing.assert_series_equal(benchmark, original)

    @pytest.mark.parametrize(("strategy_hour", "benchmark_hour"), [(0, 5), (5, 0)])
    def test_offset_weekend_gap_keeps_the_return_on_monday(
        self, strategy_hour, benchmark_hour
    ):
        period, benchmark, expected = self._weekend_gap()
        period += pd.Timedelta(hours=strategy_hour)
        benchmark.index += pd.Timedelta(hours=benchmark_hour)
        expected.index = period

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(("strategy_hour", "benchmark_hour"), [(0, 5), (5, 0)])
    def test_offset_period_compounds_intervening_returns(
        self, strategy_hour, benchmark_hour
    ):
        period = pd.to_datetime(["2024-01-01", "2024-01-03"])
        period += pd.Timedelta(hours=strategy_hour)
        benchmark = pd.Series(
            [0.0, 0.10, 0.20],
            index=pd.date_range("2024-01-01", periods=3, freq="D")
            + pd.Timedelta(hours=benchmark_hour),
            name="Benchmark",
        )
        # January 3 includes both intervening returns: 1.10 * 1.20 - 1.
        expected = pd.Series([0.0, 0.32], index=period, name="Benchmark")

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    def test_daily_labels_use_the_existing_utc_timezone_basis(self):
        period = pd.date_range("2024-01-03", periods=3, freq="D", tz="UTC")
        benchmark = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.date_range(
                "2024-01-02 23:00", periods=3, freq="D", tz="US/Eastern"
            ),
            name="Benchmark",
        )
        original = benchmark.copy(deep=True)
        # Eastern 23:00 belongs to the following UTC day. Matching local
        # calendar dates would move these observations to the preceding day.
        expected = pd.Series(
            [0.0, 0.02, 0.03],
            index=period.tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        pd.testing.assert_series_equal(benchmark, original)

    @pytest.mark.parametrize("timezone", ["Europe/London", "Asia/Tokyo"])
    @pytest.mark.parametrize(("strategy_hour", "benchmark_hour"), [(16.5, 0), (0, 5)])
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_positive_offset_daily_labels_preserve_the_calendar_day(
        self, timezone, strategy_hour, benchmark_hour, as_frame
    ):
        period = pd.date_range("2024-07-01", periods=3, freq="D", tz=timezone)
        period += pd.Timedelta(hours=strategy_hour)
        benchmark = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.date_range("2024-07-01", periods=3, freq="D", tz=timezone)
            + pd.Timedelta(hours=benchmark_hour),
            name="Benchmark",
        )
        if as_frame:
            benchmark = benchmark.to_frame()
        original = benchmark.copy(deep=True)
        expected = pd.Series(
            [0.0, 0.02, 0.03],
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        returns = pd.Series([0.0, 0.03, 0.04], index=period)
        original_returns = returns.copy(deep=True)
        # Active returns [0, .01, .01] have mean 1/150 and sample standard
        # deviation .01 / sqrt(3), giving an independent IR of 2 / sqrt(3).
        assert stats.information_ratio(returns, benchmark) == pytest.approx(
            2 / np.sqrt(3)
        )
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original)
        else:
            pd.testing.assert_series_equal(benchmark, original)
        # A future day's changed return must not affect the preceding days.
        future_changed = benchmark.copy(deep=True)
        future_changed.iloc[-1] = 0.90
        future_result = utils._prepare_benchmark(
            future_changed, period, prepare_returns=False
        )
        pd.testing.assert_series_equal(
            future_result.iloc[:-1], expected.iloc[:-1], rtol=1e-12, atol=1e-12
        )

    @pytest.mark.parametrize("timezone", ["Europe/London", "Asia/Tokyo"])
    def test_positive_offset_weekend_gap_keeps_the_return_on_monday(self, timezone):
        period = pd.date_range("2024-07-05 16:30", periods=4, freq="D", tz=timezone)
        benchmark = pd.Series(
            [0.0, 0.10],
            index=pd.to_datetime(["2024-07-05", "2024-07-08"]).tz_localize(timezone),
            name="Benchmark",
        )
        expected = pd.Series(
            [0.0, 0.0, 0.0, 0.10],
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    def test_positive_offset_daily_alignment_across_dst(self):
        period = pd.date_range(
            "2024-03-30 16:30", periods=4, freq="D", tz="Europe/London"
        )
        benchmark = pd.Series(
            [0.01, 0.02, 0.03, 0.04],
            index=pd.date_range("2024-03-30", periods=4, freq="D", tz="Europe/London"),
            name="Benchmark",
        )
        # Consecutive local midnights can share a UTC date across spring DST.
        # The public caller also removes the strategy timezone before alignment.
        expected = pd.Series(
            [0.0, 0.02, 0.03, 0.04],
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        for strategy_index in (period, expected.index):
            result = utils._prepare_benchmark(
                benchmark, strategy_index, prepare_returns=False
            )
            pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        returns = pd.Series([0.0, 0.03, 0.04, 0.05], index=period)
        assert stats.information_ratio(returns, benchmark) == pytest.approx(1.5)

    @pytest.mark.parametrize("timezone", ["Europe/London", "Asia/Tokyo"])
    def test_positive_offset_irregular_period_compounds_returns(self, timezone):
        period = pd.to_datetime(["2024-07-01 16:30", "2024-07-03 16:30"])
        period = period.tz_localize(timezone)
        benchmark = pd.Series(
            [0.0, 0.10, 0.20],
            index=pd.date_range("2024-07-01", periods=3, freq="D", tz=timezone),
            name="Benchmark",
        )
        # The two intervening local-day returns compound: 1.10 * 1.20 - 1.
        expected = pd.Series(
            [0.0, 0.32],
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        ("start", "expected_returns"),
        [("2024-07-01", [0.0, 0.32, 0.82]), ("2024-03-30", [0.0, 0.32, 0.30])],
    )
    def test_positive_offset_intraday_benchmark_keeps_timestamp_alignment(
        self, start, expected_returns
    ):
        period = pd.date_range(start, periods=3, freq="D", tz="Europe/London")
        benchmark = pd.Series(
            [0.0, 0.10, 0.20, 0.30, 0.40],
            index=pd.date_range(start, periods=5, freq="12h", tz="Europe/London"),
            name="Benchmark",
        )
        # Multiple local observations must bypass the daily calendar guard.
        # The last observation falls after the final midnight across spring DST,
        # so only 1.10 * 1.20 and then 1.30 are observed on the sampling grid.
        expected = pd.Series(
            expected_returns,
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    def test_positive_offset_intraday_strategy_keeps_timestamp_alignment(self):
        period = pd.date_range("2024-03-30", periods=5, freq="12h", tz="Europe/London")
        benchmark = pd.Series(
            [0.0, 0.10, 0.20],
            index=pd.date_range("2024-03-30", periods=3, freq="D", tz="Europe/London"),
            name="Benchmark",
        )
        # Multiple strategy observations must bypass the daily calendar guard.
        # The established daily grid leaves intervening observations missing,
        # so pct_change(fill_method=None) cannot reconstruct these returns.
        expected = pd.Series(
            0.0, index=period.tz_convert("UTC").tz_localize(None), name="Benchmark"
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        ("timezone", "expected_returns"),
        [(None, [0.0, 0.32, 0.82]), ("US/Eastern", [0.0, 0.32, 0.30])],
    )
    def test_intraday_benchmark_observations_keep_timestamp_alignment(
        self, timezone, expected_returns
    ):
        period = pd.date_range("2024-03-09", periods=3, freq="D", tz=timezone)
        benchmark = pd.Series(
            [0.0, 0.10, 0.20, 0.30, 0.40],
            index=pd.date_range("2024-03-09", periods=5, freq="12h", tz=timezone),
            name="Benchmark",
        )
        # Keep distinct observations within a day. Midnight prices include
        # 1.10 * 1.20. Across Eastern DST, the last 12-hour observation falls
        # after midnight, so only 1.30 is observed at the final strategy label.
        output_index = period
        if timezone is not None:
            output_index = period.tz_convert("UTC").tz_localize(None)
        expected = pd.Series(expected_returns, index=output_index, name="Benchmark")

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("timezone", ["Europe/London", "Asia/Tokyo"])
    @pytest.mark.parametrize("intraday_owner", ["strategy", "benchmark"])
    @pytest.mark.parametrize("prepared", [False, True])
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_native_intraday_observations_keep_timestamp_sampling(
        self, timezone, intraday_owner, prepared, as_frame
    ):
        if intraday_owner == "strategy":
            period = pd.DatetimeIndex(
                ["2024-07-01 00:00", "2024-07-01 12:00"]
            ).tz_localize(timezone)
            benchmark_index = pd.date_range("2024-07-01", periods=2, tz=timezone)
            # Both strategy rows precede the second benchmark observation.
            # The daily sampling grid omits noon, retaining its existing zero.
            expected_returns = [0.0, 0.0]
        else:
            period = pd.date_range("2024-07-01 09:00", periods=2, tz=timezone)
            benchmark_index = pd.DatetimeIndex(
                ["2024-07-01 00:00", "2024-07-01 12:00"]
            ).tz_localize(timezone)
            # At 09:00 only the midnight price is observed; the noon gain is
            # included at the following day's 09:00 sampling point.
            expected_returns = [0.0, 0.10]
        benchmark = pd.Series([0.0, 0.10], index=benchmark_index, name="Benchmark")
        if as_frame:
            benchmark = benchmark.to_frame()
        original_benchmark = benchmark.copy(deep=True)
        output_index = period.tz_convert("UTC").tz_localize(None)
        expected = pd.Series(expected_returns, index=output_index, name="Benchmark")

        result = utils._prepare_benchmark(
            benchmark,
            output_index if prepared else period,
            prepare_returns=False,
            strategy_tz=period.tz if prepared else None,
        )

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        changed = benchmark.copy(deep=True)
        changed.iloc[-1] = 0.90
        changed_result = utils._prepare_benchmark(
            changed,
            output_index if prepared else period,
            prepare_returns=False,
            strategy_tz=period.tz if prepared else None,
        )
        # The later benchmark observation remains invisible before its sample.
        pd.testing.assert_series_equal(changed_result.iloc[:1], expected.iloc[:1])
        assert changed_result.iloc[-1] == pytest.approx(
            0.0 if intraday_owner == "strategy" else 0.90
        )
        returns = pd.Series([0.02, 0.04], index=period, name="Strategy")
        original_returns = returns.copy(deep=True)
        active = np.array([0.02, 0.04]) - np.array(expected_returns)
        expected_ir = active.mean() / active.std(ddof=1)
        assert stats.information_ratio(returns, benchmark) == pytest.approx(expected_ir)
        qs.extend_pandas()
        assert returns.information_ratio(benchmark) == pytest.approx(expected_ir)
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original_benchmark)
        else:
            pd.testing.assert_series_equal(benchmark, original_benchmark)

    @pytest.mark.parametrize(
        ("timezone", "start"),
        [("America/Sao_Paulo", "2018-11-03"), ("America/Havana", "2024-11-02")],
    )
    @pytest.mark.parametrize("prepared", [False, True])
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_daily_alignment_does_not_construct_local_dst_midnight(
        self, timezone, start, prepared, as_frame
    ):
        benchmark_index = pd.date_range(start + " 12:00", periods=3, tz=timezone)
        period = benchmark_index + pd.Timedelta(hours=4)
        benchmark = pd.Series(
            [0.0, 0.02, 0.03], index=benchmark_index, name="Benchmark"
        )
        if as_frame:
            benchmark = benchmark.to_frame()
        original_benchmark = benchmark.copy(deep=True)
        output_index = period.tz_convert("UTC").tz_localize(None)
        # Noon observations are valid and have distinct local dates even when
        # midnight on the intervening DST transition is nonexistent or ambiguous.
        expected = pd.Series([0.0, 0.02, 0.03], index=output_index, name="Benchmark")

        result = utils._prepare_benchmark(
            benchmark,
            output_index if prepared else period,
            prepare_returns=False,
            strategy_tz=period.tz if prepared else None,
        )

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        returns = pd.Series([0.0, 0.03, 0.04], index=period, name="Strategy")
        original_returns = returns.copy(deep=True)
        # Active returns [0, .01, .01] give mean / sample deviation 2/sqrt(3).
        expected_ir = 2 / np.sqrt(3)
        assert stats.information_ratio(returns, benchmark) == pytest.approx(expected_ir)
        qs.extend_pandas()
        assert returns.information_ratio(benchmark) == pytest.approx(expected_ir)
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original_benchmark)
        else:
            pd.testing.assert_series_equal(benchmark, original_benchmark)

    def test_daily_alignment_preserves_strategy_labels_across_dst(self):
        period = pd.date_range("2024-03-08", periods=5, freq="D", tz="US/Eastern")
        benchmark = pd.Series(
            [0.0, 0.10, 0.20],
            index=pd.to_datetime(
                ["2024-03-08", "2024-03-11", "2024-03-12"]
            ).tz_localize("UTC"),
            name="Benchmark",
        )
        # Eastern midnight moves from 05:00 to 04:00 UTC. The working day
        # keys must not discard the last observation or replace those labels.
        expected = pd.Series(
            [0.0, 0.0, 0.0, 0.10, 0.20],
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    def test_intraday_strategy_labels_keep_the_existing_sampling_behavior(self):
        period = pd.date_range("2024-01-01", periods=5, freq="12h")
        benchmark = pd.Series(
            [0.0, 0.10, 0.20],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
            name="Benchmark",
        )
        # The existing daily grid leaves the intervening half-days missing;
        # pct_change(fill_method=None) cannot reconstruct returns across them.
        # This daily-label fix does not introduce an intraday resampling policy.
        expected = pd.Series(0.0, index=period, name="Benchmark")

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)

    @staticmethod
    def _cross_timezone_case(strategy_time="16:00", as_frame=False):
        period = pd.date_range(
            "2024-07-01 " + strategy_time,
            periods=3,
            freq="D",
            tz="America/New_York",
        )
        benchmark = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.date_range(
                "2024-07-01 15:00", periods=3, freq="D", tz="Asia/Tokyo"
            ),
            name="Benchmark",
        )
        return period, benchmark.to_frame() if as_frame else benchmark

    @pytest.mark.parametrize("strategy_time", ["10:59", "11:00", "16:00"])
    @pytest.mark.parametrize("as_frame", [False, True])
    @pytest.mark.parametrize("prepared", [False, True])
    def test_cross_timezone_daily_alignment_keeps_available_observations(
        self, strategy_time, as_frame, prepared
    ):
        period, benchmark = self._cross_timezone_case(strategy_time, as_frame)
        original = benchmark.copy(deep=True)
        output_index = period.tz_convert("UTC").tz_localize(None)
        if prepared:
            period = output_index
        # Crossing Tokyo midnight at NY 11:00 does not make another benchmark
        # observation available. All three strategy hours must use the same bars.
        expected = pd.Series([0.0, 0.02, 0.03], index=output_index, name="Benchmark")

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original)
        else:
            pd.testing.assert_series_equal(benchmark, original)

    @pytest.mark.parametrize("as_frame", [False, True])
    @pytest.mark.parametrize("prepared", [False, True])
    def test_future_cross_timezone_return_cannot_change_an_earlier_row(
        self, as_frame, prepared
    ):
        period, benchmark = self._cross_timezone_case(as_frame=as_frame)
        if prepared:
            period = period.tz_convert("UTC").tz_localize(None)
        baseline = utils._prepare_benchmark(benchmark, period, prepare_returns=False)
        changed = benchmark.copy(deep=True)
        changed.iloc[-1] = 0.90

        result = utils._prepare_benchmark(changed, period, prepare_returns=False)

        # Tokyo July 3 at 06:00 UTC follows NY July 2 at 20:00 UTC by ten hours.
        pd.testing.assert_series_equal(
            result.iloc[:-1], baseline.iloc[:-1], rtol=1e-12, atol=1e-12
        )
        assert result.iloc[-1] == pytest.approx(0.90)

    @pytest.mark.parametrize(
        ("start", "end"),
        [("2024-07-05", "2024-07-08"), ("2024-03-08", "2024-03-11")],
    )
    def test_cross_timezone_weekend_return_stays_on_monday(self, start, end):
        period = pd.date_range(
            start + " 16:00", periods=4, freq="D", tz="America/New_York"
        )
        benchmark = pd.Series(
            [0.01, 0.10],
            index=pd.to_datetime([start + " 15:00", end + " 15:00"]).tz_localize(
                "Asia/Tokyo"
            ),
            name="Benchmark",
        )
        expected = pd.Series(
            [0.0, 0.0, 0.0, 0.10],
            index=period.tz_convert("UTC").tz_localize(None),
            name="Benchmark",
        )

        result = utils._prepare_benchmark(benchmark, period, prepare_returns=False)
        changed = benchmark.copy(deep=True)
        changed.iloc[-1] = 0.90
        changed_result = utils._prepare_benchmark(
            changed, period, prepare_returns=False
        )

        # Tokyo's Monday return is unavailable on the NY weekend, including DST.
        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        pd.testing.assert_series_equal(changed_result.iloc[:-1], result.iloc[:-1])
        assert changed_result.iloc[-1] == pytest.approx(0.90)

    @pytest.mark.parametrize(
        ("timezone", "start"),
        [("Europe/London", "2024-03-30"), ("US/Eastern", "2024-03-09")],
    )
    @pytest.mark.parametrize("intraday_owner", ["benchmark", "strategy"])
    def test_strategy_calendar_context_preserves_prepared_intraday_sampling(
        self, timezone, start, intraday_owner
    ):
        period = pd.date_range(
            start,
            periods=3 if intraday_owner == "benchmark" else 5,
            freq="D" if intraday_owner == "benchmark" else "12h",
            tz=timezone,
        )
        output_index = period.tz_convert("UTC").tz_localize(None)
        benchmark = pd.Series(
            [0.0, 0.10, 0.20, 0.30, 0.40]
            if intraday_owner == "benchmark"
            else [0.0, 0.10, 0.20],
            index=pd.date_range(
                start,
                periods=5 if intraday_owner == "benchmark" else 3,
                freq="12h" if intraday_owner == "benchmark" else "D",
                tz=timezone,
            ),
            name="Benchmark",
        )
        original = benchmark.copy(deep=True)
        # An already-prepared UTC grid keeps its 24-hour steps across DST.
        # Calendar context must not restore the different native-midnight grid.
        expected = pd.Series(
            [0.0, 0.32, 0.0] if intraday_owner == "benchmark" else [0.0] * 5,
            index=output_index,
            name="Benchmark",
        )

        result = utils._prepare_benchmark(
            benchmark,
            output_index,
            prepare_returns=False,
            strategy_tz=period.tz,
        )

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        pd.testing.assert_series_equal(benchmark, original)

    @pytest.mark.parametrize(
        "metric", ["information_ratio", "r_squared", "greeks", "rolling_greeks"]
    )
    @pytest.mark.parametrize("calendar", ["cross_timezone", "shared_calendar"])
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_public_statistics_preserve_the_strategy_calendar(
        self, metric, calendar, as_frame
    ):
        if calendar == "cross_timezone":
            period, benchmark = self._cross_timezone_case(as_frame=as_frame)
        else:
            period = pd.date_range(
                "2024-07-01", periods=3, freq="D", tz="Europe/London"
            )
            benchmark = pd.Series(
                [0.01, 0.02, 0.03],
                index=period + pd.Timedelta(hours=5),
                name="Benchmark",
            )
            if as_frame:
                benchmark = benchmark.to_frame()
        returns = pd.Series([0.0, 0.03, 0.04], index=period, name="Strategy")
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)
        # Derive the statistics from the independently aligned return vector.
        expected_benchmark = np.array([0.0, 0.02, 0.03])
        strategy_values = np.array([0.0, 0.03, 0.04])
        active = strategy_values - expected_benchmark
        covariance = np.cov(strategy_values, expected_benchmark)
        beta = covariance[0, 1] / covariance[1, 1]
        alpha = strategy_values.mean() - beta * expected_benchmark.mean()

        if metric == "information_ratio":
            assert stats.information_ratio(returns, benchmark) == pytest.approx(
                active.mean() / active.std(ddof=1)
            )
        elif metric == "r_squared":
            assert stats.r_squared(returns, benchmark) == pytest.approx(
                covariance[0, 1] ** 2 / (covariance[0, 0] * covariance[1, 1])
            )
        elif metric == "greeks":
            result = stats.greeks(returns, benchmark)
            assert result["beta"] == pytest.approx(beta)
            assert result["alpha"] == pytest.approx(alpha * 252)
        else:
            result = stats.rolling_greeks(returns, benchmark, periods=3)
            pd.testing.assert_index_equal(
                result.index, period.tz_convert("UTC").tz_localize(None)
            )
            assert result["beta"].iloc[-1] == pytest.approx(beta)
            assert result["alpha"].iloc[-1] == pytest.approx(alpha)
        pd.testing.assert_series_equal(returns, original_returns)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original_benchmark)
        else:
            pd.testing.assert_series_equal(benchmark, original_benchmark)

    def test_report_metrics_use_the_causal_cross_timezone_benchmark(self):
        period, benchmark = self._cross_timezone_case()
        returns = pd.Series([0.0, 0.03, 0.04], index=period, name="Strategy")
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)

        result = reports.metrics(returns, benchmark, display=False)

        # The native table rounds fractional returns to two decimals.
        assert result.loc["Cumulative Return", "Benchmark"] == pytest.approx(
            round(1.02 * 1.03 - 1, 2)
        )
        pd.testing.assert_series_equal(returns, original_returns)
        pd.testing.assert_series_equal(benchmark, original_benchmark)

    @pytest.mark.parametrize("calendar", ["utc_boundary", "local_calendar"])
    @pytest.mark.parametrize("as_frame", [False, True])
    @pytest.mark.parametrize("mode", ["basic", "full"])
    @pytest.mark.parametrize("match_dates", [False, True])
    def test_report_metrics_preserve_original_calendar_at_utc_boundary(
        self, calendar, as_frame, mode, match_dates
    ):
        if calendar == "utc_boundary":
            period = pd.date_range("2024-07-01 23:00", periods=4, tz="America/New_York")
            # New York 23:00 falls on the following UTC date; Tokyo 15:00 does not.
            utc_index = pd.date_range("2024-07-02 03:00", periods=4)
            expected_values = [0.0, 0.03, 0.04, 0.0]
        else:
            period = pd.date_range("2024-07-01", periods=4, tz="Asia/Tokyo")
            # Tokyo midnight crosses UTC midnight, but its original local days
            # still match the benchmark's calendar and retain all daily gains.
            utc_index = pd.date_range("2024-06-30 15:00", periods=4)
            expected_values = [0.0, 0.02, 0.03, 0.04]
        returns = pd.Series([0.01, -0.02, 0.03, 0.04], index=period, name="Strategy")
        benchmark = pd.Series(
            [0.0, 0.02, 0.03, 0.04],
            index=pd.date_range("2024-07-01 15:00", periods=4, tz="Asia/Tokyo"),
            name="Benchmark",
        )
        reference_returns = pd.Series(
            [0.01, -0.02, 0.03, 0.04], index=utc_index, name="Strategy"
        )
        reference_benchmark = pd.Series(
            expected_values, index=utc_index, name="Benchmark"
        )
        if as_frame:
            returns, benchmark = returns.to_frame(), benchmark.to_frame()
            reference_returns = reference_returns.to_frame()
            reference_benchmark = reference_benchmark.to_frame()
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)

        # The control is independently pre-aligned, so it bypasses calendar
        # reconstruction while exercising the complete public report table.
        reference = reports.metrics(
            reference_returns,
            reference_benchmark,
            display=False,
            mode=mode,
            match_dates=match_dates,
        )
        result = reports.metrics(
            returns, benchmark, display=False, mode=mode, match_dates=match_dates
        )

        expected_cumulative = np.prod(1 + np.array(expected_values)) - 1
        assert result.loc["Cumulative Return", "Benchmark"] == pytest.approx(
            round(expected_cumulative, 2)
        )
        pd.testing.assert_frame_equal(result, reference)
        check = (
            pd.testing.assert_frame_equal
            if as_frame
            else pd.testing.assert_series_equal
        )
        check(returns, original_returns)
        check(benchmark, original_benchmark)

    @staticmethod
    def _mixed_clock_case(shared_calendar=False):
        strategy_tz = "Europe/London" if shared_calendar else "America/New_York"
        benchmark_tz = "Europe/London" if shared_calendar else "Asia/Tokyo"
        strategy_hour = "16:30" if shared_calendar else "16:00"
        seed_day = "2024-06-28" if shared_calendar else "2024-06-27"
        benchmark_seed = "2024-06-28 15:00" if shared_calendar else "2024-06-28 00:00"
        benchmark_hour = "00:00" if shared_calendar else "15:00"
        period = pd.to_datetime(
            [
                seed_day + " " + strategy_hour,
                "2024-07-01 " + strategy_hour,
                "2024-07-02 " + strategy_hour,
                "2024-07-03 " + strategy_hour,
            ]
        ).tz_localize(strategy_tz)
        benchmark = pd.Series(
            [0.0, 0.01, 0.02, 0.03],
            index=pd.to_datetime(
                [
                    benchmark_seed,
                    "2024-07-01 " + benchmark_hour,
                    "2024-07-02 " + benchmark_hour,
                    "2024-07-03 " + benchmark_hour,
                ]
            ).tz_localize(benchmark_tz),
            name="Benchmark",
        )
        return period, benchmark

    @pytest.mark.parametrize("as_frame", [False, True])
    @pytest.mark.parametrize("prepared", [False, True])
    @pytest.mark.parametrize("shared_calendar", [False, True])
    def test_mixed_clock_seed_preserves_causal_daily_alignment(
        self, as_frame, prepared, shared_calendar
    ):
        period, benchmark = self._mixed_clock_case(shared_calendar)
        if as_frame:
            benchmark = benchmark.to_frame()
        original = benchmark.copy(deep=True)
        output_index = period.tz_convert("UTC").tz_localize(None)
        context = {"strategy_tz": period.tz} if prepared else {}
        if prepared:
            period = output_index
        expected = pd.Series(
            [0.0, 0.01, 0.02, 0.03], index=output_index, name="Benchmark"
        )

        result = utils._prepare_benchmark(
            benchmark, period, prepare_returns=False, **context
        )
        # Both zero seeds precede the first strategy observation. Moving only
        # that historical label must not change the later calendar alignment.
        seed_shifted = benchmark.copy(deep=True)
        seed_shifted.index = pd.DatetimeIndex(
            [pd.Timestamp("2024-06-27 15:00", tz=benchmark.index.tz)]
            + list(benchmark.index[1:])
        )
        shifted_result = utils._prepare_benchmark(
            seed_shifted, period, prepare_returns=False, **context
        )
        future_changed = benchmark.copy(deep=True)
        future_changed.iloc[-1] = 0.05
        future_result = utils._prepare_benchmark(
            future_changed, period, prepare_returns=False, **context
        )

        pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=1e-12)
        pd.testing.assert_series_equal(shifted_result, expected, rtol=1e-12, atol=1e-12)
        # The last benchmark bar follows the preceding strategy observation;
        # another label crossing UTC midnight cannot make it available sooner.
        pd.testing.assert_series_equal(
            future_result.iloc[:-1], expected.iloc[:-1], rtol=1e-12, atol=1e-12
        )
        assert future_result.iloc[-1] == pytest.approx(0.05)
        if as_frame:
            pd.testing.assert_frame_equal(benchmark, original)
        else:
            pd.testing.assert_series_equal(benchmark, original)

    def test_information_ratio_uses_the_causal_mixed_clock_benchmark(self):
        period, benchmark = self._mixed_clock_case()
        returns = pd.Series([0.0, 0.02, 0.03, 0.04], index=period)
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)
        # Independently aligned active returns [0, .01, .01, .01] have
        # mean .0075 and sample standard deviation .005, giving IR 1.5.
        assert stats.information_ratio(returns, benchmark) == pytest.approx(1.5)
        pd.testing.assert_series_equal(returns, original_returns)
        pd.testing.assert_series_equal(benchmark, original_benchmark)

    def test_future_mixed_clock_return_preserves_an_earlier_rolling_window(self):
        period, benchmark = self._mixed_clock_case()
        returns = pd.Series([0.0, 0.02, 0.03, 0.04], index=period)
        original_returns = returns.copy(deep=True)
        original_benchmark = benchmark.copy(deep=True)
        future_changed = benchmark.copy(deep=True)
        future_changed.iloc[-1] = 0.05
        result = stats.rolling_greeks(returns, benchmark, periods=3)
        future_result = stats.rolling_greeks(returns, future_changed, periods=3)
        # The window ending July 2 excludes Tokyo's July 3 observation.
        # Derive its regression from the three independently aligned returns.
        strategy_values = np.array([0.0, 0.02, 0.03])
        benchmark_values = np.array([0.0, 0.01, 0.02])
        covariance = np.cov(strategy_values, benchmark_values)
        beta = covariance[0, 1] / covariance[1, 1]
        alpha = strategy_values.mean() - beta * benchmark_values.mean()

        assert result["beta"].iloc[2] == pytest.approx(beta)
        assert result["alpha"].iloc[2] == pytest.approx(alpha)
        pd.testing.assert_series_equal(result.iloc[2], future_result.iloc[2])
        pd.testing.assert_index_equal(
            result.index, period.tz_convert("UTC").tz_localize(None)
        )
        pd.testing.assert_series_equal(returns, original_returns)
        pd.testing.assert_series_equal(benchmark, original_benchmark)


class TestRiskFreeDeannualization:
    """An annual rf must never be charged once per period (#552).

    rar() passed rf to _prepare_returns() without `periods`, so a 5% annual
    rate was subtracted from every daily return. The series was wiped out and
    the reported Risk-Adjusted Return pinned to -100% for any non-zero rf.
    """

    def test_rar_with_rf_is_not_a_wipeout(self, daily_returns):
        value = stats.rar(daily_returns, rf=0.05)

        assert value > -1.0
        assert np.isfinite(value)

    def test_rar_rf_costs_roughly_the_annual_rate(self, daily_returns):
        # Exposure is 1.0 here, so RaR is just the excess CAGR: charging a 5%
        # annual rate should cost about 5 points of CAGR, not everything.
        gross = stats.rar(daily_returns, rf=0.0)
        net = stats.rar(daily_returns, rf=0.05)

        assert gross - net == pytest.approx(0.05, abs=0.02)

    def test_metrics_risk_adjusted_return_survives_rf(self, daily_returns):
        table = reports.metrics(
            daily_returns, rf=0.05, display=False, mode="full", prepare_returns=False
        )
        value = float(table.loc["Risk-Adjusted Return", "Strategy"])

        # 0.0.82 through 0.0.84 reported -1.0 here for any non-zero rf.
        assert value > -1.0

    def test_no_rf_metric_collapses_to_total_loss(self, daily_returns):
        # Guards the whole class rather than the one function: a 5% annual
        # rate must not push any rf-aware metric to -100%.
        for name in ("rar", "sharpe", "sortino", "omega", "adjusted_sortino"):
            value = getattr(stats, name)(daily_returns, rf=0.05)
            assert value > -1.0, f"{name} collapsed with a 5% risk-free rate"


class TestNoCrossColumnContamination:
    """A column's statistic must not depend on its neighbours (#556).

    avg_return/avg_win/avg_loss masked the unselected cells to NaN and then
    called .dropna(), which drops whole *rows* on a DataFrame. Each column's
    average was therefore computed only over rows where every other column
    also qualified, so a strategy's Kelly, payoff ratio and average win
    changed the moment a benchmark column sat beside it in reports.metrics().
    """

    @staticmethod
    def _frames():
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2021-01-01", periods=400)
        strat = pd.Series(rng.normal(0.0006, 0.009, 400), index=idx)
        bench = pd.Series(rng.normal(0.0004, 0.011, 400), index=idx)
        # Zeros matter: avg_return masks on `!= 0`, so the bug only shows
        # there when some row is zero in one column but not the other.
        strat.iloc[:8] = 0.0
        bench.iloc[3:11] = 0.0
        alone = pd.DataFrame({"returns_1": strat})
        with_bench = pd.DataFrame({"returns_1": strat, "benchmark": bench})
        return alone, with_bench

    @pytest.mark.parametrize(
        "name",
        [
            "avg_return",
            "avg_win",
            "avg_loss",
            "payoff_ratio",
            "win_loss_ratio",
            "kelly_criterion",
            "cpc_index",
        ],
    )
    def test_second_column_does_not_change_the_first(self, name):
        alone, with_bench = self._frames()
        fn = getattr(stats, name)

        expected = fn(alone, prepare_returns=False)["returns_1"]
        result = fn(with_bench, prepare_returns=False)["returns_1"]

        assert result == pytest.approx(expected)

    def test_metrics_table_matches_the_standalone_call(self):
        alone, with_bench = self._frames()
        strategy = alone[["returns_1"]]
        benchmark = with_bench["benchmark"]

        table = reports.metrics(
            strategy, benchmark, display=False, mode="full", prepare_returns=False
        )
        shown = float(table.loc["Kelly Criterion", "Strategy"])
        standalone = float(stats.kelly_criterion(strategy)["returns_1"])

        # The table rounds to 2dp, so compare at that resolution.
        assert shown == pytest.approx(round(standalone, 2), abs=0.01)


class TestRollingGreeksAlpha:
    """Rolling alpha used full-sample means instead of each window's (#554)."""

    def test_each_window_matches_a_regression_on_that_window(self):
        rng = np.random.RandomState(11)
        idx = pd.date_range("2020-01-01", periods=300, freq="D")
        bench = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx)
        strat = 0.7 * bench + rng.normal(0.0002, 0.006, 300)
        strat.iloc[150:] += 0.002  # alpha genuinely changes half way through

        window = 60
        rolling = stats.rolling_greeks(
            strat, bench, periods=window, prepare_returns=False
        )

        for end in (window, 150, 300):
            beta, alpha = np.polyfit(
                bench.iloc[end - window : end], strat.iloc[end - window : end], 1
            )
            assert rolling["beta"].iloc[end - 1] == pytest.approx(beta)
            assert rolling["alpha"].iloc[end - 1] == pytest.approx(alpha)

    def test_alpha_responds_to_a_regime_change(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-01", periods=504)
        bench = pd.Series(rng.normal(0.0003, 0.01, 504), index=idx)
        strat = 0.8 * bench + rng.normal(0.0, 0.006, 504)
        strat.iloc[252:] += 0.001

        rolling = stats.rolling_greeks(strat, bench, periods=126, prepare_returns=False)

        before = rolling["alpha"].iloc[251]
        after = rolling["alpha"].iloc[503]

        # Previously both read ~0.00057 regardless of the added 10bp/day.
        assert after > before + 0.0005


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
        # "ME" only exists in pandas 2.2+; translate for older versions.
        dates = pd.date_range("2016-01-31", periods=121, freq=get_frequency_alias("ME"))
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


class TestRollingRiskFreeDeannualization:
    """
    rolling_sharpe/rolling_sortino passed rolling_period where
    _prepare_returns expects periods_per_year, so rf was de-annualized over
    the window instead of the year - exactly 2x too much at the defaults.
    """

    def test_rolling_sharpe_uses_periods_per_year_for_rf(self, daily_returns):
        expected_input = utils._prepare_returns(daily_returns, 0.05, 252)
        expected = (
            expected_input.rolling(126).mean()
            / expected_input.rolling(126).std()
            * np.sqrt(252)
        )

        result = stats.rolling_sharpe(daily_returns, rf=0.05, rolling_period=126)

        pd.testing.assert_series_equal(
            result.dropna(), expected.dropna(), check_names=False
        )

    def test_window_length_does_not_change_the_rf_charged(self, daily_returns):
        # Two windows must agree on the rf they subtract, so a short window
        # cannot be penalised for being short.
        a = stats.rolling_sharpe(daily_returns, rf=0.05, rolling_period=60)
        b = stats.rolling_sharpe(daily_returns, rf=0.05, rolling_period=120)

        assert not a.dropna().empty and not b.dropna().empty
        # Same underlying excess returns => same value where windows coincide
        ref = utils._prepare_returns(daily_returns, 0.05, 252)
        for period, series in ((60, a), (120, b)):
            expected = (
                ref.rolling(period).mean() / ref.rolling(period).std() * np.sqrt(252)
            )
            pd.testing.assert_series_equal(
                series.dropna(), expected.dropna(), check_names=False
            )

    def test_rolling_sortino_uses_periods_per_year_for_rf(self, daily_returns):
        result = stats.rolling_sortino(daily_returns, rf=0.05, rolling_period=126)

        assert not result.dropna().empty
        assert np.isfinite(result.dropna()).all()


class TestMissingObservationsAreNotZeroReturns:
    """
    _prepare_returns filled gaps with 0.0, asserting the strategy was flat on
    days it had no data for. That understates volatility and drawdown and
    inflates every ratio built on them.
    """

    @staticmethod
    def _gapped(daily_returns):
        gapped = daily_returns.copy()
        gapped.iloc[50:90] = np.nan
        return gapped, daily_returns.drop(daily_returns.index[50:90])

    def test_gaps_are_not_turned_into_zeros(self, daily_returns):
        gapped, _ = self._gapped(daily_returns)

        prepared = utils._prepare_returns(gapped, apply_rf=False)

        assert prepared.isna().sum() == 40
        assert (prepared == 0).sum() == 0

    @pytest.mark.parametrize(
        "metric",
        [
            "volatility",
            "sharpe",
            "sortino",
            "cagr",
            "win_rate",
            "max_drawdown",
            "kelly_criterion",
            "ghpr",
            "exposure",
        ],
    )
    def test_metric_matches_dropping_the_gap(self, metric, daily_returns):
        gapped, baseline = self._gapped(daily_returns)

        fn = getattr(stats, metric)

        assert float(fn(gapped)) == pytest.approx(float(fn(baseline)), rel=1e-9)

    def test_volatility_is_not_understated_by_gaps(self, daily_returns):
        gapped, baseline = self._gapped(daily_returns)

        # Filling with zeros used to drag volatility down toward zero.
        filled = stats.volatility(gapped.fillna(0), prepare_returns=False)

        assert stats.volatility(gapped) > filled

    def test_equity_curve_carries_through_a_gap(self, daily_returns):
        gapped, _ = self._gapped(daily_returns)

        curve = stats.compsum(utils._prepare_returns(gapped, apply_rf=False))

        assert not curve.isna().any()


class TestConditionalValueAtRisk:
    """
    CVaR took a parametric VaR threshold and averaged the observations below
    it, mixing two estimators, and returned the VaR itself when no observation
    fell below - which overstates CVaR, since CVaR is at least as severe.
    """

    def test_cvar_is_never_milder_than_var(self, daily_returns):
        var = stats.value_at_risk(daily_returns)
        cvar = stats.conditional_value_at_risk(daily_returns)

        assert cvar <= var

    @pytest.mark.parametrize("n", [3, 5, 8, 20])
    def test_small_samples_do_not_fall_back_to_var(self, n):
        series = pd.Series(np.linspace(0.001, 0.002, n))

        var = stats.value_at_risk(series, prepare_returns=False)
        cvar = stats.conditional_value_at_risk(series, prepare_returns=False)

        assert cvar < var

    def test_no_empty_slice_warning(self):
        series = pd.Series(np.linspace(0.001, 0.002, 4))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            stats.conditional_value_at_risk(series, prepare_returns=False)

    def test_historical_method_is_available(self, daily_returns):
        historical = stats.conditional_value_at_risk(daily_returns, method="historical")

        assert historical <= stats.value_at_risk(daily_returns)

    def test_unknown_method_is_rejected(self, daily_returns):
        with pytest.raises(ValueError, match="parametric"):
            stats.conditional_value_at_risk(daily_returns, method="nope")

    def test_dataframe_returns_one_value_per_column(self, daily_returns):
        frame = pd.DataFrame({"a": daily_returns, "b": daily_returns * 2})

        result = stats.conditional_value_at_risk(frame)

        assert list(result.index) == ["a", "b"]
        assert result["b"] == pytest.approx(result["a"] * 2, rel=1e-9)

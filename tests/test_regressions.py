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
            daily_returns,
            benchmark,
            annualize=False,
            compounded=False,
            prepare_returns=False,
        )

        diff = daily_returns - benchmark
        assert result == pytest.approx(diff.mean() / diff.std())

    def test_prepared_large_returns_are_not_detected_as_prices(self):
        dates = pd.date_range("2020-01-01", periods=3)
        returns = pd.Series([1.2, 0.2, 0.4], index=dates)
        benchmark = pd.Series([0.2, 0.1, 0.3], index=dates)
        # Wealth: 2.2 * 1.2 * 1.4 = 3.696 and 1.2 * 1.1 * 1.3 = 1.716.
        # Active returns [1, 0.1, 0.1] have sample variance 0.27.
        # (3.696 - 1.716) / sqrt(0.27 * 3) = 2.2.
        expected = 2.2
        result = stats.information_ratio(
            returns, benchmark, periods=3, prepare_returns=False
        )
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("multiple", [False, True])
    @pytest.mark.parametrize("compounded, expected", [(True, 0.76), (False, 0.77)])
    def test_report_uses_periods_and_compounding(self, multiple, compounded, expected):
        dates = pd.date_range("2020-01-01", periods=4)
        returns = pd.Series([0.02, -0.01, 0.03, -0.02], index=dates)
        benchmark = pd.Series([0.01, 0.00, 0.01, -0.02], index=dates)
        if multiple:
            returns = pd.DataFrame({"first": returns, "second": returns})
        result = reports.metrics(
            returns,
            benchmark,
            display=False,
            mode="full",
            periods_per_year=4,
            compounded=compounded,
            prepare_returns=False,
        )
        values = result.loc["Information Ratio"].drop(labels="Benchmark")
        assert all(float(value) == pytest.approx(expected) for value in values)


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

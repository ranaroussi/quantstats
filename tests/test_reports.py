"""
Tests for quantstats.reports module
"""

import base64
import os
import re
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from quantstats import reports


@pytest.fixture
def sample_returns():
    """Generate sample daily returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.02, index=dates, name="Strategy")
    return returns


@pytest.fixture
def sample_benchmark():
    """Generate sample benchmark returns for testing."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.015, index=dates, name="Benchmark")
    return returns


class TestHTMLReport:
    """Test HTML report generation."""

    def test_html_generates_file(self, sample_returns, sample_benchmark):
        """Test that HTML report generates a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(sample_returns, sample_benchmark, output=output_path)
            assert os.path.exists(output_path)
            # Check file has content
            with open(output_path) as f:
                content = f.read()
                assert len(content) > 1000  # Should have substantial content
                assert "<!DOCTYPE html>" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_html_contains_title(self, sample_returns):
        """Test that HTML report contains proper title."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(sample_returns, output=output_path, title="My Strategy")
            with open(output_path) as f:
                content = f.read()
                assert "My Strategy" in content
                assert "(Compounded)" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_html_simple_returns_title(self, sample_returns):
        """Test title does NOT show (Compounded) when compounded=False."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(sample_returns, output=output_path, compounded=False)
            with open(output_path) as f:
                content = f.read()
                assert "(Compounded)" not in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_html_contains_benchmark_param(self, sample_returns, sample_benchmark):
        """Test that HTML report shows benchmark in parameters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(sample_returns, sample_benchmark, output=output_path)
            with open(output_path) as f:
                content = f.read()
                assert "Benchmark:" in content
                assert "Periods/Year: 252" in content
                assert "RF:" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_html_matched_dates_indicator(self, sample_returns, sample_benchmark):
        """Test matched dates indicator appears when appropriate."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(
                sample_returns, sample_benchmark, output=output_path, match_dates=True
            )
            with open(output_path) as f:
                content = f.read()
                assert "(matched dates)" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_html_no_dark_mode(self, sample_returns):
        """Test that dark mode CSS is not present."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(sample_returns, output=output_path)
            with open(output_path) as f:
                content = f.read()
                assert "prefers-color-scheme:dark" not in content
                assert "color-scheme" not in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_html_custom_rf(self, sample_returns):
        """Test custom risk-free rate in parameters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(sample_returns, output=output_path, rf=0.05)
            with open(output_path) as f:
                content = f.read()
                assert "RF: 5.0%" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestMetrics:
    """Test metrics function."""

    def test_metrics_basic_mode(self, sample_returns):
        """Test metrics in basic mode."""
        result = reports.metrics(sample_returns, display=False, mode="basic")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_metrics_full_mode(self, sample_returns):
        """Test metrics in full mode."""
        result = reports.metrics(sample_returns, display=False, mode="full")
        assert isinstance(result, pd.DataFrame)
        # Full mode should have more metrics than basic
        basic_result = reports.metrics(sample_returns, display=False, mode="basic")
        assert len(result) >= len(basic_result)

    def test_metrics_with_benchmark(self, sample_returns, sample_benchmark):
        """Test metrics with benchmark comparison."""
        result = reports.metrics(
            sample_returns, benchmark=sample_benchmark, display=False
        )
        assert isinstance(result, pd.DataFrame)
        # Should have at least two columns (strategy + benchmark)
        assert result.shape[1] >= 2

    def test_metrics_with_rf(self, sample_returns):
        """Test metrics with non-zero risk-free rate."""
        result_no_rf = reports.metrics(sample_returns, rf=0, display=False)
        result_with_rf = reports.metrics(sample_returns, rf=0.02, display=False)
        # Results should be different
        assert not result_no_rf.equals(result_with_rf)


class TestMatchDates:
    """Test date matching functionality."""

    def test_match_dates_aligns_series(self):
        """Test that _match_dates aligns returns and benchmark."""
        # Create returns starting later than benchmark
        dates1 = pd.date_range("2020-01-10", periods=100, freq="D")
        dates2 = pd.date_range("2020-01-01", periods=110, freq="D")

        returns = pd.Series(np.random.randn(100) * 0.01, index=dates1)
        benchmark = pd.Series(np.random.randn(110) * 0.01, index=dates2)

        # Set first values to non-zero
        returns.iloc[0] = 0.01
        benchmark.iloc[0] = 0.01

        aligned_ret, aligned_bench = reports._match_dates(returns, benchmark)

        # Both should now start from the same date
        assert aligned_ret.index[0] == aligned_bench.index[0]


class TestParametersTable:
    """Test parameters table printing."""

    def test_print_parameters_table_with_benchmark(self, capsys):
        """Test parameters table output with benchmark."""
        reports._print_parameters_table(
            benchmark_title="SPY",
            periods_per_year=252,
            rf=0.02,
            compounded=True,
            match_dates=True,
        )
        captured = capsys.readouterr()
        assert "SPY" in captured.out
        assert "252" in captured.out
        assert "2.0%" in captured.out
        assert "Yes" in captured.out

    def test_print_parameters_table_no_benchmark(self, capsys):
        """Test parameters table output without benchmark."""
        reports._print_parameters_table(
            benchmark_title=None,
            periods_per_year=252,
            rf=0.0,
            compounded=False,
            match_dates=True,
        )
        captured = capsys.readouterr()
        assert "Benchmark" not in captured.out
        assert "252" in captured.out
        assert "No" in captured.out  # Compounded = No


class TestEdgeCases:
    """Test edge cases in reports."""

    def test_html_with_dataframe(self, sample_returns, sample_benchmark):
        """Test HTML generation with DataFrame input."""
        df = pd.DataFrame({"Strategy": sample_returns})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            reports.html(df, sample_benchmark, output=output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_metrics_empty_benchmark_title(self, sample_returns, sample_benchmark):
        """Test metrics when benchmark has no name."""
        benchmark_no_name = sample_benchmark.copy()
        benchmark_no_name.name = None
        result = reports.metrics(
            sample_returns, benchmark=benchmark_no_name, display=False
        )
        assert isinstance(result, pd.DataFrame)


class TestEmbedFigure:
    """Multi-column reports pass a list of figure buffers into the HTML."""

    def test_single_svg_is_embedded_unchanged(self):
        html = reports._embed_figure(BytesIO(b"<svg>one</svg>"), "svg")

        assert html == "<svg>one</svg>"

    def test_list_keeps_every_svg(self):
        html = reports._embed_figure(
            [BytesIO(b"<svg>one</svg>"), BytesIO(b"<svg>two</svg>")],
            "svg",
        )

        assert "<svg>one</svg>" in html
        assert "<svg>two</svg>" in html

    def test_joined_svgs_do_not_share_ids(self):
        svg = (
            b'<svg><clipPath id="figure_1"><path id="figure_10"/></clipPath>'
            b'<use href="#figure_1" clip-path="url(#figure_10)"/></svg>'
        )
        html = reports._embed_figure([BytesIO(svg), BytesIO(svg)], "svg")

        ids = re.findall(r'\bid="([^"]+)"', html)
        assert len(ids) == len(set(ids)) == 4
        for ref in re.findall(r'href="#([^"]+)"', html):
            assert ref in ids
        for ref in re.findall(r"url\(#([^)]+)\)", html):
            assert ref in ids
        bare = [re.sub(r"^qs\d+-", "", ident) for ident in ids]
        assert bare.count("figure_1") == 2
        assert bare.count("figure_10") == 2

    def test_hyphenated_data_attributes_are_not_svg_ids(self):
        untouched = (
            b'<svg data-id="figure_10">'
            b'<use data-href="#figure_10" clip-path="url(#figure_10)"/></svg>'
        )
        assert reports._embed_figure(BytesIO(untouched), "svg") == untouched.decode()

        svg = (
            b'<svg id="figure_1" data-id="figure_1">'
            b'<use xlink:href="#figure_1" href="#figure_1" data-href="#figure_1" '
            b'clip-path="url(#figure_1)"/></svg>'
        )
        html = reports._embed_figure(BytesIO(svg), "svg")

        assert 'data-id="figure_1"' in html
        assert 'data-href="#figure_1"' in html
        assert html.count('href="#qs') == 2
        assert re.search(r'(?<![\w-])id="qs\d+-figure_1"', html)
        assert re.search(r"url\(#qs\d+-figure_1\)", html)

    def test_shorter_undeclared_ref_does_not_rewrite_longer_id(self):
        svg = (
            b'<svg id="figure_10">'
            b'<use href="#figure_1" clip-path="url(#figure_10)"/></svg>'
        )
        html = reports._embed_figure(BytesIO(svg), "svg")

        assert re.search(r'href="#figure_1"', html)
        assert not re.search(r'href="#qs\d+-figure_1"', html)
        assert re.search(r'id="qs\d+-figure_10"', html)
        assert re.search(r"url\(#qs\d+-figure_10\)", html)

    def test_list_keeps_every_png(self):
        html = reports._embed_figure(
            [BytesIO(b"png-one"), BytesIO(b"png-two")],
            "png",
        )

        assert html.count("<img ") == 2
        assert base64.b64encode(b"png-one").decode() in html
        assert base64.b64encode(b"png-two").decode() in html

    def test_html_dataframe_writes_one_figure_per_column(self):
        dates = pd.date_range("2020-01-01", periods=120, freq="B")
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            rng.normal(0, 0.01, size=(len(dates), 2)),
            index=dates,
            columns=["Alpha", "Beta"],
        )
        one = frame[["Alpha"]].squeeze()
        one.name = "Alpha"

        def write(returns):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as handle:
                path = handle.name
            reports.html(returns, output=path, figfmt="svg", title="Two")
            with open(path, encoding="utf-8") as handle:
                html = handle.read()
            os.remove(path)
            return html

        single = write(one)
        double = write(frame)

        # Drawdown periods, the monthly heatmap, and the distribution each
        # add one figure per column. The shared buffer used to keep only the first.
        assert double.count("<svg") == single.count("<svg") + 3
        assert "Alpha" in double and "Beta" in double
        ids = re.findall(r'\bid="([^"]+)"', double)
        assert ids
        assert len(ids) == len(set(ids))

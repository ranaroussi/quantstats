"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.quantstats.data import _Data, build_data


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture
def returns(resource_dir):
    # only feed in frames. no series
    x = pd.read_csv(resource_dir / "meta.csv", parse_dates=True, index_col=0)["Close"]
    return x.to_frame(name="Meta")


@pytest.fixture
def benchmark(resource_dir):
    x = pd.read_csv(resource_dir / "benchmark.csv", parse_dates=True, index_col=0)["Close"]
    x.name = "SPY -- Benchmark"
    return x


@pytest.fixture
def portfolio(resource_dir):
    return pd.read_csv(resource_dir / "portfolio.csv", parse_dates=True, index_col=0)


@pytest.fixture
def data(portfolio, benchmark) -> _Data:
    return build_data(returns=portfolio, benchmark=benchmark)

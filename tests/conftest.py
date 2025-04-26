"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture
def returns(resource_dir):
    return pd.read_csv(resource_dir / "meta.csv", parse_dates=True, index_col=0)["Close"].dropna()


@pytest.fixture
def benchmark(resource_dir):
    return pd.read_csv(resource_dir / "benchmark.csv", parse_dates=True, index_col=0)["Close"].dropna()

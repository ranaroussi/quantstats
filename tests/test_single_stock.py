import pandas as pd
import pytest

import quantstats as qs

qs.extend_pandas()


def test_meta(resource_dir):
    ts = pd.read_csv(resource_dir / "meta.csv", parse_dates=True, index_col=0)["Close"]
    assert ts.sharpe() == pytest.approx(0.7145916016746434)

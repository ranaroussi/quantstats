import dataclasses

import numpy as np
import pandas as pd


def build_data(returns, rf=0.0, benchmark=None, nperiods=None):
    def excess(x, rf, nperiods=None):
        if not isinstance(rf, float):
            rf = rf[rf.index.isin(x.index)]

        if nperiods is not None:
            # deannualize
            rf = np.power(1 + rf, 1.0 / nperiods) - 1.0

        df = x - rf
        df = df.tz_localize(None)
        return df

    # if isinstance(returns, pd.Series):
    #    df = returns.to_frame()
    # else:
    #    df = returns

    if benchmark is None:
        return _Data(returns=excess(returns, rf, nperiods=nperiods))

    else:
        common = sorted(list(set(returns.index) & set(benchmark.index)))
        return _Data(
            returns=excess(returns.loc[common], rf, nperiods=nperiods),
            benchmark=excess(benchmark.loc[common], rf, nperiods=nperiods),
        )

        # return excess(returns, rf, nperiods=nperiods), excess(benchmark, rf, nperiods=nperiods)


@dataclasses.dataclass
class _Data:
    returns: pd.Series | pd.DataFrame
    benchmark: pd.Series | None = None

    def all(self) -> pd.DataFrame:
        # Ensure 'r' is always a DataFrame
        if isinstance(self.returns, pd.Series):
            r = self.returns.to_frame()
        else:
            r = self.returns.copy()  # Copy to avoid mutating the original returns

        # Only add 'Benchmark' column if benchmark data is available
        if self.benchmark is not None:
            r["Benchmark"] = self.benchmark

        # Return the combined DataFrame
        return r

    @property
    def index(self):
        return self.returns.index

    @property
    def name(self):
        try:
            return list(self.returns.columns)
        except AttributeError:
            return "Strategy"

        # if isinstance(returns, _pd.DataFrame):
        #    if len(returns.columns) > 1:
        #        if isinstance(strategy_colname, str):
        #            strategy_colname = list(returns.columns)

    def apply(self, fct, **kwargs):
        return fct(self.returns, **kwargs)

#!/usr/bin/env python
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Monte Carlo Simulation Module

This module provides Monte Carlo simulation functionality for portfolio
analysis. It allows users to generate multiple simulated return paths
by shuffling historical returns, enabling probability-based risk assessment.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """
    Container for Monte Carlo simulation results.

    This class holds the results of a Monte Carlo simulation and provides
    convenient methods for analyzing and visualizing the simulated paths.

    Attributes
    ----------
    data : pd.DataFrame
        Raw simulation paths (sims x periods) as cumulative returns
    original : pd.Series
        Original cumulative returns path
    bust_threshold : float, optional
        Drawdown threshold used for bust probability calculation
    goal_threshold : float, optional
        Return threshold used for goal probability calculation

    Examples
    --------
    >>> import quantstats as qs
    >>> returns = qs.utils.download_returns("SPY")
    >>> mc = qs.stats.montecarlo(returns, sims=1000)
    >>> print(mc.stats)
    >>> mc.plot()
    """

    data: pd.DataFrame
    original: pd.Series
    bust_threshold: Optional[float] = None
    goal_threshold: Optional[float] = None
    _maxdd_cache: Optional[pd.Series] = field(default=None, repr=False)

    @property
    def stats(self) -> Dict[str, float]:
        """
        Terminal value statistics across all simulations.

        Returns
        -------
        dict
            Dictionary containing min, max, mean, median, std of terminal values
        """
        terminal = self.data.iloc[-1]
        return {
            "min": terminal.min(),
            "max": terminal.max(),
            "mean": terminal.mean(),
            "median": terminal.median(),
            "std": terminal.std(),
            "percentile_5": terminal.quantile(0.05),
            "percentile_25": terminal.quantile(0.25),
            "percentile_75": terminal.quantile(0.75),
            "percentile_95": terminal.quantile(0.95),
        }

    @property
    def maxdd(self) -> Dict[str, float]:
        """
        Maximum drawdown statistics across all simulations.

        Returns
        -------
        dict
            Dictionary containing min, max, mean, median of max drawdowns
        """
        if self._maxdd_cache is None:
            # Calculate max drawdown for each simulation path
            maxdd_values = []
            for col in self.data.columns:
                path = self.data[col]
                # Calculate drawdown from cumulative returns
                cumulative = path + 1  # Convert to growth factor
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                maxdd_values.append(drawdown.min())
            object.__setattr__(self, "_maxdd_cache", pd.Series(maxdd_values))

        dd = self._maxdd_cache
        return {
            "min": dd.min(),  # Worst (most negative) drawdown
            "max": dd.max(),  # Best (least negative) drawdown
            "mean": dd.mean(),
            "median": dd.median(),
            "std": dd.std(),
            "percentile_5": dd.quantile(0.05),
            "percentile_95": dd.quantile(0.95),
        }

    @property
    def bust_probability(self) -> Optional[float]:
        """
        Probability of exceeding the bust (drawdown) threshold.

        Returns
        -------
        float or None
            Probability of bust if threshold was set, None otherwise
        """
        if self.bust_threshold is None:
            return None

        if self._maxdd_cache is None:
            # Trigger maxdd calculation
            _ = self.maxdd

        # Count simulations where max drawdown exceeds bust threshold
        bust_count = (self._maxdd_cache <= self.bust_threshold).sum()
        return bust_count / len(self._maxdd_cache)

    @property
    def goal_probability(self) -> Optional[float]:
        """
        Probability of reaching the goal (return) threshold.

        Returns
        -------
        float or None
            Probability of reaching goal if threshold was set, None otherwise
        """
        if self.goal_threshold is None:
            return None

        terminal = self.data.iloc[-1]
        goal_count = (terminal >= self.goal_threshold).sum()
        return goal_count / len(terminal)

    def percentile(self, p: float) -> pd.Series:
        """
        Get the p-th percentile path across all simulations.

        Parameters
        ----------
        p : float
            Percentile value between 0 and 100

        Returns
        -------
        pd.Series
            The p-th percentile path
        """
        return self.data.quantile(p / 100, axis=1)

    def confidence_band(
        self, level: float = 0.95
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Get lower and upper bounds for a confidence interval.

        Parameters
        ----------
        level : float, default 0.95
            Confidence level (e.g., 0.95 for 95% confidence interval)

        Returns
        -------
        tuple
            (lower_bound, upper_bound) as pd.Series
        """
        alpha = (1 - level) / 2
        lower = self.data.quantile(alpha, axis=1)
        upper = self.data.quantile(1 - alpha, axis=1)
        return lower, upper

    def plot(self, **kwargs) -> Any:
        """
        Plot all simulation paths with the original path highlighted.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the plotting function

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        # Import here to avoid circular imports
        from . import plots

        return plots.montecarlo(self, **kwargs)


def run_montecarlo(
    returns: pd.Series,
    sims: int = 1000,
    bust: Optional[float] = None,
    goal: Optional[float] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation by shuffling returns.

    This function creates multiple simulated return paths by randomly
    shuffling the historical returns. This preserves the return distribution
    while breaking any time-series dependencies.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (not prices)
    sims : int, default 1000
        Number of simulations to run
    bust : float, optional
        Drawdown threshold for "bust" probability (e.g., -0.1 for -10%)
    goal : float, optional
        Return threshold for "goal" probability (e.g., 1.0 for +100%)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    MonteCarloResult
        Object containing simulation results with stats, maxdd, and plot methods

    Examples
    --------
    >>> import quantstats as qs
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> mc = run_montecarlo(returns, sims=100, bust=-0.1, goal=0.5)
    >>> print(mc.stats)
    >>> print(f"Bust probability: {mc.bust_probability:.1%}")
    """
    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Get returns as numpy array for efficiency
    returns_array = returns.dropna().values
    n_periods = len(returns_array)

    # Pre-allocate simulation array
    sim_returns = np.empty((n_periods, sims))

    # First column is original (unshuffled) returns
    sim_returns[:, 0] = returns_array

    # Generate shuffled paths
    for i in range(1, sims):
        sim_returns[:, i] = rng.permutation(returns_array)

    # Calculate cumulative returns for all paths
    # Using (1 + r).cumprod() - 1 formula
    cumulative = np.cumprod(1 + sim_returns, axis=0) - 1

    # Create DataFrame with simulation results
    sim_df = pd.DataFrame(
        cumulative,
        index=range(n_periods),
        columns=[f"sim_{i}" for i in range(sims)],
    )

    # Original cumulative returns
    original = pd.Series(cumulative[:, 0], index=range(n_periods), name="original")

    return MonteCarloResult(
        data=sim_df,
        original=original,
        bust_threshold=bust,
        goal_threshold=goal,
    )

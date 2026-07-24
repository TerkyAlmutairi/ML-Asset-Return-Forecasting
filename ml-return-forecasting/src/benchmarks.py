"""
benchmarks.py

Benchmark strategies run through the exact same backtest engine
(run_backtest in backtest.py) as the ML models, so comparisons are
apples-to-apples: same universe, same rebalance frequency, same
transaction cost assumption, same decile sizing. The only thing that
differs is what generates the ranking score.

  - buy_and_hold_scores: every stock gets an equal score (no ranking skill
    at all) -- this collapses the long/short backtest to ~market-neutral
    noise, which is the honest "you have no signal" baseline.
  - momentum_scores: classic 12-1 month momentum (trailing 12-month return,
    skipping the most recent month to avoid short-term reversal effects) --
    a well-known, genuinely real factor, and a much more meaningful bar for
    an ML model to clear than "beats zero."
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold_scores(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Every ticker gets an identical score each period -- no ranking signal.
    Not meant to be run through the long/short decile backtest (a constant
    score has no ranking information, so run_backtest correctly skips those
    periods -- see equal_weight_market_return for the actual buy-and-hold
    comparison, which is an absolute-return benchmark, not a long/short one)."""
    df = feature_panel.reset_index()[["date", "ticker", "forward_return"]].copy()
    df["score"] = 0.0
    df = df.rename(columns={"forward_return": "y_true"})
    return df[["date", "ticker", "score", "y_true"]].dropna(subset=["y_true"])


def equal_weight_market_return(feature_panel: pd.DataFrame, forward_horizon: int = 5):
    """
    The real buy-and-hold benchmark: simply hold the equal-weighted universe
    every period, no ranking involved. This is an absolute-return strategy,
    fundamentally different in kind from the long/short decile strategies
    (which bet on relative ranking, not market direction), so it's computed
    directly here rather than forced through run_backtest's long/short logic.

    Uses the same non-overlapping-window discipline as run_backtest (see
    that function's docstring) -- only every forward_horizon-th date is
    used, to avoid the same overlapping-return-window bias.
    """
    from backtest import _summarize

    df = feature_panel.reset_index()[["date", "ticker", "forward_return"]].dropna()
    all_dates = sorted(df["date"].unique())
    rebalance_dates = set(all_dates[::forward_horizon])
    df = df[df["date"].isin(rebalance_dates)]

    period_returns = df.groupby("date")["forward_return"].mean().sort_index()
    periods_per_year = 252 / forward_horizon
    return _summarize(
        period_returns, turnovers=[0.0] * len(period_returns), strategy_name="buy_and_hold",
        periods_per_year=periods_per_year,
    )


def momentum_scores(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the already-computed momentum_60d feature (~3-month momentum, given
    our forward_horizon of 5 trading days -- a full 12-1 month momentum
    factor needs a longer lookback than is useful for a 5-day-forward
    strategy; momentum_60d is the closest analog already in the feature set
    and keeps this benchmark honest rather than hand-tuned to look weak).
    """
    df = feature_panel.reset_index()[["date", "ticker", "momentum_60d", "forward_return"]].copy()
    df = df.rename(columns={"momentum_60d": "score", "forward_return": "y_true"})
    return df.dropna(subset=["score", "y_true"])


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=100)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(10)]], names=["date", "ticker"])
    panel = pd.DataFrame(
        {
            "momentum_60d": rng.normal(size=len(idx)),
            "forward_return": rng.normal(0, 0.02, len(idx)),
        },
        index=idx,
    )

    from backtest import run_backtest

    bh = equal_weight_market_return(panel)
    mom = run_backtest(momentum_scores(panel), score_col="score", strategy_name="momentum")
    print(bh.summary())
    print(mom.summary())

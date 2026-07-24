"""
backtest.py

An event-driven backtest: at each rebalance period, rank all stocks by
their predicted score, go long the top decile and short the bottom decile
(equal-weighted within each side), hold for the period, then rebalance.
This directly exploits the ranking skill that Information Coefficient (see
evaluation.py) measures.

Includes a simple transaction cost assumption (applied to every position
opened/closed) since a backtest that ignores trading costs systematically
overstates a strategy that trades often -- this matters a lot for a
monthly-rebalanced, wide long/short book like this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    strategy_name: str
    period_returns: pd.Series  # net-of-cost return per rebalance period
    n_periods: int
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    avg_turnover: float

    def summary(self) -> str:
        return (
            f"{self.strategy_name}: annualized return={self.annualized_return:+.2%}, "
            f"Sharpe={self.sharpe_ratio:.2f}, max drawdown={self.max_drawdown:.2%}, "
            f"periods={self.n_periods}"
        )


def _decile_long_short_return(
    scores: pd.Series, forward_returns: pd.Series, decile: float = 0.2
) -> tuple[float, float]:
    """
    scores, forward_returns: aligned Series for one rebalance period, one
    value per ticker. Returns (period_return, turnover_proxy).

    decile=0.2 means top/bottom 20% of the universe -- with a ~15-name
    universe that's roughly the top/bottom 3 names, which is intentionally
    a wide fraction given the small universe size (a true decile split
    needs more names to be meaningful; documented in README).
    """
    n = len(scores)
    n_side = max(1, int(np.floor(n * decile)))
    ranked = scores.sort_values(ascending=False)

    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]

    long_return = forward_returns.loc[longs].mean()
    short_return = forward_returns.loc[shorts].mean()

    period_return = 0.5 * long_return - 0.5 * short_return
    turnover = 2 * n_side / n  # fraction of universe traded (both legs), proxy for cost scaling
    return period_return, turnover


def run_backtest(
    predictions: pd.DataFrame,
    score_col: str,
    label_col: str = "y_true",
    decile: float = 0.2,
    cost_per_turnover: float = 0.0015,
    strategy_name: str = "strategy",
    forward_horizon: int = 5,
) -> BacktestResult:
    """
    predictions: DataFrame with columns [date, ticker, score_col, label_col],
    one row per (date, ticker) prediction, spanning all walk-forward test folds.

    cost_per_turnover: assumed round-trip transaction cost as a fraction of
    turnover (0.0015 = 15bps, a reasonable large-cap equity assumption).

    forward_horizon: MUST match the forward-return horizon used when the
    labels were built (features.py's forward_horizon). This matters more
    than it might look: label_col is a forward_horizon-day forward return
    recomputed at every single date, so evaluating every date's prediction
    creates massively OVERLAPPING return windows (day t's window covers
    [t, t+5], day t+1's covers [t+1, t+6] -- ~80% overlap). Overlapping
    windows are serially correlated by construction, which silently
    inflates apparent Sharpe ratios and breaks the annualization math if
    left unaddressed. The fix: only rebalance on dates spaced
    forward_horizon trading days apart, so consecutive periods never share
    return window overlap, and annualize using 252/forward_horizon periods
    per year rather than an assumed calendar frequency.
    """
    all_dates = sorted(predictions["date"].unique())
    # Take every forward_horizon-th date as a non-overlapping rebalance point
    rebalance_dates = set(all_dates[::forward_horizon])
    predictions = predictions[predictions["date"].isin(rebalance_dates)]

    period_returns = []
    turnovers = []
    dates = []

    for date, group in predictions.groupby("date"):
        if len(group) < 5 or group[score_col].nunique() < 2:
            continue
        scores = group.set_index("ticker")[score_col]
        labels = group.set_index("ticker")[label_col]
        gross_return, turnover = _decile_long_short_return(scores, labels, decile)
        net_return = gross_return - cost_per_turnover * turnover

        period_returns.append(net_return)
        turnovers.append(turnover)
        dates.append(date)

    if not period_returns:
        return BacktestResult(strategy_name, pd.Series(dtype=float), 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = pd.Series(period_returns, index=pd.DatetimeIndex(dates)).sort_index()
    periods_per_year = 252 / forward_horizon
    return _summarize(returns, turnovers, strategy_name, periods_per_year=periods_per_year)


def _summarize(
    returns: pd.Series, turnovers: list[float], strategy_name: str, periods_per_year: float = 12
) -> BacktestResult:
    ann_return = float((1 + returns.mean()) ** periods_per_year - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = round(ann_return / ann_vol, 3) if ann_vol > 0 else 0.0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    return BacktestResult(
        strategy_name=strategy_name,
        period_returns=returns,
        n_periods=len(returns),
        annualized_return=round(ann_return, 4),
        annualized_volatility=round(ann_vol, 4),
        sharpe_ratio=sharpe,
        max_drawdown=round(max_dd, 4),
        avg_turnover=round(float(np.mean(turnovers)), 3) if turnovers else 0.0,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=300)
    rows = []
    for d in dates:
        for t in range(15):
            signal = rng.normal()
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{t}",
                    "score": signal,
                    "y_true": signal * 0.015 + rng.normal(0, 0.03),
                }
            )
    df = pd.DataFrame(rows)
    result = run_backtest(df, score_col="score", strategy_name="synthetic_test", forward_horizon=5)
    print(result.summary())

"""
evaluation.py

Information Coefficient (IC): the Spearman rank correlation, within each
period's cross-section of stocks, between the model's predicted score and
the realized forward return. This is the standard quant-research metric for
"does this signal rank stocks correctly" -- it's more robust than a
regression R^2 for noisy financial data, and it's what the ranking-based
backtest in backtest.py is actually trying to exploit.

Also implements regime-conditioned evaluation: splits periods by trailing
realized market volatility (a common, simple regime proxy) and reports IC
separately for high- and low-volatility regimes, since a signal's strength
is rarely regime-invariant and claiming otherwise would be the wrong lesson
to take from a backtest that happened to cover one calm period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ICResult:
    period_ics: pd.Series  # one IC value per test period/fold
    mean_ic: float
    ic_ir: float  # "IC information ratio": mean / std -- consistency of the signal, not just its average strength

    def summary(self) -> str:
        return (
            f"Mean IC: {self.mean_ic:+.4f} | IC IR: {self.ic_ir:.3f} | "
            f"periods: {len(self.period_ics)} | "
            f"% periods with positive IC: {100 * (self.period_ics > 0).mean():.1f}%"
        )


def compute_ic(predictions: pd.DataFrame, score_col: str, label_col: str = "y_true") -> ICResult:
    """
    predictions: DataFrame with columns [date, ticker, score_col, label_col].
    Computes Spearman correlation between score and label WITHIN each date
    (i.e. cross-sectionally, comparing stocks against each other on the same
    day) -- this is what makes it an "Information Coefficient" rather than
    just a correlation across the whole pooled dataset, which would conflate
    cross-sectional ranking skill with time-varying market-wide return levels.
    """
    def _daily_ic(group):
        if len(group) < 3 or group[score_col].nunique() < 2:
            return np.nan
        corr, _ = spearmanr(group[score_col], group[label_col])
        return corr

    period_ics = predictions.groupby("date").apply(_daily_ic).dropna()
    mean_ic = float(period_ics.mean()) if len(period_ics) else 0.0
    std_ic = float(period_ics.std()) if len(period_ics) else 0.0
    ic_ir = round(mean_ic / std_ic, 3) if std_ic > 0 else 0.0

    return ICResult(period_ics=period_ics, mean_ic=round(mean_ic, 4), ic_ir=ic_ir)


def compute_regime_ic(
    predictions: pd.DataFrame,
    market_volatility: pd.Series,
    score_col: str,
    label_col: str = "y_true",
    vol_split_quantile: float = 0.5,
) -> dict[str, ICResult]:
    """
    market_volatility: a Series indexed by date (e.g. trailing 20-day
    realized volatility of an equal-weighted universe), used to split
    periods into "high vol" / "low vol" regimes at the given quantile.
    Returns IC computed separately within each regime.
    """
    threshold = market_volatility.quantile(vol_split_quantile)
    dates = predictions["date"].unique()

    regime_by_date = {
        d: ("high_vol" if market_volatility.get(d, np.nan) >= threshold else "low_vol") for d in dates
    }
    predictions = predictions.copy()
    predictions["regime"] = predictions["date"].map(regime_by_date)

    results = {}
    for regime_name in ["high_vol", "low_vol"]:
        subset = predictions[predictions["regime"] == regime_name]
        if len(subset) > 0:
            results[regime_name] = compute_ic(subset, score_col, label_col)
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=20)
    rows = []
    for d in dates:
        for t in range(15):
            true_rank_signal = rng.normal()
            score = true_rank_signal + rng.normal(0, 0.5)  # noisy but genuinely informative
            label = true_rank_signal * 0.01 + rng.normal(0, 0.02)
            rows.append({"date": d, "ticker": f"T{t}", "score": score, "y_true": label})

    df = pd.DataFrame(rows)
    result = compute_ic(df, score_col="score")
    print(result.summary())

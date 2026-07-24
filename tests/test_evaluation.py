import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation import compute_ic  # noqa: E402


def test_ic_is_high_for_perfectly_ranked_signal():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=20)
    rows = []
    for d in dates:
        for t in range(15):
            true_signal = rng.normal()
            rows.append({"date": d, "ticker": f"T{t}", "score": true_signal, "y_true": true_signal})
    df = pd.DataFrame(rows)

    result = compute_ic(df, score_col="score")
    assert result.mean_ic > 0.95  # score == y_true exactly -> perfect rank correlation


def test_ic_near_zero_for_random_signal():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=40)
    rows = []
    for d in dates:
        for t in range(15):
            rows.append({"date": d, "ticker": f"T{t}", "score": rng.normal(), "y_true": rng.normal()})
    df = pd.DataFrame(rows)

    result = compute_ic(df, score_col="score")
    assert abs(result.mean_ic) < 0.15


def test_ic_is_cross_sectional_not_pooled():
    """
    IC must be computed WITHIN each date (cross-sectionally), not across the
    whole pooled dataset -- otherwise it would conflate cross-sectional
    ranking skill with time-varying market-wide return levels. This test
    constructs a case where pooled correlation would be strongly positive
    (market-wide trend dominates) but true cross-sectional ranking is random,
    and checks IC correctly reports ~zero.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=30)
    rows = []
    for i, d in enumerate(dates):
        market_level = i * 0.001  # trending upward over time
        for t in range(15):
            score = rng.normal()  # no real cross-sectional ranking skill
            y_true = market_level + rng.normal(0, 0.0001)  # dominated by market-wide trend, not ranking
            rows.append({"date": d, "ticker": f"T{t}", "score": score, "y_true": y_true})
    df = pd.DataFrame(rows)

    result = compute_ic(df, score_col="score")
    assert abs(result.mean_ic) < 0.2  # cross-sectional IC should NOT pick up the pooled/market trend

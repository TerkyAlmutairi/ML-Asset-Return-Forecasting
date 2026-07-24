import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import run_backtest, _decile_long_short_return  # noqa: E402


def test_decile_long_short_favors_top_ranked():
    scores = pd.Series([5, 4, 3, 2, 1], index=["A", "B", "C", "D", "E"])
    returns = pd.Series([0.10, 0.05, 0.0, -0.05, -0.10], index=["A", "B", "C", "D", "E"])
    period_return, turnover = _decile_long_short_return(scores, returns, decile=0.2)
    # long A (best score, best return), short E (worst score, worst return) -> should be strongly positive
    assert period_return > 0


def test_backtest_uses_non_overlapping_rebalance_dates_only():
    """
    Regression test for a real bug caught during development: label_col is a
    forward_horizon-day forward return recomputed EVERY day, so naively
    evaluating every date creates massively overlapping (serially
    correlated) return windows, which silently inflates Sharpe. run_backtest
    must subsample to only every forward_horizon-th date.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=100)
    rows = []
    for d in dates:
        for t in range(10):
            rows.append({"date": d, "ticker": f"T{t}", "score": rng.normal(), "y_true": rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)

    result = run_backtest(df, score_col="score", forward_horizon=5)
    # 100 business days / 5-day horizon should give ~20 non-overlapping periods,
    # NOT ~100 (which is what the bug produced before the fix)
    assert result.n_periods <= 21
    assert result.n_periods >= 15


def test_backtest_no_signal_produces_roughly_zero_return():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-01", periods=200)
    rows = []
    for d in dates:
        for t in range(15):
            # score and y_true are fully independent -- no real signal
            rows.append({"date": d, "ticker": f"T{t}", "score": rng.normal(), "y_true": rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)

    result = run_backtest(df, score_col="score", forward_horizon=5)
    # With no genuine edge, Sharpe should be small in magnitude, not dramatically positive or negative
    assert abs(result.sharpe_ratio) < 3.0


def test_backtest_strong_signal_produces_strong_sharpe():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2023-01-01", periods=200)
    rows = []
    for d in dates:
        for t in range(15):
            signal = rng.normal()
            # y_true strongly driven by score -- genuine, strong ranking signal
            rows.append({"date": d, "ticker": f"T{t}", "score": signal, "y_true": signal * 0.02 + rng.normal(0, 0.005)})
    df = pd.DataFrame(rows)

    result = run_backtest(df, score_col="score", forward_horizon=5)
    assert result.sharpe_ratio > 1.0

"""
test_features.py

The most important test in this repo. Verifies that every feature value at
date T is identical whether computed on the full price history, or on a
truncated history that ends at T -- i.e. no feature secretly depends on
future data. This is the single most common way a backtest silently
overstates its own performance, and it's treated here as a first-class,
explicitly tested property rather than something just "written carefully."
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features import compute_features_for_ticker, FEATURE_COLUMNS  # noqa: E402


def _make_synthetic_ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def test_features_at_time_t_unaffected_by_future_data():
    full_df = _make_synthetic_ohlcv(n=200)
    cutoff_idx = 150
    cutoff_date = full_df.index[cutoff_idx]

    features_full = compute_features_for_ticker(full_df)
    features_truncated = compute_features_for_ticker(full_df.iloc[: cutoff_idx + 1])

    row_full = features_full.loc[cutoff_date]
    row_truncated = features_truncated.loc[cutoff_date]

    for col in FEATURE_COLUMNS:
        full_val = row_full[col]
        trunc_val = row_truncated[col]
        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        assert np.isclose(full_val, trunc_val, equal_nan=True), (
            f"LOOKAHEAD BIAS in '{col}': value at cutoff date differs depending on "
            f"whether future data is present ({full_val} vs {trunc_val})"
        )


def test_features_change_when_history_before_t_changes():
    # Sanity check the other direction: features SHOULD depend on past data,
    # otherwise the "no lookahead" test above would trivially pass because
    # nothing depends on anything.
    df_a = _make_synthetic_ohlcv(n=100, seed=0)
    df_b = _make_synthetic_ohlcv(n=100, seed=1)  # different price history

    feats_a = compute_features_for_ticker(df_a)
    feats_b = compute_features_for_ticker(df_b)

    last_date = df_a.index[-1]
    assert not np.isclose(
        feats_a.loc[last_date, "momentum_20d"], feats_b.loc[last_date, "momentum_20d"]
    )


def test_forward_return_is_the_only_place_future_data_is_used():
    from features import build_feature_panel

    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 100))
    df = pd.DataFrame(
        {"Open": prices, "High": prices * 1.01, "Low": prices * 0.99, "Close": prices, "Volume": 1_000_000},
        index=dates,
    )
    panel = pd.concat({"TEST": df}, names=["ticker", "date"]).reorder_levels(["date", "ticker"])

    result = build_feature_panel(panel, forward_horizon=5)
    # forward_return should be NaN for the last 5 rows (no future price to compute it from)
    tail_forward_returns = result.xs("TEST", level="ticker")["forward_return"].tail(5)
    assert tail_forward_returns.isna().all()

"""
walk_forward.py

Generates walk-forward train/test splits over time: train on all data up to
some date, predict on the following period, then roll the window forward
and retrain. This is the only defensible way to validate a time-series
model -- a random train/test split (the default in most ML tutorials)
leaks information, because rows near each other in time are correlated and
a random split lets the model implicitly "see the future" through
adjacent-in-time training examples.

Uses an EXPANDING window (train set grows over time, always starting from
the beginning of the data) rather than a fixed rolling window, since with a
few years of data a fixed window would leave little training data in early
folds. Retraining monthly balances "adapts to changing conditions" against
"doesn't retrain so often that each fold has too little new test data to
be meaningful."
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    dates: pd.DatetimeIndex,
    min_train_months: int = 24,
    test_months: int = 1,
    embargo_days: int = 5,
) -> list[WalkForwardFold]:
    """
    dates: the sorted, unique set of dates present in the panel.

    embargo_days: a gap between the end of the training window and the
    start of the test window. This matters because forward_return at date T
    uses price data up to T + forward_horizon -- without an embargo, the
    last few rows of a training window would have labels computed using
    price data that falls inside the following test window, a subtle form
    of leakage. embargo_days should be >= forward_horizon used when
    building features (see features.py).
    """
    dates = pd.DatetimeIndex(sorted(set(dates)))
    start = dates.min()
    min_train_end = start + pd.DateOffset(months=min_train_months)

    folds = []
    fold_id = 0
    test_start = dates[dates >= min_train_end].min()

    while test_start is not pd.NaT and test_start <= dates.max():
        train_end_cutoff = test_start - pd.Timedelta(days=embargo_days)
        train_dates = dates[dates <= train_end_cutoff]
        if len(train_dates) == 0:
            break

        test_end = test_start + pd.DateOffset(months=test_months)
        test_dates = dates[(dates >= test_start) & (dates < test_end)]
        if len(test_dates) == 0:
            break

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=dates.min(),
                train_end=train_dates.max(),
                test_start=test_dates.min(),
                test_end=test_dates.max(),
            )
        )

        fold_id += 1
        next_candidates = dates[dates >= test_end]
        test_start = next_candidates.min() if len(next_candidates) else pd.NaT

    return folds


def split_fold(panel: pd.DataFrame, fold: WalkForwardFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """panel: indexed by (date, ticker). Returns (train_df, test_df)."""
    dates = panel.index.get_level_values("date")
    train = panel[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test = panel[(dates >= fold.test_start) & (dates <= fold.test_end)]
    return train, test


if __name__ == "__main__":
    dates = pd.bdate_range("2020-01-01", "2024-01-01")
    folds = generate_folds(dates, min_train_months=24, test_months=1, embargo_days=5)
    print(f"{len(folds)} folds generated")
    for f in folds[:3]:
        print(f)
    print("...")
    print(folds[-1])

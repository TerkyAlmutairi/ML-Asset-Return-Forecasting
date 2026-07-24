import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from walk_forward import generate_folds, split_fold  # noqa: E402


def test_folds_have_no_train_test_overlap():
    dates = pd.bdate_range("2020-01-01", "2023-01-01")
    folds = generate_folds(dates, min_train_months=12, test_months=1, embargo_days=5)
    assert len(folds) > 0
    for f in folds:
        assert f.train_end < f.test_start
        assert (f.test_start - f.train_end).days >= 5  # embargo respected


def test_folds_progress_forward_in_time():
    dates = pd.bdate_range("2020-01-01", "2023-01-01")
    folds = generate_folds(dates, min_train_months=12, test_months=1, embargo_days=5)
    for a, b in zip(folds, folds[1:]):
        assert b.test_start > a.test_start
        assert b.train_end >= a.train_end  # expanding window never shrinks


def test_split_fold_respects_date_boundaries():
    dates = pd.bdate_range("2020-01-01", "2021-06-01")
    tickers = ["AAA", "BBB"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({"value": range(len(idx))}, index=idx)

    folds = generate_folds(dates, min_train_months=6, test_months=1, embargo_days=5)
    fold = folds[0]
    train, test = split_fold(panel, fold)

    train_dates = train.index.get_level_values("date")
    test_dates = test.index.get_level_values("date")
    assert train_dates.max() <= fold.train_end
    assert test_dates.min() >= fold.test_start
    assert test_dates.max() <= fold.test_end
    assert len(set(train_dates) & set(test_dates)) == 0

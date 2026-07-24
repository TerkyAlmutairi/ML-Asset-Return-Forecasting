"""
models.py

Three models, deliberately spanning a simple baseline to more expressive
gradient boosting:

  - LogisticRegression: predicts direction (up/down) only. The baseline
    every fancier model needs to beat to justify its complexity.
  - XGBoost / LightGBM: predict forward_return magnitude directly
    (regression), and their outputs are used as a cross-sectional RANKING
    SCORE, not a literal return forecast to be taken at face value --
    gradient boosting on noisy financial data is much more trustworthy at
    "is A likely to outperform B" (ranking) than at "return will be exactly
    X%" (point estimate).

All models are refit from scratch on each walk-forward fold's training data
-- no model ever sees a future fold's data at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPredictions:
    fold_id: int
    test_index: pd.MultiIndex  # (date, ticker)
    y_true: np.ndarray
    logistic_score: np.ndarray  # P(up), in [0, 1]
    xgboost_score: np.ndarray  # predicted forward_return
    lightgbm_score: np.ndarray  # predicted forward_return


def _clean(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with any NaN feature or missing label (e.g. early rows before
    rolling windows have enough history, or the last few rows of a fold
    where forward_return can't be computed)."""
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


def fit_and_predict_fold(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, fold_id: int
) -> FoldPredictions | None:
    X_train_c, y_train_c = _clean(X_train, y_train)
    X_test_c, y_test_c = _clean(X_test, y_test)

    if len(X_train_c) < 50 or len(X_test_c) < 5:
        return None  # not enough clean data in this fold to fit/evaluate meaningfully

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_c)
    X_test_scaled = scaler.transform(X_test_c)

    # --- Logistic Regression: direction only ---
    y_train_direction = (y_train_c > 0).astype(int)
    if y_train_direction.nunique() < 2:
        logistic_score = np.full(len(X_test_c), 0.5)
    else:
        logreg = LogisticRegression(max_iter=1000, C=1.0)
        logreg.fit(X_train_scaled, y_train_direction)
        logistic_score = logreg.predict_proba(X_test_scaled)[:, 1]

    # --- XGBoost: regression on forward_return ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train_c, y_train_c)
    xgboost_score = xgb_model.predict(X_test_c)

    # --- LightGBM: regression on forward_return ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_train_c, y_train_c)
    lightgbm_score = lgb_model.predict(X_test_c)

    return FoldPredictions(
        fold_id=fold_id,
        test_index=X_test_c.index,
        y_true=y_test_c.loc[X_test_c.index].values,
        logistic_score=logistic_score,
        xgboost_score=xgboost_score,
        lightgbm_score=lightgbm_score,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.MultiIndex.from_arrays(
        [pd.bdate_range("2023-01-01", periods=n), [f"T{i%10}" for i in range(n)]], names=["date", "ticker"]
    )
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.normal(0, 0.02, n), index=idx)

    result = fit_and_predict_fold(X.iloc[:400], y.iloc[:400], X.iloc[400:], y.iloc[400:], fold_id=0)
    print(f"Test set size: {len(result.y_true)}")
    print(f"XGBoost predictions range: [{result.xgboost_score.min():.4f}, {result.xgboost_score.max():.4f}]")

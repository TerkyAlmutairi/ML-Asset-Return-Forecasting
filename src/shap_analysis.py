"""
shap_analysis.py

Uses SHAP (SHapley Additive exPlanations) to explain the XGBoost model's
predictions: which features actually drive the ranking score, and in which
direction. This is what turns "a model that predicts returns" into
something a researcher can interrogate -- if momentum and volatility
dominate but RSI contributes almost nothing, that's a real, checkable
finding, not just a black-box number.

Fits one explainer on the FINAL walk-forward fold's trained model (the
most recent, most data-rich fold) rather than refitting a separate model
just for explanation -- the SHAP values describe the same model that
actually generated predictions in the backtest, not a different one.
"""

from __future__ import annotations

import shap
import numpy as np
import pandas as pd
import xgboost as xgb


def compute_shap_importance(
    X_train: pd.DataFrame, y_train: pd.Series, X_explain: pd.DataFrame
) -> pd.DataFrame:
    """
    Fits an XGBoost model on X_train/y_train (same hyperparameters as
    models.py, kept in sync manually -- see README note on why this isn't
    factored into one shared function) and returns mean absolute SHAP value
    per feature over X_explain, sorted descending -- the standard
    "global feature importance" summary derived from SHAP.
    """
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_explain.columns)
    importance = importance.sort_values(ascending=False)
    importance.name = "mean_abs_shap"
    return importance.to_frame()


def compute_shap_direction(
    X_train: pd.DataFrame, y_train: pd.Series, X_explain: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns, per feature, the correlation between the feature's raw value
    and its SHAP value across X_explain -- a simple, interpretable proxy for
    "does higher [feature] push the prediction up or down, on average."
    Not a substitute for looking at a real SHAP dependence plot (which can
    reveal non-monotonic relationships this collapses), but a useful,
    text-summarizable signal for a written report.
    """
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    directions = {}
    for i, col in enumerate(X_explain.columns):
        feature_vals = X_explain[col].values
        shap_col = shap_values[:, i]
        if np.std(feature_vals) > 0 and np.std(shap_col) > 0:
            directions[col] = float(np.corrcoef(feature_vals, shap_col)[0, 1])
        else:
            directions[col] = 0.0

    return pd.Series(directions, name="feature_shap_correlation").sort_values(ascending=False).to_frame()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        {
            "momentum": rng.normal(size=n),
            "volatility": rng.uniform(0.01, 0.05, n),
            "noise_feature": rng.normal(size=n),
        }
    )
    # y genuinely depends on momentum (positive) and volatility (negative), not noise_feature
    y = 0.02 * X["momentum"] - 0.3 * X["volatility"] + rng.normal(0, 0.005, n)

    importance = compute_shap_importance(X.iloc[:400], y.iloc[:400], X.iloc[400:])
    print(importance)
    print()
    direction = compute_shap_direction(X.iloc[:400], y.iloc[:400], X.iloc[400:])
    print(direction)

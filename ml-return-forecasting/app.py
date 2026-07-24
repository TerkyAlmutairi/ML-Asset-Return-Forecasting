"""
app.py

Streamlit front end. Runs the full walk-forward research pipeline live:
fetches real price data, builds features, walk-forward trains three models,
computes Information Coefficient, backtests each model against buy-and-hold
and momentum benchmarks, and shows SHAP feature importance.

Runtime note: with the default ~15-name universe and several years of
history, a full run involves dozens of walk-forward folds each fitting
three models -- this can take a couple of minutes. A progress spinner is
shown; there's no way to meaningfully shortcut this without compromising
the walk-forward methodology itself (that's kind of the point).
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DEFAULT_UNIVERSE  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="ML Asset Return Forecasting", page_icon="📊", layout="centered")

st.title("📊 ML Asset Return Forecasting")
st.caption(
    "A cross-sectional equity ranking pipeline: Logistic Regression, XGBoost, and LightGBM "
    "trained with strict walk-forward validation, evaluated with Information Coefficient, "
    "and backtested against buy-and-hold and momentum benchmarks -- with SHAP explainability."
)

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Universe", DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)
    start = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today())
    min_train_months = st.slider("Minimum training window (months)", 12, 36, 24)
    run_button = st.button("Run full pipeline", type="primary", use_container_width=True)
    st.caption("⚠️ This runs dozens of walk-forward folds live -- can take 1-3 minutes.")

if run_button:
    if len(tickers) < 5:
        st.error("Select at least 5 tickers -- cross-sectional ranking needs a reasonable universe size.")
    else:
        with st.spinner("Running walk-forward pipeline: fetching data, training models, backtesting..."):
            try:
                result = run_pipeline(
                    tickers=tickers, start=str(start), end=str(end), min_train_months=min_train_months
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        st.success(f"Completed {result.n_folds_run} walk-forward folds.")

        st.subheader("Information Coefficient (out-of-sample)")
        ic_rows = [
            {"model": name, "mean_ic": ic.mean_ic, "ic_ir": ic.ic_ir, "% positive periods": round(100 * (ic.period_ics > 0).mean(), 1)}
            for name, ic in result.ic_results.items()
        ]
        st.dataframe(pd.DataFrame(ic_rows).set_index("model"))
        st.caption(
            "IC measures whether the model correctly RANKS stocks against each other each period. "
            "Values near zero mean no detectable ranking skill -- a legitimate, common outcome, not a failed run."
        )

        st.subheader("Backtest: model strategies vs. benchmarks")
        bt_rows = [
            {
                "strategy": name,
                "annualized_return": bt.annualized_return,
                "sharpe_ratio": bt.sharpe_ratio,
                "max_drawdown": bt.max_drawdown,
                "periods": bt.n_periods,
            }
            for name, bt in result.backtest_results.items()
        ]
        st.dataframe(pd.DataFrame(bt_rows).set_index("strategy"))
        st.caption(
            "All strategies run through the identical backtest engine (same universe, same rebalance "
            "frequency, same transaction cost assumption) -- the ML models should be judged against the "
            "momentum benchmark, not against zero."
        )

        st.subheader("SHAP feature importance (XGBoost, final fold)")
        if not result.shap_importance.empty:
            st.bar_chart(result.shap_importance["mean_abs_shap"])
        else:
            st.info("SHAP importance unavailable (no successful fold to explain).")

        with st.expander("Regime-conditioned IC (high vs. low volatility periods)"):
            for model, regimes in result.regime_ic_results.items():
                st.write(f"**{model}**")
                for regime_name, ic in regimes.items():
                    st.write(f"  {regime_name}: {ic.summary()}")
else:
    st.info("Configure the universe and date range in the sidebar, then click **Run full pipeline**.")

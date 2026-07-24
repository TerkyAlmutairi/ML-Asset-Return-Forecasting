# ML Asset Return Forecasting

A cross-sectional equity ranking pipeline: Logistic Regression, XGBoost, and LightGBM trained on technical/momentum/volatility features, validated with strict walk-forward methodology, evaluated with Information Coefficient, and backtested against buy-and-hold and momentum benchmarks — with SHAP explainability and regime-conditioned robustness checks.

## The actual claim this project makes

Not "I predicted the stock market." The honest, defensible claim is: **I built a research pipeline rigorous enough to detect a genuine predictive signal if one exists, with every step of validation done correctly** — and correctly means reporting a weak-or-absent signal honestly when that's what the evidence shows, which is the most statistically likely outcome for public technical features on public tickers. The two things almost every amateur quant portfolio project gets wrong — leaking future information into training, and reporting backtest performance without a null/benchmark comparison — are treated here as first-class, explicitly tested properties, not afterthoughts.

## Pipeline

```
prices (yfinance) -> technical/momentum/volatility features
                              |
                     walk-forward folds (expanding window, embargo gap)
                              |
        Logistic Reg. (direction) / XGBoost (magnitude) / LightGBM (magnitude)
                              |
              pooled out-of-sample predictions
                              |
              Information Coefficient (cross-sectional rank correlation)
                              |
              backtest (long/short decile ranking) vs. buy-and-hold vs. momentum
                              |
                    SHAP explainability + regime-conditioned IC
```

## Why cross-sectional ranking, not single-asset forecasting

Most portfolio "stock prediction" projects forecast one ticker's price in isolation. Real equity quant strategies rank a universe of stocks against each other each period and bet on relative performance (long the top-ranked, short the bottom-ranked) — this is what Information Coefficient is designed to measure, and it's a fairer, more standard framing of "does this model have skill" than a single-series R².

## Methodological details that matter (and were verified, not assumed)

**Walk-forward validation, not a random train/test split** (`src/walk_forward.py`) — an expanding training window, retrained monthly, with an embargo gap between train and test to prevent a label computed from near-future prices leaking across the boundary. `tests/test_walk_forward.py` verifies zero date overlap between train and test in every fold.

**Zero lookahead in feature engineering** (`src/features.py`) — every feature at date T is provably unaffected by data after T. `tests/test_features.py` checks this directly: it computes features on the full price history and on a history truncated at T, and asserts the values at T are identical. This is the single most common way a backtest silently inflates its own results, and it's tested explicitly rather than just "written carefully."

**Non-overlapping backtest windows** (`src/backtest.py`) — a genuine bug caught during development: since `forward_return` is a 5-day forward return recomputed at every date, naively backtesting every date creates massively overlapping (serially correlated) return windows, which silently inflates Sharpe ratios. The fix subsamples to only every `forward_horizon`-th date and annualizes using the correct period count (252/horizon, not an assumed monthly frequency). `tests/test_backtest.py` has a regression test for exactly this.

**Honest benchmarking, same engine** (`src/benchmarks.py`) — buy-and-hold and a 60-day momentum factor run through the identical backtest engine as the ML models (same universe, same costs, same rebalance frequency), so any comparison is apples-to-apples. An ML model beating zero is not evidence of skill; beating momentum is a much more meaningful bar.

**Information Coefficient computed cross-sectionally, not pooled** (`src/evaluation.py`) — IC is the rank correlation *within each date* across the stock universe, not pooled across the whole dataset. `tests/test_evaluation.py` includes a specific test proving this distinction matters: a scenario with strong pooled correlation (driven by a market-wide trend) but zero genuine cross-sectional ranking skill correctly reports IC ~ 0.

**Regime-conditioned evaluation** — IC is also reported separately for high- and low-volatility periods (trailing realized volatility of the equal-weighted universe), since a signal's strength is rarely regime-invariant.

## Running it

```bash
git clone <this-repo>
cd ml-return-forecasting
pip install -r requirements.txt

# Fast offline tests (13 tests: features, walk-forward, backtest, IC math --
# all pure pandas/numpy/scipy, no model download or network needed)
pytest tests/ -v

# Full pipeline (real yfinance data, several minutes for the full walk-forward run)
python src/pipeline.py

# Interactive app
streamlit run app.py
```

## Testing approach — same split as my other projects, for the same reason

- **CI (`tests/`, 13 tests)**: pure logic — feature computation, fold generation, backtest math, IC computation — tested with synthetic data. No network, no model training, runs in seconds.
- **Local (`python src/pipeline.py`)**: the real, network-dependent full run — live price data via yfinance, real model training across dozens of walk-forward folds, real SHAP analysis. Not run in CI because it needs live market data and takes minutes, not seconds.

## What I'd say honestly in an interview

- **The most likely, and most honest, finding is a weak-to-absent signal.** Public technical/momentum/volatility features on liquid large-cap names are heavily arbitraged; if this pipeline finds strong out-of-sample IC on such a small universe, that result deserves *more* skepticism, not less — it's more likely multiple-testing luck than a real edge, and the honest move is to say so rather than present it as a discovered strategy.
- **XGBoost/LightGBM predict return magnitude but are used only as a ranking score**, not taken at face value as a literal return forecast — gradient boosting on noisy return data is far more trustworthy at "is A likely to outperform B" than at a precise point estimate.
- **A ~15-name universe is small for a "decile" split** — the long/short backtest effectively trades the top/bottom ~3 names, not a true decile. This is flagged deliberately rather than hidden; a production version would need a much larger universe (100+ names) for the decile framing to be statistically meaningful.
- **This isn't survivorship-bias-free** — the ticker universe is today's well-known large caps, not a point-in-time historical universe, so there's an implicit "stocks that turned out to matter" bias worth naming.

## Stack

Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, SHAP, yfinance, Streamlit.

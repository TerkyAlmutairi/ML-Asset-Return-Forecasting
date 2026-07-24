"""
data_loader.py

Fetches daily OHLCV data for a universe of tickers via yfinance (free, no
API key), and reshapes it into a long "panel" format: one row per
(date, ticker), which is the natural shape for cross-sectional
quant research (rank stocks against each other on a given day, rather than
just forecasting a single series in isolation).

Cross-sectional ranking is deliberately the design choice here rather than
single-asset time-series forecasting -- it's how equity quant strategies
are actually built in practice (long the top-ranked names, short the
bottom-ranked names, each period), and it also gives Information Coefficient
(a rank correlation, needs a cross-section to be meaningful) something real
to measure.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# A modest, liquid, well-known universe -- large-cap US equities across
# several sectors. Kept intentionally small (~15 names) so a walk-forward
# backtest with periodic retraining runs in reasonable time, including live
# in the Streamlit app.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS",
    "XOM", "CVX",
    "JNJ", "PFE",
    "WMT", "HD",
    "DIS",
]


def load_panel(
    tickers: list[str] | None = None, start: str = "2016-01-01", end: str | None = None
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame indexed by (date, ticker) with columns:
    Open, High, Low, Close, Volume.
    """
    tickers = tickers or DEFAULT_UNIVERSE
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, group_by="ticker")

    frames = []
    for ticker in tickers:
        try:
            df = raw[ticker].copy()
        except KeyError:
            continue  # ticker failed to download; skip rather than fail the whole pipeline
        df["ticker"] = ticker
        frames.append(df)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()
    return panel


if __name__ == "__main__":
    panel = load_panel(start="2023-01-01", end="2024-01-01")
    print(panel.shape)
    print(panel.head())

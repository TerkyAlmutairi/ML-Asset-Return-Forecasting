"""
features.py

Computes technical, momentum, and volatility features from the price panel.

The single most important correctness property of this module: every
feature at row (date=T, ticker=X) must be computable using only data from
dates <= T. This is checked explicitly in tests/test_features.py -- lookahead
bias (accidentally using future information) is the most common way a
backtest silently inflates its own results, and it's worth treating as a
first-class thing to test for, not just something to be "careful about."

All features are computed independently per ticker (via groupby) so no
ticker's feature values ever depend on another ticker's data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def compute_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """df: single-ticker OHLCV, sorted by date ascending. All rolling windows
    use only past-and-present data (pandas .rolling/.ewm are causal by construction)."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Momentum: trailing returns over several lookback windows
    out["momentum_5d"] = close.pct_change(5)
    out["momentum_20d"] = close.pct_change(20)
    out["momentum_60d"] = close.pct_change(60)

    # Trend: price relative to moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    out["price_to_ma20"] = close / ma20 - 1
    out["price_to_ma50"] = close / ma50 - 1
    out["ma20_to_ma50"] = ma20 / ma50 - 1

    # Volatility
    daily_ret = close.pct_change()
    out["volatility_20d"] = daily_ret.rolling(20).std()
    out["volatility_60d"] = daily_ret.rolling(60).std()
    out["atr_14d"] = _atr(df["High"], df["Low"], df["Close"], window=14) / close

    # Oscillators
    out["rsi_14d"] = _rsi(close, window=14)
    out["macd_hist"] = _macd_hist(close) / close  # normalized by price for cross-asset comparability

    # Volume
    out["volume_change_20d"] = df["Volume"].pct_change(20)
    vol_ma20 = df["Volume"].rolling(20).mean()
    out["relative_volume"] = df["Volume"] / vol_ma20

    return out


def build_feature_panel(price_panel: pd.DataFrame, forward_horizon: int = 5) -> pd.DataFrame:
    """
    price_panel: long-format panel from data_loader.py, indexed by (date, ticker).

    Returns a panel with feature columns plus `forward_return`: the
    forward_horizon-day forward return, which is the prediction target.
    forward_return at date T uses price at T+horizon -- this is the ONE
    place future information is intentionally used (it's the label, not a
    feature), and it's what walk_forward.py's time-based split guards
    against leaking into training.
    """
    feature_frames = []
    for ticker, df in price_panel.groupby(level="ticker"):
        df = df.droplevel("ticker").sort_index()
        feats = compute_features_for_ticker(df)
        feats["forward_return"] = df["Close"].shift(-forward_horizon) / df["Close"] - 1
        feats["ticker"] = ticker
        feature_frames.append(feats)

    panel = pd.concat(feature_frames)
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()
    return panel


FEATURE_COLUMNS = [
    "momentum_5d", "momentum_20d", "momentum_60d",
    "price_to_ma20", "price_to_ma50", "ma20_to_ma50",
    "volatility_20d", "volatility_60d", "atr_14d",
    "rsi_14d", "macd_hist",
    "volume_change_20d", "relative_volume",
]


if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))
    df = pd.DataFrame(
        {
            "Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
            "Close": prices, "Volume": rng.integers(1e6, 5e6, 200),
        },
        index=dates,
    )
    feats = compute_features_for_ticker(df)
    print(feats.tail())

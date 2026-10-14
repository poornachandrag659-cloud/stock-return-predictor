from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_21",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
    "ema_ratio_12", "ema_ratio_26", "rsi_14", "volatility_5",
    "volatility_10", "volatility_21", "volume_change", "volume_z_20",
    "intraday_range", "close_open_return",
]

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(loss.ne(0), 100.0)

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build features known at close t; target is close-to-close return t+1."""
    df = prices.copy().sort_index()
    close, volume = df["Close"].astype(float), df["Volume"].astype(float)
    ret = close.pct_change()
    for lag in [1, 2, 5, 10, 21]:
        df[f"ret_{lag}"] = close.pct_change(lag)
    for window in [5, 10, 20, 50]:
        df[f"sma_ratio_{window}"] = close / close.rolling(window).mean() - 1
    df["ema_ratio_12"] = close / close.ewm(span=12, adjust=False).mean() - 1
    df["ema_ratio_26"] = close / close.ewm(span=26, adjust=False).mean() - 1
    df["rsi_14"] = rsi(close) / 100.0
    for window in [5, 10, 21]:
        df[f"volatility_{window}"] = ret.rolling(window).std()
    df["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_mean, vol_std = volume.rolling(20).mean(), volume.rolling(20).std()
    df["volume_z_20"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    df["intraday_range"] = (df["High"] - df["Low"]) / close
    df["close_open_return"] = close / df["Open"] - 1
    df["target_return"] = close.shift(-1) / close - 1
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Final row has no next-day label and is intentionally excluded.
    return df.dropna(subset=FEATURE_COLUMNS + ["target_return"]).copy()

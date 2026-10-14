from __future__ import annotations
import pandas as pd

def download_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted OHLCV history using yfinance; network access is required."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `pip install -e .` to use yfinance.") from exc
    kwargs = {"tickers": ticker, "start": start, "auto_adjust": True, "progress": False}
    if end:
        kwargs["end"] = end
    frame = yf.download(**kwargs)
    if frame is None or frame.empty:
        raise ValueError(f"No price data returned for {ticker!r}. Check ticker, dates, and network access.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).strip().title() for c in frame.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Price source missing expected columns: {sorted(missing)}")
    frame = frame.sort_index().loc[~frame.index.duplicated(keep="last")]
    frame = frame[~frame["Close"].isna()].copy()
    if len(frame) < 250:
        raise ValueError(f"Only {len(frame)} rows downloaded; at least 250 are recommended.")
    return frame

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:02:31 2025

@author: Colby Jaskowiak

Data loading module for BRG Risk Metrics project.
Handles downloading and loading financial data from various sources.
"""

from pathlib import Path
import time
from typing import Optional

import pandas as pd
import yfinance as yf
import brg_risk_metrics.config.settings as cfg

#%%
try:
    from pandas_datareader.stooq import StooqDailyReader
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

__all__ = [
    "load_ticker_data",
    "load_spy_data",
    "load_multiple_tickers",
    "load_csv",
    "save_data",
]

# Sentinel for “argument not provided”
_MISSING = object()

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a standard, predictable column set when possible.
    Returns at least a 'Close' column; includes OHLCV when present.
    """
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not cols:
        # Last resort: if no standard columns, just return original
        return df.copy()
    out = df[cols].copy()
    # Make sure index is tz-naive and sorted
    if hasattr(out.index, "tz") and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    return out

def _download_yf(ticker: str, start: Optional[str], end: Optional[str],
                 retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame:
    """
    Robust yfinance download:
    - threads=False to avoid JSONDecodeError on some setups
    - auto_adjust=True, actions=False for clean OHLC
    - retries for transient Yahoo hiccups
    """
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,  # important for stability
                group_by=None,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize_ohlc(df)
        except Exception as e:
            last_err = e
        time.sleep(sleep_s)

    # Second attempt path: Ticker().history()
    try:
        t = yf.Ticker(ticker)
        df2 = t.history(start=start, end=end, interval="1d", auto_adjust=True)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            return _normalize_ohlc(df2)
    except Exception as e:
        last_err = e

    # Fall back to Stooq (via pandas-datareader) if available
    if _HAS_PDR:
        try:
            s = pd.to_datetime(start) if start else None
            e = pd.to_datetime(end) if end else None
            rdr = StooqDailyReader(symbols=ticker.lower(), start=s, end=e)
            df3 = rdr.read()
            if isinstance(df3, pd.DataFrame) and not df3.empty:
                df3 = df3.sort_index()
                return _normalize_ohlc(df3)
        except Exception as e:
            last_err = e

    # Final fallback: direct Stooq CSV (no extra deps)
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
        df4 = pd.read_csv(url)
        if not df4.empty and "Date" in df4.columns:
            df4["Date"] = pd.to_datetime(df4["Date"])
            df4 = df4.set_index("Date").sort_index()
            return _normalize_ohlc(df4)
    except Exception as e:
        last_err = e

    msg = f"No data returned for {ticker} from {start} to {end}"
    if last_err:
        msg += f" | last error: {last_err}"
    raise ValueError(msg)

def _default_end_if_missing(end_val) -> str:
    """
    If end is not provided, use 'tomorrow' so today's bar is included
    regardless of intraday timing or timezone.
    """
    if end_val is not _MISSING and end_val is not None:
        return end_val
    return (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

def _resample_if_needed(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Downsample daily prices to weekly (Fri) or monthly last if requested."""
    rule = {"weekly": "W-FRI", "monthly": "M"}.get(freq)
    if not rule:
        return df
    return df.resample(rule).last()

#%%
def load_ticker_data(ticker=_MISSING, start=_MISSING, end=_MISSING, freq=_MISSING) -> pd.DataFrame:
    """
    Load a single ticker as a DataFrame of OHLCV (when available).
    Any omitted argument is pulled from config.settings (cfg).
    """
    if ticker is _MISSING:
        ticker = cfg.ticker
    if start is _MISSING:
        start = cfg.start_date
    if end is _MISSING:
        end = cfg.end_date
    end = _default_end_if_missing(end)
    if freq is _MISSING:
        freq = cfg.frequency

    df = _download_yf(ticker, start, end)
    df = _resample_if_needed(df, freq)
    print(f"Loaded {ticker}: {len(df)} obs ({df.index[0].date()} → {df.index[-1].date()})")
    return df

def load_spy_data(start=_MISSING, end=_MISSING, freq=_MISSING) -> pd.DataFrame:
    """Convenience wrapper for SPY."""
    return load_ticker_data("SPY", start, end, freq)

def load_multiple_tickers(tickers, start=_MISSING, end=_MISSING, freq=_MISSING) -> dict[str, pd.DataFrame]:
    """Load many tickers; returns a dict[ticker -> DataFrame]. Skips failed tickers with a printed warning."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = load_ticker_data(t, start, end, freq)
        except Exception as e:
            print(f"[warn] {t}: {e}")
    return out

#%%
def load_csv(filepath) -> pd.DataFrame:
    """Load data from CSV with a DatetimeIndex in the first column."""
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"CSV not found: {fp}")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"CSV loaded but empty: {fp}")
    print(f"Loaded {fp}: {len(df)} observations")
    return df

def save_data(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame into the configured data directory."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    fp = cfg.data_dir / filename
    df.to_csv(fp, index=True)
    print(f"Saved to {fp}")
    return fp

if __name__ == "__main__":
    print("Testing data_loader.py...")
    # Keep this small on purpose (less likely to hit rate limits)
    spy = load_spy_data(start="2020-01-01")
    print("\nFirst 5 rows:")
    print(spy.head())
    print(f"\nColumns: {list(spy.columns)}")
    print(f"Shape: {spy.shape}")
    print("\nTest complete!")
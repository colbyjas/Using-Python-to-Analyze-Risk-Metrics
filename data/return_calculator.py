# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:02:32 2025

@author: Colby Jaskowiak

Return calculation module for BRG Risk Metrics project.
Converts price data to returns and handles related calculations.
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

import brg_risk_metrics.config.settings as cfg

#%%
# Sentinel for “argument not provided”
_MISSING = object()

PriceLike = Union[pd.Series, pd.DataFrame]
ReturnsLike = Union[pd.Series, pd.DataFrame]

def get_close(prices: PriceLike) -> PriceLike:
    """
    Return a price Series/DataFrame representing 'Close' (or 'Adj Close' if present).
    Handles:
      - Series of prices (returned as-is)
      - DataFrame with columns ['Open','High','Low','Close',...]
      - MultiIndex columns like [('SPY','Open'),...,('SPY','Close')]
    """
    if isinstance(prices, pd.Series):
        return prices

    cols = prices.columns

    # MultiIndex columns: pick level-1 == 'Close' or 'Adj Close'
    if isinstance(cols, pd.MultiIndex):
        for close_name in ("Adj Close", "Close"):
            if close_name in cols.get_level_values(-1):
                return prices.xs(close_name, axis=1, level=-1, drop_level=False).droplevel(-1, axis=1)

    # Single-level DataFrame
    for close_name in ("Adj Close", "Close"):
        if close_name in cols:
            return prices[close_name]

    # Fallback: if neither exists, assume input already suitable
    return prices

def _period_rf(rf_annual: float, freq: str, geometric: bool = False) -> float:
    """Convert annual RF to per-period RF under cfg.annualization_factor."""
    periods = cfg.annualization_factor[freq]
    if geometric:
        return (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    return rf_annual / periods

#%%
def calculate_returns(prices: PriceLike, method: object = _MISSING) -> ReturnsLike:
    """
    Calculate returns from price series/DataFrame.
    method: 'simple' (pct_change) or 'log' (log returns)
    Uses cfg.return_method if omitted.
    """
    if method is _MISSING:
        method = cfg.return_method

    px = get_close(prices)
    if method == "simple":
        return px.pct_change()
    elif method == "log":
        return np.log(px / px.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")

def excess_returns(returns: ReturnsLike, rf_rate: object = _MISSING, freq: object = _MISSING,
                   geometric: bool = False) -> ReturnsLike:
    """
    Excess returns over risk-free rate.
    If rf_rate/freq omitted, uses cfg.risk_free_rate and cfg.frequency.
    Set geometric=True to use geometric per-period conversion.
    """
    if rf_rate is _MISSING:
        rf_rate = cfg.risk_free_rate
    if freq is _MISSING:
        freq = cfg.frequency

    prf = _period_rf(rf_rate, freq, geometric=geometric)
    return returns - prf

def annualize_return(returns: ReturnsLike, freq: object = _MISSING) -> Union[float, pd.Series]:
    """
    Annualize using geometric mean.
    Works for Series or DataFrame (column-wise).
    """
    if freq is _MISSING:
        freq = cfg.frequency
    periods = cfg.annualization_factor[freq]

    def _ann(s: pd.Series) -> float:
        r = s.dropna()
        if r.empty:
            return 0.0
        total = (1.0 + r).prod() - 1.0
        n = len(r)
        return (1.0 + total) ** (periods / max(n, 1)) - 1.0

    if isinstance(returns, pd.DataFrame):
        return returns.apply(_ann, axis=0)
    return _ann(returns)

def cumulative_returns(returns: ReturnsLike) -> ReturnsLike:
    """Cumulative returns over time for Series/DataFrame."""
    return (1.0 + returns).cumprod() - 1.0

#%%
def resample_returns(returns: ReturnsLike, target_freq: str) -> ReturnsLike:
    """
    Resample returns to a new calendar frequency.
    target_freq: 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
    """
    prices = (1.0 + returns).cumprod()
    resampled_prices = prices.resample(target_freq).last()
    return resampled_prices.pct_change()

def clean_returns(returns: ReturnsLike, remove_outliers: bool = False, n_std: float = 5.0) -> ReturnsLike:
    """
    Drop NaNs and optionally clip outliers at mean ± n_std*std.
    Works for Series/DataFrame (column-wise mask).
    """
    r = returns.dropna()
    if not remove_outliers:
        return r

    if isinstance(r, pd.Series):
        m, s = r.mean(), r.std()
        mask = (r >= m - n_std * s) & (r <= m + n_std * s)
        removed = (~mask).sum()
        if removed > 0:
            print(f"Removed {removed} outliers (±{n_std}σ)")
        return r[mask]

    # DataFrame: apply column-wise
    m, s = r.mean(), r.std()
    lower, upper = (m - n_std * s), (m + n_std * s)
    mask = (r >= lower) & (r <= upper)
    removed = (~mask).sum().sum()
    if removed > 0:
        print(f"Removed {removed} outliers across columns (±{n_std}σ)")
    return r.where(mask).dropna()

def return_stats(returns: pd.Series) -> dict:
    """
    Summary stats for a 1D return series.
    (If you have a DataFrame, call this per column.)
    """
    r = returns.dropna()
    return {
        "count": int(len(r)),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "median": float(r.median()),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),
        "q01": float(r.quantile(0.01)),
        "q05": float(r.quantile(0.05)),
        "q25": float(r.quantile(0.25)),
        "q75": float(r.quantile(0.75)),
        "q95": float(r.quantile(0.95)),
        "q99": float(r.quantile(0.99)),
    }

def print_stats(returns: pd.Series, name: str = "Returns") -> None:
    """Pretty-print summary statistics for a 1D return series."""
    s = return_stats(returns)
    print(f"\n{'='*60}")
    print(f"{name.upper()} - SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Count:           {s['count']:>12,}")
    print(f"Mean:            {s['mean']:>12.6f}")
    print(f"Std Dev:         {s['std']:>12.6f}")
    print(f"Min:             {s['min']:>12.6f}")
    print(f"Max:             {s['max']:>12.6f}")
    print(f"Median:          {s['median']:>12.6f}")
    print(f"Skewness:        {s['skew']:>12.4f}")
    print(f"Kurtosis:        {s['kurtosis']:>12.4f}")
    print(f"\nPercentiles:")
    print(f"  1%:            {s['q01']:>12.6f}")
    print(f"  5%:            {s['q05']:>12.6f}")
    print(f"  25%:           {s['q25']:>12.6f}")
    print(f"  75%:           {s['q75']:>12.6f}")
    print(f"  95%:           {s['q95']:>12.6f}")
    print(f"  99%:           {s['q99']:>12.6f}")
    print(f"{'='*60}\n")

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data

    print("Testing return_calculator.py...\n")

    # Load OHLCV for SPY and extract a clean Close series
    spy_px = load_spy_data(start="2020-01-01")
    close = get_close(spy_px)

    # If DataFrame (e.g., multiple tickers), take first column for demo
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    retn = calculate_returns(close)  # uses cfg.return_method
    print(f"\nCalculated {retn.dropna().shape[0]} return observations")

    print_stats(retn, "SPY Daily Returns")

    cum_ret = cumulative_returns(retn)
    ann_ret = annualize_return(retn.dropna())
    ex_ret  = excess_returns(retn)

    print(f"Total cumulative return: {cum_ret.iloc[-1]:.2%}")
    print(f"Annualized return: {ann_ret:.2%}")
    print(f"Mean excess return (per {cfg.frequency}): {ex_ret.mean():.6f}")

    print("\nTest complete!")
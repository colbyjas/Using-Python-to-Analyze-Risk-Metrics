# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:04:15 2025

@author: Colby Jaskowiak

Volatility calculations for BRG Risk Metrics project.
"""

import numpy as np 
import pandas as pd

import brg_risk_metrics.config.settings as cfg

#%%
_MISSING = object()

def historical_volatility(returns, window=_MISSING, annualize=True, freq=_MISSING):
    """
    Calculate historical volatility (rolling or full sample).
    
    Parameters:
    - returns: pd.Series of returns
    - window: int or None. If None, uses full sample. If int, returns rolling vol.
    - annualize: bool, whether to annualize
    - freq: str, frequency for annualization
    """
    if freq is _MISSING:
        freq = cfg.frequency
    
    r = returns.dropna()
    
    if window is _MISSING or window is None:
        vol = r.std()
    else:
        vol = r.rolling(window=window).std()
    
    if annualize:
        factor = np.sqrt(cfg.annualization_factor[freq])
        vol = vol * factor
    
    return vol

def ewma_volatility(returns, lambda_param=_MISSING, annualize=True, freq=_MISSING):
    """
    Exponentially weighted moving average (EWMA) volatility.
    
    Parameters:
    - returns: pd.Series of returns
    - lambda_param: decay factor (default from cfg.ewma_lambda, typically 0.94)
    - annualize: bool
    - freq: str
    """
    if lambda_param is _MISSING:
        lambda_param = cfg.ewma_lambda
    if freq is _MISSING:
        freq = cfg.frequency
    
    r = returns.dropna()
    
    # EWMA variance
    var_ewma = r.ewm(alpha=1-lambda_param, adjust=False).var()
    vol_ewma = np.sqrt(var_ewma)
    
    if annualize:
        factor = np.sqrt(cfg.annualization_factor[freq])
        vol_ewma = vol_ewma * factor
    
    return vol_ewma

def realized_volatility(returns, window=_MISSING, annualize=True, freq=_MISSING):
    """
    Realized volatility over rolling windows.
    Alias for historical_volatility with a window.
    """
    if window is _MISSING:
        window = cfg.default_rolling_window
    
    return historical_volatility(returns, window=window, annualize=annualize, freq=freq)

def volatility_summary(returns, windows=None, annualize=True, freq=_MISSING):
    """
    Calculate volatility using multiple methods/windows.
    
    Returns dict with keys:
    - 'full_sample': full sample vol
    - 'ewma': EWMA vol (last value)
    - 'rolling_30d', 'rolling_90d', 'rolling_252d': rolling vols (last value)
    """
    if freq is _MISSING:
        freq = cfg.frequency
    
    if windows is None:
        windows = [30, 90, 252]
    
    summary = {}
    
    # Full sample
    summary['full_sample'] = historical_volatility(returns, window=None, annualize=annualize, freq=freq)
    
    # EWMA (last value)
    ewma = ewma_volatility(returns, annualize=annualize, freq=freq)
    summary['ewma'] = ewma.iloc[-1] if isinstance(ewma, pd.Series) else ewma
    
    # Rolling windows (last value)
    for w in windows:
        rolling = historical_volatility(returns, window=w, annualize=annualize, freq=freq)
        summary[f'rolling_{w}d'] = rolling.iloc[-1] if isinstance(rolling, pd.Series) else rolling
    
    return summary

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing volatility.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Full sample volatility
    vol_full = historical_volatility(returns)
    print(f"Full sample volatility (annualized): {vol_full:.4f} ({vol_full*100:.2f}%)")
    
    # EWMA volatility
    vol_ewma = ewma_volatility(returns)
    print(f"EWMA volatility (current): {vol_ewma.iloc[-1]:.4f} ({vol_ewma.iloc[-1]*100:.2f}%)")
    
    # Rolling 30-day volatility
    vol_30d = realized_volatility(returns, window=30)
    print(f"30-day rolling volatility (current): {vol_30d.iloc[-1]:.4f} ({vol_30d.iloc[-1]*100:.2f}%)")
    
    # Summary
    print("\nVolatility Summary:")
    summary = volatility_summary(returns)
    for key, val in summary.items():
        print(f"  {key:20s}: {val:.4f} ({val*100:.2f}%)")
    
    print("\nTest complete!")
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:04:37 2025

@author: Colby Jaskowiak

Risk-adjusted performance ratios for BRG Risk Metrics project.
"""

import numpy as np
import pandas as pd

import brg_risk_metrics.config.settings as cfg

#%%
_MISSING = object()

def sharpe_ratio(returns, rf_rate=_MISSING, freq=_MISSING, annualize=True):
    """
    Sharpe Ratio = (Return - RiskFree) / Volatility
    
    Returns annualized Sharpe by default.
    """
    from brg_risk_metrics.data.return_calculator import excess_returns, annualize_return
    
    if rf_rate is _MISSING:
        rf_rate = cfg.risk_free_rate
    if freq is _MISSING:
        freq = cfg.frequency
    
    r = returns.dropna()
    
    # Calculate excess returns
    excess = excess_returns(r, rf_rate=rf_rate, freq=freq)
    
    if annualize:
        # Annualized excess return / annualized volatility
        ann_excess = annualize_return(excess, freq=freq)
        ann_vol = r.std() * np.sqrt(cfg.annualization_factor[freq])
        
        if ann_vol == 0:
            return np.inf if ann_excess > 0 else 0
        
        return ann_excess / ann_vol
    else:
        # Period Sharpe
        mean_excess = excess.mean()
        std_excess = excess.std()
        
        if std_excess == 0:
            return np.inf if mean_excess > 0 else 0
        
        return mean_excess / std_excess

def sortino_ratio(returns, mar=_MISSING, rf_rate=_MISSING, freq=_MISSING, annualize=True):
    """
    Sortino Ratio = (Return - MAR) / Downside Deviation
    
    MAR (Minimum Acceptable Return):
    - 'zero': 0%
    - 'risk_free': uses rf_rate
    - float: custom value
    """
    from brg_risk_metrics.data.return_calculator import annualize_return
    
    if mar is _MISSING:
        mar = cfg.sortino_mar
    if rf_rate is _MISSING:
        rf_rate = cfg.risk_free_rate
    if freq is _MISSING:
        freq = cfg.frequency
    
    r = returns.dropna()
    
    # Determine MAR value
    if mar == 'zero':
        mar_value = 0.0
    elif mar == 'risk_free':
        mar_value = rf_rate / cfg.annualization_factor[freq]
    else:
        mar_value = float(mar)
    
    # Downside deviation (only negative deviations from MAR)
    downside = r[r < mar_value] - mar_value
    downside_std = np.sqrt((downside ** 2).mean())
    
    if annualize:
        ann_ret = annualize_return(r, freq=freq)
        ann_mar = mar_value * cfg.annualization_factor[freq]
        ann_downside_std = downside_std * np.sqrt(cfg.annualization_factor[freq])
        
        if ann_downside_std == 0:
            return np.inf if (ann_ret - ann_mar) > 0 else 0
        
        return (ann_ret - ann_mar) / ann_downside_std
    else:
        mean_ret = r.mean()
        
        if downside_std == 0:
            return np.inf if (mean_ret - mar_value) > 0 else 0
        
        return (mean_ret - mar_value) / downside_std

def information_ratio(returns, benchmark_returns, freq=_MISSING, annualize=True):
    """
    Information Ratio = (Return - Benchmark) / Tracking Error
    
    Measures excess return per unit of active risk.
    """
    from brg_risk_metrics.data.return_calculator import annualize_return
    
    if freq is _MISSING:
        freq = cfg.frequency
    
    r = returns.dropna()
    b = benchmark_returns.dropna()
    
    # Align indices
    common_idx = r.index.intersection(b.index)
    r = r.loc[common_idx]
    b = b.loc[common_idx]
    
    # Active returns
    active = r - b
    
    if annualize:
        ann_active = annualize_return(active, freq=freq)
        tracking_error = active.std() * np.sqrt(cfg.annualization_factor[freq])
        
        if tracking_error == 0:
            return np.inf if ann_active > 0 else 0
        
        return ann_active / tracking_error
    else:
        mean_active = active.mean()
        tracking_error = active.std()
        
        if tracking_error == 0:
            return np.inf if mean_active > 0 else 0
        
        return mean_active / tracking_error

def treynor_ratio(returns, beta, rf_rate=_MISSING, freq=_MISSING, annualize=True):
    """
    Treynor Ratio = (Return - RiskFree) / Beta
    
    Measures excess return per unit of systematic risk.
    """
    from brg_risk_metrics.data.return_calculator import excess_returns, annualize_return
    
    if rf_rate is _MISSING:
        rf_rate = cfg.risk_free_rate
    if freq is _MISSING:
        freq = cfg.frequency
    
    r = returns.dropna()
    excess = excess_returns(r, rf_rate=rf_rate, freq=freq)
    
    if annualize:
        ann_excess = annualize_return(excess, freq=freq)
        return ann_excess / beta if beta != 0 else np.inf
    else:
        mean_excess = excess.mean()
        return mean_excess / beta if beta != 0 else np.inf

def omega_ratio(returns, threshold=0.0):
    """
    Omega Ratio = Probability-weighted gains / Probability-weighted losses
    
    Ratio of area above threshold to area below threshold.
    threshold: typically 0 or MAR
    """
    r = returns.dropna()
    
    gains = r[r > threshold] - threshold
    losses = threshold - r[r < threshold]
    
    total_gains = gains.sum() if len(gains) > 0 else 0
    total_losses = losses.sum() if len(losses) > 0 else 0
    
    if total_losses == 0:
        return np.inf if total_gains > 0 else 1.0
    
    return total_gains / total_losses

def ratio_summary(returns, rf_rate=_MISSING, freq=_MISSING):
    """
    Calculate all risk-adjusted ratios.
    
    Returns dict with:
    - sharpe_ratio
    - sortino_ratio
    - omega_ratio
    """
    if rf_rate is _MISSING:
        rf_rate = cfg.risk_free_rate
    if freq is _MISSING:
        freq = cfg.frequency
    
    return {
        'sharpe_ratio': sharpe_ratio(returns, rf_rate=rf_rate, freq=freq),
        'sortino_ratio': sortino_ratio(returns, rf_rate=rf_rate, freq=freq),
        'omega_ratio': omega_ratio(returns, threshold=0.0)
    }

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing ratios.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Sharpe Ratio
    sharpe = sharpe_ratio(returns)
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Sortino Ratio
    sortino = sortino_ratio(returns)
    print(f"Sortino Ratio: {sortino:.4f}")
    
    # Omega Ratio
    omega = omega_ratio(returns)
    print(f"Omega Ratio: {omega:.4f}")
    
    # Full summary
    print("\nRatio Summary:")
    summary = ratio_summary(returns)
    for key, val in summary.items():
        print(f"  {key:20s}: {val:.4f}")
    
    print("\nTest complete!")
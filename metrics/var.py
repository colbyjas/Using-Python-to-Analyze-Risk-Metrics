# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:04:26 2025

@author: Colby Jaskowiak

Value at Risk (VaR) calculations for BRG Risk Metrics project.
"""

import numpy as np
import pandas as pd
from scipy import stats

import brg_risk_metrics.config.settings as cfg

#%%
_MISSING = object()

def historical_var(returns, confidence=_MISSING):
    """
    Historical VaR using empirical percentile.
    
    Returns positive number (e.g., 0.02 means 2% loss).
    """
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    
    r = returns.dropna()
    percentile = (1 - confidence) * 100
    var = np.percentile(r, percentile)
    
    return abs(var)

def parametric_var(returns, confidence=_MISSING, method='normal'):
    """
    Parametric VaR assuming normal or adjusted distribution.
    
    method:
    - 'normal': assumes normal distribution
    - 'cornish_fisher': adjusts for skewness and kurtosis
    
    Returns positive number.
    """
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    
    r = returns.dropna()
    
    mean = r.mean()
    std = r.std()
    
    if method == 'normal':
        # Standard normal VaR
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mean + z_score * std)
    
    elif method == 'cornish_fisher':
        # Cornish-Fisher expansion for non-normal distributions
        z = stats.norm.ppf(1 - confidence)
        skew = stats.skew(r)
        kurt = stats.kurtosis(r)
        
        # Cornish-Fisher z-score adjustment
        z_cf = (z + 
                (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * (skew**2) / 36)
        
        var = -(mean + z_cf * std)
    
    else:
        raise ValueError("method must be 'normal' or 'cornish_fisher'")
    
    return abs(var)

def monte_carlo_var(returns, confidence=_MISSING, n_sims=_MISSING, method='normal', random_seed=_MISSING):
    """
    Monte Carlo VaR using simulated returns.
    
    method:
    - 'normal': simulate from normal distribution
    - 't': simulate from t-distribution (better for fat tails)
    - 'historical': bootstrap from historical returns
    
    Returns positive number.
    """
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    if n_sims is _MISSING:
        n_sims = cfg.monte_carlo_sims
    if random_seed is _MISSING:
        random_seed = cfg.monte_carlo_random_seed
    
    np.random.seed(random_seed)
    
    r = returns.dropna()
    mean = r.mean()
    std = r.std()
    
    if method == 'normal':
        simulated = np.random.normal(mean, std, n_sims)
    
    elif method == 't':
        # Fit t-distribution
        df, loc, scale = stats.t.fit(r)
        simulated = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims)
    
    elif method == 'historical':
        # Bootstrap
        simulated = np.random.choice(r, size=n_sims, replace=True)
    
    else:
        raise ValueError("method must be 'normal', 't', or 'historical'")
    
    percentile = (1 - confidence) * 100
    var = np.percentile(simulated, percentile)
    
    return abs(var)

def rolling_var(returns, window=_MISSING, confidence=_MISSING, method='historical'):
    """
    Calculate rolling VaR over time.
    
    method: 'historical', 'parametric', or 'monte_carlo'
    
    Returns pd.Series of VaR values.
    """
    if window is _MISSING:
        window = cfg.default_rolling_window
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    
    r = returns.dropna()
    
    if method == 'historical':
        var_func = lambda x: historical_var(x, confidence=confidence)
    elif method == 'parametric':
        var_func = lambda x: parametric_var(x, confidence=confidence, method='normal')
    else:
        raise ValueError("method must be 'historical' or 'parametric' for rolling VaR")
    
    rolling_var_series = r.rolling(window=window).apply(var_func, raw=False)
    
    return rolling_var_series

def var_summary(returns, confidence=_MISSING):
    """
    Calculate VaR using all methods.
    
    Returns dict with:
    - historical_var
    - parametric_var_normal
    - parametric_var_cf (Cornish-Fisher)
    - monte_carlo_var_normal
    - monte_carlo_var_t
    """
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    
    return {
        'historical': historical_var(returns, confidence=confidence),
        'parametric_normal': parametric_var(returns, confidence=confidence, method='normal'),
        'parametric_cf': parametric_var(returns, confidence=confidence, method='cornish_fisher'),
        'monte_carlo_normal': monte_carlo_var(returns, confidence=confidence, method='normal'),
        'monte_carlo_t': monte_carlo_var(returns, confidence=confidence, method='t')
    }

#%%

if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing var.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Test different confidence levels
    for conf in [0.90, 0.95, 0.99]:
        print(f"\n{conf*100:.0f}% VaR:")
        print(f"  Historical:           {historical_var(returns, conf):.4f} ({historical_var(returns, conf)*100:.2f}%)")
        print(f"  Parametric (Normal):  {parametric_var(returns, conf, 'normal'):.4f} ({parametric_var(returns, conf, 'normal')*100:.2f}%)")
        print(f"  Parametric (CF):      {parametric_var(returns, conf, 'cornish_fisher'):.4f} ({parametric_var(returns, conf, 'cornish_fisher')*100:.2f}%)")
        print(f"  Monte Carlo (Normal): {monte_carlo_var(returns, conf, method='normal'):.4f} ({monte_carlo_var(returns, conf, method='normal')*100:.2f}%)")
        print(f"  Monte Carlo (t):      {monte_carlo_var(returns, conf, method='t'):.4f} ({monte_carlo_var(returns, conf, method='t')*100:.2f}%)")
    
    # VaR summary at 95%
    print("\n95% VaR Summary:")
    summary = var_summary(returns, confidence=0.95)
    for key, val in summary.items():
        print(f"  {key:25s}: {val:.4f} ({val*100:.2f}%)")
    
    print("\nTest complete!")
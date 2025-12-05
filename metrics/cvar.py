# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:04:32 2025

@author: Colby Jaskowiak

Conditional Value at Risk (CVaR/Expected Shortfall) calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats

import brg_risk_metrics.config.settings as cfg

#%%
_MISSING = object()

def historical_cvar(returns, confidence=_MISSING):
    """
    Historical CVaR (Expected Shortfall) using empirical distribution.
    
    CVaR = average of all losses worse than VaR.
    
    Returns positive number.
    """
    if confidence is _MISSING:
        confidence = cfg.default_cvar_confidence
    
    r = returns.dropna()
    
    # Find VaR threshold
    percentile = (1 - confidence) * 100
    var_threshold = np.percentile(r, percentile)
    
    # Average of returns worse than VaR
    tail_losses = r[r <= var_threshold]
    
    if len(tail_losses) == 0:
        return 0.0
    
    cvar = tail_losses.mean()
    
    return abs(cvar)

def parametric_cvar(returns, confidence=_MISSING, method='normal'):
    """
    Parametric CVaR assuming normal or adjusted distribution.
    
    method:
    - 'normal': assumes normal distribution
    - 'cornish_fisher': adjusts for skewness and kurtosis
    
    Returns positive number.
    """
    if confidence is _MISSING:
        confidence = cfg.default_cvar_confidence
    
    r = returns.dropna()
    
    mean = r.mean()
    std = r.std()
    
    if method == 'normal':
        # For normal distribution: CVaR = μ - σ * φ(z) / (1-α)
        # where φ is PDF and z is VaR z-score
        z = stats.norm.ppf(1 - confidence)
        phi_z = stats.norm.pdf(z)
        cvar = -(mean - std * phi_z / (1 - confidence))
    
    elif method == 'cornish_fisher':
        # Use Cornish-Fisher adjusted VaR and tail approximation
        z = stats.norm.ppf(1 - confidence)
        skew = stats.skew(r)
        kurt = stats.kurtosis(r)
        
        # Cornish-Fisher adjustment
        z_cf = (z + 
                (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * (skew**2) / 36)
        
        # Approximate CVaR adjustment factor
        phi_z = stats.norm.pdf(z_cf)
        cvar = -(mean - std * phi_z / (1 - confidence))
    
    else:
        raise ValueError("method must be 'normal' or 'cornish_fisher'")
    
    return abs(cvar)

def monte_carlo_cvar(returns, confidence=_MISSING, n_sims=_MISSING, method='normal', random_seed=_MISSING):
    """
    Monte Carlo CVaR using simulated returns.
    
    method:
    - 'normal': simulate from normal distribution
    - 't': simulate from t-distribution (better for fat tails)
    - 'historical': bootstrap from historical returns
    
    Returns positive number.
    """
    if confidence is _MISSING:
        confidence = cfg.default_cvar_confidence
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
    
    # Find VaR threshold
    percentile = (1 - confidence) * 100
    var_threshold = np.percentile(simulated, percentile)
    
    # Average of simulated returns worse than VaR
    tail_losses = simulated[simulated <= var_threshold]
    
    if len(tail_losses) == 0:
        return 0.0
    
    cvar = tail_losses.mean()
    
    return abs(cvar)

def rolling_cvar(returns, window=_MISSING, confidence=_MISSING, method='historical'):
    """
    Calculate rolling CVaR over time.
    
    method: 'historical', 'parametric'
    
    Returns pd.Series of CVaR values.
    """
    if window is _MISSING:
        window = cfg.default_rolling_window
    if confidence is _MISSING:
        confidence = cfg.default_cvar_confidence
    
    r = returns.dropna()
    
    if method == 'historical':
        cvar_func = lambda x: historical_cvar(x, confidence=confidence)
    elif method == 'parametric':
        cvar_func = lambda x: parametric_cvar(x, confidence=confidence, method='normal')
    else:
        raise ValueError("method must be 'historical' or 'parametric'")
    
    rolling_cvar_series = r.rolling(window=window).apply(cvar_func, raw=False)
    
    return rolling_cvar_series

def cvar_summary(returns, confidence=_MISSING):
    """
    Calculate CVaR using all methods.
    
    Returns dict with:
    - historical_cvar
    - parametric_cvar_normal
    - parametric_cvar_cf
    - monte_carlo_cvar_normal
    - monte_carlo_cvar_t
    """
    if confidence is _MISSING:
        confidence = cfg.default_cvar_confidence
    
    return {
        'historical': historical_cvar(returns, confidence=confidence),
        'parametric_normal': parametric_cvar(returns, confidence=confidence, method='normal'),
        'parametric_cf': parametric_cvar(returns, confidence=confidence, method='cornish_fisher'),
        'monte_carlo_normal': monte_carlo_cvar(returns, confidence=confidence, method='normal'),
        'monte_carlo_cvar_t': monte_carlo_cvar(returns, confidence=confidence, method='t')
    }

def var_cvar_comparison(returns, confidence=_MISSING):
    """
    Compare VaR and CVaR side-by-side.
    
    Returns dict with both metrics.
    """
    from brg_risk_metrics.metrics.var import var_summary
    
    if confidence is _MISSING:
        confidence = cfg.default_cvar_confidence
    
    var_results = var_summary(returns, confidence=confidence)
    cvar_results = cvar_summary(returns, confidence=confidence)
    
    comparison = {}
    for method in var_results.keys():
        comparison[method] = {
            'VaR': var_results[method],
            'CVaR': cvar_results.get(method, cvar_results.get(f"{method}_cvar", 0)),
            'Ratio': cvar_results.get(method, 0) / var_results[method] if var_results[method] != 0 else 0
        }
    
    return comparison

#%%

if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing cvar.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Test different confidence levels
    for conf in [0.95, 0.99]:
        print(f"\n{conf*100:.0f}% CVaR:")
        print(f"  Historical:           {historical_cvar(returns, conf):.4f} ({historical_cvar(returns, conf)*100:.2f}%)")
        print(f"  Parametric (Normal):  {parametric_cvar(returns, conf, 'normal'):.4f} ({parametric_cvar(returns, conf, 'normal')*100:.2f}%)")
        print(f"  Parametric (CF):      {parametric_cvar(returns, conf, 'cornish_fisher'):.4f} ({parametric_cvar(returns, conf, 'cornish_fisher')*100:.2f}%)")
        print(f"  Monte Carlo (Normal): {monte_carlo_cvar(returns, conf, method='normal'):.4f} ({monte_carlo_cvar(returns, conf, method='normal')*100:.2f}%)")
        print(f"  Monte Carlo (t):      {monte_carlo_cvar(returns, conf, method='t'):.4f} ({monte_carlo_cvar(returns, conf, method='t')*100:.2f}%)")
    
    # CVaR summary at 95%
    print("\n95% CVaR Summary:")
    summary = cvar_summary(returns, confidence=0.95)
    for key, val in summary.items():
        print(f"  {key:25s}: {val:.4f} ({val*100:.2f}%)")
    
    # VaR vs CVaR comparison
    print("\nVaR vs CVaR Comparison (95%):")
    from brg_risk_metrics.metrics.var import var_summary
    var_sum = var_summary(returns, confidence=0.95)
    cvar_sum = cvar_summary(returns, confidence=0.95)
    
    for method in ['historical', 'parametric_normal', 'monte_carlo_normal']:
        var_val = var_sum[method]
        cvar_val = cvar_sum[method]
        print(f"  {method:25s}: VaR={var_val:.4f}, CVaR={cvar_val:.4f}, Ratio={cvar_val/var_val:.2f}")
    
    print("\nTest complete!")
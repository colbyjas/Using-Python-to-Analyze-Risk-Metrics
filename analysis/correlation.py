# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:05:29 2025

@author: Colby Jaskowiak

Correlation analysis module.
Pearson, Spearman, rolling correlations, correlation matrices.
"""

import numpy as np
import pandas as pd
from scipy import stats

import brg_risk_metrics.config.settings as cfg

#%%
def pearson_correlation(returns1, returns2):
    """
    Calculate Pearson correlation coefficient.
    
    Parameters
    ----------
    returns1 : pd.Series or np.array
        First return series
    returns2 : pd.Series or np.array
        Second return series
        
    Returns
    -------
    dict
        Correlation coefficient, p-value, and interpretation
    """
    # Align data if Series
    if isinstance(returns1, pd.Series) and isinstance(returns2, pd.Series):
        common_idx = returns1.index.intersection(returns2.index)
        r1 = returns1.loc[common_idx].values
        r2 = returns2.loc[common_idx].values
    else:
        r1 = returns1
        r2 = returns2
    
    # Remove NaN
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1 = r1[mask]
    r2 = r2[mask]
    
    # Calculate correlation
    corr, pval = stats.pearsonr(r1, r2)
    
    # Interpret strength
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    return {
        'correlation': corr,
        'p_value': pval,
        'significant': pval < 0.05,
        'strength': strength,
        'n_observations': len(r1)
    }

def spearman_correlation(returns1, returns2):
    """
    Calculate Spearman rank correlation (robust to outliers).
    
    Parameters
    ----------
    returns1 : pd.Series or np.array
        First return series
    returns2 : pd.Series or np.array
        Second return series
        
    Returns
    -------
    dict
        Correlation coefficient, p-value, and interpretation
    """
    # Align data if Series
    if isinstance(returns1, pd.Series) and isinstance(returns2, pd.Series):
        common_idx = returns1.index.intersection(returns2.index)
        r1 = returns1.loc[common_idx].values
        r2 = returns2.loc[common_idx].values
    else:
        r1 = returns1
        r2 = returns2
    
    # Remove NaN
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1 = r1[mask]
    r2 = r2[mask]
    
    # Calculate correlation
    corr, pval = stats.spearmanr(r1, r2)
    
    # Interpret strength
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    return {
        'correlation': corr,
        'p_value': pval,
        'significant': pval < 0.05,
        'strength': strength,
        'n_observations': len(r1)
    }

def kendall_tau(returns1, returns2):
    """
    Calculate Kendall's tau (rank correlation, robust to ties).
    
    Parameters
    ----------
    returns1 : pd.Series or np.array
        First return series
    returns2 : pd.Series or np.array
        Second return series
        
    Returns
    -------
    dict
        Correlation coefficient, p-value, and interpretation
    """
    # Align data if Series
    if isinstance(returns1, pd.Series) and isinstance(returns2, pd.Series):
        common_idx = returns1.index.intersection(returns2.index)
        r1 = returns1.loc[common_idx].values
        r2 = returns2.loc[common_idx].values
    else:
        r1 = returns1
        r2 = returns2
    
    # Remove NaN
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1 = r1[mask]
    r2 = r2[mask]
    
    # Calculate correlation
    corr, pval = stats.kendalltau(r1, r2)
    
    # Interpret strength
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    return {
        'correlation': corr,
        'p_value': pval,
        'significant': pval < 0.05,
        'strength': strength,
        'n_observations': len(r1)
    }

def rolling_correlation(returns1, returns2, window=252):
    """
    Calculate rolling correlation over time.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    window : int
        Rolling window size (default: 252 days = 1 year)
        
    Returns
    -------
    pd.Series
        Rolling correlation series
    """
    # Align data
    common_idx = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common_idx]
    r2 = returns2.loc[common_idx]
    
    # Calculate rolling correlation
    rolling_corr = r1.rolling(window=window, min_periods=window//2).corr(r2)
    
    return rolling_corr

def correlation_matrix(returns_df, method='pearson'):
    """
    Calculate correlation matrix for multiple assets.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with returns for multiple assets
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    if method == 'pearson':
        corr_matrix = returns_df.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = returns_df.corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = returns_df.corr(method='kendall')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr_matrix

def correlation_stability(returns1, returns2, n_periods=4):
    """
    Test correlation stability across time periods.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    n_periods : int
        Number of periods to split data into (default: 4 quarters)
        
    Returns
    -------
    pd.DataFrame
        Correlation by period
    """
    # Align data
    common_idx = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common_idx]
    r2 = returns2.loc[common_idx]
    
    # Split into periods
    n_obs = len(r1)
    period_size = n_obs // n_periods
    
    results = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else n_obs
        
        period_r1 = r1.iloc[start_idx:end_idx]
        period_r2 = r2.iloc[start_idx:end_idx]
        
        corr = pearson_correlation(period_r1, period_r2)
        
        results.append({
            'period': i + 1,
            'start_date': period_r1.index[0],
            'end_date': period_r1.index[-1],
            'correlation': corr['correlation'],
            'p_value': corr['p_value'],
            'n_observations': corr['n_observations']
        })
    
    return pd.DataFrame(results)

def correlation_by_regime(returns1, returns2, regimes):
    """
    Calculate correlation within different market regimes.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    regimes : pd.Series or np.array
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Correlation by regime
    """
    # Align data
    if isinstance(regimes, pd.Series):
        common_idx = returns1.index.intersection(returns2.index).intersection(regimes.index)
        r1 = returns1.loc[common_idx]
        r2 = returns2.loc[common_idx]
        reg = regimes.loc[common_idx]
    else:
        r1 = returns1
        r2 = returns2
        reg = regimes
    
    unique_regimes = np.unique(reg)
    
    results = []
    for regime in unique_regimes:
        if isinstance(reg, pd.Series):
            mask = reg == regime
        else:
            mask = reg == regime
        
        regime_r1 = r1[mask]
        regime_r2 = r2[mask]
        
        if len(regime_r1) > 2:  # Need at least 3 observations
            corr = pearson_correlation(regime_r1, regime_r2)
            results.append({
                'regime': regime,
                'correlation': corr['correlation'],
                'p_value': corr['p_value'],
                'significant': corr['significant'],
                'n_observations': corr['n_observations']
            })
    
    return pd.DataFrame(results)

def correlation_breakdown(returns1, returns2):
    """
    Compare correlation in up vs down markets.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series (typically market benchmark)
    returns2 : pd.Series
        Second return series
        
    Returns
    -------
    dict
        Correlations in up markets, down markets, and overall
    """
    # Align data
    common_idx = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common_idx]
    r2 = returns2.loc[common_idx]
    
    # Overall correlation
    overall = pearson_correlation(r1, r2)
    
    # Up market correlation (when returns1 > 0)
    up_mask = r1 > 0
    up_corr = pearson_correlation(r1[up_mask], r2[up_mask])
    
    # Down market correlation (when returns1 < 0)
    down_mask = r1 < 0
    down_corr = pearson_correlation(r1[down_mask], r2[down_mask])
    
    return {
        'overall': overall,
        'up_market': up_corr,
        'down_market': down_corr,
        'asymmetry': down_corr['correlation'] - up_corr['correlation']
    }

def lagged_correlation(returns1, returns2, max_lag=5):
    """
    Calculate correlation at different lags (lead-lag relationship).
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    max_lag : int
        Maximum lag to test (default: 5 days)
        
    Returns
    -------
    pd.DataFrame
        Correlation at each lag
    """
    # Align data
    common_idx = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common_idx]
    r2 = returns2.loc[common_idx]
    
    results = []
    
    # Negative lags (returns2 leads returns1)
    for lag in range(-max_lag, 0):
        r1_shifted = r1.shift(-lag)  # Shift r1 forward
        valid_idx = r1_shifted.dropna().index.intersection(r2.index)
        
        if len(valid_idx) > 2:
            corr = pearson_correlation(r1_shifted.loc[valid_idx], r2.loc[valid_idx])
            results.append({
                'lag': lag,
                'correlation': corr['correlation'],
                'interpretation': f'returns2 leads by {abs(lag)} day(s)'
            })
    
    # Zero lag (contemporaneous)
    corr = pearson_correlation(r1, r2)
    results.append({
        'lag': 0,
        'correlation': corr['correlation'],
        'interpretation': 'contemporaneous'
    })
    
    # Positive lags (returns1 leads returns2)
    for lag in range(1, max_lag + 1):
        r2_shifted = r2.shift(lag)  # Shift r2 forward
        valid_idx = r1.index.intersection(r2_shifted.dropna().index)
        
        if len(valid_idx) > 2:
            corr = pearson_correlation(r1.loc[valid_idx], r2_shifted.loc[valid_idx])
            results.append({
                'lag': lag,
                'correlation': corr['correlation'],
                'interpretation': f'returns1 leads by {lag} day(s)'
            })
    
    return pd.DataFrame(results)

def dynamic_conditional_correlation(returns1, returns2, window=252):
    """
    Estimate time-varying correlation (simple rolling version).
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    window : int
        Rolling window (default: 252 days)
        
    Returns
    -------
    dict
        Rolling correlation series and summary statistics
    """
    rolling_corr = rolling_correlation(returns1, returns2, window=window)
    
    return {
        'correlation_series': rolling_corr,
        'mean': rolling_corr.mean(),
        'std': rolling_corr.std(),
        'min': rolling_corr.min(),
        'max': rolling_corr.max(),
        'median': rolling_corr.median()
    }

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing correlation.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = calculate_returns(close)
    
    # Create synthetic second asset for testing (correlated with SPY)
    np.random.seed(42)
    noise = pd.Series(np.random.normal(0, 0.005, len(returns)), index=returns.index)
    returns2 = returns * 0.7 + noise  # 70% correlated + noise
    
    print(f"Loaded SPY: {len(returns)} obs ({returns.index[0].date()} → {returns.index[-1].date()})")
    print(f"Created synthetic Asset 2 (partially correlated)")
    
    # Test 1: Pearson correlation
    print("\n1. Pearson correlation...")
    pearson = pearson_correlation(returns, returns2)
    print(f"   Correlation: {pearson['correlation']:.4f}")
    print(f"   P-value: {pearson['p_value']:.6f}")
    print(f"   Significant: {pearson['significant']}")
    print(f"   Strength: {pearson['strength']}")
    
    # Test 2: Spearman correlation
    print("\n2. Spearman correlation (rank-based)...")
    spearman = spearman_correlation(returns, returns2)
    print(f"   Correlation: {spearman['correlation']:.4f}")
    print(f"   P-value: {spearman['p_value']:.6f}")
    
    # Test 3: Kendall's tau
    print("\n3. Kendall's tau...")
    kendall = kendall_tau(returns, returns2)
    print(f"   Correlation: {kendall['correlation']:.4f}")
    print(f"   P-value: {kendall['p_value']:.6f}")
    
    # Test 4: Rolling correlation
    print("\n4. Rolling correlation (252-day window)...")
    rolling = rolling_correlation(returns, returns2, window=252)
    print(f"   Mean: {rolling.mean():.4f}")
    print(f"   Std: {rolling.std():.4f}")
    print(f"   Min: {rolling.min():.4f}")
    print(f"   Max: {rolling.max():.4f}")
    
    # Test 5: Correlation stability
    print("\n5. Correlation stability (4 periods)...")
    stability = correlation_stability(returns, returns2, n_periods=4)
    print(stability.to_string(index=False))
    
    # Test 6: Correlation breakdown
    print("\n6. Correlation breakdown (up vs down markets)...")
    breakdown = correlation_breakdown(returns, returns2)
    print(f"   Overall: {breakdown['overall']['correlation']:.4f}")
    print(f"   Up market: {breakdown['up_market']['correlation']:.4f}")
    print(f"   Down market: {breakdown['down_market']['correlation']:.4f}")
    print(f"   Asymmetry: {breakdown['asymmetry']:.4f}")
    
    # Test 7: Lagged correlation
    print("\n7. Lagged correlation (±3 days)...")
    lagged = lagged_correlation(returns, returns2, max_lag=3)
    print(lagged.to_string(index=False))
    
    # Test 8: Dynamic correlation
    print("\n8. Dynamic conditional correlation...")
    dcc = dynamic_conditional_correlation(returns, returns2, window=126)
    print(f"   Mean: {dcc['mean']:.4f}")
    print(f"   Std: {dcc['std']:.4f}")
    print(f"   Range: [{dcc['min']:.4f}, {dcc['max']:.4f}]")
    
    print("\nTest complete!")
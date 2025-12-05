# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:04:45 2025

@author: Colby Jaskowiak

Drawdown calculations for BRG Risk Metrics project.
"""

import pandas as pd
import numpy as np

#%%
def drawdown_series(returns):
    """
    Calculate drawdown series (running drawdown from peak).
    
    Returns pd.Series of drawdowns (negative values, 0 at peaks).
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown

def max_drawdown(returns):
    """
    Calculate maximum drawdown (MDD).
    
    Returns float (positive number representing magnitude).
    Example: -0.35 drawdown returns 0.35
    """
    dd = drawdown_series(returns)
    return abs(dd.min())

def drawdown_periods(returns, threshold=0.0):
    """
    Identify all drawdown periods.
    
    Returns list of dicts with keys:
    - 'start': start date of drawdown
    - 'trough': date of maximum drawdown
    - 'end': recovery date (or None if not recovered)
    - 'depth': magnitude of drawdown (positive)
    - 'duration': days from start to recovery (or to present)
    - 'recovery_time': days from trough to recovery (or None)
    """
    dd = drawdown_series(returns)
    
    periods = []
    in_drawdown = False
    start_idx = None
    trough_idx = None
    trough_val = 0
    
    for i in range(len(dd)):
        if dd.iloc[i] < threshold and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_idx = i
            trough_idx = i
            trough_val = dd.iloc[i]
        
        elif in_drawdown:
            # Update trough if deeper
            if dd.iloc[i] < trough_val:
                trough_idx = i
                trough_val = dd.iloc[i]
            
            # Check for recovery
            if dd.iloc[i] >= threshold:
                # Recovered
                periods.append({
                    'start': dd.index[start_idx],
                    'trough': dd.index[trough_idx],
                    'end': dd.index[i],
                    'depth': abs(trough_val),
                    'duration': (dd.index[i] - dd.index[start_idx]).days,
                    'recovery_time': (dd.index[i] - dd.index[trough_idx]).days
                })
                in_drawdown = False
    
    # Handle ongoing drawdown
    if in_drawdown:
        periods.append({
            'start': dd.index[start_idx],
            'trough': dd.index[trough_idx],
            'end': None,
            'depth': abs(trough_val),
            'duration': (dd.index[-1] - dd.index[start_idx]).days,
            'recovery_time': None
        })
    
    return periods

def average_drawdown(returns):
    """Calculate average of all drawdowns."""
    periods = drawdown_periods(returns)
    if not periods:
        return 0.0
    return np.mean([p['depth'] for p in periods])

def max_drawdown_duration(returns):
    """Calculate longest drawdown duration in days."""
    periods = drawdown_periods(returns)
    if not periods:
        return 0
    return max(p['duration'] for p in periods)

def ulcer_index(returns):
    """
    Calculate Ulcer Index (UI).
    
    UI = sqrt(mean(drawdown^2))
    Measures both depth and duration of drawdowns.
    """
    dd = drawdown_series(returns)
    return np.sqrt((dd ** 2).mean())

def calmar_ratio(returns, freq='daily'):
    """
    Calmar Ratio = Annualized Return / Max Drawdown
    
    Higher is better. Typical values: 0.5-3.0
    """
    from brg_risk_metrics.data.return_calculator import annualize_return
    
    ann_ret = annualize_return(returns.dropna(), freq=freq)
    mdd = max_drawdown(returns)
    
    if mdd == 0:
        return np.inf if ann_ret > 0 else 0
    
    return ann_ret / mdd

def drawdown_summary(returns):
    """
    Calculate all drawdown metrics.
    
    Returns dict with:
    - max_drawdown
    - average_drawdown
    - max_duration (days)
    - ulcer_index
    - calmar_ratio
    - num_periods
    """
    periods = drawdown_periods(returns)
    
    return {
        'max_drawdown': max_drawdown(returns),
        'average_drawdown': average_drawdown(returns),
        'max_duration': max_drawdown_duration(returns),
        'ulcer_index': ulcer_index(returns),
        'calmar_ratio': calmar_ratio(returns),
        'num_periods': len(periods)
    }


#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing drawdown.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Maximum drawdown
    mdd = max_drawdown(returns)
    print(f"Maximum Drawdown: {mdd:.4f} ({mdd*100:.2f}%)")
    
    # Average drawdown
    avg_dd = average_drawdown(returns)
    print(f"Average Drawdown: {avg_dd:.4f} ({avg_dd*100:.2f}%)")
    
    # Ulcer Index
    ui = ulcer_index(returns)
    print(f"Ulcer Index: {ui:.4f} ({ui*100:.2f}%)")
    
    # Calmar Ratio
    calmar = calmar_ratio(returns)
    print(f"Calmar Ratio: {calmar:.4f}")
    
    # Drawdown periods
    periods = drawdown_periods(returns)
    print(f"\nNumber of drawdown periods: {len(periods)}")
    print(f"Longest drawdown duration: {max_drawdown_duration(returns)} days")
    
    # Top 3 worst drawdowns
    print("\nTop 3 Worst Drawdowns:")
    sorted_periods = sorted(periods, key=lambda x: x['depth'], reverse=True)[:3]
    for i, p in enumerate(sorted_periods, 1):
        print(f"  {i}. {p['depth']*100:.2f}% | {p['start'].date()} to {p['end'].date() if p['end'] else 'ongoing'} ({p['duration']} days)")
    
    # Full summary
    print("\nDrawdown Summary:")
    summary = drawdown_summary(returns)
    for key, val in summary.items():
        if 'ratio' in key:
            print(f"  {key:20s}: {val:.4f}")
        elif 'duration' in key or 'num' in key:
            print(f"  {key:20s}: {val}")
        else:
            print(f"  {key:20s}: {val:.4f} ({val*100:.2f}%)")
    
    print("\nTest complete!")
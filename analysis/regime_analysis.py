# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:05:48 2025

@author: Colby Jaskowiak

Regime analysis module.
Simple threshold-based regime detection and analysis.
"""

import numpy as np
import pandas as pd

import brg_risk_metrics.config.settings as cfg

#%%
def detect_regimes_volatility(returns, threshold=None, window=21):
    """
    Detect regimes based on rolling volatility threshold.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    threshold : float, optional
        Volatility threshold. If None, uses median as split
    window : int
        Rolling window for volatility calculation (default: 21 days)
        
    Returns
    -------
    dict
        'regimes': Series with regime labels (0=low vol, 1=high vol)
        'threshold': Threshold used
        'volatility': Rolling volatility series
        'regime_stats': DataFrame with statistics per regime
    """
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=window, min_periods=window//2).std()
    rolling_vol = rolling_vol.bfill().ffill()
    
    # Set threshold
    if threshold is None:
        threshold = rolling_vol.median()
    
    # Assign regimes (0 = low vol, 1 = high vol)
    regimes = (rolling_vol > threshold).astype(int)
    
    # Calculate regime statistics
    regime_stats = []
    for regime in [0, 1]:
        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            # Duration analysis
            regime_changes = np.diff(np.concatenate([[0], regime_mask.values.astype(int), [0]]))
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                durations = ends - starts
                duration_avg = np.mean(durations)
                duration_median = np.median(durations)
            else:
                duration_avg = np.nan
                duration_median = np.nan
            
            regime_stats.append({
                'regime': 'Low Vol' if regime == 0 else 'High Vol',
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'n_observations': len(regime_returns),
                'pct_time': len(regime_returns) / len(returns),
                'duration_avg': duration_avg,
                'duration_median': duration_median
            })
    
    return {
        'regimes': regimes,
        'threshold': threshold,
        'volatility': rolling_vol,
        'regime_stats': pd.DataFrame(regime_stats)
    }

def detect_regimes_trend(returns, ma_fast=50, ma_slow=200):
    """
    Detect regimes based on moving average crossover.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    ma_fast : int
        Fast moving average window (default: 50 days)
    ma_slow : int
        Slow moving average window (default: 200 days)
        
    Returns
    -------
    dict
        'regimes': Series with regime labels (0=bearish, 1=bullish)
        'ma_fast': Fast MA series
        'ma_slow': Slow MA series
        'regime_stats': DataFrame with statistics per regime
    """
    # Calculate price from returns
    price = (1 + returns).cumprod()
    
    # Calculate moving averages
    fast_ma = price.rolling(window=ma_fast, min_periods=ma_fast//2).mean()
    slow_ma = price.rolling(window=ma_slow, min_periods=ma_slow//2).mean()
    
    # Fill NaN
    fast_ma = fast_ma.bfill().ffill()
    slow_ma = slow_ma.bfill().ffill()
    
    # Assign regimes (0 = bearish/below, 1 = bullish/above)
    regimes = (fast_ma > slow_ma).astype(int)
    
    # Calculate regime statistics
    regime_stats = []
    for regime in [0, 1]:
        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            # Duration analysis
            regime_changes = np.diff(np.concatenate([[0], regime_mask.values.astype(int), [0]]))
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                durations = ends - starts
                duration_avg = np.mean(durations)
                duration_median = np.median(durations)
            else:
                duration_avg = np.nan
                duration_median = np.nan
            
            regime_stats.append({
                'regime': 'Bearish' if regime == 0 else 'Bullish',
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'n_observations': len(regime_returns),
                'pct_time': len(regime_returns) / len(returns),
                'duration_avg': duration_avg,
                'duration_median': duration_median
            })
    
    return {
        'regimes': regimes,
        'ma_fast': fast_ma,
        'ma_slow': slow_ma,
        'regime_stats': pd.DataFrame(regime_stats)
    }

def detect_regimes_quantile(returns, n_regimes=3, window=252):
    """
    Detect regimes based on return quantiles.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    n_regimes : int
        Number of regimes (default: 3 for bear/neutral/bull)
    window : int
        Rolling window for quantile calculation (default: 252 days)
        
    Returns
    -------
    dict
        'regimes': Series with regime labels (0, 1, 2, ...)
        'quantiles': Quantile boundaries
        'regime_stats': DataFrame with statistics per regime
    """
    # Calculate rolling mean
    rolling_mean = returns.rolling(window=window, min_periods=window//2).mean()
    rolling_mean = rolling_mean.bfill().ffill()
    
    # Define quantile boundaries
    quantiles = [rolling_mean.quantile(i/n_regimes) for i in range(1, n_regimes)]
    
    # Assign regimes based on quantiles
    regimes = pd.Series(index=returns.index, dtype=int)
    regimes[:] = n_regimes - 1  # Default to highest regime
    
    for i, q in enumerate(quantiles):
        regimes[rolling_mean <= q] = i
    
    # Calculate regime statistics
    regime_stats = []
    regime_names = ['Bear', 'Neutral', 'Bull'] if n_regimes == 3 else [f'Regime {i}' for i in range(n_regimes)]
    
    for regime in range(n_regimes):
        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            # Duration analysis
            regime_changes = np.diff(np.concatenate([[0], regime_mask.values.astype(int), [0]]))
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                durations = ends - starts
                duration_avg = np.mean(durations)
                duration_median = np.median(durations)
            else:
                duration_avg = np.nan
                duration_median = np.nan
            
            regime_stats.append({
                'regime': regime_names[regime],
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'n_observations': len(regime_returns),
                'pct_time': len(regime_returns) / len(returns),
                'duration_avg': duration_avg,
                'duration_median': duration_median
            })
    
    return {
        'regimes': regimes,
        'quantiles': quantiles,
        'regime_stats': pd.DataFrame(regime_stats)
    }

def detect_regimes_drawdown(returns, threshold=-0.10):
    """
    Detect crisis regimes based on drawdown threshold.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    threshold : float
        Drawdown threshold for crisis regime (default: -10%)
        
    Returns
    -------
    dict
        'regimes': Series with regime labels (0=normal, 1=crisis)
        'drawdown': Drawdown series
        'threshold': Threshold used
        'regime_stats': DataFrame with statistics per regime
    """
    from brg_risk_metrics.metrics.drawdown import drawdown_series
    
    # Calculate drawdown
    dd = drawdown_series(returns)
    
    # Assign regimes (0 = normal, 1 = crisis)
    regimes = (dd < threshold).astype(int)
    
    # Calculate regime statistics
    regime_stats = []
    for regime in [0, 1]:
        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            # Duration analysis
            regime_changes = np.diff(np.concatenate([[0], regime_mask.values.astype(int), [0]]))
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                durations = ends - starts
                duration_avg = np.mean(durations)
                duration_median = np.median(durations)
            else:
                duration_avg = np.nan
                duration_median = np.nan
            
            regime_stats.append({
                'regime': 'Normal' if regime == 0 else 'Crisis',
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'n_observations': len(regime_returns),
                'pct_time': len(regime_returns) / len(returns),
                'duration_avg': duration_avg,
                'duration_median': duration_median
            })
    
    return {
        'regimes': regimes,
        'drawdown': dd,
        'threshold': threshold,
        'regime_stats': pd.DataFrame(regime_stats)
    }

def calculate_regime_metrics(returns, regimes, metric_func, **kwargs):
    """
    Calculate a metric for each regime.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    regimes : pd.Series or np.array
        Regime labels
    metric_func : callable
        Risk metric function to calculate
    **kwargs : dict
        Additional arguments for metric_func
        
    Returns
    -------
    pd.DataFrame
        Metric values by regime
    """
    unique_regimes = np.unique(regimes)
    
    results = []
    for regime in unique_regimes:
        if isinstance(regimes, pd.Series):
            regime_mask = regimes == regime
        else:
            regime_mask = regimes == regime
        
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            try:
                metric_value = metric_func(regime_returns, **kwargs)
            except:
                metric_value = np.nan
        else:
            metric_value = np.nan
        
        results.append({
            'regime': regime,
            'metric_value': metric_value,
            'n_observations': len(regime_returns)
        })
    
    return pd.DataFrame(results)

def regime_transition_count(regimes):
    """
    Count transitions between regimes.
    
    Parameters
    ----------
    regimes : pd.Series or np.array
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Transition count matrix
    """
    if isinstance(regimes, pd.Series):
        regimes = regimes.values
    
    unique_regimes = np.unique(regimes)
    n_regimes = len(unique_regimes)
    
    # Initialize transition matrix
    transition_counts = pd.DataFrame(
        np.zeros((n_regimes, n_regimes)),
        index=unique_regimes,
        columns=unique_regimes
    )
    
    # Count transitions
    for i in range(len(regimes) - 1):
        from_regime = regimes[i]
        to_regime = regimes[i + 1]
        transition_counts.loc[from_regime, to_regime] += 1
    
    return transition_counts

def regime_transition_probability(regimes):
    """
    Calculate transition probabilities between regimes.
    
    Parameters
    ----------
    regimes : pd.Series or np.array
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Transition probability matrix
    """
    counts = regime_transition_count(regimes)
    
    # Convert counts to probabilities
    row_sums = counts.sum(axis=1)
    transition_probs = counts.div(row_sums, axis=0)
    
    return transition_probs

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    
    print("Testing regime_analysis.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2015-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = calculate_returns(close)
    
    print(f"Loaded SPY: {len(returns)} obs ({returns.index[0].date()} → {returns.index[-1].date()})")
    
    # Test 1: Volatility-based regimes
    print("\n1. Volatility-based regime detection...")
    vol_regimes = detect_regimes_volatility(returns, window=21)
    print(f"   Threshold: {vol_regimes['threshold']:.4f}")
    print(f"\n   Regime Statistics:")
    print(vol_regimes['regime_stats'].to_string(index=False))
    
    # Test 2: Trend-based regimes
    print("\n2. Trend-based regime detection (MA crossover)...")
    trend_regimes = detect_regimes_trend(returns, ma_fast=50, ma_slow=200)
    print(f"\n   Regime Statistics:")
    print(trend_regimes['regime_stats'].to_string(index=False))
    
    # Test 3: Quantile-based regimes
    print("\n3. Quantile-based regime detection (3 regimes)...")
    quant_regimes = detect_regimes_quantile(returns, n_regimes=3, window=252)
    print(f"   Quantile boundaries: {quant_regimes['quantiles']}")
    print(f"\n   Regime Statistics:")
    print(quant_regimes['regime_stats'].to_string(index=False))
    
    # Test 4: Drawdown-based regimes
    print("\n4. Drawdown-based regime detection (crisis threshold: -10%)...")
    dd_regimes = detect_regimes_drawdown(returns, threshold=-0.10)
    print(f"\n   Regime Statistics:")
    print(dd_regimes['regime_stats'].to_string(index=False))
    
    # Test 5: Calculate metrics by regime
    print("\n5. Calculating volatility by volatility regime...")
    vol_by_regime = calculate_regime_metrics(
        returns, 
        vol_regimes['regimes'], 
        historical_volatility,
        annualize=True
    )
    print(vol_by_regime.to_string(index=False))
    
    # Test 6: Transition probabilities
    print("\n6. Regime transition probabilities (volatility regimes)...")
    trans_probs = regime_transition_probability(vol_regimes['regimes'])
    print(trans_probs)
    
    print("\nTest complete!")
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:06:42 2025

@author: Colby Jaskowiak

Validation module for risk metrics.
Out-of-sample testing, walk-forward analysis, train/test splits.
"""

import numpy as np
import pandas as pd

import brg_risk_metrics.config.settings as cfg

#%%
_MISSING = object()

def train_test_split(returns, train_pct=0.7):
    """
    Split returns into training and test sets.
    
    Parameters:
    - returns: pd.Series
    - train_pct: fraction for training (default 0.7 = 70%)
    
    Returns:
    - tuple of (train_returns, test_returns)
    """
    r = returns.dropna()
    split_idx = int(len(r) * train_pct)
    
    train = r.iloc[:split_idx]
    test = r.iloc[split_idx:]
    
    return train, test

def expanding_window_validation(returns, metric_func, min_window=252, **metric_kwargs):
    """
    Expanding window (anchored) walk-forward validation.
    
    Uses all data from start up to time t to predict t+1.
    Window grows over time (anchored to start).
    
    Parameters:
    - returns: pd.Series
    - metric_func: function that calculates metric (takes returns, returns value)
    - min_window: minimum observations before starting validation
    - **metric_kwargs: arguments passed to metric_func
    
    Returns:
    - pd.Series of out-of-sample predictions
    """
    r = returns.dropna()
    predictions = []
    dates = []
    
    for i in range(min_window, len(r)):
        # Expanding window: use all data from start to i
        train_data = r.iloc[:i]
        
        # Calculate metric on training data
        prediction = metric_func(train_data, **metric_kwargs)
        
        predictions.append(prediction)
        dates.append(r.index[i])
    
    return pd.Series(predictions, index=dates)

def rolling_window_validation(returns, metric_func, window=252, **metric_kwargs):
    """
    Rolling window walk-forward validation.
    
    Uses fixed window of most recent data to predict next period.
    
    Parameters:
    - returns: pd.Series
    - metric_func: function that calculates metric
    - window: size of rolling window
    - **metric_kwargs: arguments passed to metric_func
    
    Returns:
    - pd.Series of out-of-sample predictions
    """
    r = returns.dropna()
    predictions = []
    dates = []
    
    for i in range(window, len(r)):
        # Rolling window: use last 'window' observations
        train_data = r.iloc[i-window:i]
        
        # Calculate metric
        prediction = metric_func(train_data, **metric_kwargs)
        
        predictions.append(prediction)
        dates.append(r.index[i])
    
    return pd.Series(predictions, index=dates)

def walk_forward_analysis(returns, metric_func, train_size=252, test_size=63, **metric_kwargs):
    """
    Walk-forward analysis with fixed train/test windows.
    
    Trains on 'train_size' observations, tests on next 'test_size' observations,
    then steps forward.
    
    Parameters:
    - returns: pd.Series
    - metric_func: function to calculate metric
    - train_size: training window size (default 252 = 1 year)
    - test_size: test period size (default 63 = 3 months)
    - **metric_kwargs: arguments for metric_func
    
    Returns:
    - dict with train predictions, test predictions, test dates
    """
    r = returns.dropna()
    
    train_predictions = []
    test_predictions = []
    test_dates = []
    train_dates = []
    
    i = 0
    while i + train_size + test_size <= len(r):
        # Training window
        train_data = r.iloc[i:i+train_size]
        
        # Calculate metric on training data
        train_metric = metric_func(train_data, **metric_kwargs)
        
        # Test window - assume metric stays constant
        for j in range(i+train_size, i+train_size+test_size):
            if j < len(r):
                train_predictions.append(train_metric)
                train_dates.append(r.index[i+train_size-1])
                test_predictions.append(train_metric)
                test_dates.append(r.index[j])
        
        # Step forward
        i += test_size
    
    return {
        'test_predictions': pd.Series(test_predictions, index=test_dates),
        'train_predictions': pd.Series(train_predictions, index=train_dates),
        'train_size': train_size,
        'test_size': test_size
    }

def time_series_cv(returns, metric_func, n_splits=5, test_size=0.2, **metric_kwargs):
    """
    Time series cross-validation.
    
    Splits data into n_splits, maintaining temporal order.
    Each split uses progressively more training data.
    
    Parameters:
    - returns: pd.Series
    - metric_func: function to calculate metric
    - n_splits: number of CV folds
    - test_size: fraction of data for testing in each split
    - **metric_kwargs: arguments for metric_func
    
    Returns:
    - list of dicts with train/test results for each fold
    """
    r = returns.dropna()
    results = []
    
    # Calculate split points
    total_size = len(r)
    test_points = int(total_size * test_size)
    
    for i in range(1, n_splits + 1):
        # Progressive train size
        train_end = int(total_size * i / (n_splits + 1))
        test_start = train_end
        test_end = min(test_start + test_points, total_size)
        
        if test_start >= total_size:
            break
        
        # Split data
        train_data = r.iloc[:train_end]
        test_data = r.iloc[test_start:test_end]
        
        # Calculate metrics
        train_metric = metric_func(train_data, **metric_kwargs)
        test_metric = metric_func(test_data, **metric_kwargs)
        
        results.append({
            'fold': i,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_metric': train_metric,
            'test_metric': test_metric,
            'train_period': (train_data.index[0], train_data.index[-1]),
            'test_period': (test_data.index[0], test_data.index[-1])
        })
    
    return results

def compare_in_sample_out_sample(returns, metric_func, train_pct=0.7, **metric_kwargs):
    """
    Simple in-sample vs out-of-sample comparison.
    
    Calculates metric on training data and test data separately.
    
    Returns:
    - dict with in-sample and out-of-sample metrics
    """
    train, test = train_test_split(returns, train_pct)
    
    in_sample = metric_func(train, **metric_kwargs)
    out_sample = metric_func(test, **metric_kwargs)
    
    return {
        'in_sample': in_sample,
        'out_of_sample': out_sample,
        'difference': out_sample - in_sample,
        'pct_difference': (out_sample - in_sample) / in_sample if in_sample != 0 else np.inf,
        'train_size': len(train),
        'test_size': len(test),
        'train_period': (train.index[0], train.index[-1]),
        'test_period': (test.index[0], test.index[-1])
    }

def stability_test(returns, metric_func, n_periods=10, **metric_kwargs):
    """
    Test metric stability across different time periods.
    
    Divides data into n_periods and calculates metric for each.
    
    Returns:
    - dict with stability statistics
    """
    r = returns.dropna()
    period_size = len(r) // n_periods
    
    period_metrics = []
    period_dates = []
    
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else len(r)
        
        period_data = r.iloc[start_idx:end_idx]
        metric_value = metric_func(period_data, **metric_kwargs)
        
        period_metrics.append(metric_value)
        period_dates.append((period_data.index[0], period_data.index[-1]))
    
    period_series = pd.Series(period_metrics)
    
    return {
        'period_metrics': period_metrics,
        'period_dates': period_dates,
        'mean': period_series.mean(),
        'std': period_series.std(),
        'min': period_series.min(),
        'max': period_series.max(),
        'coefficient_of_variation': period_series.std() / period_series.mean() if period_series.mean() != 0 else np.inf
    }

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    
    print("Testing validation.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Test 1: Train/test split
    print("Test 1: Train/Test Split")
    train, test = train_test_split(returns, train_pct=0.7)
    print(f"  Train size: {len(train)} ({train.index[0]} to {train.index[-1]})")
    print(f"  Test size: {len(test)} ({test.index[0]} to {test.index[-1]})")
    
    # Test 2: In-sample vs out-of-sample
    print("\nTest 2: In-Sample vs Out-of-Sample (Volatility)")
    comparison = compare_in_sample_out_sample(returns, historical_volatility, train_pct=0.7)
    print(f"  In-Sample Vol: {comparison['in_sample']:.4f}")
    print(f"  Out-of-Sample Vol: {comparison['out_of_sample']:.4f}")
    print(f"  Difference: {comparison['difference']:.4f} ({comparison['pct_difference']:.2%})")
    
    # Test 3: Stability test
    print("\nTest 3: Stability Test (VaR across 5 periods)")
    stability = stability_test(returns, historical_var, n_periods=5, confidence=0.95)
    print(f"  Mean VaR: {stability['mean']:.4f}")
    print(f"  Std Dev: {stability['std']:.4f}")
    print(f"  Range: {stability['min']:.4f} to {stability['max']:.4f}")
    print(f"  Coefficient of Variation: {stability['coefficient_of_variation']:.4f}")
    
    # Test 4: Time series CV
    print("\nTest 4: Time Series Cross-Validation (Volatility)")
    cv_results = time_series_cv(returns, historical_volatility, n_splits=3)
    for result in cv_results:
        print(f"  Fold {result['fold']}: Train={result['train_metric']:.4f}, Test={result['test_metric']:.4f}")
    
    print("\nTest complete!")
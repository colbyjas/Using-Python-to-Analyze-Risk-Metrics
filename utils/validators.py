# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:10:05 2025

@author: Colby Jaskowiak

Validators module.
Input validation, data quality checks, and error handling.
"""

import numpy as np
import pandas as pd

import brg_risk_metrics.config.settings as cfg

#%% DATA VALIDATION
def validate_returns(returns):
    """
    Validate return series.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series to validate
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # Check if empty
    if len(returns) == 0:
        errors.append("Returns series is empty")
        return False, errors
    
    # Check for all NaN
    if isinstance(returns, pd.Series):
        if returns.isna().all():
            errors.append("All values are NaN")
    else:
        if np.isnan(returns).all():
            errors.append("All values are NaN")
    
    # Check for infinite values
    if isinstance(returns, pd.Series):
        inf_count = np.isinf(returns).sum()
    else:
        inf_count = np.isinf(returns).sum()
    
    if inf_count > 0:
        errors.append(f"Contains {inf_count} infinite values")
    
    # Check for suspicious patterns
    if isinstance(returns, pd.Series):
        zero_count = (returns == 0).sum()
        nan_count = returns.isna().sum()
    else:
        zero_count = (returns == 0).sum()
        nan_count = np.isnan(returns).sum()
    
    zero_pct = zero_count / len(returns)
    nan_pct = nan_count / len(returns)
    
    if zero_pct > 0.5:
        errors.append(f"More than 50% zeros ({zero_pct:.1%})")
    
    if nan_pct > 0.3:
        errors.append(f"More than 30% NaN values ({nan_pct:.1%})")
    
    # Check for extreme values
    if isinstance(returns, pd.Series):
        valid_returns = returns.dropna()
    else:
        valid_returns = returns[~np.isnan(returns)]
    
    if len(valid_returns) > 0:
        if valid_returns.max() > 1.0:
            errors.append(f"Extreme positive return: {valid_returns.max():.2%}")
        if valid_returns.min() < -0.5:
            errors.append(f"Extreme negative return: {valid_returns.min():.2%}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def validate_confidence_level(confidence):
    """
    Validate confidence level parameter.
    
    Parameters
    ----------
    confidence : float
        Confidence level
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not isinstance(confidence, (int, float)):
        return False, "Confidence must be numeric"
    
    if confidence <= 0 or confidence >= 1:
        return False, f"Confidence must be between 0 and 1, got {confidence}"
    
    if confidence < 0.5:
        return False, f"Confidence unusually low: {confidence}"
    
    return True, None

def validate_window_size(window, data_length):
    """
    Validate window size for rolling calculations.
    
    Parameters
    ----------
    window : int
        Window size
    data_length : int
        Length of data series
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not isinstance(window, int):
        return False, "Window must be integer"
    
    if window < 2:
        return False, f"Window too small: {window} (minimum: 2)"
    
    if window > data_length:
        return False, f"Window ({window}) larger than data length ({data_length})"
    
    if window > data_length * 0.5:
        return False, f"Window ({window}) uses more than 50% of data ({data_length})"
    
    if window < 10:
        return False, f"Window ({window}) may be too small for reliable estimates"
    
    return True, None

def validate_date_range(start_date, end_date):
    """
    Validate date range.
    
    Parameters
    ----------
    start_date : pd.Timestamp
        Start date
    end_date : pd.Timestamp
        End date
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if start_date >= end_date:
        return False, f"Start date ({start_date}) must be before end date ({end_date})"
    
    # Check if range is too short
    days_diff = (end_date - start_date).days
    if days_diff < 30:
        return False, f"Date range too short: {days_diff} days (minimum: 30)"
    
    # Check if dates are in future
    now = pd.Timestamp.now()
    if start_date > now:
        return False, f"Start date is in future: {start_date}"
    
    return True, None

#%% DATA QUALITY CHECKS
def check_data_quality(returns, verbose=True):
    """
    Comprehensive data quality check.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    verbose : bool
        Print detailed report (default: True)
        
    Returns
    -------
    dict
        Quality metrics and flags
    """
    report = {}
    
    # Basic stats
    report['n_observations'] = len(returns)
    report['n_missing'] = returns.isna().sum()
    report['pct_missing'] = report['n_missing'] / len(returns)
    report['n_zeros'] = (returns == 0).sum()
    report['pct_zeros'] = report['n_zeros'] / len(returns)
    report['n_infinite'] = np.isinf(returns).sum()
    
    # Valid data
    valid_returns = returns.dropna()
    report['n_valid'] = len(valid_returns)
    
    if len(valid_returns) > 0:
        # Distribution stats
        report['mean'] = valid_returns.mean()
        report['std'] = valid_returns.std()
        report['min'] = valid_returns.min()
        report['max'] = valid_returns.max()
        report['skewness'] = valid_returns.skew()
        report['kurtosis'] = valid_returns.kurtosis()
        
        # Quality flags
        report['has_missing'] = report['n_missing'] > 0
        report['has_zeros'] = report['n_zeros'] > 0
        report['has_infinite'] = report['n_infinite'] > 0
        report['has_extreme_values'] = (valid_returns.max() > 1.0) or (valid_returns.min() < -0.5)
        report['sufficient_data'] = report['n_valid'] >= 100
        
        # Overall quality score (0-100)
        score = 100
        if report['pct_missing'] > 0.1:
            score -= 20
        if report['pct_zeros'] > 0.1:
            score -= 10
        if report['has_infinite']:
            score -= 20
        if report['has_extreme_values']:
            score -= 10
        if not report['sufficient_data']:
            score -= 30
        
        report['quality_score'] = max(0, score)
    else:
        report['quality_score'] = 0
    
    if verbose:
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Total observations:    {report['n_observations']}")
        print(f"Valid observations:    {report['n_valid']}")
        print(f"Missing values:        {report['n_missing']} ({report['pct_missing']:.1%})")
        print(f"Zero values:           {report['n_zeros']} ({report['pct_zeros']:.1%})")
        print(f"Infinite values:       {report['n_infinite']}")
        
        if len(valid_returns) > 0:
            print(f"\nDistribution:")
            print(f"  Mean:      {report['mean']:.6f}")
            print(f"  Std:       {report['std']:.6f}")
            print(f"  Min:       {report['min']:.6f}")
            print(f"  Max:       {report['max']:.6f}")
            print(f"  Skewness:  {report['skewness']:.3f}")
            print(f"  Kurtosis:  {report['kurtosis']:.3f}")
            
            print(f"\nQuality Flags:")
            print(f"  Missing data:      {'⚠️ Yes' if report['has_missing'] else '✓ No'}")
            print(f"  Zero inflation:    {'⚠️ Yes' if report['has_zeros'] else '✓ No'}")
            print(f"  Infinite values:   {'⚠️ Yes' if report['has_infinite'] else '✓ No'}")
            print(f"  Extreme values:    {'⚠️ Yes' if report['has_extreme_values'] else '✓ No'}")
            print(f"  Sufficient data:   {'✓ Yes' if report['sufficient_data'] else '⚠️ No'}")
            
            print(f"\nOverall Quality Score: {report['quality_score']}/100")
            if report['quality_score'] >= 80:
                print("  Status: ✓ Good")
            elif report['quality_score'] >= 60:
                print("  Status: ⚠️ Acceptable")
            else:
                print("  Status: ❌ Poor")
        
        print("=" * 60)
    
    return report

def check_stationarity(returns, max_lag=10):
    """
    Check for stationarity (basic test).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    max_lag : int
        Maximum lag for autocorrelation test
        
    Returns
    -------
    dict
        Stationarity indicators
    """
    from scipy import stats as sp_stats
    
    report = {}
    
    # Rolling mean and std
    window = min(len(returns) // 4, 252)
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Check if mean/std are relatively stable
    mean_variation = rolling_mean.std() / returns.mean() if returns.mean() != 0 else np.inf
    std_variation = rolling_std.std() / returns.std() if returns.std() != 0 else np.inf
    
    report['mean_variation'] = mean_variation
    report['std_variation'] = std_variation
    report['appears_stationary'] = (mean_variation < 2.0) and (std_variation < 1.0)
    
    return report

#%% PARAMETER VALIDATION
def validate_metric_params(metric_name, **params):
    """
    Validate parameters for specific metric.
    
    Parameters
    ----------
    metric_name : str
        Name of metric
    **params : dict
        Parameters to validate
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # VaR/CVaR validation
    if metric_name.lower() in ['var', 'cvar']:
        if 'confidence' in params:
            is_valid, msg = validate_confidence_level(params['confidence'])
            if not is_valid:
                errors.append(msg)
    
    # Window-based metrics
    if metric_name.lower() in ['volatility', 'rolling_var']:
        if 'window' in params and 'data_length' in params:
            is_valid, msg = validate_window_size(params['window'], params['data_length'])
            if not is_valid:
                errors.append(msg)
    
    # Monte Carlo validation
    if metric_name.lower() == 'monte_carlo':
        if 'n_paths' in params:
            n_paths = params['n_paths']
            if n_paths < 100:
                errors.append(f"n_paths too small: {n_paths} (minimum: 100)")
            elif n_paths > 100000:
                errors.append(f"n_paths very large: {n_paths} (may be slow)")
        
        if 'horizon_days' in params:
            horizon = params['horizon_days']
            if horizon < 1:
                errors.append(f"horizon_days must be positive: {horizon}")
            elif horizon > 1000:
                errors.append(f"horizon_days very large: {horizon}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def validate_input_types(returns=None, confidence=None, window=None):
    """
    Validate input data types.
    
    Parameters
    ----------
    returns : pd.Series or np.array, optional
        Return series
    confidence : float, optional
        Confidence level
    window : int, optional
        Window size
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    if returns is not None:
        if not isinstance(returns, (pd.Series, np.ndarray)):
            errors.append(f"Returns must be pd.Series or np.array, got {type(returns)}")
    
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            errors.append(f"Confidence must be numeric, got {type(confidence)}")
    
    if window is not None:
        if not isinstance(window, int):
            errors.append(f"Window must be int, got {type(window)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

#%%
if __name__ == "__main__":
    print("Testing validators.py...\n")
    
    # Create test data
    test_returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
    test_returns.iloc[10] = np.nan  # Add missing value
    test_returns.iloc[50] = 0  # Add zero
    
    # Test 1: Validate returns
    print("1. Validating returns...")
    is_valid, errors = validate_returns(test_returns)
    print(f"   Valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"   - {error}")
    
    # Test 2: Validate confidence
    print("\n2. Validating confidence levels...")
    for conf in [0.50, 0.95, 0.99, 1.5]:
        is_valid, msg = validate_confidence_level(conf)
        status = "✓" if is_valid else "✗"
        print(f"   {status} {conf}: {msg if msg else 'OK'}")
    
    # Test 3: Validate window
    print("\n3. Validating window sizes...")
    for window in [5, 21, 252, 1000, 2000]:
        is_valid, msg = validate_window_size(window, len(test_returns))
        status = "✓" if is_valid else "✗"
        print(f"   {status} {window}: {msg if msg else 'OK'}")
    
    # Test 4: Data quality check
    print("\n4. Data quality check...")
    quality = check_data_quality(test_returns, verbose=True)
    
    print("\nTest complete!")
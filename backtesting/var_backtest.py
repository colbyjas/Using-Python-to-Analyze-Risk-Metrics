# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:06:24 2025

@author: Colby Jaskowiak

VaR backtesting module.
Tests VaR model accuracy using historical data.
"""

import numpy as np
import pandas as pd

import brg_risk_metrics.config.settings as cfg
from brg_risk_metrics.backtesting.statistical_tests import (
    kupiec_pof_test, christoffersen_test, conditional_coverage_test,
    traffic_light_test
)

#%%
_MISSING = object()

def backtest_var(returns, var_method, confidence=_MISSING, window=_MISSING, **method_kwargs):
    """
    Backtest a VaR model using rolling window.
    
    Parameters:
    - returns: pd.Series of returns
    - var_method: function that calculates VaR (e.g., historical_var)
    - confidence: confidence level
    - window: rolling window size for estimation
    - **method_kwargs: additional arguments for var_method
    
    Returns:
    - dict with backtest results
    """
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    if window is _MISSING:
        window = cfg.default_rolling_window
    
    r = returns.dropna()
    
    var_estimates = []
    actual_returns = []
    violations = []
    dates = []
    
    # Rolling window backtest
    for i in range(window, len(r)):
        # In-sample data (estimation window)
        train_data = r.iloc[i-window:i]
        
        # Out-of-sample data (test point)
        test_return = r.iloc[i]
        
        # Estimate VaR on in-sample data
        var_estimate = var_method(train_data, confidence=confidence, **method_kwargs)
        
        # Check if violation occurred
        violation = test_return < -var_estimate
        
        var_estimates.append(var_estimate)
        actual_returns.append(test_return)
        violations.append(violation)
        dates.append(r.index[i])
    
    # Convert to series
    var_estimates = pd.Series(var_estimates, index=dates)
    actual_returns = pd.Series(actual_returns, index=dates)
    violations_series = pd.Series(violations, index=dates)
    
    n_violations = violations_series.sum()
    n_observations = len(violations_series)
    
    # Statistical tests
    pof_test = kupiec_pof_test(n_violations, n_observations, confidence)
    independence_test = christoffersen_test(violations_series)
    coverage_test = conditional_coverage_test(violations_series, n_observations, confidence)
    traffic_test = traffic_light_test(n_violations, n_observations, confidence)
    
    return {
        'var_estimates': var_estimates,
        'actual_returns': actual_returns,
        'violations': violations_series,
        'n_violations': n_violations,
        'n_observations': n_observations,
        'violation_rate': n_violations / n_observations,
        'expected_rate': 1 - confidence,
        'pof_test': pof_test,
        'independence_test': independence_test,
        'coverage_test': coverage_test,
        'traffic_light': traffic_test,
        'window': window,
        'confidence': confidence
    }

def compare_var_methods(returns, methods_dict, confidence=_MISSING, window=_MISSING):
    """
    Compare multiple VaR methods via backtesting.
    
    Parameters:
    - returns: pd.Series
    - methods_dict: dict of {method_name: method_function}
    - confidence: confidence level
    - window: rolling window
    
    Returns:
    - dict of backtest results for each method
    """
    if confidence is _MISSING:
        confidence = cfg.default_var_confidence
    if window is _MISSING:
        window = cfg.default_rolling_window
    
    results = {}
    
    for method_name, method_func in methods_dict.items():
        print(f"Backtesting {method_name}...")
        results[method_name] = backtest_var(returns, method_func, confidence=confidence, window=window)
    
    return results

def summarize_backtest_results(backtest_results):
    """
    Create summary table of backtest results.
    
    Parameters:
    - backtest_results: dict from compare_var_methods
    
    Returns:
    - pd.DataFrame with summary
    """
    summary_data = []
    
    for method_name, results in backtest_results.items():
        summary_data.append({
            'Method': method_name,
            'Violations': results['n_violations'],
            'Expected': f"{results['expected_rate']:.2%}",
            'Actual': f"{results['violation_rate']:.2%}",
            'POF Test': results['pof_test']['result'],
            'Independence': results['independence_test']['result'],
            'Coverage Test': results['coverage_test']['result'],
            'Traffic Light': results['traffic_light']['zone']
        })
    
    return pd.DataFrame(summary_data)

def get_violation_dates(backtest_result):
    """
    Get dates when VaR violations occurred.
    
    Returns:
    - pd.DataFrame with violation details
    """
    violations = backtest_result['violations']
    var_estimates = backtest_result['var_estimates']
    actual_returns = backtest_result['actual_returns']
    
    violation_dates = violations[violations == True].index
    
    violation_details = pd.DataFrame({
        'Date': violation_dates,
        'Actual_Return': [actual_returns.loc[d] for d in violation_dates],
        'VaR_Estimate': [var_estimates.loc[d] for d in violation_dates],
        'Excess_Loss': [abs(actual_returns.loc[d]) - var_estimates.loc[d] for d in violation_dates]
    })
    
    violation_details = violation_details.sort_values('Excess_Loss', ascending=False)
    
    return violation_details

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.metrics.var import historical_var, parametric_var
    
    print("Testing var_backtest.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Backtest historical VaR
    print("Backtesting Historical VaR...")
    backtest_result = backtest_var(returns, historical_var, confidence=0.95, window=252)
    
    print(f"\nBacktest Results:")
    print(f"  Observations: {backtest_result['n_observations']}")
    print(f"  Violations: {backtest_result['n_violations']}")
    print(f"  Expected Rate: {backtest_result['expected_rate']:.2%}")
    print(f"  Actual Rate: {backtest_result['violation_rate']:.2%}")
    print(f"\n  POF Test: {backtest_result['pof_test']['result']}")
    print(f"  Independence Test: {backtest_result['independence_test']['result']}")
    print(f"  Traffic Light: {backtest_result['traffic_light']['zone']}")
    
    # Compare methods
    print("\n" + "="*60)
    print("Comparing VaR Methods...")
    
    methods = {
        'Historical': historical_var,
        'Parametric': lambda r, confidence: parametric_var(r, confidence, method='normal')
    }
    
    comparison = compare_var_methods(returns, methods, confidence=0.95, window=252)
    summary = summarize_backtest_results(comparison)
    
    print("\nComparison Summary:")
    print(summary.to_string(index=False))
    
    # Violation details
    print("\n" + "="*60)
    print("Top 5 Worst Violations (Historical VaR):")
    violations = get_violation_dates(backtest_result)
    print(violations.head().to_string(index=False))
    
    print("\nTest complete!")
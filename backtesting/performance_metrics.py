# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:07:12 2025

@author: Colby Jaskowiak

Performance metrics for backtesting.
Evaluate prediction accuracy, forecast quality, etc.
"""

import numpy as np
import pandas as pd
from scipy import stats

#%%
def mean_absolute_error(actual, predicted):
    """
    MAE - Average absolute difference between actual and predicted.
    Lower is better.
    """
    return np.mean(np.abs(actual - predicted))

def mean_squared_error(actual, predicted):
    """
    MSE - Average squared difference.
    Lower is better.
    """
    return np.mean((actual - predicted) ** 2)

def root_mean_squared_error(actual, predicted):
    """
    RMSE - Square root of MSE.
    Same units as original data. Lower is better.
    """
    return np.sqrt(mean_squared_error(actual, predicted))

def mean_absolute_percentage_error(actual, predicted):
    """
    MAPE - Average absolute percentage error.
    Returns value between 0-100% (lower is better).
    """
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def r_squared(actual, predicted):
    """
    R² - Coefficient of determination.
    1.0 = perfect prediction, 0 = no better than mean.
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return 0
    
    return 1 - (ss_res / ss_tot)

def direction_accuracy(actual_returns, predicted_sign):
    """
    Fraction of times predicted direction matches actual.
    
    Parameters:
    - actual_returns: actual return values
    - predicted_sign: predicted direction (1 for up, -1 for down, 0 for flat)
    
    Returns:
    - accuracy between 0 and 1
    """
    actual_sign = np.sign(actual_returns)
    return np.mean(actual_sign == predicted_sign)

def hit_rate(violations_expected, violations_actual):
    """
    Fraction of time actual violations match expectations.
    Used for VaR validation.
    """
    return 1 - np.abs(violations_expected - violations_actual) / violations_expected

def sharpe_of_forecast_errors(actual, predicted):
    """
    Sharpe ratio of forecast errors.
    Measures consistency of forecast accuracy.
    Higher is better (more consistent errors).
    """
    errors = actual - predicted
    if errors.std() == 0:
        return 0
    return errors.mean() / errors.std()

def tracking_error(actual, predicted):
    """
    Standard deviation of prediction errors.
    Lower is better.
    """
    errors = actual - predicted
    return errors.std()

def information_coefficient(actual, predicted):
    """
    Correlation between actual and predicted.
    1.0 = perfect correlation, 0 = no correlation.
    """
    if len(actual) < 2:
        return 0
    return np.corrcoef(actual, predicted)[0, 1]

def max_forecast_error(actual, predicted):
    """
    Maximum absolute error in forecasts.
    Lower is better.
    """
    return np.max(np.abs(actual - predicted))

def forecast_bias(actual, predicted):
    """
    Average forecast error (measures systematic over/under prediction).
    
    Positive = overforecasting (predicted > actual)
    Negative = underforecasting (predicted < actual)
    Close to 0 is best.
    """
    return np.mean(predicted - actual)

def create_performance_report(actual, predicted, metric_name="Metric"):
    """
    Generate comprehensive performance report.
    
    Parameters:
    - actual: pd.Series or np.array of actual values
    - predicted: pd.Series or np.array of predicted values
    - metric_name: name of metric being evaluated
    
    Returns:
    - dict with all performance metrics
    """
    # Convert to numpy arrays
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(predicted, pd.Series):
        predicted = predicted.values
    
    # Align arrays (in case of different lengths)
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    report = {
        'metric_name': metric_name,
        'n_observations': len(actual),
        'mae': mean_absolute_error(actual, predicted),
        'rmse': root_mean_squared_error(actual, predicted),
        'r_squared': r_squared(actual, predicted),
        'correlation': information_coefficient(actual, predicted),
        'tracking_error': tracking_error(actual, predicted),
        'max_error': max_forecast_error(actual, predicted),
        'forecast_bias': forecast_bias(actual, predicted),
        'actual_mean': np.mean(actual),
        'predicted_mean': np.mean(predicted),
        'actual_std': np.std(actual),
        'predicted_std': np.std(predicted)
    }
    
    # Add MAPE if no zeros in actual
    if not np.any(actual == 0):
        report['mape'] = mean_absolute_percentage_error(actual, predicted)
    
    return report

def print_performance_report(report):
    """
    Print formatted performance report.
    """
    print(f"\n{'='*60}")
    print(f"PERFORMANCE REPORT: {report['metric_name']}")
    print(f"{'='*60}")
    print(f"Observations:        {report['n_observations']:>12,}")
    print(f"\nAccuracy Metrics:")
    print(f"  MAE:               {report['mae']:>12.6f}")
    print(f"  RMSE:              {report['rmse']:>12.6f}")
    print(f"  Max Error:         {report['max_error']:>12.6f}")
    if 'mape' in report:
        print(f"  MAPE:              {report['mape']:>12.2f}%")
    print(f"\nPredictive Power:")
    print(f"  R²:                {report['r_squared']:>12.4f}")
    print(f"  Correlation:       {report['correlation']:>12.4f}")
    print(f"\nBias & Consistency:")
    print(f"  Forecast Bias:     {report['forecast_bias']:>12.6f}")
    print(f"  Tracking Error:    {report['tracking_error']:>12.6f}")
    print(f"\nDistribution:")
    print(f"  Actual Mean:       {report['actual_mean']:>12.6f}")
    print(f"  Predicted Mean:    {report['predicted_mean']:>12.6f}")
    print(f"  Actual Std:        {report['actual_std']:>12.6f}")
    print(f"  Predicted Std:     {report['predicted_std']:>12.6f}")
    print(f"{'='*60}\n")

def compare_forecast_methods(actual, forecasts_dict):
    """
    Compare multiple forecasting methods.
    
    Parameters:
    - actual: pd.Series of actual values
    - forecasts_dict: dict of {method_name: forecast_series}
    
    Returns:
    - pd.DataFrame with comparison
    """
    comparison_data = []
    
    for method_name, forecast in forecasts_dict.items():
        report = create_performance_report(actual, forecast, method_name)
        
        comparison_data.append({
            'Method': method_name,
            'MAE': report['mae'],
            'RMSE': report['rmse'],
            'R²': report['r_squared'],
            'Correlation': report['correlation'],
            'Bias': report['forecast_bias']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Rank methods (lower is better for MAE, RMSE, Bias)
    df['MAE_Rank'] = df['MAE'].rank()
    df['RMSE_Rank'] = df['RMSE'].rank()
    df['R²_Rank'] = df['R²'].rank(ascending=False)
    
    return df

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.backtesting.validation import expanding_window_validation, rolling_window_validation
    
    print("Testing performance_metrics.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Calculate actual realized volatility (using 30-day windows)
    actual_vol = historical_volatility(returns, window=30, annualize=True)
    
    # Generate forecasts using different methods
    print("Generating forecasts...")
    
    # Method 1: Expanding window
    forecast_expanding = expanding_window_validation(
        returns, historical_volatility, min_window=252, annualize=True
    )
    
    # Method 2: Rolling window
    forecast_rolling = rolling_window_validation(
        returns, historical_volatility, window=252, annualize=True
    )
    
    # Align indices
    common_idx = actual_vol.index.intersection(forecast_expanding.index).intersection(forecast_rolling.index)
    actual_aligned = actual_vol.loc[common_idx]
    forecast_exp_aligned = forecast_expanding.loc[common_idx]
    forecast_roll_aligned = forecast_rolling.loc[common_idx]
    
    # Test 1: Single method report
    print("\nTest 1: Performance Report (Expanding Window)")
    report = create_performance_report(actual_aligned, forecast_exp_aligned, "Volatility Forecast (Expanding)")
    print_performance_report(report)
    
    # Test 2: Compare methods
    print("\nTest 2: Comparing Forecast Methods")
    forecasts = {
        'Expanding Window': forecast_exp_aligned,
        'Rolling Window': forecast_roll_aligned
    }
    
    comparison = compare_forecast_methods(actual_aligned, forecasts)
    print(comparison.to_string(index=False))
    
    # Test 3: Specific metrics
    print("\nTest 3: Individual Metrics")
    mae = mean_absolute_error(actual_aligned.values, forecast_exp_aligned.values)
    rmse = root_mean_squared_error(actual_aligned.values, forecast_exp_aligned.values)
    corr = information_coefficient(actual_aligned.values, forecast_exp_aligned.values)
    
    print(f"  MAE:         {mae:.6f}")
    print(f"  RMSE:        {rmse:.6f}")
    print(f"  Correlation: {corr:.4f}")
    
    print("\nTest complete!")
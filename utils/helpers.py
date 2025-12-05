# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:10:01 2025

@author: Colby Jaskowiak

Helper utilities module.
Common utility functions for date handling, data cleaning, and conversions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

import brg_risk_metrics.config.settings as cfg

#%% DATE UTILITIES
def parse_date(date_input):
    """
    Parse date input into pd.Timestamp.
    
    Parameters
    ----------
    date_input : str, datetime, pd.Timestamp
        Date in various formats
        
    Returns
    -------
    pd.Timestamp
        Parsed date
    """
    if isinstance(date_input, pd.Timestamp):
        return date_input
    elif isinstance(date_input, datetime):
        return pd.Timestamp(date_input)
    elif isinstance(date_input, str):
        return pd.to_datetime(date_input)
    else:
        raise ValueError(f"Cannot parse date: {date_input}")

def get_date_range(start, end=None, days=None):
    """
    Get date range between two dates or for specified number of days.
    
    Parameters
    ----------
    start : str, datetime, pd.Timestamp
        Start date
    end : str, datetime, pd.Timestamp, optional
        End date
    days : int, optional
        Number of trading days from start
        
    Returns
    -------
    tuple
        (start_date, end_date) as pd.Timestamp
    """
    start_date = parse_date(start)
    
    if end is not None:
        end_date = parse_date(end)
    elif days is not None:
        # Approximate: assume ~252 trading days per year
        calendar_days = int(days * 365.25 / 252)
        end_date = start_date + timedelta(days=calendar_days)
    else:
        end_date = pd.Timestamp.now()
    
    return start_date, end_date

def count_trading_days(start, end):
    """
    Count trading days between two dates.
    
    Parameters
    ----------
    start : str, datetime, pd.Timestamp
        Start date
    end : str, datetime, pd.Timestamp
        End date
        
    Returns
    -------
    int
        Number of trading days (approximate)
    """
    start_date = parse_date(start)
    end_date = parse_date(end)
    
    calendar_days = (end_date - start_date).days
    # Approximate: ~252 trading days per year
    trading_days = int(calendar_days * 252 / 365.25)
    
    return trading_days

#%% DATA CLEANING UTILITIES
def remove_outliers(data, method='iqr', threshold=3.0):
    """
    Remove outliers from data.
    
    Parameters
    ----------
    data : pd.Series or np.array
        Data series
    method : str
        'iqr' (Interquartile Range) or 'zscore'
    threshold : float
        For IQR: multiplier (default: 3.0)
        For z-score: number of std devs (default: 3.0)
        
    Returns
    -------
    pd.Series or np.array
        Data with outliers removed
    """
    if isinstance(data, pd.Series):
        clean_data = data.copy()
        
        if method == 'iqr':
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (clean_data >= lower_bound) & (clean_data <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_data.dropna()))
            mask = z_scores < threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return clean_data[mask]
    else:
        # For numpy arrays, convert to Series
        series_data = pd.Series(data)
        clean_series = remove_outliers(series_data, method=method, threshold=threshold)
        return clean_series.values

def handle_missing_data(data, method='drop'):
    """
    Handle missing data in series.
    
    Parameters
    ----------
    data : pd.Series
        Data series
    method : str
        'drop', 'ffill' (forward fill), 'bfill' (backward fill), or 'interpolate'
        
    Returns
    -------
    pd.Series
        Data with missing values handled
    """
    if method == 'drop':
        return data.dropna()
    elif method == 'ffill':
        return data.fillna(method='ffill')
    elif method == 'bfill':
        return data.fillna(method='bfill')
    elif method == 'interpolate':
        return data.interpolate()
    else:
        raise ValueError(f"Unknown method: {method}")

def clean_returns(returns, remove_zeros=True, handle_inf=True, 
                 outlier_method=None, outlier_threshold=3.0):
    """
    Comprehensive returns cleaning.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    remove_zeros : bool
        Remove exact zero returns (default: True)
    handle_inf : bool
        Replace inf/-inf with NaN (default: True)
    outlier_method : str, optional
        'iqr' or 'zscore' to remove outliers
    outlier_threshold : float
        Threshold for outlier removal
        
    Returns
    -------
    pd.Series
        Cleaned returns
    """
    clean = returns.copy()
    
    # Replace inf with NaN
    if handle_inf:
        clean = clean.replace([np.inf, -np.inf], np.nan)
    
    # Remove NaN
    clean = clean.dropna()
    
    # Remove zeros
    if remove_zeros:
        clean = clean[clean != 0]
    
    # Remove outliers
    if outlier_method is not None:
        clean = remove_outliers(clean, method=outlier_method, threshold=outlier_threshold)
    
    return clean

#%% CONVERSION UTILITIES
def annualize_return(returns, periods_per_year=252):
    """
    Annualize returns.
    
    Parameters
    ----------
    returns : float or pd.Series
        Return(s) to annualize
    periods_per_year : int
        Number of periods per year (default: 252 for daily)
        
    Returns
    -------
    float or pd.Series
        Annualized return(s)
    """
    if isinstance(returns, pd.Series):
        return returns * periods_per_year
    else:
        return returns * periods_per_year

def annualize_volatility(volatility, periods_per_year=252):
    """
    Annualize volatility.
    
    Parameters
    ----------
    volatility : float or pd.Series
        Volatility to annualize
    periods_per_year : int
        Number of periods per year (default: 252 for daily)
        
    Returns
    -------
    float or pd.Series
        Annualized volatility
    """
    if isinstance(volatility, pd.Series):
        return volatility * np.sqrt(periods_per_year)
    else:
        return volatility * np.sqrt(periods_per_year)

def convert_frequency(data, from_freq='D', to_freq='M', method='last'):
    """
    Convert time series frequency.
    
    Parameters
    ----------
    data : pd.Series
        Data series with DatetimeIndex
    from_freq : str
        Current frequency ('D', 'W', 'M')
    to_freq : str
        Target frequency
    method : str
        'last', 'first', 'mean', 'sum'
        
    Returns
    -------
    pd.Series
        Resampled data
    """
    if method == 'last':
        return data.resample(to_freq).last()
    elif method == 'first':
        return data.resample(to_freq).first()
    elif method == 'mean':
        return data.resample(to_freq).mean()
    elif method == 'sum':
        return data.resample(to_freq).sum()
    else:
        raise ValueError(f"Unknown method: {method}")

def to_percentage(value, decimals=2):
    """
    Convert decimal to percentage string.
    
    Parameters
    ----------
    value : float
        Decimal value
    decimals : int
        Decimal places (default: 2)
        
    Returns
    -------
    str
        Percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def from_percentage(percentage_str):
    """
    Convert percentage string to decimal.
    
    Parameters
    ----------
    percentage_str : str
        Percentage string (e.g., "5.5%")
        
    Returns
    -------
    float
        Decimal value
    """
    return float(percentage_str.strip('%')) / 100

#%% FORMATTING UTILITIES
def format_number(value, decimals=2, thousands_sep=True):
    """
    Format number with commas and decimals.
    
    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Decimal places
    thousands_sep : bool
        Include thousands separator
        
    Returns
    -------
    str
        Formatted number
    """
    if thousands_sep:
        return f"{value:,.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"

def format_metric(metric_name, value, metric_type='float'):
    """
    Format metric value with appropriate display.
    
    Parameters
    ----------
    metric_name : str
        Name of metric
    value : float
        Metric value
    metric_type : str
        'float', 'percentage', 'ratio'
        
    Returns
    -------
    str
        Formatted string
    """
    if metric_type == 'percentage':
        return f"{metric_name}: {to_percentage(value)}"
    elif metric_type == 'ratio':
        return f"{metric_name}: {value:.3f}"
    else:
        return f"{metric_name}: {value:.4f}"

def create_summary_dict(returns, metrics_dict):
    """
    Create formatted summary dictionary.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    metrics_dict : dict
        Dict of {metric_name: metric_value}
        
    Returns
    -------
    dict
        Formatted summary
    """
    summary = {
        'Period': f"{returns.index[0].date()} to {returns.index[-1].date()}",
        'Observations': len(returns)
    }
    
    for metric, value in metrics_dict.items():
        if 'return' in metric.lower():
            summary[metric] = to_percentage(value)
        elif 'ratio' in metric.lower():
            summary[metric] = f"{value:.3f}"
        else:
            summary[metric] = f"{value:.4f}"
    
    return summary

#%% STATISTICAL UTILITIES
def calculate_percentile_range(data, lower=5, upper=95):
    """
    Calculate percentile range.
    
    Parameters
    ----------
    data : pd.Series or np.array
        Data series
    lower : float
        Lower percentile (default: 5)
    upper : float
        Upper percentile (default: 95)
        
    Returns
    -------
    tuple
        (lower_value, upper_value)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    lower_val = np.percentile(data, lower)
    upper_val = np.percentile(data, upper)
    
    return lower_val, upper_val

def calculate_rolling_statistic(data, window, statistic='mean'):
    """
    Calculate rolling statistic.
    
    Parameters
    ----------
    data : pd.Series
        Data series
    window : int
        Rolling window size
    statistic : str
        'mean', 'std', 'min', 'max', 'median'
        
    Returns
    -------
    pd.Series
        Rolling statistic
    """
    if statistic == 'mean':
        return data.rolling(window=window).mean()
    elif statistic == 'std':
        return data.rolling(window=window).std()
    elif statistic == 'min':
        return data.rolling(window=window).min()
    elif statistic == 'max':
        return data.rolling(window=window).max()
    elif statistic == 'median':
        return data.rolling(window=window).median()
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

#%%
if __name__ == "__main__":
    print("Testing helpers.py...\n")
    
    # Test date utilities
    print("1. Date utilities...")
    start = parse_date("2020-01-01")
    end = parse_date("2023-12-31")
    print(f"   Date range: {start.date()} to {end.date()}")
    print(f"   Trading days: {count_trading_days(start, end)}")
    
    # Test data cleaning
    print("\n2. Data cleaning...")
    test_data = pd.Series([1, 2, 3, 100, 4, 5, -50, 6, np.nan, 7])
    clean = clean_returns(test_data, outlier_method='zscore', outlier_threshold=2.0)
    print(f"   Original: {len(test_data)} obs")
    print(f"   Cleaned: {len(clean)} obs")
    
    # Test conversions
    print("\n3. Conversions...")
    daily_return = 0.001
    annual_return = annualize_return(daily_return, periods_per_year=252)
    print(f"   Daily return: {to_percentage(daily_return)}")
    print(f"   Annual return: {to_percentage(annual_return)}")
    
    daily_vol = 0.01
    annual_vol = annualize_volatility(daily_vol, periods_per_year=252)
    print(f"   Daily volatility: {to_percentage(daily_vol)}")
    print(f"   Annual volatility: {to_percentage(annual_vol)}")
    
    # Test formatting
    print("\n4. Formatting...")
    print(f"   {format_number(1234567.89, decimals=2)}")
    print(f"   {format_metric('Sharpe Ratio', 1.234, 'ratio')}")
    print(f"   {format_metric('Return', 0.1566, 'percentage')}")
    
    print("\nTest complete!")
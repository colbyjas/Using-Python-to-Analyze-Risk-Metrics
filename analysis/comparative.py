# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:05:57 2025

@author: Colby Jaskowiak

Comparative analysis module.
Compare VaR methods, time periods, metrics, and benchmarks.
"""

import numpy as np
import pandas as pd

import brg_risk_metrics.config.settings as cfg

#%%
def compare_var_methods(returns, confidence=0.95):
    """
    Compare different VaR calculation methods.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence : float
        Confidence level (default: 0.95)
        
    Returns
    -------
    pd.DataFrame
        VaR estimates from different methods
    """
    from brg_risk_metrics.metrics.var import historical_var, parametric_var
    
    results = []
    
    # Historical VaR
    hist_var = historical_var(returns, confidence=confidence)
    results.append({
        'method': 'Historical',
        'var': hist_var,
        'description': 'Empirical quantile'
    })
    
    # Parametric VaR (assumes normality)
    param_var = parametric_var(returns, confidence=confidence)
    results.append({
        'method': 'Parametric',
        'var': param_var,
        'description': 'Normal distribution'
    })
    
    # Try Cornish-Fisher if available
    try:
        from brg_risk_metrics.metrics.var import cornish_fisher_var
        cf_var = cornish_fisher_var(returns, confidence=confidence)
        results.append({
            'method': 'Cornish-Fisher',
            'var': cf_var,
            'description': 'Adjusted for moments'
        })
    except ImportError:
        pass  # Cornish-Fisher not available
    
    df = pd.DataFrame(results)
    
    # Add percentage difference from historical
    df['pct_diff_from_hist'] = ((df['var'] - hist_var) / abs(hist_var)) * 100
    
    return df

def compare_cvar_methods(returns, confidence=0.95):
    """
    Compare different CVaR calculation methods.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence : float
        Confidence level (default: 0.95)
        
    Returns
    -------
    pd.DataFrame
        CVaR estimates from different methods
    """
    from brg_risk_metrics.metrics.cvar import historical_cvar
    
    results = []
    
    # Historical CVaR
    hist_cvar = historical_cvar(returns, confidence=confidence)
    results.append({
        'method': 'Historical',
        'cvar': hist_cvar,
        'description': 'Mean of tail losses'
    })
    
    # Try Parametric CVaR if available
    try:
        from brg_risk_metrics.metrics.cvar import parametric_cvar
        param_cvar = parametric_cvar(returns, confidence=confidence)
        results.append({
            'method': 'Parametric',
            'cvar': param_cvar,
            'description': 'Normal distribution'
        })
    except ImportError:
        pass  # Parametric CVaR not available
    
    df = pd.DataFrame(results)
    
    # Add percentage difference from historical
    if len(df) > 1:
        df['pct_diff_from_hist'] = ((df['cvar'] - hist_cvar) / abs(hist_cvar)) * 100
    
    return df

def compare_time_periods(returns, split_date=None, split_ratio=0.7):
    """
    Compare metrics between in-sample and out-of-sample periods.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    split_date : str or pd.Timestamp, optional
        Date to split at. If None, uses split_ratio
    split_ratio : float
        Ratio for train/test split (default: 0.7)
        
    Returns
    -------
    dict
        Metrics for both periods and comparison
    """
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    from brg_risk_metrics.metrics.drawdown import max_drawdown
    from brg_risk_metrics.metrics.ratios import sharpe_ratio
    
    # Split data
    if split_date is not None:
        in_sample = returns.loc[:split_date]
        out_sample = returns.loc[split_date:]
    else:
        split_idx = int(len(returns) * split_ratio)
        in_sample = returns.iloc[:split_idx]
        out_sample = returns.iloc[split_idx:]
    
    # Calculate metrics for both periods
    metrics = ['volatility', 'var_95', 'cvar_95', 'max_drawdown', 'sharpe']
    
    in_sample_metrics = {
        'volatility': historical_volatility(in_sample, annualize=True),
        'var_95': historical_var(in_sample, confidence=0.95),
        'cvar_95': historical_cvar(in_sample, confidence=0.95),
        'max_drawdown': max_drawdown(in_sample),
        'sharpe': sharpe_ratio(in_sample, annualize=True)
    }
    
    out_sample_metrics = {
        'volatility': historical_volatility(out_sample, annualize=True),
        'var_95': historical_var(out_sample, confidence=0.95),
        'cvar_95': historical_cvar(out_sample, confidence=0.95),
        'max_drawdown': max_drawdown(out_sample),
        'sharpe': sharpe_ratio(out_sample, annualize=True)
    }
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'in_sample': in_sample_metrics,
        'out_sample': out_sample_metrics
    })
    
    # Add percentage change
    comparison['pct_change'] = ((comparison['out_sample'] - comparison['in_sample']) / 
                                abs(comparison['in_sample'])) * 100
    
    return {
        'comparison': comparison,
        'in_sample_period': (in_sample.index[0], in_sample.index[-1]),
        'out_sample_period': (out_sample.index[0], out_sample.index[-1]),
        'in_sample_n': len(in_sample),
        'out_sample_n': len(out_sample)
    }

def compare_across_regimes(returns, regimes):
    """
    Compare metrics across different market regimes.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    regimes : pd.Series or np.array
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Metrics by regime
    """
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    from brg_risk_metrics.metrics.ratios import sharpe_ratio
    
    # Align data
    if isinstance(regimes, pd.Series):
        common_idx = returns.index.intersection(regimes.index)
        ret = returns.loc[common_idx]
        reg = regimes.loc[common_idx]
    else:
        ret = returns
        reg = regimes
    
    unique_regimes = np.unique(reg)
    
    results = []
    for regime in unique_regimes:
        if isinstance(reg, pd.Series):
            mask = reg == regime
        else:
            mask = reg == regime
        
        regime_returns = ret[mask]
        
        if len(regime_returns) > 10:  # Need sufficient data
            results.append({
                'regime': regime,
                'mean_return': regime_returns.mean(),
                'volatility': historical_volatility(regime_returns, annualize=True),
                'var_95': historical_var(regime_returns, confidence=0.95),
                'cvar_95': historical_cvar(regime_returns, confidence=0.95),
                'sharpe': sharpe_ratio(regime_returns, annualize=True),
                'n_observations': len(regime_returns)
            })
    
    return pd.DataFrame(results)

def compare_windows(returns, metric_func, windows=[21, 63, 126, 252], **kwargs):
    """
    Compare metric estimates using different lookback windows.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    metric_func : callable
        Metric function to calculate
    windows : list
        List of window sizes (default: [21, 63, 126, 252] days)
    **kwargs : dict
        Additional arguments for metric_func
        
    Returns
    -------
    pd.DataFrame
        Metric values for different windows
    """
    results = []
    
    for window in windows:
        # Use most recent window
        window_returns = returns.iloc[-window:]
        
        if len(window_returns) >= window * 0.5:  # At least 50% of window
            try:
                metric_value = metric_func(window_returns, **kwargs)
                results.append({
                    'window': window,
                    'window_label': f'{window}d',
                    'metric_value': metric_value,
                    'n_observations': len(window_returns)
                })
            except:
                pass
    
    df = pd.DataFrame(results)
    
    # Add percentage difference from longest window
    if len(df) > 0:
        base_value = df.iloc[-1]['metric_value']
        df['pct_diff_from_longest'] = ((df['metric_value'] - base_value) / abs(base_value)) * 100
    
    return df

def compare_metrics_summary(returns):
    """
    Generate comprehensive metrics comparison summary.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
        
    Returns
    -------
    dict
        Summary of all key metrics
    """
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    from brg_risk_metrics.metrics.drawdown import max_drawdown, average_drawdown
    from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio
    
    # Try to import calmar_ratio if available
    try:
        from brg_risk_metrics.metrics.ratios import calmar_ratio
        has_calmar = True
    except ImportError:
        has_calmar = False
    
    # Helper to safely calculate metrics
    def safe_calc(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return np.nan
    
    summary = {
        # Return metrics
        'mean_return': returns.mean(),
        'median_return': returns.median(),
        'annualized_return': returns.mean() * 252,
        
        # Risk metrics
        'volatility': safe_calc(historical_volatility, returns, annualize=True),
        'var_95': safe_calc(historical_var, returns, confidence=0.95),
        'var_99': safe_calc(historical_var, returns, confidence=0.99),
        'cvar_95': safe_calc(historical_cvar, returns, confidence=0.95),
        'cvar_99': safe_calc(historical_cvar, returns, confidence=0.99),
        
        # Drawdown metrics
        'max_drawdown': safe_calc(max_drawdown, returns),
        'avg_drawdown': safe_calc(average_drawdown, returns),
        
        # Risk-adjusted returns
        'sharpe_ratio': safe_calc(sharpe_ratio, returns, annualize=True),
        'sortino_ratio': safe_calc(sortino_ratio, returns, annualize=True),
        
        # Observations
        'n_observations': len(returns),
        'start_date': returns.index[0],
        'end_date': returns.index[-1]
    }
    
    # Add calmar ratio if available
    if has_calmar:
        summary['calmar_ratio'] = safe_calc(calmar_ratio, returns)
    
    # Try to add additional metrics if available
    try:
        from brg_risk_metrics.metrics.additional import (
            skewness, kurtosis, positive_periods, negative_periods
        )
        summary['skewness'] = safe_calc(skewness, returns)
        summary['kurtosis'] = safe_calc(kurtosis, returns)
        summary['positive_periods'] = safe_calc(positive_periods, returns)
        summary['negative_periods'] = safe_calc(negative_periods, returns)
        
        pos_per = summary['positive_periods']
        if not np.isnan(pos_per):
            summary['win_rate'] = pos_per / len(returns)
    except ImportError:
        pass  # Additional metrics not available
    
    return summary

def benchmark_comparison(returns, benchmark_returns, metrics=['volatility', 'sharpe', 'max_drawdown']):
    """
    Compare asset performance against benchmark.
    
    Parameters
    ----------
    returns : pd.Series
        Asset return series
    benchmark_returns : pd.Series
        Benchmark return series
    metrics : list
        Metrics to compare (default: ['volatility', 'sharpe', 'max_drawdown'])
        
    Returns
    -------
    pd.DataFrame
        Side-by-side comparison
    """
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.ratios import sharpe_ratio
    from brg_risk_metrics.metrics.drawdown import max_drawdown
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    
    # Align data
    common_idx = returns.index.intersection(benchmark_returns.index)
    asset_ret = returns.loc[common_idx]
    bench_ret = benchmark_returns.loc[common_idx]
    
    # Available metrics with error handling
    def safe_metric(func, ret, **kwargs):
        try:
            return func(ret, **kwargs)
        except:
            return np.nan
    
    metric_funcs = {
        'volatility': lambda r: safe_metric(historical_volatility, r, annualize=True),
        'sharpe': lambda r: safe_metric(sharpe_ratio, r, annualize=True),
        'max_drawdown': lambda r: safe_metric(max_drawdown, r),
        'var_95': lambda r: safe_metric(historical_var, r, confidence=0.95),
        'cvar_95': lambda r: safe_metric(historical_cvar, r, confidence=0.95)
    }
    
    results = []
    for metric in metrics:
        if metric in metric_funcs:
            asset_value = metric_funcs[metric](asset_ret)
            bench_value = metric_funcs[metric](bench_ret)
            
            if not np.isnan(asset_value) and not np.isnan(bench_value):
                results.append({
                    'metric': metric,
                    'asset': asset_value,
                    'benchmark': bench_value,
                    'difference': asset_value - bench_value,
                    'pct_difference': ((asset_value - bench_value) / abs(bench_value)) * 100 if bench_value != 0 else np.nan
                })
    
    return pd.DataFrame(results)

def method_ranking(returns, methods_dict, metric='mse', higher_is_better=False):
    """
    Rank different methods based on performance metric.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    methods_dict : dict
        Dict of {method_name: method_function}
    metric : str
        Metric to use for ranking (default: 'mse')
    higher_is_better : bool
        If True, higher values are better (default: False)
        
    Returns
    -------
    pd.DataFrame
        Ranked methods
    """
    results = []
    
    for method_name, method_func in methods_dict.items():
        try:
            result = method_func(returns)
            results.append({
                'method': method_name,
                'value': result
            })
        except Exception as e:
            print(f"Warning: {method_name} failed - {e}")
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df['rank'] = df['value'].rank(ascending=not higher_is_better)
        df = df.sort_values('rank')
    
    return df

def compare_confidence_levels(returns, metric_func, confidence_levels=[0.90, 0.95, 0.99]):
    """
    Compare metric across different confidence levels.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    metric_func : callable
        Metric function that accepts confidence parameter
    confidence_levels : list
        Confidence levels to test (default: [0.90, 0.95, 0.99])
        
    Returns
    -------
    pd.DataFrame
        Metric values at different confidence levels
    """
    results = []
    
    for conf in confidence_levels:
        try:
            value = metric_func(returns, confidence=conf)
            results.append({
                'confidence': conf,
                'confidence_pct': f'{conf*100:.0f}%',
                'metric_value': value
            })
        except Exception as e:
            print(f"Warning: confidence {conf} failed - {e}")
    
    return pd.DataFrame(results)

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.analysis.regime_analysis import detect_regimes_volatility
    
    print("Testing comparative.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = calculate_returns(close)
    
    print(f"Loaded SPY: {len(returns)} obs ({returns.index[0].date()} → {returns.index[-1].date()})")
    
    # Test 1: Compare VaR methods
    print("\n1. Comparing VaR methods (95% confidence)...")
    var_comparison = compare_var_methods(returns, confidence=0.95)
    print(var_comparison.to_string(index=False))
    
    # Test 2: Compare CVaR methods
    print("\n2. Comparing CVaR methods (95% confidence)...")
    cvar_comparison = compare_cvar_methods(returns, confidence=0.95)
    print(cvar_comparison.to_string(index=False))
    
    # Test 3: Compare time periods
    print("\n3. Comparing in-sample vs out-of-sample...")
    period_comparison = compare_time_periods(returns, split_ratio=0.7)
    print(f"   In-sample:  {period_comparison['in_sample_n']} obs")
    print(f"   Out-sample: {period_comparison['out_sample_n']} obs")
    print(period_comparison['comparison'])
    
    # Test 4: Compare across regimes
    print("\n4. Comparing metrics across volatility regimes...")
    vol_regimes = detect_regimes_volatility(returns, window=21)
    regime_comparison = compare_across_regimes(returns, vol_regimes['regimes'])
    print(regime_comparison.to_string(index=False))
    
    # Test 5: Compare windows
    print("\n5. Comparing volatility across different windows...")
    window_comparison = compare_windows(
        returns, 
        historical_volatility,
        windows=[21, 63, 126, 252],
        annualize=True
    )
    print(window_comparison.to_string(index=False))
    
    # Test 6: Metrics summary
    print("\n6. Comprehensive metrics summary...")
    summary = compare_metrics_summary(returns)
    print(f"   Annualized return: {summary['annualized_return']:.2%}")
    print(f"   Volatility: {summary['volatility']:.2%}")
    print(f"   Sharpe ratio: {summary['sharpe_ratio']:.3f}")
    print(f"   Max drawdown: {summary['max_drawdown']:.2%}")
    print(f"   VaR 95%: {summary['var_95']:.4f}")
    print(f"   CVaR 95%: {summary['cvar_95']:.4f}")
    
    # Test 7: Confidence level comparison
    print("\n7. Comparing VaR at different confidence levels...")
    conf_comparison = compare_confidence_levels(returns, historical_var, [0.90, 0.95, 0.99])
    print(conf_comparison.to_string(index=False))
    
    print("\nTest complete!")
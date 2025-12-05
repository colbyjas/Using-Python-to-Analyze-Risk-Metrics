# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:05:38 2025

@author: Colby Jaskowiak

Distribution analysis module.
Statistical tests, distribution fitting, goodness-of-fit tests.
"""

import numpy as np
import pandas as pd
from scipy import stats

import brg_risk_metrics.config.settings as cfg

#%%
def normality_tests(returns):
    """
    Comprehensive normality tests.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
        
    Returns
    -------
    dict
        Results from multiple normality tests
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    results = {}
    
    # 1. Jarque-Bera test
    jb_stat, jb_pval = stats.jarque_bera(returns)
    results['jarque_bera'] = {
        'statistic': jb_stat,
        'p_value': jb_pval,
        'result': 'FAIL' if jb_pval < 0.05 else 'PASS',
        'interpretation': 'Reject normality' if jb_pval < 0.05 else 'Cannot reject normality'
    }
    
    # 2. Shapiro-Wilk test
    sw_stat, sw_pval = stats.shapiro(returns)
    results['shapiro_wilk'] = {
        'statistic': sw_stat,
        'p_value': sw_pval,
        'result': 'FAIL' if sw_pval < 0.05 else 'PASS',
        'interpretation': 'Reject normality' if sw_pval < 0.05 else 'Cannot reject normality'
    }
    
    # 3. Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
    results['kolmogorov_smirnov'] = {
        'statistic': ks_stat,
        'p_value': ks_pval,
        'result': 'FAIL' if ks_pval < 0.05 else 'PASS',
        'interpretation': 'Reject normality' if ks_pval < 0.05 else 'Cannot reject normality'
    }
    
    # 4. Anderson-Darling test
    ad_result = stats.anderson(returns, dist='norm')
    # Use 5% significance level (index 2)
    ad_critical = ad_result.critical_values[2]
    ad_pass = ad_result.statistic < ad_critical
    results['anderson_darling'] = {
        'statistic': ad_result.statistic,
        'critical_value_5pct': ad_critical,
        'result': 'PASS' if ad_pass else 'FAIL',
        'interpretation': 'Cannot reject normality' if ad_pass else 'Reject normality'
    }
    
    return results

def distribution_moments(returns):
    """
    Calculate distribution moments.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
        
    Returns
    -------
    dict
        Mean, variance, skewness, kurtosis (excess)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    return {
        'mean': np.mean(returns),
        'variance': np.var(returns, ddof=1),
        'std': np.std(returns, ddof=1),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),  # Excess kurtosis (normal = 0)
        'n_observations': len(returns)
    }

def fit_normal_distribution(returns):
    """
    Fit normal distribution to returns.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
        
    Returns
    -------
    dict
        Fitted parameters and goodness-of-fit
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(returns)
    
    # Kolmogorov-Smirnov goodness-of-fit
    ks_stat, ks_pval = stats.kstest(returns, 'norm', args=(mu, sigma))
    
    # Log-likelihood
    log_likelihood = np.sum(stats.norm.logpdf(returns, mu, sigma))
    
    # AIC and BIC
    n_params = 2  # mu and sigma
    n_obs = len(returns)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    
    return {
        'distribution': 'Normal',
        'mu': mu,
        'sigma': sigma,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'ks_statistic': ks_stat,
        'ks_p_value': ks_pval
    }

def fit_t_distribution(returns):
    """
    Fit Student's t-distribution to returns (captures fat tails).
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
        
    Returns
    -------
    dict
        Fitted parameters and goodness-of-fit
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    # Fit t-distribution
    df, loc, scale = stats.t.fit(returns)
    
    # Kolmogorov-Smirnov goodness-of-fit
    ks_stat, ks_pval = stats.kstest(returns, 't', args=(df, loc, scale))
    
    # Log-likelihood
    log_likelihood = np.sum(stats.t.logpdf(returns, df, loc, scale))
    
    # AIC and BIC
    n_params = 3  # df, loc, scale
    n_obs = len(returns)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    
    return {
        'distribution': 'Student-t',
        'df': df,
        'loc': loc,
        'scale': scale,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'ks_statistic': ks_stat,
        'ks_p_value': ks_pval
    }

def compare_distributions(returns):
    """
    Compare multiple distribution fits.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
        
    Returns
    -------
    pd.DataFrame
        Comparison of distribution fits
    """
    normal_fit = fit_normal_distribution(returns)
    t_fit = fit_t_distribution(returns)
    
    comparison = pd.DataFrame([
        {
            'distribution': 'Normal',
            'log_likelihood': normal_fit['log_likelihood'],
            'aic': normal_fit['aic'],
            'bic': normal_fit['bic'],
            'ks_statistic': normal_fit['ks_statistic'],
            'ks_p_value': normal_fit['ks_p_value']
        },
        {
            'distribution': 'Student-t',
            'log_likelihood': t_fit['log_likelihood'],
            'aic': t_fit['aic'],
            'bic': t_fit['bic'],
            'ks_statistic': t_fit['ks_statistic'],
            'ks_p_value': t_fit['ks_p_value']
        }
    ])
    
    # Add ranking (lower AIC/BIC is better)
    comparison['aic_rank'] = comparison['aic'].rank()
    comparison['bic_rank'] = comparison['bic'].rank()
    
    return comparison

def qq_plot_statistics(returns, distribution='norm'):
    """
    Calculate Q-Q plot statistics.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
    distribution : str
        Distribution to compare against ('norm' or 't')
        
    Returns
    -------
    dict
        Theoretical quantiles, sample quantiles, R-squared
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    # Sort returns
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    
    # Theoretical quantiles
    if distribution == 'norm':
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    elif distribution == 't':
        df, loc, scale = stats.t.fit(returns)
        theoretical = stats.t.ppf(np.linspace(0.01, 0.99, n), df, loc, scale)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Calculate R-squared
    correlation = np.corrcoef(theoretical, sorted_returns)[0, 1]
    r_squared = correlation ** 2
    
    return {
        'theoretical_quantiles': theoretical,
        'sample_quantiles': sorted_returns,
        'r_squared': r_squared,
        'correlation': correlation
    }

def tail_analysis(returns, threshold_pct=5):
    """
    Analyze distribution tails.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
    threshold_pct : float
        Percentile for tail analysis (default: 5%)
        
    Returns
    -------
    dict
        Tail statistics
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    # Left tail (losses)
    left_threshold = np.percentile(returns, threshold_pct)
    left_tail = returns[returns <= left_threshold]
    
    # Right tail (gains)
    right_threshold = np.percentile(returns, 100 - threshold_pct)
    right_tail = returns[returns >= right_threshold]
    
    return {
        'left_tail_threshold': left_threshold,
        'left_tail_mean': left_tail.mean(),
        'left_tail_std': left_tail.std(),
        'left_tail_count': len(left_tail),
        'right_tail_threshold': right_threshold,
        'right_tail_mean': right_tail.mean(),
        'right_tail_std': right_tail.std(),
        'right_tail_count': len(right_tail),
        'tail_ratio': abs(left_tail.mean() / right_tail.mean())  # Asymmetry measure
    }

def test_autocorrelation(returns, lags=10):
    """
    Test for autocorrelation in returns (and squared returns).
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
    lags : int
        Number of lags to test (default: 10)
        
    Returns
    -------
    dict
        Autocorrelation test results
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return {
            'error': 'statsmodels not installed',
            'returns_significant': None,
            'arch_effects': None
        }
    
    if isinstance(returns, pd.Series):
        returns_series = returns.dropna()
    else:
        returns_series = pd.Series(returns)
    
    # Ljung-Box test on returns
    lb_returns = acorr_ljungbox(returns_series, lags=lags, return_df=True)
    
    # Ljung-Box test on squared returns (test for ARCH effects)
    squared_returns = returns_series ** 2
    lb_squared = acorr_ljungbox(squared_returns, lags=lags, return_df=True)
    
    return {
        'returns_ljungbox': lb_returns,
        'squared_returns_ljungbox': lb_squared,
        'returns_significant': (lb_returns['lb_pvalue'] < 0.05).any(),
        'arch_effects': (lb_squared['lb_pvalue'] < 0.05).any()
    }

def comprehensive_distribution_report(returns):
    """
    Generate comprehensive distribution analysis report.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
        
    Returns
    -------
    dict
        Complete distribution analysis
    """
    report = {}
    
    # Basic moments
    report['moments'] = distribution_moments(returns)
    
    # Normality tests
    report['normality_tests'] = normality_tests(returns)
    
    # Distribution fitting
    report['normal_fit'] = fit_normal_distribution(returns)
    report['t_fit'] = fit_t_distribution(returns)
    
    # Distribution comparison
    report['distribution_comparison'] = compare_distributions(returns)
    
    # Tail analysis
    report['tail_analysis'] = tail_analysis(returns)
    
    # Q-Q statistics
    report['qq_normal'] = qq_plot_statistics(returns, distribution='norm')
    
    return report

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing distribution.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = calculate_returns(close)
    
    print(f"Loaded SPY: {len(returns)} obs ({returns.index[0].date()} → {returns.index[-1].date()})")
    
    # Test 1: Distribution moments
    print("\n1. Distribution moments...")
    moments = distribution_moments(returns)
    for key, val in moments.items():
        if key != 'n_observations':
            print(f"   {key}: {val:.6f}")
        else:
            print(f"   {key}: {val}")
    
    # Test 2: Normality tests
    print("\n2. Normality tests...")
    norm_tests = normality_tests(returns)
    for test_name, result in norm_tests.items():
        print(f"\n   {test_name.replace('_', ' ').title()}:")
        print(f"      Statistic: {result['statistic']:.4f}")
        if 'p_value' in result:
            print(f"      P-value: {result['p_value']:.4f}")
        print(f"      Result: {result['result']}")
        print(f"      {result['interpretation']}")
    
    # Test 3: Distribution fitting
    print("\n3. Distribution fitting...")
    normal_fit = fit_normal_distribution(returns)
    print(f"\n   Normal Distribution:")
    print(f"      μ: {normal_fit['mu']:.6f}")
    print(f"      σ: {normal_fit['sigma']:.6f}")
    print(f"      AIC: {normal_fit['aic']:.2f}")
    print(f"      BIC: {normal_fit['bic']:.2f}")
    
    t_fit = fit_t_distribution(returns)
    print(f"\n   Student-t Distribution:")
    print(f"      df: {t_fit['df']:.2f}")
    print(f"      loc: {t_fit['loc']:.6f}")
    print(f"      scale: {t_fit['scale']:.6f}")
    print(f"      AIC: {t_fit['aic']:.2f}")
    print(f"      BIC: {t_fit['bic']:.2f}")
    
    # Test 4: Distribution comparison
    print("\n4. Distribution comparison...")
    comparison = compare_distributions(returns)
    print(comparison.to_string(index=False))
    
    # Test 5: Tail analysis
    print("\n5. Tail analysis (5% threshold)...")
    tails = tail_analysis(returns, threshold_pct=5)
    print(f"   Left tail (losses):")
    print(f"      Threshold: {tails['left_tail_threshold']:.4f}")
    print(f"      Mean: {tails['left_tail_mean']:.4f}")
    print(f"      Count: {tails['left_tail_count']}")
    print(f"   Right tail (gains):")
    print(f"      Threshold: {tails['right_tail_threshold']:.4f}")
    print(f"      Mean: {tails['right_tail_mean']:.4f}")
    print(f"      Count: {tails['right_tail_count']}")
    print(f"   Tail ratio: {tails['tail_ratio']:.3f}")
    
    # Test 6: Q-Q statistics
    print("\n6. Q-Q plot statistics...")
    qq_stats = qq_plot_statistics(returns, distribution='norm')
    print(f"   R-squared: {qq_stats['r_squared']:.4f}")
    print(f"   Correlation: {qq_stats['correlation']:.4f}")
    
    # Test 7: Autocorrelation
    print("\n7. Autocorrelation tests...")
    try:
        autocorr = test_autocorrelation(returns, lags=5)
        print(f"   Returns autocorrelation significant: {autocorr['returns_significant']}")
        print(f"   ARCH effects detected: {autocorr['arch_effects']}")
    except Exception as e:
        print(f"   Autocorrelation test skipped: {e}")
    
    print("\nTest complete!")
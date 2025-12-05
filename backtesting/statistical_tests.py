# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:06:31 2025

@author: Colby Jaskowiak

Statistical tests for backtesting risk metrics.
Includes Kupiec POF, Christoffersen, coverage tests.
"""

import numpy as np
import pandas as pd
from scipy import stats

#%%
def kupiec_pof_test(violations, n_observations, confidence_level):
    """
    Kupiec Proportion of Failures (POF) Test.
    
    Tests if observed violation rate matches expected rate.
    H0: violation rate = (1 - confidence_level)
    
    Parameters:
    - violations: number of VaR violations
    - n_observations: total number of observations
    - confidence_level: VaR confidence level (e.g., 0.95)
    
    Returns:
    - dict with test_statistic, p_value, result, critical_value
    """
    expected_rate = 1 - confidence_level
    observed_rate = violations / n_observations
    
    # Likelihood ratio test statistic
    if violations == 0:
        lr_stat = -2 * np.log((1 - expected_rate) ** n_observations)
    elif violations == n_observations:
        lr_stat = -2 * np.log(expected_rate ** n_observations)
    else:
        lr_stat = -2 * (
            n_observations * np.log(1 - expected_rate) + violations * np.log(expected_rate) -
            (n_observations - violations) * np.log(1 - observed_rate) - violations * np.log(observed_rate)
        )
    
    # Chi-square with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    # Critical value at 5% significance
    critical_value = stats.chi2.ppf(0.95, df=1)
    
    # Result
    reject_null = lr_stat > critical_value
    
    return {
        'test_statistic': lr_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject_null': reject_null,
        'result': 'FAIL' if reject_null else 'PASS',
        'expected_violations': n_observations * expected_rate,
        'observed_violations': violations,
        'expected_rate': expected_rate,
        'observed_rate': observed_rate
    }

def christoffersen_test(violation_series):
    """
    Christoffersen Independence Test.
    
    Tests if violations are independent (not clustered).
    H0: violations are independent
    
    Parameters:
    - violation_series: pd.Series of boolean (True = violation, False = no violation)
    
    Returns:
    - dict with test results
    """
    violations = violation_series.astype(int).values
    
    # Count transitions
    n00 = 0  # no violation -> no violation
    n01 = 0  # no violation -> violation
    n10 = 0  # violation -> no violation
    n11 = 0  # violation -> violation
    
    for i in range(len(violations) - 1):
        if violations[i] == 0 and violations[i+1] == 0:
            n00 += 1
        elif violations[i] == 0 and violations[i+1] == 1:
            n01 += 1
        elif violations[i] == 1 and violations[i+1] == 0:
            n10 += 1
        elif violations[i] == 1 and violations[i+1] == 1:
            n11 += 1
    
    # Probabilities
    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Likelihood ratio test
    if pi_0 == 0 or pi_1 == 0 or pi == 0 or pi == 1:
        lr_stat = 0
    else:
        lr_stat = -2 * (
            (n00 + n01) * np.log(1 - pi) + (n01 + n11) * np.log(pi) -
            n00 * np.log(1 - pi_0) - n01 * np.log(pi_0) -
            n10 * np.log(1 - pi_1) - n11 * np.log(pi_1)
        )
    
    # Chi-square with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    critical_value = stats.chi2.ppf(0.95, df=1)
    
    reject_null = lr_stat > critical_value
    
    return {
        'test_statistic': lr_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject_null': reject_null,
        'result': 'FAIL' if reject_null else 'PASS',
        'clustering_detected': reject_null,
        'transition_matrix': {
            '00': n00, '01': n01,
            '10': n10, '11': n11
        }
    }

def conditional_coverage_test(violations, n_observations, confidence_level):
    """
    Christoffersen Conditional Coverage Test.
    
    Combines POF and independence tests.
    H0: correct unconditional coverage AND independence
    
    Returns:
    - dict with combined test results
    """
    violation_series = pd.Series(violations) if not isinstance(violations, pd.Series) else violations
    
    # POF test
    pof_result = kupiec_pof_test(violation_series.sum(), n_observations, confidence_level)
    
    # Independence test
    ind_result = christoffersen_test(violation_series)
    
    # Combined test statistic
    combined_stat = pof_result['test_statistic'] + ind_result['test_statistic']
    combined_p_value = 1 - stats.chi2.cdf(combined_stat, df=2)
    critical_value = stats.chi2.ppf(0.95, df=2)
    
    reject_null = combined_stat > critical_value
    
    return {
        'test_statistic': combined_stat,
        'p_value': combined_p_value,
        'critical_value': critical_value,
        'reject_null': reject_null,
        'result': 'FAIL' if reject_null else 'PASS',
        'pof_test': pof_result,
        'independence_test': ind_result
    }

def traffic_light_test(violations, n_observations, confidence_level):
    """
    Basel Traffic Light Test (adapted for different confidence levels).
    
    Original Basel: 99% VaR over 250 days
    - Green: 0-4 violations
    - Yellow: 5-9 violations  
    - Red: 10+ violations
    
    Adapted: scales thresholds based on confidence level and sample size
    
    Returns:
    - dict with zone classification
    """
    expected_violations = n_observations * (1 - confidence_level)
    
    # Scale thresholds based on expected violations
    # Basel zones assume ~2.5 expected violations (99% over 250 days)
    basel_expected = 2.5
    scale_factor = expected_violations / basel_expected
    
    green_threshold = max(int(4 * scale_factor), expected_violations * 0.8)
    yellow_threshold = max(int(9 * scale_factor), expected_violations * 1.5)
    
    if violations <= green_threshold:
        zone = 'GREEN'
        interpretation = 'Acceptable'
    elif violations <= yellow_threshold:
        zone = 'YELLOW'
        interpretation = 'Concern - Review model'
    else:
        zone = 'RED'
        interpretation = 'Unacceptable - Revise model'
    
    return {
        'zone': zone,
        'interpretation': interpretation,
        'violations': violations,
        'expected_violations': expected_violations,
        'green_threshold': green_threshold,
        'yellow_threshold': yellow_threshold
    }

def mean_absolute_error(actual_losses, var_estimates):
    """
    MAE between actual losses and VaR estimates.
    Lower is better.
    """
    return np.mean(np.abs(actual_losses - var_estimates))

def root_mean_squared_error(actual_losses, var_estimates):
    """
    RMSE between actual losses and VaR estimates.
    Lower is better.
    """
    return np.sqrt(np.mean((actual_losses - var_estimates) ** 2))

def quantile_loss(actual_losses, var_estimates, confidence_level):
    """
    Quantile loss function (tick loss).
    Asymmetric loss that penalizes under/over-prediction differently.
    
    Lower is better.
    """
    alpha = 1 - confidence_level
    errors = actual_losses - var_estimates
    loss = np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)
    return np.mean(loss)

#%%
if __name__ == "__main__":
    print("Testing statistical_tests.py...\n")
    
    # Simulated data
    np.random.seed(42)
    n = 250
    confidence = 0.95
    
    # Case 1: Correct model (5% violations expected)
    violations_correct = np.random.random(n) < 0.05
    
    print("Case 1: Correct Model (expected 5% violations)")
    print(f"Violations: {violations_correct.sum()} out of {n}")
    
    pof = kupiec_pof_test(violations_correct.sum(), n, confidence)
    print(f"\nKupiec POF Test:")
    print(f"  Test Statistic: {pof['test_statistic']:.4f}")
    print(f"  P-value: {pof['p_value']:.4f}")
    print(f"  Result: {pof['result']}")
    
    chris = christoffersen_test(pd.Series(violations_correct))
    print(f"\nChristoffersen Independence Test:")
    print(f"  Test Statistic: {chris['test_statistic']:.4f}")
    print(f"  P-value: {chris['p_value']:.4f}")
    print(f"  Result: {chris['result']}")
    
    traffic = traffic_light_test(violations_correct.sum(), n, confidence)
    print(f"\nTraffic Light Test:")
    print(f"  Zone: {traffic['zone']}")
    print(f"  Interpretation: {traffic['interpretation']}")
    
    # Case 2: Too many violations (bad model)
    violations_bad = np.random.random(n) < 0.15
    
    print("\n" + "="*50)
    print("Case 2: Bad Model (15% violations, should be 5%)")
    print(f"Violations: {violations_bad.sum()} out of {n}")
    
    pof_bad = kupiec_pof_test(violations_bad.sum(), n, confidence)
    print(f"\nKupiec POF Test:")
    print(f"  Result: {pof_bad['result']}")
    
    traffic_bad = traffic_light_test(violations_bad.sum(), n, confidence)
    print(f"\nTraffic Light Test:")
    print(f"  Zone: {traffic_bad['zone']}")
    print(f"  Interpretation: {traffic_bad['interpretation']}")
    
    print("\nTest complete!")
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:04:54 2025

@author: Colby Jaskowiak

Additional risk metrics for BRG Risk Metrics project.
"""

import numpy as np
import pandas as pd
from scipy import stats

#%%
def gain_to_pain_ratio(returns):
    """
    Gain-to-Pain Ratio = Sum of gains / Sum of losses
    
    Similar to Omega but simpler calculation.
    """
    r = returns.dropna()
    
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0
    
    return gains / losses

def win_rate(returns):
    """
    Win Rate = Percentage of positive return periods.
    """
    r = returns.dropna()
    return (r > 0).sum() / len(r)

def payoff_ratio(returns):
    """
    Payoff Ratio = Average Win / Average Loss
    
    Measures average magnitude of wins vs losses.
    """
    r = returns.dropna()
    
    wins = r[r > 0]
    losses = r[r < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0
    
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    return avg_win / avg_loss if avg_loss != 0 else np.inf

def profit_factor(returns):
    """
    Profit Factor = Gross Profit / Gross Loss
    
    Similar to gain-to-pain ratio.
    """
    return gain_to_pain_ratio(returns)

def tail_ratio(returns, percentile=0.05):
    """
    Tail Ratio = 95th percentile / 5th percentile
    
    Measures asymmetry in tails.
    Higher is better (larger positive tail vs negative).
    """
    r = returns.dropna()
    
    upper = r.quantile(1 - percentile)
    lower = abs(r.quantile(percentile))
    
    if lower == 0:
        return np.inf if upper > 0 else 0
    
    return upper / lower

def value_at_risk_ratio(returns, confidence=0.95):
    """
    VaR Ratio = Mean Return / VaR
    
    Risk-adjusted return using VaR as risk measure.
    """
    from brg_risk_metrics.metrics.var import historical_var
    
    r = returns.dropna()
    mean_ret = r.mean()
    var = historical_var(r, confidence=confidence)
    
    if var == 0:
        return np.inf if mean_ret > 0 else 0
    
    return mean_ret / var

def conditional_sharpe_ratio(returns, threshold=0, rf_rate=0):
    """
    Conditional Sharpe Ratio = (Mean - RF) / Std(returns below threshold)
    
    Focuses on downside volatility.
    """
    r = returns.dropna()
    
    excess = r - rf_rate
    downside = excess[excess < threshold]
    
    if len(downside) == 0 or downside.std() == 0:
        return 0
    
    return excess.mean() / downside.std()

def skewness(returns):
    """
    Skewness of return distribution.
    
    Positive skew = more extreme positive returns (good)
    Negative skew = more extreme negative returns (bad)
    """
    return stats.skew(returns.dropna())

def kurtosis(returns):
    """
    Excess kurtosis of return distribution.
    
    Higher = fatter tails (more extreme events)
    Normal distribution has kurtosis = 0
    """
    return stats.kurtosis(returns.dropna())

def jarque_bera_test(returns):
    """
    Jarque-Bera test for normality.
    
    Returns (statistic, p_value).
    p_value < 0.05 means reject normality at 5% significance.
    """
    r = returns.dropna()
    statistic, p_value = stats.jarque_bera(r)
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': p_value > 0.05
    }

def beta(returns, market_returns):
    """
    Beta = Cov(asset, market) / Var(market)
    
    Measures systematic risk relative to market.
    """
    r = returns.dropna()
    m = market_returns.dropna()
    
    # Align indices
    common_idx = r.index.intersection(m.index)
    r = r.loc[common_idx]
    m = m.loc[common_idx]
    
    covariance = np.cov(r, m)[0, 1]
    market_variance = m.var()
    
    if market_variance == 0:
        return 0
    
    return covariance / market_variance

def additional_summary(returns):
    """
    Calculate all additional metrics.
    
    Returns dict with various metrics.
    """
    return {
        'win_rate': win_rate(returns),
        'payoff_ratio': payoff_ratio(returns),
        'gain_to_pain': gain_to_pain_ratio(returns),
        'tail_ratio': tail_ratio(returns),
        'skewness': skewness(returns),
        'kurtosis': kurtosis(returns),
        'jarque_bera_pval': jarque_bera_test(returns)['p_value']
    }

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing additional.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    # Individual metrics
    print("Win Rate:", f"{win_rate(returns):.2%}")
    print("Payoff Ratio:", f"{payoff_ratio(returns):.4f}")
    print("Gain-to-Pain:", f"{gain_to_pain_ratio(returns):.4f}")
    print("Tail Ratio:", f"{tail_ratio(returns):.4f}")
    print("Skewness:", f"{skewness(returns):.4f}")
    print("Kurtosis:", f"{kurtosis(returns):.4f}")
    
    # Normality test
    jb = jarque_bera_test(returns)
    print(f"\nJarque-Bera Test:")
    print(f"  Statistic: {jb['statistic']:.2f}")
    print(f"  P-value: {jb['p_value']:.4f}")
    print(f"  Is Normal? {jb['is_normal']}")
    
    # Full summary
    print("\nAdditional Metrics Summary:")
    summary = additional_summary(returns)
    for key, val in summary.items():
        if 'rate' in key:
            print(f"  {key:20s}: {val:.2%}")
        else:
            print(f"  {key:20s}: {val:.4f}")
    
    print("\nTest complete!")
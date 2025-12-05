# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:07:44 2025

@author: Colby Jaskowiak

Heatmap plotting functions for BRG Risk Metrics project.
Correlation matrices, calendar heatmaps, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_correlation_heatmap(returns_dict, title="Return Correlation Matrix", 
                             save=False, filename=None):
    """
    Correlation heatmap for multiple return series.
    
    Parameters:
    - returns_dict: dict of {name: returns_series}
    """
    # Create DataFrame from dict
    df = pd.DataFrame(returns_dict)
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, vmin=-1, vmax=1, square=True, 
               linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_metric_correlation_heatmap(returns, title="Risk Metric Correlations", 
                                   save=False, filename=None):
    """
    Correlation heatmap between different risk metrics calculated on same data.
    """
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio
    from brg_risk_metrics.metrics.drawdown import max_drawdown
    from brg_risk_metrics.metrics.additional import skewness, kurtosis
    
    # Calculate metrics on rolling window
    window = 252
    r = returns.dropna()
    
    rolling_metrics = pd.DataFrame(index=r.index[window-1:])
    
    for i in range(window-1, len(r)):
        window_returns = r.iloc[i-window+1:i+1]
        rolling_metrics.loc[r.index[i], 'Volatility'] = historical_volatility(window_returns, annualize=True)
        rolling_metrics.loc[r.index[i], 'VaR_95'] = historical_var(window_returns, 0.95)
        rolling_metrics.loc[r.index[i], 'CVaR_95'] = historical_cvar(window_returns, 0.95)
        rolling_metrics.loc[r.index[i], 'Sharpe'] = sharpe_ratio(window_returns)
        rolling_metrics.loc[r.index[i], 'Sortino'] = sortino_ratio(window_returns)
        rolling_metrics.loc[r.index[i], 'Max_DD'] = max_drawdown(window_returns)
        rolling_metrics.loc[r.index[i], 'Skewness'] = skewness(window_returns)
        rolling_metrics.loc[r.index[i], 'Kurtosis'] = kurtosis(window_returns)
    
    corr_matrix = rolling_metrics.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
               center=0, vmin=-1, vmax=1, square=True,
               linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_rolling_correlation(returns1, returns2, window=90, 
                             title="Rolling Correlation", save=False, filename=None):
    """
    Plot rolling correlation between two return series.
    """
    # Align series
    common_idx = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common_idx]
    r2 = returns2.loc[common_idx]
    
    # Calculate rolling correlation
    rolling_corr = r1.rolling(window=window).corr(r2)
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    rolling_corr.plot(ax=ax, color='darkblue', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=rolling_corr.mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {rolling_corr.mean():.3f}')
    
    ax.set_title(f'{title} ({window}-day window)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_calendar_heatmap(returns, title="Return Calendar Heatmap", 
                          save=False, filename=None):
    """
    Calendar heatmap showing returns by month and year.
    """
    r = returns.dropna()
    
    # Aggregate to monthly returns
    monthly = r.resample('M').apply(lambda x: (1+x).prod() - 1)
    
    # Create pivot table: rows=year, cols=month
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0,
               linewidths=1, cbar_kws={"shrink": 0.8, "label": "Monthly Return"},
               ax=ax)
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    ax.set_xlabel('')
    ax.set_ylabel('Year')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_var_heatmap(returns, confidence_levels=[0.90, 0.95, 0.99],
                    title="VaR Heatmap (Methods vs Confidence)", 
                    save=False, filename=None):
    """
    Heatmap showing VaR values across methods and confidence levels.
    """
    from brg_risk_metrics.metrics.var import historical_var, parametric_var, monte_carlo_var
    
    methods = {
        'Historical': historical_var,
        'Parametric': lambda r, c: parametric_var(r, c, 'normal'),
        'Cornish-Fisher': lambda r, c: parametric_var(r, c, 'cornish_fisher'),
        'Monte Carlo': lambda r, c: monte_carlo_var(r, c, method='normal')
    }
    
    # Build matrix
    data = []
    for method_name, method_func in methods.items():
        row = [method_func(returns, conf) for conf in confidence_levels]
        data.append(row)
    
    var_matrix = pd.DataFrame(data, 
                             index=list(methods.keys()),
                             columns=[f'{int(c*100)}%' for c in confidence_levels])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(var_matrix, annot=True, fmt='.2%', cmap='Reds',
               linewidths=1, cbar_kws={"shrink": 0.8, "label": "VaR"},
               ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Method')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing heatmaps.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Plotting metric correlation heatmap...")
    print("   (This may take a moment - calculating rolling metrics...)")
    plot_metric_correlation_heatmap(returns, save=True, filename='heat_metric_corr.png')
    
    print("2. Plotting calendar heatmap...")
    plot_calendar_heatmap(returns, save=True, filename='heat_calendar.png')
    
    print("3. Plotting VaR heatmap...")
    plot_var_heatmap(returns, save=True, filename='heat_var.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
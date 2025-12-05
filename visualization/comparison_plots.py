# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:07:38 2025

@author: Colby Jaskowiak

Comparison plotting functions for BRG Risk Metrics project.
Compare multiple metrics, assets, or time periods.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_risk_return_scatter(returns_dict, title="Risk-Return Profile", save=False, filename=None):
    """
    Scatter plot of risk vs return for multiple assets/strategies.
    
    Parameters:
    - returns_dict: dict of {name: returns_series}
    """
    from brg_risk_metrics.data.return_calculator import annualize_return
    from brg_risk_metrics.metrics.volatility import historical_volatility
    
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    for name, returns in returns_dict.items():
        r = returns.dropna()
        ann_return = annualize_return(r)
        ann_vol = historical_volatility(r, annualize=True)
        
        ax.scatter(ann_vol, ann_return, s=100, alpha=0.7, label=name)
        ax.text(ann_vol, ann_return, f'  {name}', fontsize=9, verticalalignment='center')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualized Volatility (Risk)')
    ax.set_ylabel('Annualized Return')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_metric_comparison(returns, metrics_dict, title="Risk Metrics Comparison", 
                          save=False, filename=None):
    """
    Bar chart comparing multiple risk metrics.
    
    Parameters:
    - metrics_dict: dict of {metric_name: metric_value}
    """
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(names, values, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_ratio_comparison(returns, title="Risk-Adjusted Ratios", save=False, filename=None):
    """Compare Sharpe, Sortino, Calmar, and other ratios."""
    from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio, omega_ratio
    from brg_risk_metrics.metrics.drawdown import calmar_ratio
    
    ratios = {
        'Sharpe': sharpe_ratio(returns),
        'Sortino': sortino_ratio(returns),
        'Calmar': calmar_ratio(returns),
        'Omega': omega_ratio(returns)
    }
    
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    names = list(ratios.keys())
    values = list(ratios.values())
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Ratio Value')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_period_comparison(returns, periods_dict, title="Performance by Period", 
                          save=False, filename=None):
    """
    Compare metrics across different time periods.
    
    Parameters:
    - periods_dict: dict of {period_name: (start_date, end_date)}
    """
    from brg_risk_metrics.data.return_calculator import annualize_return
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.ratios import sharpe_ratio
    
    period_names = list(periods_dict.keys())
    returns_by_period = []
    vols_by_period = []
    sharpes_by_period = []
    
    for name, (start, end) in periods_dict.items():
        period_returns = returns[start:end].dropna()
        returns_by_period.append(annualize_return(period_returns))
        vols_by_period.append(historical_volatility(period_returns, annualize=True))
        sharpes_by_period.append(sharpe_ratio(period_returns))
    
    x = np.arange(len(period_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    ax.bar(x - width, returns_by_period, width, label='Ann. Return', color='green', alpha=0.8)
    ax.bar(x, vols_by_period, width, label='Ann. Volatility', color='red', alpha=0.8)
    ax.bar(x + width, sharpes_by_period, width, label='Sharpe Ratio', color='blue', alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(period_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_metric_evolution(returns_dict, metric_func, title="Metric Evolution", 
                         save=False, filename=None):
    """
    Plot how a metric evolves for multiple assets/strategies.
    
    Parameters:
    - returns_dict: dict of {name: returns_series}
    - metric_func: function that takes returns and returns a value
    """
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    for name, returns in returns_dict.items():
        metric_value = metric_func(returns)
        if isinstance(metric_value, pd.Series):
            metric_value.plot(ax=ax, label=name, linewidth=2, alpha=0.8)
        else:
            ax.axhline(y=metric_value, label=name, linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Metric Value')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing comparison_plots.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Plotting ratio comparison...")
    plot_ratio_comparison(returns, save=True, filename='comp_ratios.png')
    
    print("2. Plotting period comparison...")
    periods = {
        '2020': ('2020-01-01', '2020-12-31'),
        '2021': ('2021-01-01', '2021-12-31'),
        '2022': ('2022-01-01', '2022-12-31'),
        '2023': ('2023-01-01', '2023-12-31'),
        '2024': ('2024-01-01', '2024-12-31')
    }
    plot_period_comparison(returns, periods, save=True, filename='comp_periods.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
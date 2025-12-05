# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:07:12 2025

@author: Colby Jaskowiak

Time series plotting functions for BRG Risk Metrics project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_returns(returns, title="Daily Returns", save=False, filename=None):
    """Plot returns time series."""
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    returns.plot(ax=ax, color='steelblue', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_cumulative_returns(returns, title="Cumulative Returns", save=False, filename=None):
    """Plot cumulative returns over time."""
    from brg_risk_metrics.data.return_calculator import cumulative_returns
    
    cum_ret = cumulative_returns(returns)
    
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    cum_ret.plot(ax=ax, color='darkgreen', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    
    # Annotation
    final_ret = cum_ret.iloc[-1]
    ax.text(0.02, 0.98, f'Total: {final_ret:.2%}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_price_series(prices, title="Price Series", save=False, filename=None):
    """Plot price time series."""
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    prices.plot(ax=ax, color='navy', linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_multiple_returns(returns_dict, title="Multiple Asset Returns", save=False, filename=None):
    """
    Plot multiple return series on same chart.
    
    Parameters:
    - returns_dict: dict of {name: returns_series}
    """
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    for name, returns in returns_dict.items():
        from brg_risk_metrics.data.return_calculator import cumulative_returns
        cum_ret = cumulative_returns(returns)
        cum_ret.plot(ax=ax, label=name, linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_rolling_returns(returns, window=252, title="Rolling Annual Returns", save=False, filename=None):
    """Plot rolling returns over specified window."""
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    rolling = returns.rolling(window=window).apply(lambda x: (1+x).prod() - 1)
    rolling.plot(ax=ax, color='darkblue', linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(f'{title} ({window}-day window)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Return')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing time_series_plots.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Plotting returns...")
    plot_returns(returns, save=True, filename='ts_returns.png')
    
    print("2. Plotting cumulative returns...")
    plot_cumulative_returns(returns, save=True, filename='ts_cumulative.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
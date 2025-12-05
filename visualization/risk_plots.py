# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:48:30 2025

@author: Colby Jaskowiak

Risk metric plotting functions for BRG Risk Metrics project.
Includes volatility, VaR, CVaR visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_rolling_volatility(returns, windows=[30, 90, 252], 
                            title="Rolling Volatility", save=False, filename=None):
    """Plot rolling volatility with multiple windows."""
    from brg_risk_metrics.metrics.volatility import historical_volatility
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    colors = ['blue', 'orange', 'green']
    for window, color in zip(windows, colors):
        vol = historical_volatility(returns, window=window, annualize=True)
        vol.plot(ax=ax, label=f'{window}-day', linewidth=2, color=color, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_volatility_comparison(returns, title="Volatility Comparison", save=False, filename=None):
    """Compare different volatility estimation methods."""
    from brg_risk_metrics.metrics.volatility import (
        historical_volatility, ewma_volatility, realized_volatility
    )
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    # EWMA
    ewma = ewma_volatility(returns, annualize=True)
    ewma.plot(ax=ax, label='EWMA', linewidth=2, color='red', alpha=0.8)
    
    # Rolling 90-day
    rolling = realized_volatility(returns, window=90, annualize=True)
    rolling.plot(ax=ax, label='90-day Rolling', linewidth=2, color='blue', alpha=0.8)
    
    # Full sample (horizontal line)
    full_sample = historical_volatility(returns, window=None, annualize=True)
    ax.axhline(y=full_sample, color='green', linestyle='--', linewidth=2, 
              label=f'Full Sample: {full_sample:.2%}', alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_var_levels(returns, confidence_levels=[0.90, 0.95, 0.99], 
                    title="VaR at Different Confidence Levels", save=False, filename=None):
    """Compare VaR across confidence levels and methods."""
    from brg_risk_metrics.metrics.var import historical_var, parametric_var, monte_carlo_var
    
    methods = {
        'Historical': lambda c: historical_var(returns, c),
        'Parametric': lambda c: parametric_var(returns, c, 'normal'),
        'Cornish-Fisher': lambda c: parametric_var(returns, c, 'cornish_fisher'),
        'Monte Carlo': lambda c: monte_carlo_var(returns, c, method='normal')
    }
    
    n_conf = len(confidence_levels)
    fig, axes = plt.subplots(1, n_conf, figsize=(5*n_conf, 5))
    if n_conf == 1:
        axes = [axes]
    
    for idx, conf in enumerate(confidence_levels):
        ax = axes[idx]
        
        method_names = list(methods.keys())
        var_values = [methods[m](conf) for m in method_names]
        
        bars = ax.bar(range(len(method_names)), var_values, 
                     color='steelblue', alpha=0.7, edgecolor='black')
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{conf*100:.0f}% VaR', fontsize=12, fontweight='bold')
        ax.set_ylabel('VaR' if idx == 0 else '')
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_var_cvar_comparison(returns, confidence=0.95, 
                             title="VaR vs CVaR Comparison", save=False, filename=None):
    """Side-by-side comparison of VaR and CVaR."""
    from brg_risk_metrics.metrics.var import var_summary
    from brg_risk_metrics.metrics.cvar import cvar_summary
    
    var_results = var_summary(returns, confidence=confidence)
    cvar_results = cvar_summary(returns, confidence=confidence)
    
    methods = ['historical', 'parametric_normal', 'parametric_cf', 'monte_carlo_normal']
    var_vals = [var_results[m] for m in methods]
    cvar_vals = [cvar_results.get(m, cvar_results.get(f"{m}_cvar", 0)) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    bars1 = ax.bar(x - width/2, var_vals, width, label='VaR', 
                  color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, cvar_vals, width, label='CVaR', 
                  color='darkred', alpha=0.8, edgecolor='black')
    
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title(f'{title} ({confidence*100:.0f}% Confidence)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Risk Measure')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_rolling_var(returns, window=252, confidence=0.95, 
                    title="Rolling VaR", save=False, filename=None):
    """Plot VaR evolution over time using rolling window."""
    from brg_risk_metrics.metrics.var import rolling_var
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    var_rolling = rolling_var(returns, window=window, confidence=confidence, method='historical')
    var_rolling.plot(ax=ax, label=f'{confidence*100:.0f}% VaR', linewidth=2, color='red')
    
    # Add actual returns for context
    ax2 = ax.twinx()
    returns.plot(ax=ax2, color='gray', alpha=0.3, linewidth=0.5, label='Daily Returns')
    ax2.set_ylabel('Daily Return', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax.set_title(f'{title} ({window}-day window)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('VaR', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing risk_plots.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Plotting rolling volatility...")
    plot_rolling_volatility(returns, save=True, filename='risk_rolling_vol.png')
    
    print("2. Plotting VaR levels...")
    plot_var_levels(returns, save=True, filename='risk_var_levels.png')
    
    print("3. Plotting VaR vs CVaR...")
    plot_var_cvar_comparison(returns, save=True, filename='risk_var_cvar.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
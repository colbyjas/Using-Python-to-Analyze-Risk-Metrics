# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:07:22 2025

@author: Colby Jaskowiak

Distribution plotting functions for BRG Risk Metrics project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_histogram(returns, bins=50, title="Return Distribution", save=False, filename=None):
    """Plot return histogram with normal distribution overlay."""
    r = returns.dropna()
    
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    # Histogram
    ax.hist(r, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Normal overlay
    mu, std = r.mean(), r.std()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'r-', linewidth=2, label='Normal Dist.')
    
    # Statistics
    skew = stats.skew(r)
    kurt = stats.kurtosis(r)
    stats_text = f'μ: {mu:.4f}\nσ: {std:.4f}\nSkew: {skew:.2f}\nKurt: {kurt:.2f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_qq(returns, title="Q-Q Plot (Normality Test)", save=False, filename=None):
    """Q-Q plot to assess normality."""
    r = returns.dropna()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    _, p_val = stats.shapiro(r)
    interp = "Normal" if p_val > 0.05 else "Non-Normal"
    ax.text(0.05, 0.95, f'Shapiro-Wilk p={p_val:.4f}\n({interp} at 5%)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_kde(returns, title="Kernel Density Estimate", save=False, filename=None):
    """Plot KDE of returns."""
    r = returns.dropna()
    
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    r.plot.kde(ax=ax, color='darkblue', linewidth=2, label='KDE')
    
    # Normal overlay for comparison
    mu, std = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r--', linewidth=2, label='Normal')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_boxplot(returns, freq='M', title="Return Boxplot by Period", save=False, filename=None):
    """
    Boxplot of returns grouped by time period.
    
    freq: 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
    """
    r = returns.dropna()
    
    # Group by period
    grouped = r.groupby(pd.Grouper(freq=freq))
    data = [group.dropna().values for name, group in grouped if len(group.dropna()) > 0]
    labels = [name.strftime('%Y-%m') for name, group in grouped if len(group.dropna()) > 0]
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Period')
    ax.set_ylabel('Return')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_tail_analysis(returns, title="Tail Analysis", save=False, filename=None):
    """Plot focusing on distribution tails."""
    r = returns.dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.large_figure_size)
    
    # Left tail (losses)
    left_tail = r[r < r.quantile(0.05)]
    ax1.hist(left_tail, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax1.set_title('Left Tail (5% worst)', fontweight='bold')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.axvline(left_tail.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {left_tail.mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right tail (gains)
    right_tail = r[r > r.quantile(0.95)]
    ax2.hist(right_tail, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax2.set_title('Right Tail (5% best)', fontweight='bold')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.axvline(right_tail.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {right_tail.mean():.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing distribution_plots.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Plotting histogram...")
    plot_histogram(returns, save=True, filename='dist_histogram.png')
    
    print("2. Plotting Q-Q plot...")
    plot_qq(returns, save=True, filename='dist_qq.png')
    
    print("3. Plotting tail analysis...")
    plot_tail_analysis(returns, save=True, filename='dist_tails.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
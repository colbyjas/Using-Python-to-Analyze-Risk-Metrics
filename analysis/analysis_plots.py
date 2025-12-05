# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 2025

@author: Colby Jaskowiak

Analysis visualization module.
Plots for regime analysis, distribution analysis, correlation analysis, and comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
# =============================================================================
# REGIME ANALYSIS PLOTS
# =============================================================================

def plot_regime_timeline(returns, regimes, regime_labels=None,
                        title="Market Regime Timeline",
                        save=False, filename=None):
    """
    Plot cumulative returns with regime shading.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    regimes : pd.Series or np.array
        Regime labels (0, 1, 2, ...)
    regime_labels : dict, optional
        Dict mapping regime numbers to names {0: 'Low Vol', 1: 'High Vol'}
    """
    from brg_risk_metrics.data.return_calculator import cumulative_returns
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=cfg.large_figure_size,
                                   height_ratios=[3, 1])
    
    # Align data
    if isinstance(regimes, pd.Series):
        common_idx = returns.index.intersection(regimes.index)
        ret = returns.loc[common_idx]
        reg = regimes.loc[common_idx].values
    else:
        ret = returns
        reg = regimes
    
    # Calculate cumulative returns
    cum_ret = cumulative_returns(ret)
    
    # Plot cumulative returns
    ax1.plot(ret.index, cum_ret.values * 100, color='darkblue', linewidth=2, zorder=10)
    
    # Shade regimes
    unique_regimes = np.unique(reg)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_regimes)))
    
    # Create shaded regions
    current_regime = reg[0]
    start_idx = 0
    
    for i in range(1, len(reg)):
        if reg[i] != current_regime:
            # Shade the region
            ax1.axvspan(ret.index[start_idx], ret.index[i-1],
                       alpha=0.2, color=colors[int(current_regime)], zorder=1)
            current_regime = reg[i]
            start_idx = i
    
    # Shade final region
    ax1.axvspan(ret.index[start_idx], ret.index[-1],
               alpha=0.2, color=colors[int(current_regime)], zorder=1)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_title('Cumulative Returns by Market Regime', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Create legend
    if regime_labels is None:
        regime_labels = {i: f'Regime {i}' for i in unique_regimes}
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[int(i)], alpha=0.2, 
                            label=regime_labels.get(i, f'Regime {i}'))
                      for i in unique_regimes]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Bottom: Regime sequence
    ax2.fill_between(ret.index, reg, alpha=0.6, color='steelblue', step='post')
    ax2.plot(ret.index, reg, color='darkblue', linewidth=1.5, drawstyle='steps-post')
    ax2.set_title('Detected Regimes Over Time', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Regime', fontsize=11)
    ax2.set_yticks(unique_regimes)
    if regime_labels:
        ax2.set_yticklabels([regime_labels.get(i, str(i)) for i in unique_regimes])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_regime_metrics_comparison(regime_comparison_df, 
                                   title="Risk Metrics by Regime",
                                   save=False, filename=None):
    """
    Bar chart comparing metrics across regimes.
    
    Parameters
    ----------
    regime_comparison_df : pd.DataFrame
        Output from compare_across_regimes (from comparative.py)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    regimes = regime_comparison_df['regime'].values
    x_pos = np.arange(len(regimes))
    
    metrics = [
        ('volatility', 'Volatility', '%', 100),
        ('var_95', 'VaR 95%', '', 1),
        ('cvar_95', 'CVaR 95%', '', 1),
        ('sharpe', 'Sharpe Ratio', '', 1)
    ]
    
    for idx, (metric, label, unit, multiplier) in enumerate(metrics):
        ax = axes[idx]
        values = regime_comparison_df[metric].values * multiplier
        
        bars = ax.bar(x_pos, values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}{unit}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_ylabel(f'{label} {unit}' if unit else label, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regimes, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
# =============================================================================
# DISTRIBUTION ANALYSIS PLOTS
# =============================================================================

def plot_distribution_fit_comparison(returns, 
                                     title="Return Distribution with Fitted Models",
                                     save=False, filename=None):
    """
    Histogram with Normal and t-distribution overlays.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
    """
    if isinstance(returns, pd.Series):
        ret = returns.dropna().values
    else:
        ret = returns
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    # Histogram
    n, bins, patches = ax.hist(ret, bins=50, density=True, alpha=0.6, 
                               color='steelblue', edgecolor='black', 
                               label='Actual Returns')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(ret)
    x = np.linspace(ret.min(), ret.max(), 200)
    normal_pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, normal_pdf, 'r-', linewidth=3, label=f'Normal (μ={mu:.4f}, σ={sigma:.4f})')
    
    # Fit t-distribution
    df, loc, scale = stats.t.fit(ret)
    t_pdf = stats.t.pdf(x, df, loc, scale)
    ax.plot(x, t_pdf, 'g-', linewidth=3, label=f'Student-t (df={df:.2f})')
    
    # Add statistics text
    from brg_risk_metrics.analysis.distribution import distribution_moments
    moments = distribution_moments(ret)
    
    stats_text = (f"Statistics:\n"
                 f"Mean: {moments['mean']:.4f}\n"
                 f"Std: {moments['std']:.4f}\n"
                 f"Skew: {moments['skewness']:.3f}\n"
                 f"Kurt: {moments['kurtosis']:.3f}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_tail_comparison(returns, threshold_pct=5,
                        title="Tail Analysis: Left vs Right Tails",
                        save=False, filename=None):
    """
    Visualize asymmetry between loss and gain tails.
    
    Parameters
    ----------
    returns : pd.Series or np.array
        Return series
    threshold_pct : float
        Percentile for tail definition (default: 5%)
    """
    if isinstance(returns, pd.Series):
        ret = returns.dropna().values
    else:
        ret = returns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.large_figure_size)
    
    # Calculate tail thresholds
    left_threshold = np.percentile(ret, threshold_pct)
    right_threshold = np.percentile(ret, 100 - threshold_pct)
    
    # Left plot: Histogram with tail highlighting
    n, bins, patches = ax1.hist(ret, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Color the tails
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val <= left_threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)
        elif bin_val >= right_threshold:
            patch.set_facecolor('green')
            patch.set_alpha(0.8)
    
    ax1.axvline(left_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Left tail ({threshold_pct}%)')
    ax1.axvline(right_threshold, color='green', linestyle='--', linewidth=2,
               label=f'Right tail ({threshold_pct}%)')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_title('Return Distribution with Tails', fontweight='bold')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Tail comparison
    left_tail = ret[ret <= left_threshold]
    right_tail = ret[ret >= right_threshold]
    
    tail_stats = pd.DataFrame({
        'Metric': ['Mean', 'Std', 'Count'],
        'Left Tail (Losses)': [left_tail.mean(), left_tail.std(), len(left_tail)],
        'Right Tail (Gains)': [right_tail.mean(), right_tail.std(), len(right_tail)]
    })
    
    x = np.arange(len(tail_stats))
    width = 0.35
    
    # Plot for Mean and Std only
    for idx in range(2):  # Mean and Std
        ax2_temp = ax2 if idx == 0 else ax2.twinx()
        
        metric_data = tail_stats.iloc[idx]
        bars1 = ax2_temp.bar(idx - width/2, abs(metric_data['Left Tail (Losses)']), 
                            width, label='Left Tail', color='red', alpha=0.7)
        bars2 = ax2_temp.bar(idx + width/2, metric_data['Right Tail (Gains)'], 
                            width, label='Right Tail', color='green', alpha=0.7)
        
        # Add value labels
        for bar in [bars1, bars2]:
            for b in bar:
                height = b.get_height()
                ax2_temp.text(b.get_x() + b.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
    
    ax2.set_title('Tail Statistics Comparison', fontweight='bold')
    ax2.set_ylabel('Absolute Value')
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(['Mean', 'Std Dev'])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add asymmetry ratio
    tail_ratio = abs(left_tail.mean() / right_tail.mean())
    fig.text(0.7, 0.25, f'Tail Ratio: {tail_ratio:.3f}\n(Losses/Gains)',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
# =============================================================================
# CORRELATION ANALYSIS PLOTS
# =============================================================================

def plot_rolling_correlation(returns1, returns2, window=252,
                            labels=('Asset 1', 'Asset 2'),
                            title="Rolling Correlation Over Time",
                            save=False, filename=None):
    """
    Plot rolling correlation time series.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    window : int
        Rolling window (default: 252 days)
    labels : tuple
        Labels for assets
    """
    from brg_risk_metrics.analysis.correlation import rolling_correlation
    
    rolling_corr = rolling_correlation(returns1, returns2, window=window)
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    # Plot rolling correlation
    ax.plot(rolling_corr.index, rolling_corr.values, color='darkblue', linewidth=2)
    
    # Add mean line
    mean_corr = rolling_corr.mean()
    ax.axhline(mean_corr, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_corr:.3f}')
    
    # Add confidence bands (±1 std)
    std_corr = rolling_corr.std()
    ax.axhline(mean_corr + std_corr, color='orange', linestyle=':', linewidth=1.5,
              label=f'±1σ: [{mean_corr-std_corr:.3f}, {mean_corr+std_corr:.3f}]')
    ax.axhline(mean_corr - std_corr, color='orange', linestyle=':', linewidth=1.5)
    
    # Shade high correlation periods (> mean + 0.5*std)
    high_corr_threshold = mean_corr + 0.5 * std_corr
    high_periods = rolling_corr > high_corr_threshold
    
    # Find continuous periods
    in_high = False
    start_idx = None
    for i, (idx, val) in enumerate(zip(rolling_corr.index, high_periods)):
        if val and not in_high:
            start_idx = idx
            in_high = True
        elif not val and in_high:
            ax.axvspan(start_idx, rolling_corr.index[i-1], alpha=0.2, color='red')
            in_high = False
    
    ax.set_title(f'{title}\n{labels[0]} vs {labels[1]} ({window}-day window)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_ylim(-1, 1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_correlation_breakdown(breakdown_dict,
                               title="Correlation: Up vs Down Markets",
                               save=False, filename=None):
    """
    Bar chart comparing correlations in different market conditions.
    
    Parameters
    ----------
    breakdown_dict : dict
        Output from correlation_breakdown function
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Overall', 'Up Market', 'Down Market']
    correlations = [
        breakdown_dict['overall']['correlation'],
        breakdown_dict['up_market']['correlation'],
        breakdown_dict['down_market']['correlation']
    ]
    
    colors = ['steelblue', 'green', 'red']
    bars = ax.bar(categories, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{corr:.3f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    # Add asymmetry annotation
    asymmetry = breakdown_dict['asymmetry']
    ax.text(0.5, 0.95, f'Asymmetry: {asymmetry:+.4f}\n(Down - Up)',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
# =============================================================================
# COMPARATIVE ANALYSIS PLOTS
# =============================================================================

def plot_var_method_comparison(var_comparison_df,
                               title="VaR Method Comparison",
                               save=False, filename=None):
    """
    Bar chart comparing VaR estimates from different methods.
    
    Parameters
    ----------
    var_comparison_df : pd.DataFrame
        Output from compare_var_methods
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = var_comparison_df['method'].values
    vars = var_comparison_df['var'].values * 100  # Convert to %
    
    colors = ['steelblue', 'coral', 'lightgreen'][:len(methods)]
    bars = ax.bar(methods, np.abs(vars), color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, var_val in zip(bars, vars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{abs(var_val):.2f}%', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    
    # Add percentage difference annotations
    if 'pct_diff_from_hist' in var_comparison_df.columns:
        for i, (bar, pct_diff) in enumerate(zip(bars, var_comparison_df['pct_diff_from_hist'].values)):
            if abs(pct_diff) > 0.01:  # Only show if meaningful difference
                ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                       f'{pct_diff:+.1f}%', ha='center', va='center',
                       fontsize=9, style='italic',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('VaR (% loss)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation note
    note = "Higher VaR = More conservative (predicts larger losses)"
    ax.text(0.5, 0.02, note, transform=ax.transAxes, ha='center',
           fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_window_sensitivity(window_comparison_df, metric_name='Metric',
                           title="Metric Sensitivity to Window Size",
                           save=False, filename=None):
    """
    Line plot showing how metric changes with window size.
    
    Parameters
    ----------
    window_comparison_df : pd.DataFrame
        Output from compare_windows
    metric_name : str
        Name of metric being analyzed
    """
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    windows = window_comparison_df['window'].values
    values = window_comparison_df['metric_value'].values
    
    # Line plot with markers
    ax.plot(windows, values, color='steelblue', linewidth=3, marker='o', 
           markersize=10, markerfacecolor='coral', markeredgecolor='black', 
           markeredgewidth=2)
    
    # Highlight min and max
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)
    
    ax.scatter(windows[min_idx], values[min_idx], s=200, color='green', 
              marker='v', zorder=10, label=f'Min: {values[min_idx]:.4f}')
    ax.scatter(windows[max_idx], values[max_idx], s=200, color='red',
              marker='^', zorder=10, label=f'Max: {values[max_idx]:.4f}')
    
    # Add value labels
    for window, value in zip(windows, values):
        ax.text(window, value, f'{value:.3f}', ha='center', va='bottom',
               fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Window Size (days)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xticks(windows)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_period_comparison(period_comparison_dict,
                          title="In-Sample vs Out-of-Sample Performance",
                          save=False, filename=None):
    """
    Bar chart comparing metrics across time periods.
    
    Parameters
    ----------
    period_comparison_dict : dict
        Output from compare_time_periods
    """
    comparison_df = period_comparison_dict['comparison']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = comparison_df.index.values
    in_sample = comparison_df['in_sample'].values
    out_sample = comparison_df['out_sample'].values
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, np.abs(in_sample), width, label='In-Sample',
                   color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, np.abs(out_sample), width, label='Out-of-Sample',
                   color='coral', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add period info
    period_text = (f"In-Sample: {period_comparison_dict['in_sample_n']} obs\n"
                  f"Out-Sample: {period_comparison_dict['out_sample_n']} obs")
    ax.text(0.02, 0.98, period_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
if __name__ == "__main__":
    print("Analysis plots module loaded successfully!")
    print("\nAvailable plot functions:")
    print("  Regime Analysis:")
    print("    - plot_regime_timeline()")
    print("    - plot_regime_metrics_comparison()")
    print("  Distribution Analysis:")
    print("    - plot_distribution_fit_comparison()")
    print("    - plot_tail_comparison()")
    print("  Correlation Analysis:")
    print("    - plot_rolling_correlation()")
    print("    - plot_correlation_breakdown()")
    print("  Comparative Analysis:")
    print("    - plot_var_method_comparison()")
    print("    - plot_window_sensitivity()")
    print("    - plot_period_comparison()")
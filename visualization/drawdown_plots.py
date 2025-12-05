# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:07:30 2025

@author: Colby Jaskowiak

Drawdown plotting functions for BRG Risk Metrics project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_underwater(returns, title="Underwater Plot (Drawdown)", save=False, filename=None):
    """Underwater plot showing all drawdowns."""
    from brg_risk_metrics.metrics.drawdown import drawdown_series, max_drawdown
    
    dd = drawdown_series(returns)
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    ax.fill_between(dd.index, dd.values, 0, where=(dd.values<0), color='red', alpha=0.3, label='Drawdown')
    ax.plot(dd.index, dd.values, color='darkred', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Annotate max drawdown
    mdd = max_drawdown(returns)
    mdd_idx = dd.idxmin()
    ax.scatter(mdd_idx, dd.loc[mdd_idx], color='red', s=100, zorder=5, marker='v')
    ax.text(mdd_idx, dd.loc[mdd_idx]*1.1, f'Max DD: {mdd:.2%}', 
            fontsize=10, ha='center', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_drawdown_periods(returns, top_n=5, title="Major Drawdown Periods", save=False, filename=None):
    """Highlight top N worst drawdown periods."""
    from brg_risk_metrics.metrics.drawdown import drawdown_series, drawdown_periods
    
    dd = drawdown_series(returns)
    periods = drawdown_periods(returns)
    
    # Sort by depth and take top N
    periods_sorted = sorted(periods, key=lambda x: x['depth'], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    # Plot full drawdown series
    ax.plot(dd.index, dd.values, color='gray', linewidth=1, alpha=0.5, label='All Drawdowns')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Highlight major periods
    colors = plt.cm.Reds(np.linspace(0.5, 0.9, top_n))
    for i, period in enumerate(periods_sorted):
        start = period['start']
        end = period['end'] if period['end'] else dd.index[-1]
        mask = (dd.index >= start) & (dd.index <= end)
        ax.fill_between(dd.index[mask], dd.values[mask], 0, 
                       color=colors[i], alpha=0.6, label=f"#{i+1}: {period['depth']:.1%}")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_recovery_analysis(returns, title="Drawdown Recovery Analysis", save=False, filename=None):
    """Analyze recovery times from drawdowns."""
    from brg_risk_metrics.metrics.drawdown import drawdown_periods
    
    periods = [p for p in drawdown_periods(returns) if p['recovery_time'] is not None]
    
    if len(periods) == 0:
        print("No completed drawdown periods to analyze")
        return None
    
    depths = [p['depth'] for p in periods]
    recovery_times = [p['recovery_time'] for p in periods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.large_figure_size)
    
    # Scatter: depth vs recovery time
    ax1.scatter(depths, recovery_times, alpha=0.6, s=50, color='steelblue')
    ax1.set_title('Depth vs Recovery Time', fontweight='bold')
    ax1.set_xlabel('Drawdown Depth')
    ax1.set_ylabel('Recovery Time (days)')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.grid(True, alpha=0.3)
    
    # Histogram of recovery times
    ax2.hist(recovery_times, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(recovery_times), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(recovery_times):.0f} days')
    ax2.set_title('Recovery Time Distribution', fontweight='bold')
    ax2.set_xlabel('Recovery Time (days)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

def plot_drawdown_statistics(returns, title="Drawdown Statistics", save=False, filename=None):
    """Summary statistics visualization for drawdowns."""
    from brg_risk_metrics.metrics.drawdown import (
        max_drawdown, average_drawdown, max_drawdown_duration, ulcer_index
    )
    
    mdd = max_drawdown(returns)
    avg_dd = average_drawdown(returns)
    max_dur = max_drawdown_duration(returns)
    ui = ulcer_index(returns)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    stats_data = [
        ['Maximum Drawdown', f'{mdd:.2%}'],
        ['Average Drawdown', f'{avg_dd:.2%}'],
        ['Max Duration', f'{max_dur} days'],
        ['Ulcer Index', f'{ui:.2%}']
    ]
    
    table = ax.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                    cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Style
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#d32f2f')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#ffebee' if i % 2 == 0 else 'white')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing drawdown_plots.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Plotting underwater...")
    plot_underwater(returns, save=True, filename='dd_underwater.png')
    
    print("2. Plotting major periods...")
    plot_drawdown_periods(returns, save=True, filename='dd_periods.png')
    
    print("3. Plotting recovery analysis...")
    plot_recovery_analysis(returns, save=True, filename='dd_recovery.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
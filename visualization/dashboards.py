# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:07:12 2025

@author: Colby Jaskowiak

Dashboard wrapper functions for BRG Risk Metrics project.
Combines multiple visualizations into comprehensive dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy import stats

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_overview_dashboard(returns, title="Risk Metrics Overview Dashboard", 
                           save=False, filename=None):
    """
    Comprehensive dashboard showing all major risk metrics.
    Layout: 3x3 grid with key visualizations.
    """
    from brg_risk_metrics.data.return_calculator import cumulative_returns, annualize_return
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.drawdown import drawdown_series, max_drawdown
    from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    from brg_risk_metrics.metrics.additional import skewness, kurtosis
    
    r = returns.dropna()
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative Returns (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    cum_ret = cumulative_returns(r)
    cum_ret.plot(ax=ax1, color='darkgreen', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Cumulative Returns', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Return')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Total: {cum_ret.iloc[-1]:.2%}', 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Return Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(r, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    mu, std = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
    ax2.set_title('Distribution', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Return', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown (Underwater)
    ax3 = fig.add_subplot(gs[1, 1])
    dd = drawdown_series(r)
    ax3.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax3.plot(dd.index, dd.values, color='darkred', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Drawdown', fontweight='bold', fontsize=11)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax3.grid(True, alpha=0.3)
    
    # 4. Rolling Volatility
    ax4 = fig.add_subplot(gs[1, 2])
    vol_90 = historical_volatility(r, window=90, annualize=True)
    vol_90.plot(ax=ax4, color='purple', linewidth=2)
    ax4.set_title('90-Day Volatility', fontweight='bold', fontsize=11)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax4.grid(True, alpha=0.3)
    
    # 5. Key Metrics Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    metrics = {
        'Annualized Return': f"{annualize_return(r):.2%}",
        'Annualized Volatility': f"{historical_volatility(r):.2%}",
        'Sharpe Ratio': f"{sharpe_ratio(r):.3f}",
        'Sortino Ratio': f"{sortino_ratio(r):.3f}",
        'Max Drawdown': f"{max_drawdown(r):.2%}",
        '95% VaR': f"{historical_var(r, 0.95):.2%}",
        '95% CVaR': f"{historical_cvar(r, 0.95):.2%}",
        'Skewness': f"{skewness(r):.3f}",
        'Kurtosis': f"{kurtosis(r):.3f}",
        'Win Rate': f"{(r > 0).sum() / len(r):.2%}"
    }
    
    # Split into 2 columns for better layout
    items = list(metrics.items())
    mid = len(items) // 2
    col1 = items[:mid]
    col2 = items[mid:]
    
    table_data = [[k1, v1, k2, v2] for (k1, v1), (k2, v2) in zip(col1, col2)]
    
    table = ax5.table(cellText=table_data, 
                     colLabels=['Metric', 'Value', 'Metric', 'Value'],
                     cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_risk_analysis_dashboard(returns, title="Risk Analysis Dashboard", 
                                 save=False, filename=None):
    """
    Dashboard focused on risk metrics: VaR, CVaR, volatility, drawdowns.
    """
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import var_summary
    from brg_risk_metrics.metrics.cvar import cvar_summary
    from brg_risk_metrics.metrics.drawdown import drawdown_series, drawdown_periods
    
    r = returns.dropna()
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Rolling Volatility (Multiple Windows)
    ax1 = fig.add_subplot(gs[0, 0])
    for window, color in zip([30, 90, 252], ['blue', 'orange', 'green']):
        vol = historical_volatility(r, window=window, annualize=True)
        vol.plot(ax=ax1, label=f'{window}-day', linewidth=2, color=color, alpha=0.7)
    ax1.set_title('Rolling Volatility', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Ann. Volatility')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. VaR vs CVaR Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    var_results = var_summary(r, confidence=0.95)
    cvar_results = cvar_summary(r, confidence=0.95)
    
    methods = ['historical', 'parametric_normal', 'monte_carlo_normal']
    x = np.arange(len(methods))
    width = 0.35
    
    var_vals = [var_results[m] for m in methods]
    cvar_vals = [cvar_results[m] for m in methods]
    
    ax2.bar(x - width/2, var_vals, width, label='VaR', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, cvar_vals, width, label='CVaR', color='darkred', alpha=0.8)
    ax2.set_title('95% VaR vs CVaR', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Risk Measure')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=30, ha='right', fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Drawdown Plot
    ax3 = fig.add_subplot(gs[1, 0])
    dd = drawdown_series(r)
    ax3.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax3.plot(dd.index, dd.values, color='darkred', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Drawdown Evolution', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Drawdown')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax3.grid(True, alpha=0.3)
    
    # 4. Top Drawdown Periods Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    periods = drawdown_periods(r)
    periods_sorted = sorted(periods, key=lambda x: x['depth'], reverse=True)[:5]
    
    table_data = [
        [f"{p['depth']:.2%}", 
         p['start'].strftime('%Y-%m-%d'), 
         p['end'].strftime('%Y-%m-%d') if p['end'] else 'Ongoing',
         f"{p['duration']} days"]
        for p in periods_sorted
    ]
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Depth', 'Start', 'End', 'Duration'],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#d32f2f')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#ffebee' if i % 2 == 0 else 'white')
    
    ax4.set_title('Top 5 Drawdown Periods', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_performance_dashboard(returns, title="Performance Dashboard", 
                               save=False, filename=None):
    """
    Dashboard focused on performance: returns, ratios, efficiency.
    """
    from brg_risk_metrics.data.return_calculator import cumulative_returns, annualize_return
    from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio, omega_ratio
    from brg_risk_metrics.metrics.drawdown import calmar_ratio
    from brg_risk_metrics.metrics.additional import win_rate, payoff_ratio
    
    r = returns.dropna()
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative Returns with Benchmark
    ax1 = fig.add_subplot(gs[0, :])
    cum_ret = cumulative_returns(r)
    cum_ret.plot(ax=ax1, color='darkgreen', linewidth=2.5, label='Portfolio')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Cumulative Performance', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Cumulative Return')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk-Adjusted Ratios
    ax2 = fig.add_subplot(gs[1, 0])
    
    ratios = {
        'Sharpe': sharpe_ratio(r),
        'Sortino': sortino_ratio(r),
        'Calmar': calmar_ratio(r),
        'Omega': omega_ratio(r)
    }
    
    names = list(ratios.keys())
    values = list(ratios.values())
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#FFC107']
    
    bars = ax2.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_title('Risk-Adjusted Ratios', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Ratio Value')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Performance Metrics Table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.drawdown import max_drawdown
    
    metrics = [
        ['Annualized Return', f"{annualize_return(r):.2%}"],
        ['Annualized Volatility', f"{historical_volatility(r):.2%}"],
        ['Sharpe Ratio', f"{sharpe_ratio(r):.3f}"],
        ['Sortino Ratio', f"{sortino_ratio(r):.3f}"],
        ['Max Drawdown', f"{max_drawdown(r):.2%}"],
        ['Win Rate', f"{win_rate(r):.2%}"],
        ['Payoff Ratio', f"{payoff_ratio(r):.3f}"],
        ['Total Return', f"{cum_ret.iloc[-1]:.2%}"]
    ]
    
    table = ax3.table(cellText=metrics, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#E8F5E9' if i % 2 == 0 else 'white')
    
    ax3.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_full_report(returns, title="Complete Risk Analysis Report", 
                    save=False, filename=None):
    """
    Mega-dashboard with everything. Use for comprehensive reports.
    """
    from brg_risk_metrics.data.return_calculator import cumulative_returns, annualize_return
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.drawdown import drawdown_series, max_drawdown
    from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.cvar import historical_cvar
    from brg_risk_metrics.metrics.additional import skewness, kurtosis, win_rate
    
    r = returns.dropna()
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Cumulative Returns (full width)
    ax1 = fig.add_subplot(gs[0, :])
    cum_ret = cumulative_returns(r)
    cum_ret.plot(ax=ax1, color='darkgreen', linewidth=2.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Cumulative Returns', fontweight='bold', fontsize=13)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(True, alpha=0.3)
    
    # Row 2: Distribution, Drawdown, Volatility
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(r, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    mu, std = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
    ax2.set_title('Distribution', fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    dd = drawdown_series(r)
    ax3.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax3.plot(dd.index, dd.values, color='darkred', linewidth=1)
    ax3.set_title('Drawdown', fontweight='bold', fontsize=11)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2])
    vol_90 = historical_volatility(r, window=90, annualize=True)
    vol_90.plot(ax=ax4, color='purple', linewidth=2)
    ax4.set_title('90-Day Volatility', fontweight='bold', fontsize=11)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax4.grid(True, alpha=0.3)
    
    # Row 3: VaR/CVaR, Ratios, Rolling Returns
    ax5 = fig.add_subplot(gs[2, 0])
    from brg_risk_metrics.metrics.var import var_summary
    var_res = var_summary(r, 0.95)
    methods = ['historical', 'parametric_normal', 'monte_carlo_normal']
    ax5.bar(range(len(methods)), [var_res[m] for m in methods], 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax5.set_title('95% VaR Methods', fontweight='bold', fontsize=11)
    ax5.set_xticks(range(len(methods)))
    ax5.set_xticklabels([m.replace('_', ' ')[:8] for m in methods], rotation=30, fontsize=8)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(gs[2, 1])
    ratios = {
        'Sharpe': sharpe_ratio(r),
        'Sortino': sortino_ratio(r),
    }
    ax6.bar(ratios.keys(), ratios.values(), color=['#2196F3', '#FF9800'], alpha=0.8)
    for i, (k, v) in enumerate(ratios.items()):
        ax6.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    ax6.set_title('Risk-Adj. Ratios', fontweight='bold', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    
    ax7 = fig.add_subplot(gs[2, 2])
    rolling_ret = r.rolling(252).apply(lambda x: (1+x).prod() - 1)
    rolling_ret.plot(ax=ax7, color='darkblue', linewidth=2)
    ax7.set_title('Rolling Annual Return', fontweight='bold', fontsize=11)
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax7.grid(True, alpha=0.3)
    
    # Row 4: Comprehensive Metrics Table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    metrics = {
        'Total Return': f"{cum_ret.iloc[-1]:.2%}",
        'Ann. Return': f"{annualize_return(r):.2%}",
        'Ann. Volatility': f"{historical_volatility(r):.2%}",
        'Sharpe Ratio': f"{sharpe_ratio(r):.3f}",
        'Sortino Ratio': f"{sortino_ratio(r):.3f}",
        'Max Drawdown': f"{max_drawdown(r):.2%}",
        '95% VaR': f"{historical_var(r, 0.95):.2%}",
        '95% CVaR': f"{historical_cvar(r, 0.95):.2%}",
        'Skewness': f"{skewness(r):.3f}",
        'Kurtosis': f"{kurtosis(r):.3f}",
        'Win Rate': f"{win_rate(r):.2%}",
        'Best Day': f"{r.max():.2%}",
        'Worst Day': f"{r.min():.2%}",
        'Observations': f"{len(r):,}"
    }
    
    items = list(metrics.items())
    mid = len(items) // 2
    col1 = items[:mid]
    col2 = items[mid:]
    
    table_data = [[k1, v1, k2, v2] for (k1, v1), (k2, v2) in zip(col1, col2)]
    
    table = ax8.table(cellText=table_data, 
                     colLabels=['Metric', 'Value', 'Metric', 'Value'],
                     cellLoc='left', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1976D2')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#BBDEFB' if i % 2 == 0 else 'white')
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.998)
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    
    print("Testing dashboards.py...\n")
    
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    returns = calculate_returns(close)
    
    print("1. Generating overview dashboard...")
    plot_overview_dashboard(returns, save=True, filename='dashboard_overview.png')
    
    print("2. Generating risk analysis dashboard...")
    plot_risk_analysis_dashboard(returns, save=True, filename='dashboard_risk.png')
    
    print("3. Generating performance dashboard...")
    plot_performance_dashboard(returns, save=True, filename='dashboard_performance.png')
    
    print("4. Generating full report dashboard...")
    plot_full_report(returns, save=True, filename='dashboard_full_report.png')
    
    print("\nAll dashboards saved to:", cfg.figures_dir)
    print("Test complete!")
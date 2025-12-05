# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:07:44 2025

@author: Colby Jaskowiak

Backtesting dashboard - comprehensive validation summary.
Compares in-sample vs out-of-sample across all metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_validation_dashboard(returns, title="Validation Dashboard", 
                              save=False, filename=None):
    """
    Comprehensive validation dashboard comparing in-sample vs out-of-sample.
    Shows performance across multiple metrics and methods.
    NOW INCLUDES: Deployment simulation line starting at split.
    """
    from brg_risk_metrics.backtesting.validation import train_test_split, compare_in_sample_out_sample
    from brg_risk_metrics.metrics.volatility import historical_volatility
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.ratios import sharpe_ratio
    from brg_risk_metrics.metrics.drawdown import max_drawdown
    from brg_risk_metrics.data.return_calculator import cumulative_returns
    
    # Split data
    train, test = train_test_split(returns, train_pct=0.7)
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative returns with split
    ax1 = fig.add_subplot(gs[0, :])
    all_returns = pd.concat([train, test])
    cum_ret = cumulative_returns(all_returns)
    
    # NEW: Calculate out-of-sample only cumulative returns (deployment simulation)
    test_cum_ret = cumulative_returns(test)
    
    # Plot full period cumulative returns
    ax1.plot(cum_ret.index, cum_ret.values, color='darkblue', linewidth=2, 
            label='Full Period')
    
    # NEW: Plot deployment simulation (starts at 0 at split)
    ax1.plot(test_cum_ret.index, test_cum_ret.values, color='purple', linewidth=2,
            linestyle='--', label='Deployment (from split)', alpha=0.8)
    
    ax1.axvline(x=test.index[0], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    ax1.axvspan(train.index[0], train.index[-1], alpha=0.15, color='green', label='In-Sample')
    ax1.axvspan(test.index[0], test.index[-1], alpha=0.15, color='orange', label='Out-of-Sample')
    ax1.set_title('Cumulative Returns: In-Sample vs Out-of-Sample', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Cumulative Return')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility comparison
    ax2 = fig.add_subplot(gs[1, 0])
    vol_comp = compare_in_sample_out_sample(returns, historical_volatility, train_pct=0.7, annualize=True)
    categories = ['In-Sample', 'Out-of-Sample']
    vol_values = [vol_comp['in_sample'], vol_comp['out_of_sample']]
    bars = ax2.bar(categories, vol_values, color=['green', 'orange'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, vol_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_title('Volatility', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Annualized Volatility')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. VaR comparison
    ax3 = fig.add_subplot(gs[1, 1])
    var_comp = compare_in_sample_out_sample(returns, historical_var, train_pct=0.7, confidence=0.95)
    var_values = [var_comp['in_sample'], var_comp['out_of_sample']]
    bars = ax3.bar(categories, var_values, color=['green', 'orange'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, var_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_title('95% VaR', fontweight='bold', fontsize=11)
    ax3.set_ylabel('VaR')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Sharpe Ratio comparison
    ax4 = fig.add_subplot(gs[2, 0])
    sharpe_comp = compare_in_sample_out_sample(returns, sharpe_ratio, train_pct=0.7)
    sharpe_values = [sharpe_comp['in_sample'], sharpe_comp['out_of_sample']]
    bars = ax4.bar(categories, sharpe_values, color=['green', 'orange'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, sharpe_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_title('Sharpe Ratio', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Sharpe Ratio')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Max Drawdown comparison
    ax5 = fig.add_subplot(gs[2, 1])
    dd_comp = compare_in_sample_out_sample(returns, max_drawdown, train_pct=0.7)
    dd_values = [dd_comp['in_sample'], dd_comp['out_of_sample']]
    bars = ax5.bar(categories, dd_values, color=['green', 'orange'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, dd_values):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax5.set_title('Maximum Drawdown', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Max Drawdown')
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_comprehensive_backtest_summary(var_backtest_result, cv_results,
                                       title="Comprehensive Backtesting Summary",
                                       save=False, filename=None):
    """
    Master dashboard combining VaR backtest and cross-validation results.
    Ultimate summary of model validation.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. VaR violations (top, 2/3 width)
    ax1 = fig.add_subplot(gs[0, :2])
    returns = var_backtest_result['actual_returns']
    var_estimates = var_backtest_result['var_estimates']
    violations = var_backtest_result['violations']
    
    ax1.plot(returns.index, returns.values, color='gray', alpha=0.4, linewidth=0.5)
    ax1.plot(var_estimates.index, -var_estimates.values, color='red', linewidth=2, 
            label='VaR Threshold')
    violation_dates = violations[violations == True].index
    ax1.scatter(violation_dates, returns.loc[violation_dates], color='red', 
               s=30, zorder=5, marker='v')
    ax1.set_title('VaR Violations Over Time', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Return')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Test statistics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    test_data = [
        ['Violations', f"{var_backtest_result['n_violations']}/{var_backtest_result['n_observations']}"],
        ['Rate', f"{var_backtest_result['violation_rate']:.2%}"],
        ['Expected', f"{var_backtest_result['expected_rate']:.2%}"],
        ['POF Test', var_backtest_result['pof_test']['result']],
        ['Independence', var_backtest_result['independence_test']['result']],
        ['Traffic Light', var_backtest_result['traffic_light']['zone']]
    ]
    
    table = ax2.table(cellText=test_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#FF5722')
            cell.set_text_props(weight='bold', color='white')
        else:
            # Color code based on result
            if j == 1 and i >= 4:  # Result cells
                text = cell.get_text().get_text()
                if text == 'PASS' or text == 'GREEN':
                    cell.set_facecolor('#C8E6C9')
                elif text == 'FAIL' or text == 'RED':
                    cell.set_facecolor('#FFCDD2')
                elif text == 'YELLOW':
                    cell.set_facecolor('#FFF9C4')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax2.set_title('Test Results', fontweight='bold', fontsize=11, pad=20)
    
    # 3. Cross-validation results
    ax3 = fig.add_subplot(gs[1, :])
    folds = [r['fold'] for r in cv_results]
    train_metrics = [r['train_metric'] for r in cv_results]
    test_metrics = [r['test_metric'] for r in cv_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax3.bar(x - width/2, train_metrics, width, label='Train', 
           color='green', alpha=0.7, edgecolor='black')
    ax3.bar(x + width/2, test_metrics, width, label='Test', 
           color='orange', alpha=0.7, edgecolor='black')
    ax3.axhline(y=np.mean(train_metrics), color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=np.mean(test_metrics), color='darkorange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax3.set_title('Time Series Cross-Validation', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Metric Value')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Fold {f}' for f in folds])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Key insights
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate insights
    avg_train = np.mean(train_metrics)
    avg_test = np.mean(test_metrics)
    cv_diff = ((avg_test - avg_train) / avg_train * 100) if avg_train != 0 else 0
    
    insights_text = (
        f"VALIDATION SUMMARY\n\n"
        f"VaR Model Performance:\n"
        f"• Violation rate: {var_backtest_result['violation_rate']:.2%} "
        f"(Target: {var_backtest_result['expected_rate']:.2%})\n"
        f"• POF Test: {var_backtest_result['pof_test']['result']} "
        f"(p={var_backtest_result['pof_test']['p_value']:.4f})\n"
        f"• Traffic Light: {var_backtest_result['traffic_light']['zone']}\n\n"
        f"Cross-Validation:\n"
        f"• Average Train: {avg_train:.4f}\n"
        f"• Average Test: {avg_test:.4f}\n"
        f"• Train/Test Difference: {cv_diff:+.1f}%\n\n"
        f"Conclusion:\n"
        f"• {'No signs of overfitting' if abs(cv_diff) < 20 else 'Potential overfitting detected'}\n"
        f"• {'Model generalizes well' if var_backtest_result['traffic_light']['zone'] == 'GREEN' else 'Model needs review'}"
    )
    
    ax4.text(0.5, 0.5, insights_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.998)
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.backtesting.var_backtest import backtest_var
    from brg_risk_metrics.backtesting.validation import time_series_cv
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.metrics.volatility import historical_volatility
    
    print("Testing backtest_dashboard.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = calculate_returns(close)
    
    # Generate backtesting results
    print("1. Generating validation dashboard...")
    plot_validation_dashboard(returns, save=True, filename='bt_dash_validation.png')
    
    print("2. Generating comprehensive summary...")
    var_result = backtest_var(returns, historical_var, confidence=0.95, window=252)
    cv_results = time_series_cv(returns, historical_volatility, n_splits=3)
    plot_comprehensive_backtest_summary(var_result, cv_results, 
                                       save=True, filename='bt_dash_comprehensive.png')
    
    print("\nDashboards saved to:", cfg.figures_dir)
    print("Test complete!")
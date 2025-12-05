# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:07:44 2025

@author: Colby Jaskowiak

Backtesting visualization module.
Plots for validation, forecast accuracy, in-sample vs out-sample.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

import brg_risk_metrics.config.settings as cfg

plt.style.use(cfg.plot_style)

#%%
def plot_var_violations(backtest_result, title="VaR Backtesting - Violations", 
                       save=False, filename=None):
    """
    Plot VaR estimates vs actual returns, highlighting violations.
    
    Key Plot #1: Shows where VaR model failed.
    """
    returns = backtest_result['actual_returns']
    var_estimates = backtest_result['var_estimates']
    violations = backtest_result['violations']
    
    fig, ax = plt.subplots(figsize=cfg.large_figure_size)
    
    # Plot returns
    ax.plot(returns.index, returns.values, color='gray', alpha=0.5, 
           linewidth=0.5, label='Daily Returns')
    
    # Plot VaR threshold
    ax.plot(var_estimates.index, -var_estimates.values, color='red', 
           linewidth=2, label=f'{backtest_result["confidence"]*100:.0f}% VaR', linestyle='--')
    
    # Highlight violations
    violation_dates = violations[violations == True].index
    violation_returns = returns.loc[violation_dates]
    ax.scatter(violation_dates, violation_returns, color='red', s=50, 
              zorder=5, label=f'Violations ({len(violation_dates)})', marker='v')
    
    # Add statistics box
    stats_text = (f"Violations: {backtest_result['n_violations']}/{backtest_result['n_observations']}\n"
                 f"Rate: {backtest_result['violation_rate']:.2%} (Expected: {backtest_result['expected_rate']:.2%})\n"
                 f"POF Test: {backtest_result['pof_test']['result']}\n"
                 f"Traffic Light: {backtest_result['traffic_light']['zone']}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_method_comparison(comparison_results, title="VaR Method Comparison", 
                          save=False, filename=None):
    """
    Compare multiple VaR methods side-by-side.
    
    Key Plot #2: Shows which method performs best.
    """
    methods = list(comparison_results.keys())
    violation_rates = [comparison_results[m]['violation_rate'] for m in methods]
    expected_rate = comparison_results[methods[0]]['expected_rate']
    pof_results = [comparison_results[m]['pof_test']['result'] for m in methods]
    traffic_lights = [comparison_results[m]['traffic_light']['zone'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.large_figure_size)
    
    # Left: Violation rates
    colors = ['green' if pof == 'PASS' else 'red' for pof in pof_results]
    bars = ax1.bar(methods, violation_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=expected_rate, color='blue', linestyle='--', linewidth=2, 
               label=f'Expected: {expected_rate:.2%}')
    
    # Add value labels
    for bar, rate in zip(bars, violation_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Violation Rates', fontweight='bold')
    ax1.set_ylabel('Violation Rate')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Traffic light zones
    zone_colors = {'GREEN': 'green', 'YELLOW': 'yellow', 'RED': 'red'}
    bar_colors = [zone_colors[zone] for zone in traffic_lights]
    ax2.bar(methods, [1]*len(methods), color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Add zone labels
    for i, (method, zone) in enumerate(zip(methods, traffic_lights)):
        ax2.text(i, 0.5, zone, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    ax2.set_title('Traffic Light Test', fontweight='bold')
    ax2.set_ylabel('Status')
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_forecast_accuracy(actual, predicted, method_name="Forecast", 
                          title="Forecast Accuracy", save=False, filename=None):
    """
    Scatter plot: actual vs predicted values.
    
    Key Plot #3: Shows forecast quality visually.
    """
    from brg_risk_metrics.backtesting.performance_metrics import (
        r_squared, information_coefficient, mean_absolute_error
    )
    
    # Align data
    common_idx = actual.index.intersection(predicted.index)
    actual_aligned = actual.loc[common_idx].values
    predicted_aligned = predicted.loc[common_idx].values
    
    # Calculate metrics
    r2 = r_squared(actual_aligned, predicted_aligned)
    corr = information_coefficient(actual_aligned, predicted_aligned)
    mae = mean_absolute_error(actual_aligned, predicted_aligned)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(actual_aligned, predicted_aligned, alpha=0.5, s=20, color='steelblue')
    
    # Perfect prediction line
    min_val = min(actual_aligned.min(), predicted_aligned.min())
    max_val = max(actual_aligned.max(), predicted_aligned.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
           label='Perfect Forecast')
    
    # Add regression line
    z = np.polyfit(actual_aligned, predicted_aligned, 1)
    p = np.poly1d(z)
    ax.plot(actual_aligned, p(actual_aligned), 'g-', linewidth=2, alpha=0.7,
           label='Fitted Line')
    
    # Metrics box
    metrics_text = f'R²: {r2:.3f}\nCorr: {corr:.3f}\nMAE: {mae:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'{title} - {method_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_in_sample_out_sample(train_data, test_data, train_metric, test_metric,
                              metric_name="Metric", title="In-Sample vs Out-of-Sample",
                              save=False, filename=None):
    """
    Compare in-sample vs out-of-sample performance.
    
    Key Plot #4: Shows if model generalizes.
    NOW INCLUDES: Deployment simulation line starting at split.
    """
    from brg_risk_metrics.data.return_calculator import cumulative_returns
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=cfg.large_figure_size, 
                                   height_ratios=[2, 1])
    
    # Top: Returns with train/test split
    all_returns = pd.concat([train_data, test_data])
    cum_ret = cumulative_returns(all_returns)
    
    # NEW: Calculate out-of-sample only cumulative returns (deployment simulation)
    test_cum_ret = cumulative_returns(test_data)
    
    # Plot full period cumulative returns
    ax1.plot(cum_ret.index, cum_ret.values, color='darkblue', linewidth=2, 
            label='Full Period')
    
    # NEW: Plot deployment simulation (starts at 0 at split)
    ax1.plot(test_cum_ret.index, test_cum_ret.values, color='purple', linewidth=2,
            linestyle='--', label='Deployment (from split)', alpha=0.8)
    
    ax1.axvline(x=test_data.index[0], color='red', linestyle='--', linewidth=2,
               label='Train/Test Split')
    
    # Shade regions
    ax1.axvspan(train_data.index[0], train_data.index[-1], alpha=0.2, color='green',
               label='In-Sample (Train)')
    ax1.axvspan(test_data.index[0], test_data.index[-1], alpha=0.2, color='orange',
               label='Out-of-Sample (Test)')
    
    ax1.set_title('Cumulative Returns with Train/Test Split', fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Metric comparison
    categories = ['In-Sample\n(Train)', 'Out-of-Sample\n(Test)']
    values = [train_metric, test_metric]
    colors_bar = ['green', 'orange']
    
    bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add difference annotation
    diff = test_metric - train_metric
    pct_diff = (diff / train_metric * 100) if train_metric != 0 else 0
    ax2.text(0.5, 0.95, f'Difference: {diff:+.4f} ({pct_diff:+.1f}%)',
            transform=ax2.transAxes, ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.set_title(f'{metric_name} Comparison', fontweight='bold')
    ax2.set_ylabel(metric_name)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_time_series_cv_results(cv_results, metric_name="Metric",
                                title="Time Series Cross-Validation Results",
                                save=False, filename=None):
    """
    Visualize cross-validation fold performance.
    
    Key Plot #5: Shows consistency across folds.
    """
    folds = [r['fold'] for r in cv_results]
    train_metrics = [r['train_metric'] for r in cv_results]
    test_metrics = [r['test_metric'] for r in cv_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=cfg.default_figure_size)
    
    bars1 = ax.bar(x - width/2, train_metrics, width, label='Train', 
                  color='green', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_metrics, width, label='Test', 
                  color='orange', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add average lines
    avg_train = np.mean(train_metrics)
    avg_test = np.mean(test_metrics)
    ax.axhline(y=avg_train, color='darkgreen', linestyle='--', linewidth=2,
              label=f'Avg Train: {avg_train:.3f}')
    ax.axhline(y=avg_test, color='darkorange', linestyle='--', linewidth=2,
              label=f'Avg Test: {avg_test:.3f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(cfg.figures_dir / filename, dpi=cfg.save_dpi, bbox_inches='tight')
    
    plt.show()
    return fig

#%%
if __name__ == "__main__":
    from brg_risk_metrics.data.data_loader import load_spy_data
    from brg_risk_metrics.data.return_calculator import calculate_returns, get_close
    from brg_risk_metrics.metrics.var import historical_var
    from brg_risk_metrics.backtesting.var_backtest import backtest_var, compare_var_methods
    from brg_risk_metrics.backtesting.validation import train_test_split, time_series_cv
    from brg_risk_metrics.metrics.volatility import historical_volatility
    
    print("Testing backtesting_plots.py...\n")
    
    # Load data
    spy_px = load_spy_data(start='2020-01-01')
    close = get_close(spy_px)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = calculate_returns(close)
    
    # Test 1: VaR violations plot
    print("1. Plotting VaR violations...")
    backtest_result = backtest_var(returns, historical_var, confidence=0.95, window=252)
    plot_var_violations(backtest_result, save=True, filename='bt_var_violations.png')
    
    # Test 2: Method comparison (not running to save time, but code is ready)
    print("2. Plotting method comparison...")
    methods = {'Historical': historical_var}
    comparison = compare_var_methods(returns, methods, confidence=0.95, window=252)
    plot_method_comparison(comparison, save=True, filename='bt_method_comparison.png')
    
    # Test 3: In-sample vs out-sample
    print("3. Plotting in-sample vs out-sample...")
    train, test = train_test_split(returns, train_pct=0.7)
    train_vol = historical_volatility(train, annualize=True)
    test_vol = historical_volatility(test, annualize=True)
    plot_in_sample_out_sample(train, test, train_vol, test_vol,
                              metric_name="Volatility",
                              save=True, filename='bt_in_out_sample.png')
    
    # Test 4: Time series CV
    print("4. Plotting time series CV...")
    cv_results = time_series_cv(returns, historical_volatility, n_splits=3)
    plot_time_series_cv_results(cv_results, metric_name="Volatility",
                                save=True, filename='bt_cv_results.png')
    
    print("\nPlots saved to:", cfg.figures_dir)
    print("Test complete!")
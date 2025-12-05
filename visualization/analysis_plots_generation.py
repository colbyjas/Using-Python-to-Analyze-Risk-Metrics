# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 2025

@author: Colby Jaskowiak

Analysis plots generation script.
Generates all visualizations from analysis modules.
"""

import numpy as np
import pandas as pd

# Import data loading
from brg_risk_metrics.data.data_loader import load_spy_data
from brg_risk_metrics.data.return_calculator import calculate_returns, get_close

# Import analysis modules
from brg_risk_metrics.analysis.regime_analysis import detect_regimes_volatility
from brg_risk_metrics.analysis.correlation import (
    rolling_correlation,
    correlation_breakdown
)
from brg_risk_metrics.analysis.comparative import (
    compare_var_methods,
    compare_windows,
    compare_time_periods,
    compare_across_regimes
)

# Import plot functions from /analysis
from brg_risk_metrics.analysis.analysis_plots import (
    plot_regime_timeline,
    plot_regime_metrics_comparison,
    plot_distribution_fit_comparison,
    plot_tail_comparison,
    plot_rolling_correlation,
    plot_correlation_breakdown,
    plot_var_method_comparison,
    plot_window_sensitivity,
    plot_period_comparison
)

# Import metrics
from brg_risk_metrics.metrics.volatility import historical_volatility

print("="*80)
print("GENERATING ANALYSIS VISUALIZATIONS")
print("="*80)

#%%
# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading SPY data...")
spy_px = load_spy_data(start='2020-01-01')
close = get_close(spy_px)
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
returns = calculate_returns(close)
print(f"   Loaded {len(returns)} observations ({returns.index[0].date()} → {returns.index[-1].date()})")

# Create synthetic second asset for correlation plots
np.random.seed(42)
noise = pd.Series(np.random.normal(0, 0.005, len(returns)), index=returns.index)
returns2 = returns * 0.7 + noise

#%%
# =============================================================================
# REGIME ANALYSIS PLOTS
# =============================================================================
print("\n[2/4] Generating regime analysis plots...")

# Plot 1: Regime timeline
print("   1a. Regime timeline (volatility-based)...")
vol_regimes = detect_regimes_volatility(returns, window=21)
plot_regime_timeline(
    returns, 
    vol_regimes['regimes'],
    regime_labels={0: 'Low Volatility', 1: 'High Volatility'},
    title="Market Regime Timeline (Volatility-Based)",
    save=True, 
    filename='analysis_regime_timeline.png'
)
print("      ✓ Saved: analysis_regime_timeline.png")

# Plot 2: Regime metrics comparison
print("   1b. Metrics comparison by regime...")
regime_comparison = compare_across_regimes(returns, vol_regimes['regimes'])
plot_regime_metrics_comparison(
    regime_comparison,
    title="Risk Metrics by Volatility Regime",
    save=True,
    filename='analysis_regime_metrics.png'
)
print("      ✓ Saved: analysis_regime_metrics.png")

#%%
# =============================================================================
# DISTRIBUTION ANALYSIS PLOTS
# =============================================================================
print("\n[3/4] Generating distribution analysis plots...")

# Plot 3: Distribution fit comparison
print("   2a. Distribution fit comparison (Normal vs t)...")
plot_distribution_fit_comparison(
    returns,
    title="SPY Return Distribution with Fitted Models",
    save=True,
    filename='analysis_distribution_fit.png'
)
print("      ✓ Saved: analysis_distribution_fit.png")

# Plot 4: Tail comparison
print("   2b. Tail analysis (5% tails)...")
plot_tail_comparison(
    returns,
    threshold_pct=5,
    title="Tail Analysis: Loss vs Gain Asymmetry",
    save=True,
    filename='analysis_tail_comparison.png'
)
print("      ✓ Saved: analysis_tail_comparison.png")

#%%
# =============================================================================
# CORRELATION ANALYSIS PLOTS
# =============================================================================
print("\n[4/4] Generating correlation analysis plots...")

# Plot 5: Rolling correlation
print("   3a. Rolling correlation (252-day window)...")
plot_rolling_correlation(
    returns,
    returns2,
    window=252,
    labels=('SPY', 'Synthetic Asset'),
    title="Rolling Correlation Over Time",
    save=True,
    filename='analysis_rolling_correlation.png'
)
print("      ✓ Saved: analysis_rolling_correlation.png")

# Plot 6: Correlation breakdown
print("   3b. Correlation breakdown (up vs down markets)...")
breakdown = correlation_breakdown(returns, returns2)
plot_correlation_breakdown(
    breakdown,
    title="Correlation Asymmetry: Up vs Down Markets",
    save=True,
    filename='analysis_correlation_breakdown.png'
)
print("      ✓ Saved: analysis_correlation_breakdown.png")

#%%
# =============================================================================
# COMPARATIVE ANALYSIS PLOTS
# =============================================================================
print("\n[5/5] Generating comparative analysis plots...")

# Plot 7: VaR method comparison
print("   4a. VaR method comparison...")
var_comparison = compare_var_methods(returns, confidence=0.95)
plot_var_method_comparison(
    var_comparison,
    title="VaR 95% Method Comparison",
    save=True,
    filename='analysis_var_methods.png'
)
print("      ✓ Saved: analysis_var_methods.png")

# Plot 8: Window sensitivity
print("   4b. Window sensitivity analysis...")
window_comparison = compare_windows(
    returns,
    historical_volatility,
    windows=[21, 63, 126, 252],
    annualize=True
)
plot_window_sensitivity(
    window_comparison,
    metric_name='Annualized Volatility',
    title="Volatility Sensitivity to Window Size",
    save=True,
    filename='analysis_window_sensitivity.png'
)
print("      ✓ Saved: analysis_window_sensitivity.png")

# Plot 9: Period comparison
print("   4c. In-sample vs out-of-sample comparison...")
period_comparison = compare_time_periods(returns, split_ratio=0.7)
plot_period_comparison(
    period_comparison,
    title="In-Sample vs Out-of-Sample Performance",
    save=True,
    filename='analysis_period_comparison.png'
)
print("      ✓ Saved: analysis_period_comparison.png")

#%%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("GENERATION COMPLETE!")
print("="*80)
print("\nGenerated visualizations:")
print("  Regime Analysis (2 plots):")
print("    - analysis_regime_timeline.png")
print("    - analysis_regime_metrics.png")
print("\n  Distribution Analysis (2 plots):")
print("    - analysis_distribution_fit.png")
print("    - analysis_tail_comparison.png")
print("\n  Correlation Analysis (2 plots):")
print("    - analysis_rolling_correlation.png")
print("    - analysis_correlation_breakdown.png")
print("\n  Comparative Analysis (3 plots):")
print("    - analysis_var_methods.png")
print("    - analysis_window_sensitivity.png")
print("    - analysis_period_comparison.png")
print("\nTotal: 9 analysis visualizations")
print(f"\nAll files saved to: {__import__('brg_risk_metrics.config.settings', fromlist=['figures_dir']).figures_dir}")
print("="*80)
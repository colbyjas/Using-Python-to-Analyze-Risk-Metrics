# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 2025

@author: Colby Jaskowiak

Test script for advanced features and visualizations.
Generates all plots from monte_carlo, stress_testing, and optimization modules.
"""

import numpy as np
import pandas as pd

# Import data loading
from brg_risk_metrics.data.data_loader import load_spy_data
from brg_risk_metrics.data.return_calculator import calculate_returns, get_close

# Import advanced modules
from brg_risk_metrics.advanced.monte_carlo import (
    simulate_gbm_paths, 
    simulate_t_distribution_paths,
    scenario_analysis,
    expected_shortfall_distribution
)

from brg_risk_metrics.advanced.stress_testing import (
    stress_test_metrics,
    sensitivity_analysis,
    HISTORICAL_SCENARIOS
)

from brg_risk_metrics.advanced.optimization import (
    efficient_frontier,
    compare_portfolios,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    risk_parity_portfolio
)

# Import metrics for testing
from brg_risk_metrics.metrics.var import historical_var

# Import advanced plots functions from /advanced
from brg_risk_metrics.advanced.advanced_plots_functions import (
    plot_monte_carlo_paths,
    plot_final_distribution,
    plot_scenario_comparison,
    plot_stress_comparison,
    plot_sensitivity_curve,
    plot_stress_heatmap,
    plot_efficient_frontier,
    plot_portfolio_comparison,
    plot_weight_allocation
)

print("="*80)
print("TESTING ADVANCED FEATURES AND VISUALIZATIONS")
print("="*80)

#%% LOAD DATA

print("\n[1/3] Loading SPY data...")
spy_px = load_spy_data(start='2020-01-01')
close = get_close(spy_px)
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
returns = calculate_returns(close)
print(f"   Loaded {len(returns)} observations ({returns.index[0].date()} → {returns.index[-1].date()})")

#%% MONTE CARLO SIMULATIONS + PLOTS

print("\n[2/3] Running Monte Carlo simulations and generating plots...")

# Test 1: GBM simulation with paths plot
print("\n   1a. GBM simulation (1000 paths, 252 days)...")
gbm_results = simulate_gbm_paths(returns, n_paths=1000, horizon_days=252, seed=42)
plot_monte_carlo_paths(gbm_results, n_paths_show=100, 
                      title="Monte Carlo Simulation - GBM Paths",
                      save=True, filename='adv_mc_paths_gbm.png')
print("      ✓ Saved: adv_mc_paths_gbm.png")

# Test 2: Final distribution plot
print("   1b. Final distribution analysis...")
plot_final_distribution(gbm_results,
                       title="Monte Carlo - Final Portfolio Distribution",
                       save=True, filename='adv_mc_distribution.png')
print("      ✓ Saved: adv_mc_distribution.png")

# Test 3: t-distribution simulation
print("   1c. t-distribution simulation (fat tails)...")
t_results = simulate_t_distribution_paths(returns, n_paths=1000, horizon_days=252, seed=42)
plot_monte_carlo_paths(t_results, n_paths_show=100,
                      title="Monte Carlo Simulation - Student t Paths",
                      save=True, filename='adv_mc_paths_t.png')
print("      ✓ Saved: adv_mc_paths_t.png")

# Test 4: Scenario comparison
print("   1d. Scenario analysis (bull/bear/base)...")
scenarios = {
    'base': {'mu': returns.mean(), 'sigma': returns.std()},
    'bull': {'mu': returns.mean() * 1.5, 'sigma': returns.std() * 0.8},
    'bear': {'mu': returns.mean() * -1, 'sigma': returns.std() * 1.5}
}
scenario_results = scenario_analysis(returns, scenarios, n_paths=1000, seed=42)
plot_scenario_comparison(scenario_results,
                        title="Monte Carlo - Scenario Comparison",
                        save=True, filename='adv_mc_scenarios.png')
print("      ✓ Saved: adv_mc_scenarios.png")

#%% STRESS TESTING + PLOTS

print("\n[3/3] Running stress tests and generating plots...")

# Test 5: Stress test metrics
print("\n   2a. Stress testing metrics (3 scenarios)...")
stress_scenarios = ['covid_crash_2020', 'financial_crisis_2008', 'black_monday_1987']
stress_metrics = stress_test_metrics(returns, scenarios=stress_scenarios)
plot_stress_comparison(stress_metrics,
                      title="Stress Test Impact Analysis",
                      save=True, filename='adv_stress_comparison.png')
print("      ✓ Saved: adv_stress_comparison.png")

# Test 6: Stress heatmap
print("   2b. Stress test heatmap...")
plot_stress_heatmap(stress_metrics,
                   title="Stress Test Heatmap - Metric Changes",
                   save=True, filename='adv_stress_heatmap.png')
print("      ✓ Saved: adv_stress_heatmap.png")

# Test 7: Sensitivity analysis
print("   2c. VaR sensitivity to confidence level...")
sensitivity_df = sensitivity_analysis(
    returns,
    historical_var,
    parameter='confidence',
    values=[0.90, 0.95, 0.99],
    base_kwargs={}
)
plot_sensitivity_curve(sensitivity_df, 
                      parameter_name='confidence',
                      metric_name='VaR',
                      title="VaR Sensitivity to Confidence Level",
                      save=True, filename='adv_sensitivity_var.png')
print("      ✓ Saved: adv_sensitivity_var.png")

#%% OPTIMIZATION + PLOTS
print("\n[4/4] Running portfolio optimization and generating plots...")

# Create synthetic multi-asset data for demonstration
print("\n   3a. Creating synthetic 3-asset portfolio...")
np.random.seed(42)
n_obs = len(returns)
asset1 = returns.values
asset2 = returns.values + np.random.normal(0, 0.005, n_obs)
asset3 = returns.values + np.random.normal(0, 0.007, n_obs)

multi_returns = pd.DataFrame({
    'Asset1': asset1,
    'Asset2': asset2,
    'Asset3': asset3
}, index=returns.index)

expected_returns = multi_returns.mean()
cov_matrix = multi_returns.cov()
print(f"      Expected returns: {expected_returns.values}")

# Test 8: Efficient frontier
print("   3b. Generating efficient frontier (20 portfolios)...")
frontier = efficient_frontier(expected_returns, cov_matrix, 
                             n_portfolios=20, allow_short=False)
plot_efficient_frontier(frontier,
                       title="Efficient Frontier - 3 Asset Portfolio",
                       save=True, filename='adv_opt_frontier.png')
print("      ✓ Saved: adv_opt_frontier.png")

# Test 9: Portfolio comparison
print("   3c. Comparing portfolio strategies...")
comparison = compare_portfolios(multi_returns, cov_matrix)
plot_portfolio_comparison(comparison,
                         title="Portfolio Strategy Comparison",
                         save=True, filename='adv_opt_comparison.png')
print("      ✓ Saved: adv_opt_comparison.png")

# Test 10: Weight allocation
print("   3d. Portfolio weight allocations...")
min_var = minimum_variance_portfolio(cov_matrix, allow_short=False)
max_sharpe = maximum_sharpe_portfolio(expected_returns, cov_matrix, allow_short=False)
risk_par = risk_parity_portfolio(cov_matrix)

weights_dict = {
    'Min Variance': min_var['weights'],
    'Max Sharpe': max_sharpe['weights'],
    'Risk Parity': risk_par['weights']
}

plot_weight_allocation(weights_dict, 
                      asset_names=['Asset 1', 'Asset 2', 'Asset 3'],
                      title="Portfolio Weight Allocations",
                      save=True, filename='adv_opt_weights.png')
print("      ✓ Saved: adv_opt_weights.png")

#%% SUMMARY

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)
print("\nGenerated visualizations:")
print("  Monte Carlo (4 plots):")
print("    - adv_mc_paths_gbm.png")
print("    - adv_mc_distribution.png")
print("    - adv_mc_paths_t.png")
print("    - adv_mc_scenarios.png")
print("\n  Stress Testing (3 plots):")
print("    - adv_stress_comparison.png")
print("    - adv_stress_heatmap.png")
print("    - adv_sensitivity_var.png")
print("\n  Optimization (3 plots):")
print("    - adv_opt_frontier.png")
print("    - adv_opt_comparison.png")
print("    - adv_opt_weights.png")
print("\nTotal: 10 advanced visualizations")
print(f"\nAll files saved to: {__import__('brg_risk_metrics.config.settings', fromlist=['figures_dir']).figures_dir}")
print("="*80)
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 2025

@author: Colby Jaskowiak

BRG Project 1: Using Python to Analyze Risk Metrics
Main orchestration script - runs complete analysis pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import configuration
import brg_risk_metrics.config.settings as cfg

# Import utilities
from brg_risk_metrics.utils.logger import ExecutionLog, Timer
from brg_risk_metrics.utils.validators import check_data_quality

# Import data loading
from brg_risk_metrics.data.data_loader import load_spy_data
from brg_risk_metrics.data.return_calculator import calculate_returns, get_close

# Import metrics
from brg_risk_metrics.metrics.volatility import historical_volatility
from brg_risk_metrics.metrics.var import historical_var
from brg_risk_metrics.metrics.cvar import historical_cvar
from brg_risk_metrics.metrics.drawdown import max_drawdown, average_drawdown, drawdown_series
from brg_risk_metrics.metrics.ratios import sharpe_ratio, sortino_ratio

# Import backtesting
from brg_risk_metrics.backtesting.var_backtest import backtest_var

# Import analysis
from brg_risk_metrics.analysis.regime_analysis import detect_regimes_volatility
from brg_risk_metrics.analysis.distribution import normality_tests, compare_distributions
from brg_risk_metrics.analysis.correlation import pearson_correlation
from brg_risk_metrics.analysis.comparative import (
    compare_var_methods,
    compare_time_periods,
    compare_metrics_summary
)

#%%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Data parameters
START_DATE = '2020-01-01'
END_DATE = None  # None = most recent

# Analysis parameters
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
BACKTEST_WINDOW = 252
VAR_METHOD = 'historical'

# Output directories
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

#%%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def split_data(returns, train_ratio=0.7):
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    train_ratio : float
        Proportion for training (default: 0.7)
        
    Returns
    -------
    tuple
        (train_returns, test_returns)
    """
    split_idx = int(len(returns) * train_ratio)
    train = returns.iloc[:split_idx]
    test = returns.iloc[split_idx:]
    return train, test

#%%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete risk metrics analysis pipeline."""
    
    # Initialize execution log
    log = ExecutionLog("BRG Project 1: Risk Metrics Analysis")
    log.start()
    
    print("="*80)
    print("BRG PROJECT 1: USING PYTHON TO ANALYZE RISK METRICS")
    print("="*80)
    print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {}
    
    #%%
    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================
    print("\n[PHASE 1/6] LOADING DATA")
    print("-"*80)
    
    with Timer("Data loading"):
        spy_px = load_spy_data(start=START_DATE, end=END_DATE)
        close = get_close(spy_px)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = calculate_returns(close)
        
        log.log_step("Data loaded", data=f"{len(returns)} observations")
    
    print(f"✓ Loaded SPY data: {len(returns)} observations")
    print(f"  Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Total return: {(close.iloc[-1] / close.iloc[0] - 1) * 100:.2f}%")
    
    # Data quality check
    print("\n  Running data quality check...")
    quality = check_data_quality(returns, verbose=False)
    print(f"  Quality score: {quality['quality_score']}/100")
    
    results['data'] = {
        'n_observations': len(returns),
        'start_date': returns.index[0],
        'end_date': returns.index[-1],
        'quality_score': quality['quality_score']
    }
    
    #%%
    # =========================================================================
    # PHASE 2: CORE METRICS CALCULATION
    # =========================================================================
    print("\n[PHASE 2/6] CALCULATING CORE METRICS")
    print("-"*80)
    
    with Timer("Metrics calculation"):
        # Volatility
        vol = historical_volatility(returns, annualize=True)
        print(f"✓ Volatility (annualized): {vol:.2%}")
        
        # VaR at multiple confidence levels
        var_results = {}
        for conf in CONFIDENCE_LEVELS:
            var_val = historical_var(returns, confidence=conf)
            var_results[f'var_{int(conf*100)}'] = var_val
            print(f"✓ VaR {int(conf*100)}%: {abs(var_val):.4f} ({abs(var_val)*100:.2f}%)")
        
        # CVaR at multiple confidence levels
        cvar_results = {}
        for conf in CONFIDENCE_LEVELS:
            cvar_val = historical_cvar(returns, confidence=conf)
            cvar_results[f'cvar_{int(conf*100)}'] = cvar_val
            print(f"✓ CVaR {int(conf*100)}%: {abs(cvar_val):.4f} ({abs(cvar_val)*100:.2f}%)")
        
        # Drawdown metrics
        max_dd = max_drawdown(returns)
        avg_dd = average_drawdown(returns)
        print(f"✓ Maximum Drawdown: {abs(max_dd):.2%}")
        print(f"✓ Average Drawdown: {abs(avg_dd):.2%}")
        
        # Risk-adjusted returns
        sharpe = sharpe_ratio(returns, annualize=True)
        sortino = sortino_ratio(returns, annualize=True)
        print(f"✓ Sharpe Ratio: {sharpe:.3f}")
        print(f"✓ Sortino Ratio: {sortino:.3f}")
        
        log.log_step("Core metrics calculated")
    
    results['metrics'] = {
        'volatility': vol,
        **var_results,
        **cvar_results,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino
    }
    
    #%%
    # =========================================================================
    # PHASE 3: BACKTESTING
    # =========================================================================
    print("\n[PHASE 3/6] BACKTESTING VAR MODELS")
    print("-"*80)
    
    with Timer("Backtesting"):
        # Train/test split
        train_returns, test_returns = split_data(returns, train_ratio=0.7)
        print(f"✓ Data split: Train={len(train_returns)} obs, Test={len(test_returns)} obs")
        
        # Calculate VaR on training data (returns positive magnitude of loss)
        from brg_risk_metrics.metrics.var import historical_var as var_func
        var_magnitude = var_func(train_returns, confidence=0.95)
        
        # Convert to threshold (negative for losses)
        var_threshold = -abs(var_magnitude)
        
        # Backtest on test data - violations occur when returns fall below VaR threshold
        violations = test_returns < var_threshold
        n_violations = violations.sum()
        violation_rate = n_violations / len(test_returns)
        expected_rate = 1 - 0.95
        
        backtest_results = {
            'var_magnitude': var_magnitude,
            'var_threshold': var_threshold,
            'n_violations': n_violations,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate
        }
        
        print(f"✓ VaR 95% Backtest:")
        print(f"  VaR magnitude: {abs(var_magnitude):.4f}")
        print(f"  VaR threshold (loss): {var_threshold:.4f}")
        print(f"  Violations: {n_violations}/{len(test_returns)} ({violation_rate:.2%})")
        print(f"  Expected: {expected_rate:.2%}")
        print(f"  Status: {'✓ Pass' if abs(violation_rate - expected_rate) < 0.02 else '⚠️ Check'}")
        
        log.log_step("Backtesting completed", data=f"{n_violations} violations")
    
    results['backtest'] = {
        'n_violations': n_violations,
        'violation_rate': violation_rate,
        'expected_rate': expected_rate,
        'train_size': len(train_returns),
        'test_size': len(test_returns)
    }
    
    #%%
    # =========================================================================
    # PHASE 4: REGIME ANALYSIS
    # =========================================================================
    print("\n[PHASE 4/6] REGIME ANALYSIS")
    print("-"*80)
    
    with Timer("Regime analysis"):
        # Detect volatility regimes
        regime_results = detect_regimes_volatility(returns, window=21)
        regimes = regime_results['regimes']
        regime_stats = regime_results['regime_stats']
        
        print(f"✓ Detected {len(regime_stats)} regimes:")
        for _, row in regime_stats.iterrows():
            print(f"  {row['regime']}: {row['pct_time']:.1%} of time, "
                  f"Return={row['mean_return']:.4f}, Vol={row['std_return']:.4f}")
        
        log.log_step("Regime analysis completed")
    
    results['regimes'] = {
        'n_regimes': len(regime_stats),
        'regime_stats': regime_stats.to_dict('records')
    }
    
    #%%
    # =========================================================================
    # PHASE 5: DISTRIBUTION ANALYSIS
    # =========================================================================
    print("\n[PHASE 5/6] DISTRIBUTION ANALYSIS")
    print("-"*80)
    
    with Timer("Distribution analysis"):
        # Normality tests
        norm_tests = normality_tests(returns)
        
        print(f"✓ Normality tests:")
        for test_name, result in norm_tests.items():
            print(f"  {test_name}: {result['result']} (p={result.get('p_value', 'N/A')})")
        
        # Distribution comparison
        dist_comparison = compare_distributions(returns)
        best_dist = dist_comparison.loc[dist_comparison['aic'].idxmin(), 'distribution']
        print(f"✓ Best-fit distribution: {best_dist}")
        
        log.log_step("Distribution analysis completed")
    
    results['distribution'] = {
        'normality_rejected': all(r['result'] == 'FAIL' for r in norm_tests.values()),
        'best_distribution': best_dist
    }
    
    #%%
    # =========================================================================
    # PHASE 6: COMPARATIVE ANALYSIS
    # =========================================================================
    print("\n[PHASE 6/6] COMPARATIVE ANALYSIS")
    print("-"*80)
    
    with Timer("Comparative analysis"):
        # VaR method comparison
        var_comparison = compare_var_methods(returns, confidence=0.95)
        print(f"✓ VaR method comparison:")
        for _, row in var_comparison.iterrows():
            print(f"  {row['method']}: {abs(row['var']):.4f} ({row['pct_diff_from_hist']:+.1f}%)")
        
        # Period comparison
        period_comparison = compare_time_periods(returns, split_ratio=0.7)
        print(f"✓ In-sample vs Out-of-sample:")
        print(f"  Volatility: {period_comparison['comparison'].loc['volatility', 'pct_change']:+.1f}%")
        print(f"  Sharpe: {period_comparison['comparison'].loc['sharpe', 'pct_change']:+.1f}%")
        
        log.log_step("Comparative analysis completed")
    
    results['comparative'] = {
        'var_methods': var_comparison.to_dict('records'),
        'period_comparison': period_comparison['comparison'].to_dict()
    }
    
    #%%
    # =========================================================================
    # EXPORT RESULTS
    # =========================================================================
    print("\n[EXPORT] SAVING RESULTS")
    print("-"*80)
    
    with Timer("Results export"):
        # Save comprehensive metrics summary
        summary = compare_metrics_summary(returns)
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['Value']
        summary_df.to_csv(RESULTS_DIR / 'metrics_summary.csv')
        print(f"✓ Saved metrics summary to {RESULTS_DIR / 'metrics_summary.csv'}")
        
        # Save regime statistics
        regime_stats.to_csv(RESULTS_DIR / 'regime_statistics.csv', index=False)
        print(f"✓ Saved regime statistics to {RESULTS_DIR / 'regime_statistics.csv'}")
        
        # Save VaR comparison
        var_comparison.to_csv(RESULTS_DIR / 'var_method_comparison.csv', index=False)
        print(f"✓ Saved VaR comparison to {RESULTS_DIR / 'var_method_comparison.csv'}")
        
        # Save backtest results
        backtest_df = pd.DataFrame([results['backtest']])
        backtest_df.to_csv(RESULTS_DIR / 'backtest_results.csv', index=False)
        print(f"✓ Saved backtest results to {RESULTS_DIR / 'backtest_results.csv'}")
        
        log.log_step("Results exported")
    
    #%%
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\nKey Findings:")
    print(f"  • Analyzed {len(returns)} daily returns ({returns.index[0].date()} to {returns.index[-1].date()})")
    print(f"  • Annual volatility: {vol:.2%}")
    print(f"  • VaR 95%: {abs(var_results['var_95']):.4f} ({abs(var_results['var_95'])*100:.2f}%)")
    print(f"  • CVaR 95%: {abs(cvar_results['cvar_95']):.4f} ({abs(cvar_results['cvar_95'])*100:.2f}%)")
    print(f"  • Maximum drawdown: {abs(max_dd):.2%}")
    print(f"  • Sharpe ratio: {sharpe:.3f}")
    print(f"  • VaR backtest: {n_violations}/{len(test_returns)} violations ({violation_rate:.2%} vs {expected_rate:.2%} expected)")
    print(f"  • Normality rejected: {'Yes' if results['distribution']['normality_rejected'] else 'No'}")
    print(f"  • Best distribution: {best_dist}")
    print(f"  • Detected {len(regime_stats)} market regimes")
    
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
    print(f"Figures saved to: {cfg.figures_dir.absolute()}")
    
    log.finish(success=True)
    
    print("\n" + "="*80)
    print("EXECUTION LOG")
    print("="*80)
    print(log.get_summary())
    
    return results

#%%
if __name__ == "__main__":
    results = main()
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 2025

@author: Colby Jaskowiak

BRG Project 1: Full Analysis Runner
Runs complete analysis pipeline + generates all visualizations.
"""

import subprocess
import sys
from pathlib import Path

from brg_risk_metrics.utils.logger import Timer, ExecutionLog

#%%
def run_script(script_path, description):
    """Run a Python script and report status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*80}")
    
    try:
        with Timer(description):
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True,
                check=True
            )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in {description}!")
        print(f"Error: {e}")
        return False

#%%
def main():
    """Run complete analysis pipeline with all visualizations."""
    
    log = ExecutionLog("BRG Project 1: Full Analysis Pipeline")
    log.start()
    
    print("="*80)
    print("BRG PROJECT 1: FULL ANALYSIS PIPELINE")
    print("="*80)
    print("This will run:")
    print("  1. Main analysis (metrics, backtesting, regime analysis)")
    print("  2. Advanced features plots (Monte Carlo, stress testing, optimization)")
    print("  3. Analysis plots (regime, distribution, correlation, comparative)")
    print("="*80)
    
    success_count = 0
    total_count = 3
    
    #%%
    # =========================================================================
    # STEP 1: MAIN ANALYSIS
    # =========================================================================
    
    main_script = Path("main.py")
    if main_script.exists():
        if run_script(main_script, "Main Analysis Pipeline"):
            success_count += 1
            log.log_step("Main analysis completed")
        else:
            log.log_step("Main analysis failed", data="ERROR")
    else:
        print(f"\n⚠️ Warning: {main_script} not found, skipping...")
    
    #%%
    # =========================================================================
    # STEP 2: ADVANCED FEATURES VISUALIZATIONS
    # =========================================================================
    
    advanced_plots = Path("brg_risk_metrics/visualization/advanced_plots_generation.py")
    if advanced_plots.exists():
        if run_script(advanced_plots, "Advanced Features Visualizations"):
            success_count += 1
            log.log_step("Advanced plots generated")
        else:
            log.log_step("Advanced plots failed", data="ERROR")
    else:
        print(f"\n⚠️ Warning: {advanced_plots} not found, skipping...")
    
    #%%
    # =========================================================================
    # STEP 3: ANALYSIS VISUALIZATIONS
    # =========================================================================
    
    analysis_plots = Path("brg_risk_metrics/visualization/analysis_plots_generation.py")
    if analysis_plots.exists():
        if run_script(analysis_plots, "Analysis Visualizations"):
            success_count += 1
            log.log_step("Analysis plots generated")
        else:
            log.log_step("Analysis plots failed", data="ERROR")
    else:
        print(f"\n⚠️ Warning: {analysis_plots} not found, skipping...")
    
    #%%
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("FULL ANALYSIS PIPELINE COMPLETE!")
    print("="*80)
    print(f"Success rate: {success_count}/{total_count} steps completed")
    
    if success_count == total_count:
        print("✓ All steps completed successfully!")
        log.finish(success=True)
    else:
        print(f"⚠️ {total_count - success_count} steps failed")
        log.finish(success=False)
    
    print(f"\nResults location:")
    print(f"  • CSV files: ./results/")
    print(f"  • Figures: ./brg_risk_metrics/visualization/figures/")
    
    log.print_summary()
    
    return success_count == total_count

#%%
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
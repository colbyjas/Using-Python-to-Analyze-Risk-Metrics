# BRG Project 1: Execution Guide

## Quick Start

### Option 1: Run Everything (Recommended)
```bash
python run_full_analysis.py
```

This runs:
1. **Main analysis** - Calculates all metrics, backtesting, regime analysis
2. **Advanced plots** - Monte Carlo, stress testing, optimization visualizations
3. **Analysis plots** - Regime, distribution, correlation, comparative visualizations

**Output:**
- CSV results → `./results/`
- Figures → `./brg_risk_metrics/visualization/figures/`

---

### Option 2: Run Components Separately

#### Main Analysis Only (No Visualizations)
```bash
python main.py
```

**Outputs:**
- `results/metrics_summary.csv` - All key metrics
- `results/regime_statistics.csv` - Regime analysis results
- `results/var_method_comparison.csv` - VaR method comparison
- `results/backtest_results.csv` - Backtesting performance

**What it does:**
- Loads SPY data (2020-present)
- Calculates: Volatility, VaR, CVaR, Drawdown, Sharpe, Sortino
- Backtests VaR models
- Detects market regimes
- Tests distribution normality
- Compares methods and periods

---

#### Advanced Visualizations Only
```bash
python brg_risk_metrics/visualization/advanced_plots_generation.py
```

**Generates (10 plots):**
- Monte Carlo simulations (4 plots)
- Stress testing analysis (3 plots)
- Portfolio optimization (3 plots)

---

#### Analysis Visualizations Only
```bash
python brg_risk_metrics/visualization/analysis_plots_generation.py
```

**Generates (9 plots):**
- Regime timeline and metrics (2 plots)
- Distribution fitting and tails (2 plots)
- Rolling correlation analysis (2 plots)
- VaR methods and sensitivity (3 plots)

---

## Configuration

### Data Parameters
Edit in `main.py`:
```python
START_DATE = '2020-01-01'  # Start date for analysis
END_DATE = None            # None = most recent data
```

### Analysis Parameters
```python
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]  # VaR/CVaR confidence levels
BACKTEST_WINDOW = 252                    # Rolling window size
VAR_METHOD = 'historical'                # VaR calculation method
```

---

## Project Structure

```
brg-project-1-risk-metrics/
├── main.py                              # Main analysis script
├── run_full_analysis.py                 # Master runner (all components)
├── results/                             # Output directory (CSV files)
│   ├── metrics_summary.csv
│   ├── regime_statistics.csv
│   ├── var_method_comparison.csv
│   └── backtest_results.csv
│
└── brg_risk_metrics/
    ├── config/
    │   └── settings.py                  # Global configuration
    │
    ├── data/
    │   ├── data_loader.py               # Data loading
    │   └── return_calculator.py         # Return calculations
    │
    ├── metrics/
    │   ├── volatility.py                # Volatility metrics
    │   ├── var.py                       # Value at Risk
    │   ├── cvar.py                      # Conditional VaR
    │   ├── drawdown.py                  # Drawdown analysis
    │   ├── ratios.py                    # Sharpe, Sortino ratios
    │   └── additional.py                # Additional metrics
    │
    ├── backtesting/
    │   ├── var_backtest.py              # VaR backtesting
    │   ├── validation.py                # Model validation
    │   └── performance_metrics.py       # Backtest metrics
    │
    ├── advanced/
    │   ├── monte_carlo.py               # Monte Carlo simulations
    │   ├── stress_testing.py            # Stress scenarios
    │   ├── optimization.py              # Portfolio optimization
    │   ├── regime_switching.py          # HMM (sidelined)
    │   └── advanced_plots_functions.py  # Advanced plot functions
    │
    ├── analysis/
    │   ├── regime_analysis.py           # Regime detection
    │   ├── distribution.py              # Distribution analysis
    │   ├── correlation.py               # Correlation analysis
    │   ├── comparative.py               # Method comparisons
    │   └── analysis_plots.py            # Analysis plot functions
    │
    ├── visualization/
    │   ├── basic_plots.py               # Basic visualizations
    │   ├── distribution_plots.py        # Distribution plots
    │   ├── risk_plots.py                # Risk metric plots
    │   ├── heatmaps.py                  # Correlation heatmaps
    │   ├── advanced_plots_generation.py # Advanced plot runner
    │   ├── analysis_plots_generation.py # Analysis plot runner
    │   └── figures/                     # Output directory for plots
    │
    └── utils/
        ├── helpers.py                   # Utility functions
        ├── validators.py                # Input validation
        └── logger.py                    # Execution tracking
```

---

## Execution Times

| Component | Estimated Time |
|-----------|----------------|
| Main analysis | ~10-20 seconds |
| Advanced plots | ~30-60 seconds |
| Analysis plots | ~20-40 seconds |
| **Total (full pipeline)** | **~1-2 minutes** |

---

## Outputs Summary

### CSV Files (in `results/`)
1. **metrics_summary.csv** - Comprehensive metrics (volatility, VaR, CVaR, Sharpe, etc.)
2. **regime_statistics.csv** - Low vol vs high vol regime analysis
3. **var_method_comparison.csv** - Historical vs Parametric VaR comparison
4. **backtest_results.csv** - VaR backtest performance (violations, rates)

### Figures (in `brg_risk_metrics/visualization/figures/`)

**Advanced Features (10 plots):**
- `adv_mc_paths_gbm.png` - Monte Carlo GBM paths
- `adv_mc_distribution.png` - Final outcome distribution
- `adv_mc_paths_t.png` - Student-t distribution paths
- `adv_mc_scenarios.png` - Bull/bear/base scenarios
- `adv_stress_comparison.png` - Stress test impact
- `adv_stress_heatmap.png` - Stress scenario heatmap
- `adv_sensitivity_var.png` - VaR sensitivity curve
- `adv_opt_frontier.png` - Efficient frontier
- `adv_opt_comparison.png` - Portfolio strategy comparison
- `adv_opt_weights.png` - Weight allocations

**Analysis Plots (9 plots):**
- `analysis_regime_timeline.png` - Regime timeline with returns
- `analysis_regime_metrics.png` - Metrics by regime
- `analysis_distribution_fit.png` - Normal vs t-distribution fit
- `analysis_tail_comparison.png` - Left vs right tail analysis
- `analysis_rolling_correlation.png` - Time-varying correlation
- `analysis_correlation_breakdown.png` - Up vs down market correlation
- `analysis_var_methods.png` - VaR method comparison bars
- `analysis_window_sensitivity.png` - Window size sensitivity
- `analysis_period_comparison.png` - In-sample vs out-of-sample

---

## Troubleshooting

### "Module not found" error
```bash
# Ensure you're in the project root directory
cd brg-project-1-risk-metrics
python run_full_analysis.py
```

### Data download issues
- Requires internet connection for yfinance
- Default period: 2020-01-01 to present
- Modify START_DATE in main.py if needed

### Missing dependencies
```bash
pip install pandas numpy scipy matplotlib seaborn yfinance openpyxl python-pptx python-docx statsmodels
```

---

## Notes

- All scripts use relative paths - run from project root
- Figures directory created automatically if it doesn't exist
- Results directory created automatically if it doesn't exist
- Progress bars and timing information displayed during execution
- Comprehensive execution log printed at end of run_full_analysis.py

---

## For Your Report

After running `python run_full_analysis.py`:

1. **Check results/** - Use CSV files for tables in your report
2. **Check figures/** - Use PNG files in your LaTeX document
3. **Review console output** - Copy key findings into report
4. **Execution log** - Documents what was run and when

**Report-ready outputs:**
- ✅ All metrics calculated and validated
- ✅ 19 publication-quality figures
- ✅ 4 CSV files with detailed results
- ✅ Execution logs for reproducibility
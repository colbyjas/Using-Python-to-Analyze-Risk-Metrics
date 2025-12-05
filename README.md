# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:12:55 2025

@author: Colby Jaskowiak
"""

# BRG Risk Metrics

A comprehensive Python framework for analyzing portfolio risk metrics including volatility, VaR, CVaR, Sharpe ratio, Sortino ratio, and drawdown analysis.

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/brg-project-1-risk-metrics.git
cd brg-project-1-risk-metrics

# Install the package
pip install -e .

# Run test
python test_data.py
```

### Manual Installation

If you prefer not to install:

```bash
# Just install dependencies
pip install -r requirements.txt

# Run from project root
python test_data.py
```

## Usage

### Basic Example

```python
from data.data_loader import load_spy_data
from data.return_calculator import calculate_returns, print_stats

# Load data
spy = load_spy_data(start='2020-01-01')

# Calculate returns
returns = calculate_returns(spy['Adj Close'])

# Print statistics
print_stats(returns, "SPY Returns")
```

### Configuration

Edit `config/settings.py` to customize:
- Ticker symbols and date ranges
- Risk-free rate
- VaR/CVaR confidence levels
- Rolling window sizes
- Visualization settings

## Project Structure

```
brg_risk_metrics/
├── config/          # Configuration settings
├── data/            # Data loading and processing
├── metrics/         # Risk metric calculations
├── visualization/   # Plotting functions
├── backtesting/     # Model validation
└── reporting/       # Report generation
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.28
- scipy >= 1.10.0
- matplotlib >= 3.7.0

See `requirements.txt` for complete list.

## Author

Colby Jaskowiak - BRG Project 1

## License

MIT License - feel free to use and modify.
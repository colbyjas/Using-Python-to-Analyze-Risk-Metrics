# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:02:01 2025

@author: Colby Jaskowiak

Configuration settings for BRG Risk Metrics project.
Centralized location for all parameters and constants.
"""

ticker = 'SPY'
start_date = '2020-01-01'
end_date = None
frequency = 'daily'

#%%
return_method = 'simple' # 'simple' or 'log'

annualization_factor = {
    'daily': 252,
    'weekly': 52,
    'monthly': 12,
    'quarterly': 4,
    'annual': 1
}

risk_free_rate = 0.0412

#%%
var_confidence_levels = [0.90, 0.95, 0.99]
default_var_confidence = var_confidence_levels[1]

cvar_confidence_levels = [0.95, 0.99]
default_cvar_confidence = cvar_confidence_levels[0]

rolling_windows = {"short": 30, "medium": 90, "long": 252}
default_rolling_window = rolling_windows["long"]

ewma_lambda = 0.94

monte_carlo_sims = 10000
monte_carlo_random_seed = 42

#%%
sortino_mar = 'zero'
# Minimum Acceptable Return (MAR) for Sortino Ratio
# Options: 'zero', 'risk_free', or a specific value, or 0.0, or RISK_FREE_RATE

#%% Visual Settings
default_figure_size = (12,6)
large_figure_size = (14,8)

plot_style = 'seaborn-v0_8-darkgrid'
color_palette = 'husl'
save_dpi = 300

#%%
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
pkg_root = Path(__file__).resolve().parents[1]

figures_dir = pkg_root / 'visualization' / 'figures'
data_dir    = pkg_root / 'data' / 'saved'
reports_dir = pkg_root / 'reporting' / 'reports'

for p in (figures_dir, data_dir, reports_dir):
    p.mkdir(parents=True, exist_ok=True)

#%% Numerical Settings
min_observations = 30
numerical_tolerance = 1e-10
max_iterations = 1000

#%%d
display_decimals = 4
pandas_display_rows = 50
pandas_display_columns = 20

#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 23:54:49 2025

@author: Colby Jaskowiak

Setup file for BRG Risk Metrics package.
Makes the project installable with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="brg_risk_metrics",
    version="1.0.0",
    author="Colby Jaskowiak",
    description="Python-based portfolio risk metrics analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.28",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        'dev': ['pytest>=7.3.0', 'pytest-cov>=4.1.0'],
    },
)
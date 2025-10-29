
# This is a test file

import pandas as pd
import numpy as np
import yfinance as yf

#%%
tickers = ['NVDA', 'AAPL', 'META']
price_data = yf.download(tickers, start = '2020-10-29', end = pd.Timestamp.today().strftime('%Y-%m-%d'))['Close']
price_data.dropna(inplace=True)
#%%


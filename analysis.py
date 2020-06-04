# -*- coding: utf-8 -*-
"""
@author: Wangyun Luo
email: samlwy@bu.edu
"""
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from datetime import timedelta
from datetime import date


# ---------------------------------------------------------------------------------------
""" download data """
""" change df to analysis """
ticker = 'tsla'
time_interv = timedelta(days=365 * 5)
# min_date_allowed = datetime(2015, 4, 1)
df = yf.download(f'{ticker}', date.today()-time_interv, date.today())
""" write file to store data """
# data.to_csv(raw_data_path)
df_adj_close = df['Adj Close']
df_volume = df['Volume']
# df_volume.plot()
# ---------------------------------------------------------------------------------------
""" MA """
df = yf.download(f'{ticker}', date.today()-time_interv, date.today())
print('adding min_periods=0 did the same thing as df.dropna')
df['10D MA'] = df['Adj Close'].rolling(window=10, min_periods=0).mean()
print(df.head())
# df.dropna(inplace=True)
# print(df.head())
""" resample """
df_ohlc = df['Adj Close'].resample('1M').ohlc()
print(df_ohlc.head())
df
plt.show()
""" Visualization """
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=2, colspan=1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['10D MA'])
ax2.plot(df.index, df['Volume'])
plt.show()


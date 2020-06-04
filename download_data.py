# -*- coding: utf-8 -*-
"""
@author: Wangyun Luo
email: samlwy@bu.edu
"""
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
from args import tickers, raw_data_path, input_data_path
if __name__ == "__main__":
    # -------------------- Download data ---------------------------------
    time_interv = timedelta(days=365 * 5)
    data = yf.download(tickers, date.today() - time_interv, date.today())
    data.to_csv(raw_data_path)
    input_data = data['Adj Close']
    input_data.to_csv(input_data_path)

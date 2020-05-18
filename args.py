# -*- coding: utf-8 -*-
"""
@author: Wangyun Luo
email: samlwy@bu.edu
"""
tickers = ['SPY', 'XLV', 'XLF', 'XLE', 'XLK',
           'XLRE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU']
indicators = ['log_return', '5D_MA','250D_MA','5D_EWMA','250D_EWMA', '20D_Vol','120D_Vol','5D_Momentum','20D_Momentum']
raw_data_path = "data/etfs.csv"
input_data_path = "data/etfs_input.csv"
processed_data_path = "data/etfs_processed.csv"

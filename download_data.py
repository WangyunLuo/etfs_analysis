# -*- coding: utf-8 -*-
"""
@author: Wangyun Luo
email: samlwy@bu.edu
"""
import pandas as pd
import pymongo
from datetime import date
import yfinance as yf
import os
from args import tickers, raw_data_path, input_data_path
from datetime import timedelta
import dns
if __name__ == "__main__":
# -------------------- Download data and updated daily. ---------------------------------
    time_interv = timedelta(days=365 * 5)
    data = yf.download(tickers, date.today() - time_interv, date.today())
    data.to_csv(raw_data_path)
    input_data = data['Adj Close']
    input_data.to_csv(input_data_path)
    """
    # -------------------- Put the input data in MongoDB. -----------------------------------
    client = pymongo.MongoClient(
        "mongodb+srv://Newuser:GLOBALAI@cluster0-ujbuf.mongodb.net/test?retryWrites=true&w=majority")
    col = client.datebase.collection
    csv_to_dict = pd.read_csv(input_data_path).to_dict('records')
    col.insert_many(csv_to_dict)
    """
    

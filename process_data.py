# -*- coding: utf-8 -*-
"""
@author: Wangyun Luo
email: samlwy@bu.edu
"""
import numpy as np
import pandas as pd
from args import tickers, raw_data_path, input_data_path, processed_data_path, indicators
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
from itertools import product

def cal_df_stats(df, tickers, indicators):
    """
    df: a multi-dimensional dataframe of increasing time series data
    """
    for indicator in indicators:
        if indicator == "log_return":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = np.log(df[tickers]/df[tickers].shift(1))
        elif indicator == "5D_MA":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers].rolling(5).mean()
        elif indicator == "250D_MA":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers].rolling(250).mean()
        elif indicator == "5D_EWMA":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers].ewm(span=5).mean()
        elif indicator == "250D_EWMA":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers].ewm(span=250).mean()
        elif indicator == "20D_Vol":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers].rolling(20).std(ddof=0)
        elif indicator == "120D_Vol":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers].rolling(120).std(ddof=0)
        elif indicator == "5D_Momentum":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers] - df[tickers].shift(5)
        elif indicator == "20D_Momentum":
            tempt = [ticker + f"_{indicator}" for ticker in tickers]
            df[tempt] = df[tickers] - df[tickers].shift(20)

    return df


df = pd.read_csv(input_data_path, index_col=0)


def cal_df_stats_multiprocess(indicator):
    """
    input: indicator
    df: a multi-dimensional dataframe of increasing time series data
    tickers: global variable
    """
    results = pd.DataFrame()
    if indicator == "log_return":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = np.log(df[tickers]/df[tickers].shift(1))
    elif indicator == "5D_MA":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers].rolling(7).mean()
    elif indicator == "250D_MA":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers].rolling(250).mean()
    elif indicator == "5D_EWMA":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers].ewm(span=5).mean()
    elif indicator == "250D_EWMA":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers].ewm(span=250).mean()
    elif indicator == "20D_Vol":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers].rolling(20).std(ddof=0)
    elif indicator == "120D_Vol":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers].rolling(120).std(ddof=0)
    elif indicator == "5D_Momentum":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers] - df[tickers].shift(5)
    elif indicator == "20D_Momentum":
        tempt = [ticker + f"_{indicator}" for ticker in tickers]
        results[tempt] = df[tickers] - df[tickers].shift(20)
    return results


if __name__ == '__main__':
    # ---------------------------- Descriptive statistics -------------------------------
    df = pd.read_csv(input_data_path, index_col=0)
    t1 = time.time()
    processed_data = cal_df_stats(df, tickers, indicators)
    t2 = time.time()
    f1_time = t2 - t1
    print(f'Without using multiprocessing, elapsed time:{(f1_time)*1000} ms')
    # processed_data.to_csv(processed_data_path)
    # --------------------------------- Multiprocessing ---------------------------------
    t1 = time.time()
    pool = Pool(processes=2)
    results = pool.map(cal_df_stats_multiprocess, indicators)
    tempt = pd.concat(results, axis=1)
    t2 = time.time()
    f2_time = t2 - t1
    print(f'Using multiprocessing, elapsed time:{(f2_time)*1000} ms')
    print(
        f'Time improvement is {(f1_time - f2_time)*1000} ms by multiprocessing')
    # --------------------------------- Merge DataFrame ---------------------------------
    df_data = pd.read_csv(input_data_path, index_col=0)
    processed_data_2 = df_data.merge(tempt, left_on='Date', right_index=True)
    processed_data_2.to_csv(processed_data_path)
    print(processed_data_2.head())

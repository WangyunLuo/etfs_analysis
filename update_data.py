from datetime import timedelta, date
from datetime import datetime
from args import tickers, indicators, raw_data_path, input_data_path, processed_data_path
import yfinance as yf
import pandas as pd
from process_data import cal_df_stats

if __name__ == "__main__":
    end_date = date.today()
    time_interval = timedelta(days=1)
    processed_data = pd.read_csv(processed_data_path, index_col=0)
    history_date = datetime.strptime(
        processed_data.index[-1], '%Y-%m-%d').date()
    if history_date == (end_date - time_interval):
        print("Database don't need to be updated")
    else:
        history_date = datetime.strptime(
            processed_data.index[-1], '%Y-%m-%d').date()
        start_date = history_date + time_interval
        # when the arrange is not trading days, this function returns the last trading day with a Timestamp data
        new_raw_data = yf.download(tickers, start_date, end_date)
        index_new_raw_data = new_raw_data.index[0]
        if index_new_raw_data == history_date:
            print("Database don't need to be updated")
        else:
            raw_data = pd.read_csv(raw_data_path)
            input_data = pd.read_csv(input_data_path)
            new_input_data = new_raw_data['Adj Close']
            updated_raw_data = pd.concat([raw_data, new_raw_data], axis=1)
            updated_input_data = pd.concat(
                [input_data, new_input_data], axis=1)
            new_processed_data = cal_df_stats(
                new_input_data, tickers, indicators)
            updated_processed_data = pd.concat(
                [processed_data, new_processed_data], axis=1)
            updated_raw_data.to_csv(raw_data_path)
            updated_input_data.to_csv(input_data_path)
            updated_processed_data.to_csv(processed_data_path)

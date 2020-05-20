import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import yfinance as yf
from datetime import datetime
import pandas as pd
import plotly.figure_factory as ff
import plotly.offline as pyo
import numpy as np
import random
from args import tickers, indicators, raw_data_path, input_data_path, processed_data_path
from process_data import cal_df_stats, cal_df_stats_multiprocess
import timedelta
from datetime import date
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
from itertools import product
import dash_auth

# ---------------------------------------------------------------------------------------
# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
USERNAME_PASSWORD_PAIRS = [['wangyun', '0415']]
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server
# ---------------------------------------------------------------------------------------
options = [{'label': tic, 'value': tic} for tic in tickers]
# ---------------------------------------------------------------------------------------
# download the data
# time_interv = timedelta(days=365 * 5)
min_date_allowed = datetime(2015, 4, 1)
data = yf.download(tickers, min_date_allowed, date.today())
# data.to_csv(raw_data_path)
input_data = data['Adj Close']
volumn_data = data['Volume']
# input_data.to_csv(input_data_path)
# ---------------------------------------------------------------------------------------
# process the data
# df = pd.read_csv(input_data_path, index_col=0)
processed_data = cal_df_stats(input_data, tickers, indicators)
# --------------------------------- Multiprocessing ---------------------------------
# df = input_data
# import process_data
# pool = Pool(processes=2)
# results = pool.map(cal_df_stats_multiprocess, indicators)
# tempt = pd.concat(results, axis=1)
# --------------------------------- Merge DataFrame ---------------------------------
# df_data = pd.read_csv(input_data_path, index_col=0)
# processed_data_2 = input_data.merge(tempt, left_on='Date', right_index=True)
# processed_data_2.to_csv(processed_data_path)
print(processed_data.tail())
df = processed_data
# ---------------------------------------------------------------------------------------
# set up
# server = app.server
# ---------------------------------------------------------------------------------------
app.layout = html.Div(
    [
        html.H1("ETFs Dashboard", style={'text-align': 'center'}),
        html.Hr(),
        html.Div([
            dbc.Row([
                dbc.Col(html.Div([html.H3('Enter a ETFs ticker:', style={'paddingRight': '30px'}),
                                  dcc.Dropdown(id='my-ticker', value='XLV', options=options)],
                                 style={'verticalAlign': 'top', 'display': 'inline-block'})),
                dbc.Col(html.Div([html.H3('Select Date:'),
                                  dcc.DatePickerRange(id='date-range',
                                                      min_date_allowed=datetime(
                                                          2015, 4, 1),
                                                      max_date_allowed=datetime.today(),
                                                      start_date=datetime(
                                                          2018, 1, 1),
                                                      end_date=datetime.today())],
                                 style={'display': 'inline-block'}))
            ])]),
        html.Hr(),
        dbc.Button("Generate Graphs", color="primary",
                   block=True, id="button", className="mb-3"),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="trend"), style={
                'border-style': 'solid'}, width=6),
            dbc.Col(dcc.Graph(id="timeseries"), style={
                'border-style': 'solid'}, width=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="table"),  style={
                'border-style': 'solid'}),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="histogram"), style={
                'border-style': 'solid'}, width=4),
            dbc.Col(dcc.Graph(id="bar"),  style={
                'border-style': 'solid'}, width=4),
            dbc.Col(dcc.Graph(id="scatter"), style={
                'border-style': 'solid'}, width=4),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="heatmap"),  style={
                'border-style': 'solid'}, width=4),
            dbc.Col(dcc.Graph(id="radar"),  style={
                'border-style': 'solid'}, width=4),
            dbc.Col(dcc.Graph(id="speedometer"), style={
                'border-style': 'solid'}, width=4),
        ])
    ])


@ app.callback(
    [Output("trend", "figure"),
     Output("timeseries", "figure"),
     Output("scatter", "figure"),
     Output("histogram", "figure"),
     Output("bar", "figure"),
     Output("table", "figure"),
     Output("heatmap", "figure"),
     Output("radar", "figure"),
     Output("speedometer", "figure")],
    [Input("button", "n_clicks")],
    [State("my-ticker", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date")])
def update_trend1_graph(n_clicks, etfs_ticker, start_date, end_date):
    # start = datetime.strptime(start_date[: 10], "%Y-%m-%d")
    # end = datetime.strptime(end_date[: 10], "%Y-%m-%d")
    df = processed_data.loc[(processed_data.index >= start_date) & (
        processed_data.index <= end_date)]
    # df = yf.download(etfs_ticker, start_date, end_date)
    # processed_df = cal_df_stats(df, etfs_ticker, indicators)
    # -----------------------------------------------------------------------------------
    # update data if necessary
    # spy = df['SPY']

    # -----------------------------------------------------------------------------------
    # ticker_indicators = [
    #     '{}_'.format(etfs_ticker) + indicator for indicator in indicators]
    # filtered_df = df[ticker_indicators]
    # # -------------------- set up the output trace --------------------------------------
    # traces = []
    # traces = [go.Scatter(
    #     x=filtered_df.index,
    #     y=filtered_df[col],
    #     mode='lines',
    #     name=col
    # ) for col in ticker_indicators]
    traces = [
        go.Scatter(x=df.index, y=df[etfs_ticker],
                   mode="lines", name=etfs_ticker),
        go.Scatter(x=df.index, y=df[etfs_ticker].rolling(
            20).mean(), mode="lines", name=f"{etfs_ticker} 20_D_MA"),
        go.Scatter(x=df.index, y=df["SPY"],
                   mode="lines", name="SPY"),
        go.Scatter(x=df.index, y=df["SPY"].rolling(
            20).mean(), mode="lines", name="SPY 20_D_MA"),
    ]
    # go.Indicator(value=df[etfs_ticker].iloc[-1],
    #              mode="number+delta",
    #              delta={
    #                  'reference': df[etfs_ticker].iloc[-2], "valueformat": ".of"},
    #              domain={'x': [0, 1], 'y': [0, 1]},
    #              title={'text': 'Adj Close Price',
    #                     'xanchor': 'center',
    #                     'x': 0.5})
    # 'xanchor': 'center', 'x': 0.5,
    trend = go.Figure(data=traces,
                      layout=go.Layout(title=dict(text=f"{etfs_ticker} & SPY",
                                                  xanchor='center',
                                                  x=0.5),
                                       xaxis=dict(title='Date'),
                                       yaxis=dict(title='Price'),
                                       paper_bgcolor="#222",
                                       plot_bgcolor='#222',
                                       font=dict(color='#fff')))

    # time series
    timeseries = go.Figure(data=[go.Scatter(x=df.index, y=df[etfs_ticker])],
                           layout=go.Layout(title=dict(text=f"{etfs_ticker} Time Series",
                                                       xanchor='center',
                                                       x=0.5),
                                            xaxis=dict(title='Date'),
                                            yaxis=dict(title='Price')))
    timeseries.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])))
    timeseries.update_layout(title=f"{etfs_ticker} Time Series", paper_bgcolor="#222", plot_bgcolor="#222",
                             font={"color": "#fff"})

    # scatter plot
    scatter = go.Figure(data=[go.Scatter(x=df['SPY'].pct_change(),
                                         y=df[etfs_ticker].pct_change(),
                                         mode='markers',
                                         marker=dict(color=df['SPY'].pct_change(),
                                                     colorscale='Viridis', showscale=True,
                                                     size=3))],
                        layout=go.Layout(title=f"{etfs_ticker} Return v.s. SPY Return",
                                         xaxis={'title': 'SPY Return'},
                                         yaxis={
                                             'title': f'{etfs_ticker} Return'},
                                         paper_bgcolor="#222", plot_bgcolor="#222",
                                         font={"color": "#fff"}))

    # histogram
    hist_data = [df[etfs_ticker].tolist(), df['SPY'].tolist()]
    group_labels = [etfs_ticker, "SPY"]
    hist = ff.create_distplot(hist_data, group_labels, bin_size=.2)
    hist.update_layout(title=dict(text=f"{etfs_ticker} Price Distribution Histogram",
                                  xanchor='center',
                                  x=0.5),
                       xaxis={'title': 'Price'},
                       yaxis={'title': 'Distribution'},
                       paper_bgcolor="#222",
                       plot_bgcolor="#222",
                       font={"color": "#fff"})

    # horizontal Bar
    # Calculate total volume of every month of the single stock and create dummy data
    tic_bar = volumn_data[etfs_ticker].reset_index()
    tic_bar['Date'] = tic_bar['Date'].apply(
        lambda x: datetime.strftime(x, '%Y-%m'))
    tic_bar = tic_bar[etfs_ticker].groupby(tic_bar['Date']).mean()
    spy_bar = volumn_data["SPY"].reset_index()
    spy_bar['Date'] = spy_bar['Date'].apply(
        lambda x: datetime.strftime(x, '%Y-%m'))
    spy_bar = spy_bar["SPY"].groupby(spy_bar['Date']).mean()
    # random_bar = pd.Series(data=np.random.randint(low=tic_bar.min(), high=tic_bar.max(), size=len(tic_bar)),
    #                        index=tic_bar.index)
    bar = go.Figure(data=[go.Bar(y=tic_bar.index, x=tic_bar.tolist(), name=f"{etfs_ticker} Volume", orientation='h'),
                          go.Bar(y=spy_bar.index, x=spy_bar.tolist(), name="SPY Volume", orientation='h')],
                    layout=go.Layout(title=dict(text=f"{etfs_ticker} vs SPY Monthly Average Volume",
                                                xanchor='center',
                                                x=0.5),
                                     yaxis={'title': 'Month'},
                                     xaxis={'title': 'Volume'}, paper_bgcolor="#222", plot_bgcolor="#222",
                                     font={"color": "#fff"}))

    # table

    tickers_indicators = [
        '{}_'.format(etfs_ticker) + indicator for indicator in indicators]
    tmp = df[tickers_indicators]
    tmp = df[tickers_indicators].reset_index()
    table = go.Figure(data=[go.Table(header=dict(values=tmp.columns.tolist(), font=dict(size=10, color="#fff"),
                                                 align="left", fill_color='#222'),
                                     cells=dict(values=[tmp[k].tolist() for k in tmp.columns.tolist()],
                                                fill_color='grey', font=dict(color='white'),
                                                align="left"))],
                      layout=go.Layout(title=dict(text=f"{etfs_ticker} Data",
                                                  xanchor='center',
                                                  x=0.5),
                                       paper_bgcolor="#222",
                                       plot_bgcolor="#222",
                                       font={"color": "#fff"}))

    # heatmap
    # Calculate covariance matrix of the stock with other randomly chosen stocks
    # random_list = random.sample(nasdq.index.tolist(), 10)
    # if etfs_ticker in random_list:
    #     random_list = random_list
    # else:
    #     random_list.append(etfs_ticker)
    # matrix_df = yf.download(random_list, start, end)[
    #     'Adj Close'].fillna(0).pct_change()
    matrix_df = df[tickers].fillna(0)
    # matrix_df = yf.download(tickers)['Adj Close'].fillna(0).pct_change()
    heatmap = go.Figure(data=[go.Heatmap(x=matrix_df.columns, y=matrix_df.columns,
                                         z=matrix_df.corr(), colorscale='Viridis')],
                        layout=go.Layout(title=dict(text="Correlation Heatmap of 11 ETFs",
                                                    xanchor='center',
                                                    x=0.5),
                                         paper_bgcolor="#222", plot_bgcolor="#222",
                                         font={"color": "#fff"}))

    # radar Chart
    # Company Analysis (random data)
    criteria = ['Price', 'Leadership', 'Human Resources & Training', 'Organization for Change',
                'Production System', 'Information & Communication Systems', 'Safety Practices',
                'State of Technology', 'Subcontractor Management', 'Purchasing & Inventory Control',
                'Understandable Goals', 'Planning Programming', 'Cost & Due Date Control']
    grades = list(np.random.randint(1, 11, size=len(criteria)))
    radar = go.Figure(data=[go.Scatterpolar(r=grades, theta=criteria, fill='toself')],
                      layout=go.Layout(title=dict(text=f"{etfs_ticker} Company Analysis (Random Data)",
                                                  xanchor='center',
                                                  x=0.5),
                                       paper_bgcolor="#222", font={"color": "#fff"}))

    # speedometer
    speed = go.Figure(data=[go.Indicator(mode="gauge+number", value=df[etfs_ticker].iloc[-1],
                                         title={'text': "Last ADJ Close"})],
                      layout=go.Layout(title=dict(text=f"{etfs_ticker} Speedometer",
                                                  xanchor='center',
                                                  x=0.5),
                                       paper_bgcolor="#222",
                                       plot_bgcolor="#222",
                                       font={"color": "#fff"}))

    return trend, timeseries, scatter, hist, bar, table, heatmap, radar, speed


if __name__ == '__main__':
    # pyo.plot(trend, filename='line1.html')
    # pyo.plot(heatmap, filename='line2.html')
    # pyo.plot(table, filename='line2.html')
    app.run_server()  # debug=True)

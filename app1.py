# -*- coding: utf-8 -*-
"""
@author: Wangyun Luo
email: samlwy@bu.edu
"""
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from args import processed_data_path, tickers, indicators
from dash.dependencies import Input, Output
import plotly.offline as pyo

# --------------------------------- Auto Setup ------------------------------------------
USERNAME_PASSWORD_PAIRS = [
    ['Wangyun','007']
]
app = dash.Dash()
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server
# --------------------------------- df should be defined first ------------------------------------------
df = pd.read_csv(processed_data_path, index_col=0)
# ----------------------------------- dash layout ---------------------------------------
# We need to construct a dictionary of dropdown values for the indicators
indicators_options = []
for indicator in indicators:
    indicators_options.append({'label': str(indicator), 'value': indicator})

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Dropdown(id='indicator-picker', options=indicators_options,
                 value=indicators[0])
])


@app.callback(Output('graph-with-slider', 'figure'),
              [Input('indicator-picker', 'value')])
def update_figure(selected_indicator):
    ticker_selected_indicator = [
        ticker + '_{}'.format(selected_indicator) for ticker in tickers]
    filtered_df = df[ticker_selected_indicator]
    # -------------------- set up the output trace --------------------------------------
    traces = []
    traces = [go.Scatter(
        x=filtered_df.index,
        y=filtered_df[col],
        mode='lines',
        name=col
    ) for col in ticker_selected_indicator]

    layout = go.Layout(
        title='Trend of ETFs {}'.format(selected_indicator),
        hovermode='closest'
    )
    """
    plotly reference:
    fig = go.Figure(data=traces, layout=layout)
    pyo.plot(fig, filename='line2.html')
    # -------------------- set up the output layout  ------------------------------------
    layout = go.layout(
        xaxis={'type': 'log', 'title': 'depend on indicator'},
        yaxis={'title': 'depend on indicator'},
        hovermode='closest'
    )
    """
    # ------------------------------------- return --------------------------------------
    return{
        'data': traces,
        'layout': layout
    }


if __name__ == '__main__':
    df = pd.read_csv(processed_data_path, index_col=0)
    app.run_server()

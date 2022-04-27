#!/usr/bin/env python
import time

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.subplots

import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None
    config_fn = None # 'config/DPXConfig_11.conf'
    bin_edges = None # 'config/bin_edges.json'
    params_fn = None # 'config/Chip11_Ikrum35.json'
    dpx = dch.Dosepix(
        port_name=PORT,
        config_fn=config_fn,
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    # Configure dash app
    app = dash.Dash(__name__, update_title=None)
    app.layout = html.Div(
        [
            dcc.Graph(id='live-update-graph'),
            dcc.Interval(
                id="interval",
                interval=100,
                n_intervals=0
            )
        ]
    )

    # Init readout
    start_time = time.time()
    data = {'time': [], 'volt': []}

    # Update plot
    @app.callback(Output('live-update-graph', 'figure'),
    [Input('interval', 'n_intervals')])
    def update_data(n):
        data['volt'].append( int(dpx.dpf.read_bias(), 16) )
        data['time'].append( time.time() - start_time )

        fig = plotly.subplots.make_subplots(vertical_spacing=0.2)
        fig['layout']['margin'] = {
            'l': 30, 'r': 10, 'b': 30, 't': 10
        }
        fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

        fig.append_trace({
            'x': data['time'],
            'y': data['volt'],
            'name': 'Bias Voltage',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
        return fig

    # Run dash server
    app.run_server(debug=True)

if __name__ == '__main__':
    main()

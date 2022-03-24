#!/usr/bin/env python
import time

import dash
import dash_html_components as html
import dash_core_components as dcc

import numpy as np
import matplotlib.pyplot as plt

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

    # Create interactive plot
    figure = dict(
        data=[{'x': [], 'y': []}],
        layout=dict(
            xaxis=dict(
                range=[-1, 1]
            ),
            yaxis=dict(
                range=[-1, 1]
            )
        )
    )

    app = dash.Dash(__name__, update_title=None)
    app.layout = html.Div(
        [
            dcc.Graph(id='graph', figure=figure),
            dcc.Interval(id="interval")
        ]
    )

    voltages = []
    meas_time = []
    start_time = time.time()
    for _ in range(1000):
        voltages.append( int(dpx.dpf.read_bias(), 16) )
        meas_time.append( time.time() - start_time )

        # line.set_data(meas_time, voltages)
        # fig.canvas.draw()
        time.sleep(0.01)
        # plt.show()

    plt.plot(meas_time, voltages)
    plt.show()

    '''
    while True:
        omr = ''.join([np.random.choice(list('0123456789abcdef')) for n in range(6)])
        dpx.dpf.write_omr(omr)

        print('OMR set:', omr)
        print('OMR get:', dpx.dpf.read_omr())
        print()

        time.sleep(0.1)
    '''

    # *_, last = dpx.equalization(config_fn=config_fn)
    # *_, last = dpx.dpf.measure_tot(out_dir='tot_measurement/', int_plot=False)
    # *_, last = dpx.dpf.measure_dosi(frame_time=1, frames=None, freq=False)

if __name__ == '__main__':
    main()

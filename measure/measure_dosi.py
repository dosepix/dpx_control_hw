#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    port = dch.find_port()
    if port is None:
        port = '/dev/ttyACM0'

    thl_calib_fn = None
    config_fn = 'config.conf'
    bin_edges = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=PORT,
        config_fn=config_fn,
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    dpx.dpm.measure_dosi(
        frame_time=1,
        frames=10,
        freq=False,
        out_fn='dose_measurement.json',
        use_gui=False
    )

if __name__ == '__main__':
    main()

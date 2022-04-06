#!/usr/bin/env python
""" Perform integration-measurements. See documentation of
dpm.measure_integration for more information"""
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch

CONFIG = 'config.conf'
BIN_EDGES = None
PARAMS_FN = None
OUT_FN = None # 'integration_measurement.json'
def main():
    port = dch.find_port()
    if port is None:
        port = '/dev/ttyACM0'

    dpx = dch.Dosepix(
        port_name=port,
        config_fn=CONFIG,
        thl_calib_fn=None,
        params_fn=PARAMS_FN,
        bin_edges_fn=BIN_EDGES
    )

    out_dict = dpx.dpm.measure_integration(
        out_fn=OUT_FN,
        meas_time=10,
        frame_time=0,
        use_gui=False
    )

    frames = np.asarray( out_dict['frames'] )
    frames = frames.flatten()
    frames = frames[frames > 0]
    print(frames)
    plt.hist(frames, bins=100)
    plt.show()

if __name__ == '__main__':
    main()

#!/usr/bin/env python
""" Perform dosi-measurements. See documentation of
dpm.measure_dosi for more information"""
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch

CONFIG = 'config.conf'
BIN_EDGES = np.asarray(np.linspace(10, 800, 16), dtype=int).tolist()
PARAMS_FN = None
OUT_FN = None # 'dose_measurement.json'
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

    out_dict = dpx.dpm.measure_dosi(
        frame_time=1,
        frames=10,
        freq=False,
        out_fn=OUT_FN,
        use_gui=False
    )

    frames = np.asarray( out_dict['frames'] )
    hist = np.sum(frames, axis=(0, 2))
    plt.step(BIN_EDGES, hist, where='post')
    plt.show()

if __name__ == '__main__':
    main()

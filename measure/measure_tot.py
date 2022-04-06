#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch

CONFIG = 'config.conf'
OUT_DIR = None # 'tot_measurement/'
def main():
    port = dch.find_port()
    if port is None:
        port = '/dev/ttyACM0'

    thl_calib_fn = None
    bin_edges = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=port,
        config_fn=CONFIG,
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    tot_d = dpx.dpm.measure_tot(
        frame_time=0,
        save_frames=None,
        out_dir=OUT_DIR,
        meas_time=5,
        make_hist=True,
        use_gui=False
    )

    plt.plot(np.sum(tot_d, axis=0))
    plt.xlim(0, 400)
    plt.xlabel('ToT (10 ns)')
    plt.show()

if __name__ == '__main__':
    main()

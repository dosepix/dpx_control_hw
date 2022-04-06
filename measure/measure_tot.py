#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None
    config_fn = 'config.conf'
    bin_edges = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=PORT,
        config_fn='config.conf',
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    tot_d = dpx.dpm.measure_tot(
        frame_time=0,
        save_frames=None,
        out_dir='tot_measurement/',
        meas_time=5,
        make_hist=True,
        use_gui=False
    )

    plt.plot(np.sum(tot_d, axis=0))
    plt.xlim(0, 400)
    plt.show()

if __name__ == '__main__':
    main()

#!/usr/bin/env python
import dpx_control_hw as dch
import numpy as np
import matplotlib.pyplot as plt

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None
    config_fn = None
    bin_edges = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=PORT,
        config_fn='config.conf',
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    tot_frames = dpx.dpf.measure_tot(
        frame_time=0,
        save_frames=None,
        out_dir='tot_measurement/',
        meas_time=1,
        use_gui=False
    )

    # Plot
    tot = np.asarray(tot_frames).flatten()
    tot = tot[tot > 0]
    plt.hist(tot, bins=np.arange(1000))
    plt.show()

if __name__ == '__main__':
    main()

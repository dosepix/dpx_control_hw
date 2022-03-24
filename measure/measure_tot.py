#!/usr/bin/env python
import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None
    config_fn = None
    bin_edges = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=PORT,
        config_fn=config_fn,
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    dpx.dpf.measure_tot(
        frame_time=0,
        save_frames=None,
        out_dir='tot_measurement/',
        meas_time=5,
        use_gui=False
    )

if __name__ == '__main__':
    main()

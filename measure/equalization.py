#!/usr/bin/env python
import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None
    bin_edges = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=PORT,
        config_fn=None,
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=bin_edges
    )

    config_fn = 'config.conf'
    thl_calib_d = dpx.dpf.measure_thl(out_fn=None, plot=False)
    dpx.set_thl_calib(thl_calib_d)

    dpx.equalization(
        config_fn,
        thl_step=1,
        noise_limit=10,
        n_evals=3,
        num_dacs=2,
        i_pixeldac=50,
        thl_offset=30,
        plot=True
    )

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Perform a detector equalization. Keep the
detector dark for best results. The equalization
finds the best global threshold to minimize the
number of noisy pixels, while maintaining a low
energy threshold"""
import dpx_control_hw as dch

# Output file for configuration
CONFIG = 'config.conf'
def main():
    port = dch.find_port()
    if port is None:
        port = '/dev/ttyACM0'

    dpx = dch.Dosepix(
        port_name=port,
        config_fn=None,
        thl_calib_fn=None,
        params_fn=None,
        bin_edges_fn=None
    )

    thl_calib_d = dpx.dpm.measure_thl(
        out_fn=None, plot=False)
    dpx.set_thl_calib(thl_calib_d)

    dpx.equalization(
        CONFIG,
        thl_step=1,
        noise_limit=10,
        n_evals=3,
        num_dacs=2,
        i_pixeldac=60,
        thl_offset=30,
        plot=True
    )

if __name__ == '__main__':
    main()

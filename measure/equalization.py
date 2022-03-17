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
    dpx.equalization(config_fn=config_fn)

if __name__ == '__main__':
    main()
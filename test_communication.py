#!/usr/bin/env python
import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None # 'thl_calib.json'
    config_fn = 'dpx_config_small.conf'
    dpx = dch.Dosepix(port_name=PORT,
        config_fn=None,
        thl_calib_fn=thl_calib_fn)
    *_, last = dpx.equalization(config_fn=config_fn)
    # *_, last = dpx.dpf.measure_tot(out_dir='tot_measurement/', int_plot=False)

if __name__ == '__main__':
    main()

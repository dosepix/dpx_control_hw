#!/usr/bin/env python
import time
import numpy as np

import dpx_control_hw as dch

PORT = '/dev/ttyACM0'
def main():
    thl_calib_fn = None
    config_fn = None
    dpx = dch.Dosepix(port_name=PORT,
        config_fn=None,
        thl_calib_fn=thl_calib_fn)

    
    while True:
        omr = ''.join([np.random.choice(list('0123456789abcdef')) for n in range(6)])
        dpx.dpf.write_omr(omr)

        print('OMR set:', omr)
        print('OMR get:', dpx.dpf.read_omr())
        print()

        time.sleep(0.1)

    # *_, last = dpx.equalization(config_fn=config_fn)
    # *_, last = dpx.dpf.measure_tot(out_dir='tot_measurement/', int_plot=False)

if __name__ == '__main__':
    main()

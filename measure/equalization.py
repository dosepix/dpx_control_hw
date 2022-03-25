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

    '''
    periphery_d = dpx.support.split_perihpery_dacs(dpx.periphery_dacs)
    periphery_d['i_pixeldac'] = 80
    dpx.periphery_dacs = dpx.support.perihery_dacs_dict_to_code(periphery_d)
    '''

    thl_calib_d = dpx.dpf.measure_thl(out_fn=None, plot=False)
    dpx.set_thl_calib(thl_calib_d)
    dpx.equalization(config_fn=config_fn)

if __name__ == '__main__':
    main()

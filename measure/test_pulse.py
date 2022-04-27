#!/usr/bin/env python
""" Perform test pulse measurements"""
import time
from tqdm import tqdm
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

    scan_dac_range(dpx)

    print(dpx.pixel_dacs)
    print(dpx.dpf.read_pixel_dacs())

    # Start test pulses
    dpx.dtp.start(columns=[3])
    print(dpx.dpf.read_conf_bits())

    dpx.dpf.set_dosi_mode()
    dpx.dtp.set_test_pulse_voltage(300)

    dpx.dpf.data_reset()
    tot_frames = []

    start_time = time.time()
    for _ in range(1000):
        dpx.dpf.data_reset()
        dpx.dpf.generate_test_pulse()

        tot = dpx.dpf.read_tot()[0:16]
        tot_frames.append( tot )
    tot_frames = np.asarray( tot_frames ).T

    for pixel in range(16):
        print(np.mean(tot_frames[pixel]), np.std(tot_frames[pixel]))
        plt.hist(tot_frames[pixel], bins=30)
        plt.show()
    print(tot)


def scan_dac_range(dpx, column=0):
    dpx.dtp.start(columns=[column])
    dpx.dpf.set_dosi_mode()

    dac_range = np.arange(0, 512)
    tot_frames = []
    for voltage in tqdm(dac_range):
        dpx.dtp.set_test_pulse_voltage(voltage)
        dpx.dpf.data_reset()
        dpx.dpf.generate_test_pulse()
        tot = dpx.dpf.read_tot()[column * 16:(column + 1) * 16]
        tot_frames.append( tot )
    tot_frames = np.asarray( tot_frames ).T

    for pixel in range(16):
        plt.plot(dac_range, tot_frames[pixel], label=pixel)

    plt.xlabel('DAC')
    plt.ylabel('ToT (10 ns)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

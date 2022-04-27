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

    # while True:
    # scan_dac_range(dpx, repeats=10, column=7)
    show_tot_dist(dpx, dac=100, duration=10, column=0)

def show_tot_dist(dpx,
        dac=100,
        duration=10,
        column=0
    ):
    dpx.dtp.start(columns=[column])
    dpx.dpf.set_dosi_mode()
    dpx.dtp.set_test_pulse_voltage(dac)

    dpx.dpf.data_reset()
    tot_frames = []
    start_time = time.time()
    while (time.time() - start_time) < abs(duration):
        dpx.dpf.data_reset()
        dpx.dpf.generate_test_pulse()

        tot = dpx.dpf.read_tot()[column*16:(column+1)*16]
        tot_frames.append( tot )
    tot_frames = np.asarray( tot_frames ).T

    _, ax = plt.subplots(16, 1,
        figsize=(5, 50),
        sharex=True)
    for pixel in range(16):
        median, std = np.median(tot_frames[pixel]), np.std(tot_frames[pixel])
        print(pixel, median, std)
        hist, bins = np.histogram(tot_frames[pixel], bins=30)
        ax[pixel].step(bins[:-1], hist, label=pixel)
        ax[pixel].axvline(x=median, ls='--', color='k')
    plt.xlabel('ToT (10 ns)')
    plt.show()

def scan_dac_range(dpx, repeats=1, column=0):
    dpx.dtp.start(columns=[column])
    dpx.dpf.set_dosi_mode()

    dac_range = np.arange(0, 512)
    tot_frames = []
    for voltage in tqdm(dac_range):
        tot_repeats = []
        for _ in range(repeats):
            dpx.dtp.set_test_pulse_voltage(voltage)
            dpx.dpf.data_reset()
            dpx.dpf.generate_test_pulse()
            tot = dpx.dpf.read_tot()[column * 16:(column + 1) * 16]
            tot_repeats.append( tot )
        tot_repeats = np.asarray( tot_repeats )
        tot_frames.append( np.median(tot_repeats, axis=0) )
    tot_frames = np.asarray( tot_frames ).T

    for pixel in range(16):
        plt.plot(dac_range, tot_frames[pixel], label=pixel)

    plt.legend()
    plt.xlabel('DAC')
    plt.ylabel('ToT (10 ns)')
    plt.show()

if __name__ == '__main__':
    main()

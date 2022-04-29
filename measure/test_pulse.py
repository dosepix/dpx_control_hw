#!/usr/bin/env python
""" Perform test pulse measurements"""
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch

CONFIG = 'config.conf'
OUT_DIR = None # 'tot_measurement/'
BIN_EDGES = np.asarray(np.linspace(5, 800, 16), dtype=int).tolist()

def main():
    port = dch.find_port()
    if port is None:
        port = '/dev/ttyACM0'

    thl_calib_fn = None
    params_fn = None
    dpx = dch.Dosepix(
        port_name=port,
        config_fn=CONFIG,
        thl_calib_fn=thl_calib_fn,
        params_fn=params_fn,
        bin_edges_fn=BIN_EDGES
    )

    tpt = TestPulseTest(dpx)
    column = 0

    print('=== Scanning DAC range ===')
    tpt.scan_dac_range(repeats=3, column=column)
    print('=== Testing ToT distribution ===')
    tpt.show_tot_dist(dac=100, duration=10, column=column)
    print('=== Testing Dosi distribution ===')
    tpt.show_dosi_dist(BIN_EDGES)

class TestPulseTest():
    """Class containing test functions involving test pulses"""
    def __init__(self, dpx):
        self.dpx = dpx

    def show_dosi_dist(self, bin_edges):
        """Determine the event distribution of the different
        bins in dosi-mode for the different columns. Tests,
        if ToT-values are correctly sorted into the specified
        bins `BIN_EDGES`"""
        fig, ax = plt.subplots(4, 4,
            figsize=(10, 10),
            sharex=True,
            sharey=True)
        ax = ax.flatten()

        for column in range(16):
            self.dpx.dpf.set_dosi_mode()
            self.dpx.dpf.clear_bins()
            self.dpx.dtp.start(columns=[column])

            dac = int((16 - column) / 16 * 460)
            print('DAC = %d' % dac)
            self.dpx.dtp.set_test_pulse_voltage(dac)
            for _ in range(100 * (column + 1)):
                self.dpx.dpf.generate_test_pulse()

            self.dpx.dpf.write_column_select(15 - column)
            col_read = self.dpx.dpf.read_column_select()
            dosi = np.asarray( self.dpx.dpf.read_dosi() ).reshape((16, 16))
            dosi = np.flip(dosi, axis=1)
            print('ToT')
            tot = self.dpx.dpf.read_tot()[column*16:(column+1)*16]
            print(tot)
            ax[column].set_title(col_read)
            ax[column].imshow(
                dosi,
                extent=[bin_edges[0], bin_edges[-1], 0, 15],
                origin='lower',
                aspect='auto'
            )
            ax[column].axvline(x=np.median(tot), color='white', ls='--')
            print()

        fig.supylabel('Pixel idx')
        fig.supxlabel('ToT (10 ns)')

        plt.tight_layout()
        plt.show()
        self.dpx.dtp.stop()

    def show_tot_dist(self,
            dac=100,
            duration=10,
            column=0
        ):
        """Generates test pulses for the specified `duration`
        and pixel-`column` with a test pulse energy according
        to `dac`. Afterwards, the ToT-distribution for each
        pixel in the column is shown"""
        self.dpx.dtp.start(columns=[column])
        self.dpx.dpf.set_dosi_mode()
        self.dpx.dtp.set_test_pulse_voltage(dac)

        self.dpx.dpf.data_reset()
        tot_frames = []
        start_time = time.time()
        while (time.time() - start_time) < abs(duration):
            self.dpx.dpf.data_reset()
            self.dpx.dpf.generate_test_pulse()

            tot = self.dpx.dpf.read_tot()[column*16:(column+1)*16]
            tot_frames.append( tot )
        tot_frames = np.asarray( tot_frames ).T

        _, ax = plt.subplots(16, 1,
            figsize=(5, 20),
            sharex=True)
        for pixel in range(16):
            median, std = np.median(tot_frames[pixel]), np.std(tot_frames[pixel])
            print(pixel, median, std)
            hist, bins = np.histogram(tot_frames[pixel], bins=30)
            ax[pixel].step(bins[:-1], hist, label=pixel)
            ax[pixel].axvline(x=median, ls='--', color='k')
        plt.xlabel('ToT (10 ns)')
        plt.show()

        self.dpx.dtp.stop()

    def scan_dac_range(self, repeats=1, column=0):
        """Test pulse energies are evaluated for the whole
        dac-range, i.e. from 0 to 511. Finally, the trends
        of the resulting ToT vs. dac curves are shown for
        each pixel in the specified `column`"""
        self.dpx.dtp.start(columns=[column])
        self.dpx.dpf.set_dosi_mode()

        dac_range = np.arange(0, 512)
        tot_frames = []
        for voltage in tqdm(dac_range):
            tot_repeats = []
            for _ in range(repeats):
                self.dpx.dtp.set_test_pulse_voltage(voltage)
                self.dpx.dpf.data_reset()
                self.dpx.dpf.generate_test_pulse()
                tot = self.dpx.dpf.read_tot()[column * 16:(column + 1) * 16]
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

        self.dpx.dtp.stop()

if __name__ == '__main__':
    main()

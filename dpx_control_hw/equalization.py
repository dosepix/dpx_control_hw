from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import dpx_functions
from . import support

class Equalization(dpx_functions.DPXFunctions):
    def get_thl_range(self, thl_low=4000, thl_high=6000, thl_step=1):
        if (self.dpx.thl_edges is None) or (len(self.dpx.thl_edges) == 0):
            thl_range = np.arange(thl_low, thl_high, thl_step)
        else:
            thl_range = np.asarray(self.dpx.thl_edges)
            thl_range = np.around(
                    thl_range[np.logical_and(thl_range >= thl_low, thl_range <= thl_high)]
                )
        return thl_range

    def threshold_equalization(self,
            thl_step=1,
            noise_limit=0,
            n_evals=3,
            use_gui=False
        ):

        thl_range = self.get_thl_range(thl_step=thl_step)
        print('== Threshold equalization ==')
        if use_gui:
            yield {'stage': 'Init'}

        # Set PC Mode in OMR in order to read kVp values
        self.set_pc_mode()
        print('OMR set to:', self.dpx.dpf.read_omr())

        counts_dict_gen = self.get_thl_level(
            thl_range, ['3f'], n_evals=n_evals, use_gui=use_gui)

        if use_gui:
            yield {'stage': 'THL_pre_start'}
            for res in counts_dict_gen:
                if 'status' in res.keys():
                    yield {
                        'stage': 'THL_pre',
                        'status': np.round(res['status'], 4)
                        }
                elif 'DAC' in res.keys():
                    yield {
                        'stage': 'THL_pre_loop_start',
                        'status': res['DAC']
                        }
                else:
                    counts_dict = res['countsDict']
        else:
            counts_dict = deque(counts_dict_gen, maxlen=1).pop()

        gauss_dict, _ = support.get_noise_level(
            counts_dict, thl_range, ['3f'], noise_limit)
        print(np.asarray(gauss_dict['3f']).shape)
        plt.hist(gauss_dict['3f'], bins=100)
        plt.show()

        # thl_new = int(np.median(gauss_dict['3f']) - sigma * np.std(gauss_dict['3f']))
        thl_new = int( np.min(gauss_dict['3f']) )
        self.write_periphery(self.dpx.periphery_dacs[:-4] + '%04x' % thl_new)
        pixel_dacs = np.asarray([63] * 256, dtype=int)

        noisy_pixels = 256
        loop_cnt = 0
        while noisy_pixels > 0 and loop_cnt < 63:
            pc_data = np.zeros(256)
            for _ in range(3):
                self.data_reset()
                time.sleep(0.01)
                pc_data += np.asarray(self.read_pc())

            # Noisy pixels
            pc_noisy = np.argwhere(pc_data > 0).flatten()
            noisy_pixels = len(pc_noisy)

            pixel_dacs[pc_noisy] -= 1
            self.write_pixel_dacs(''.join(['%02x' % pixel_dac for pixel_dac in pixel_dacs]))
            loop_cnt += 1

        pixel_dacs[pixel_dacs < 63] -= 10
        pixel_dacs[pixel_dacs < 0] = 0

        conf_mask = np.zeros(256).astype(str)
        conf_mask.fill('00')
        conf_mask[pixel_dacs <= 0] = '%02x' % (0b1 << 2)
        conf_mask = ''.join(conf_mask)

        # Convert to code string
        pixel_dacs = ''.join(['%02x' % pixel_dac for pixel_dac in pixel_dacs])
        print('THL:', '%04x' % int(thl_new))
        print('Pixel DACs:', pixel_dacs)
        print('Conf mask:', conf_mask)

        if use_gui:
            yield {'stage': 'finished',
                    'pixelDAC': pixel_dacs,
                    'THL': '%04x' % int(thl_new),
                    'confMask': conf_mask}
        else:
            yield pixel_dacs, '%04x' % int(thl_new), conf_mask

    def get_thl_level(
            self,
            thl_range,
            pixel_dacs=['00', '3f'],
            n_evals=10,
            use_gui=False
        ):
        counts_dict = {}
        # Loop over pixel_dac values
        for pixel_dac in pixel_dacs:
            counts_dict[pixel_dac] = {}
            print('Set pixel DACs to %s' % pixel_dac)

            # Set pixel DAC values to every pixel
            pixel_code = pixel_dac * 256
            self.write_pixel_dacs(pixel_code)

            # Dummy readout
            self.read_pc()

            # Noise measurement
            # Loop over THL values
            print('Loop over THLs')

            # Fast loop
            counts_list = []
            thl_range_fast = thl_range[::10]
            for cnt, thl in enumerate(thl_range_fast):
                self.write_periphery(
                    self.dpx.periphery_dacs[:-4] + ('%04x' % int(thl))
                )
                self.data_reset()
                # time.sleep(0.01)

                # Read ToT values into matrix
                pc_meas = self.read_pc()
                counts_list.append( pc_meas )
            counts_list = np.asarray(counts_list).T
            thl_range_fast = [thl_range_fast[item[0]] if np.any(item) else np.nan for item in [
                    np.argwhere(counts > 3) for counts in counts_list]]

            # Precise loop
            if use_gui:
                yield {'DAC': pixel_dac}

            thl_range_slow = np.around(thl_range[np.logical_and(thl_range >= (
                np.nanmin(thl_range_fast) - 10), thl_range <= np.nanmax(thl_range_fast))])

            # Do not use tqdm with GUI
            if use_gui:
                loop_range = thl_range_slow
            else:
                loop_range = tqdm(thl_range_slow)
            for cnt, thl in enumerate(loop_range):
                # Repeat multiple times since data is noisy
                self.write_periphery(self.dpx.periphery_dacs[:-4] + ('%04x' % int(thl)))

                counts = np.zeros(256)
                for _ in range( n_evals ):
                    self.data_reset()
                    counts += self.read_pc()
                counts /= n_evals
                # print(counts)
                counts_dict[pixel_dac][int(thl)] = counts

                # Return status as generator when using GUI
                if use_gui:
                    yield {'status': float(cnt) / len(loop_range)}
            print()
        if use_gui:
            yield {'countsDict': counts_dict}
        else:
            yield counts_dict

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
            noise_limit=10,
            n_evals=3,
            use_gui=False,
            plot=True
        ):
        noise_limit = 10
        n_evals = 1

        num_dacs = 8
        pixel_dac_settings = ['%02x' % pixel_dac for pixel_dac in np.arange(0, 63 + 1, 64 // num_dacs)] + ['3f']
        thl_range = self.get_thl_range(thl_step=thl_step)
        print('== Threshold equalization ==')
        if use_gui:
            yield {'stage': 'Init'}

        # Set PC Mode in OMR in order to read kVp values
        self.set_pc_mode()
        print('OMR set to:', self.dpx.dpf.read_omr())

        counts_dict_gen = self.get_thl_level(
            thl_range, pixel_dac_settings, n_evals=n_evals, use_gui=use_gui)

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

        gauss_dict, noise_thl = get_noise_level(
            counts_dict, thl_range, pixel_dac_settings, noise_limit)

        # Transform values to indices and get mean_dict
        mean_dict, noise_thl = val_to_idx(
            pixel_dac_settings, gauss_dict, noise_thl,
            self.dpx.thl_edges_low,
            self.dpx.thl_edges_high,
            self.dpx.thl_fit_params)

        # New THL
        thl_mean = int( (mean_dict['00'] + mean_dict['3f']) // 2 )

        # Calculate slope, offset and mean
        pixel_dac_nums = np.asarray([int(pixel_dac, 16) for pixel_dac in pixel_dac_settings])
        noise_thls = np.asarray( [noise_thl[pixel_dac] for pixel_dac in pixel_dac_settings] ).T
        close_idx = np.argsort(np.abs(noise_thls - thl_mean), axis=1)[:,:2]
        close_idx = np.sort(close_idx, axis=1)

        noise_thl_first = [noise_thls[pixel][[close_idx[pixel, 0]]][0] for pixel in np.arange(256)]
        noise_thl_second = [noise_thls[pixel][[close_idx[pixel, 1]]][0] for pixel in np.arange(256)]
        noise_thl_first, noise_thl_second = np.asarray(noise_thl_first), np.asarray(noise_thl_second)
        slope = np.abs(noise_thl_first - noise_thl_second) / (64 / (num_dacs - 1))
        offset = np.nanmax([noise_thl_first, noise_thl_second], axis=0)

        # Get adjustment value for each pixel
        adjust = np.asarray((offset - thl_mean) / slope, dtype=int)
        adjust += pixel_dac_nums[close_idx[:,0]]

        # Consider extreme values
        adjust[np.isnan(adjust)] = 0
        adjust[adjust > 63] = 63
        adjust[adjust < 0] = 0

        if not use_gui and plot:
            plot_x = [int(pixel_dac, 16) for pixel_dac in pixel_dac_settings]
            for idx in range(16):
                color = 'C%d' % idx
                plot_y = [noise_thl[pixel_dac][idx] for pixel_dac in pixel_dac_settings]
                plt.plot(
                    plot_x,
                    plot_y,
                    marker='x',
                    color=color
                )
                plt.axvline(x=adjust[idx], ls='--', color=color)
            plt.axhline(y=thl_mean, ls='--', color='k')
            plt.show()

        # Set new pixel dacs and find THL again
        pixel_dac_new = ''.join(['%02x' % entry for entry in adjust])
        print('New pixel dac', pixel_dac_new)

        counts_dict_new_gen = self.get_thl_level(
            thl_range, [pixel_dac_new], n_evals=n_evals, use_gui=use_gui)
        if use_gui:
            yield {'stage': 'THL_start'}
            for res in counts_dict_new_gen:
                if 'status' in res.keys():
                    yield {
                        'stage': 'THL',
                        'status': np.round(res['status'], 4)
                        }
                elif 'DAC' in res.keys():
                    yield {
                        'stage': 'THL_loop_start',
                        'status': res['DAC']
                        }
                else:
                    counts_dict_new = res['countsDict']
        else:
            counts_dict_new = deque(counts_dict_new_gen, maxlen=1).pop()
        gauss_dict_new, noise_thl_new = get_noise_level(
            counts_dict_new, thl_range, [pixel_dac_new], noise_limit)
        _, noise_thl_new = val_to_idx(
            [pixel_dac_new], gauss_dict_new, noise_thl_new,
            self.dpx.thl_edges_low,
            self.dpx.thl_edges_high,
            self.dpx.thl_fit_params)

        # Plot THL distributions for different pixel dac settings
        if not use_gui and plot:
            bins = 30
            for pixel_dac in pixel_dac_settings:
                plt.hist(noise_thl[pixel_dac].flatten(), bins=bins, alpha=.5)
            plt.hist(noise_thl_new[pixel_dac_new].flatten(),
                bins=bins, alpha=.5, color='k')
            plt.axvline(x=thl_mean, ls='--', color='k')
            plt.xlabel('THL')
            plt.show()

        # sigma = 2
        # thl_new = int(np.median(gauss_dict[pixel_dac_setting]) - sigma * np.std(gauss_dict[pixel_dac_setting]))
        # thl_new = int( np.median(gauss_dict[pixel_dac_setting]) )
        # thl_new = np.min(gauss_dict[pixel_dac_setting])
        thl_new = int(np.median(gauss_dict_new[pixel_dac_new]))
        self.dpx.dpf.write_pixel_dacs(pixel_dac_new)
        pixel_dacs = adjust.flatten()

        noisy_pixels = 256
        loop_cnt = 0
        while noisy_pixels > 0 and loop_cnt < 63:
            self.write_periphery(self.dpx.periphery_dacs[:-4] + '%04x' % thl_new)
            pc_data = np.zeros(256)
            for _ in range(3):
                self.data_reset()
                time.sleep(0.01)
                pc_data += np.asarray(self.read_pc())

            noisy_pixels = len(pc_data[pc_data > 0])
            thl_new -= 1
        thl_new -= 10

        conf_mask = np.zeros(256).astype(str)
        conf_mask.fill('00')
        conf_mask[noisy_pixels > 0] = '%02x' % (0b1 << 2)
        conf_mask = ''.join(conf_mask)

        # Convert to code string
        pixel_dacs = ''.join(['%02x' % pixel_dac for pixel_dac in pixel_dacs])

        print('THL:', thl_new) # '%04x' % int(thl_new))
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
            if len(pixel_dac) > 2:
                pixel_code = pixel_dac
            else:
                pixel_code = pixel_dac * 256
            self.write_pixel_dacs(pixel_code)

            # Noise measurement
            # Loop over THL values
            print('Loop over THLs')

            # Fast loop
            counts_list = []
            thl_range_fast = thl_range[::10]
            for cnt, thl in enumerate(thl_range_fast):
                self.write_periphery(
                    self.dpx.periphery_dacs[:-4] + '%04x' % int(thl)
                )
                self.data_reset()
                # time.sleep(0.01)

                # Read PC values into matrix
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
                self.write_periphery(
                    self.dpx.periphery_dacs[:-4] + '%04x' % int(thl)
                )

                counts = np.zeros(256)
                for _ in range( n_evals ):
                    self.data_reset()
                    # time.sleep(0.01)
                    counts += self.read_pc()
                counts_dict[pixel_dac][int(thl)] = counts

                # Return status as generator when using GUI
                if use_gui:
                    yield {'status': float(cnt) / len(loop_range)}

            # for pixel in range(8):
            #     plt.plot(thl_range_slow, [counts_dict[pixel_dac][int(thl)][pixel] for thl in thl_range_slow])
            # plt.show()
        if use_gui:
            yield {'countsDict': counts_dict}
        else:
            yield counts_dict

# === Support functions ===
def val_to_idx(
        pixel_dacs,
        gauss_dict,
        noise_thl,
        thl_edges_low=None,
        thl_edges_high=None,
        thl_fit_params=None
    ):
    # Transform values to indices
    mean_dict = {}
    for pixel_dac in pixel_dacs:
        idxs = np.asarray(
            [get_volt_from_thl_fit(thl_edges_low, thl_edges_high, thl_fit_params, elm) if elm \
            else np.nan for elm in gauss_dict[pixel_dac] ], dtype=np.float)
        mean_dict[pixel_dac] = np.nanmean(idxs)

        for pixel in range(256):
            elm = noise_thl[pixel_dac][pixel]
            if elm:
                noise_thl[pixel_dac][pixel] = get_volt_from_thl_fit(
                    thl_edges_low, thl_edges_high, thl_fit_params, elm)
            else:
                noise_thl[pixel_dac][pixel] = np.nan

    return mean_dict, noise_thl

def get_volt_from_thl_fit(
        thl_edges_low,
        thl_edges_high,
        thl_fit_params,
        thl
    ):
    if (thl_edges_low is None) or (len(thl_edges_low) == 0):
        return thl

    edges = zip(thl_edges_low, thl_edges_high)
    for idx, edge in enumerate(edges):
        if edge[1] > thl >= edge[0]:
            params = thl_fit_params[idx]
            if idx == 0:
                return support.erf_std_fit(thl, *params)
            return support.linear_fit(thl, *params)

def get_noise_level(
        counts_dict,
        thl_range,
        pixel_dacs=['00', '3f'],
        noise_limt=100
    ):
    # Get noise THL for each pixel
    noise_thl = {key: np.zeros(256) for key in pixel_dacs}
    gauss_dict = {key: [] for key in pixel_dacs}

    # Loop over each pixel in countsDict
    for pixel_dac in pixel_dacs:
        for pixel in range(256):
            for thl in thl_range:
                if thl not in counts_dict[pixel_dac].keys():
                    continue

                if counts_dict[pixel_dac][thl][pixel] > noise_limt:
                    noise_thl[pixel_dac][pixel] = thl
                    gauss_dict[pixel_dac].append(thl)
                    break

    return gauss_dict, noise_thl

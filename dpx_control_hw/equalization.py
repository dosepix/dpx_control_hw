from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import dpx_functions
from . import support

class Equalization(dpx_functions.DPXFunctions):
    def get_thl_range(self,
            thl_low=5200,
            thl_high=6000,
            thl_step=1
        ):
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
            thl_offset=0,
            use_gui=False,
            plot=True
        ):
        # Set pixel-dacs to central value
        pixel_dac_setting = ['1f']

        # Get range to evaluate thl in
        thl_range = self.get_thl_range(thl_step=thl_step)

        print('== Threshold equalization ==')
        if use_gui:
            yield {'stage': 'Init'}

        # Set PC Mode in OMR in order to read kVp values
        self.set_pc_mode()
        print('OMR set to:', self.dpx.dpf.read_omr())

        counts_dict_gen = self.get_thl_level(
            thl_range, pixel_dac_setting, n_evals=n_evals, use_gui=use_gui)

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

        gauss_dict, _ = get_noise_level(
            counts_dict, thl_range, pixel_dac_setting, noise_limit)

        # New THL
        thl_mean = int( np.median(gauss_dict[pixel_dac_setting[0]]) - thl_offset)
        self.write_periphery(self.dpx.periphery_dacs[:-4] + '%04x' % thl_mean)

        # Plot THL distributions for different pixel dac settings
        if not use_gui and plot:
            bins = 30
            plt.hist(gauss_dict[pixel_dac_setting[0]], bins=bins, alpha=.5)
            plt.axvline(x=thl_mean, ls='--', color='k')
            plt.xlabel('THL')
            plt.show()

        pixel_dacs = np.asarray([int(pixel_dac_setting[0], 16)] * 256)
        noisy_pixels = np.asarray([True] * 256)
        n_noisy_pixels = 256

        loop_cnt = 0

        pixel_dacs_states = []
        while n_noisy_pixels > 0 and loop_cnt < 100:
            # np.random.shuffle( pixel_dacs )
            pixel_dac_str = ''.join(['%02x' % pixel_dac for pixel_dac in pixel_dacs])
            self.dpx.dpf.write_pixel_dacs(pixel_dac_str)
            self.data_reset()
            pc_data = np.asarray(self.read_pc())

            # Number of noisy pixels
            n_noisy_pixels = len(noisy_pixels[noisy_pixels])
            pc_noisy = pc_data > 0

            # Increase pixel-dac of noisy pixels
            pixel_dacs[pc_noisy] -= 1

            # Decrease pixel-dac of non-noisy pixels
            pixel_dacs[~pc_noisy] += 1

            # Ensure min and max values
            pixel_dacs[pixel_dacs < 0] = 0
            pixel_dacs[pixel_dacs > 63] = 63

            pixel_dacs_states.append( pixel_dacs.tolist() )

            loop_cnt += 1

        # Calculate median of 20 last pixel-dac values per pixel
        pixel_dacs_states = np.asarray(pixel_dacs_states).T
        pixel_dacs_medians = np.median(pixel_dacs_states[:,-20:], axis=1)

        # Reduce thl to ensure robustness
        thl_mean -= 20

        if not use_gui and plot:
            print('-- Evolution of pixel-dacs for first 16 pixels --')
            for pixel in range(16):
                color = 'C%d' % pixel
                plt.step(
                    np.arange(pixel_dacs_states.shape[1]),
                    pixel_dacs_states[pixel], color=color
                )
                plt.axhline(
                    y=np.median(pixel_dacs_medians[pixel]),
                    ls='--', color=color
                )
            plt.xlabel('Iterations')
            plt.ylabel('pixel-dac')
            plt.show()

        # Create conf mask to switch noisy pixels off
        noisy_pixels = (pixel_dacs_medians >= 63) | (pixel_dacs_medians <= 0)
        conf_mask = np.zeros(256).astype(str)
        conf_mask.fill('00')
        conf_mask[noisy_pixels] = '%02x' % (0b1 << 2)
        conf_mask = ''.join(conf_mask)

        # Convert to code string
        pixel_dacs = ''.join(['%02x' % pixel_dac for pixel_dac in pixel_dacs])

        print('Bad pixels:', len(noisy_pixels[noisy_pixels]))
        print('THL:', thl_mean) # '%04x' % int(thl_new))
        print('Pixel DACs:', pixel_dacs)
        print('Conf mask:', conf_mask)

        if use_gui:
            yield {'stage': 'finished',
                    'pixelDAC': pixel_dacs,
                    'THL': '%04x' % int(thl_mean),
                    'confMask': conf_mask}
        else:
            yield pixel_dacs, '%04x' % int(thl_mean), conf_mask

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

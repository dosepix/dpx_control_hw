# pylint: disable=missing-function-docstring
from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import communicate
from . import support
from . import dpx_support

class DPXFunctions():
    def __init__(self, dpx, comm: communicate.Communicate):
        self.dpx = dpx
        self.comm = comm

    # === HARDWARE ===
    def enable_vdd(self):
        self.comm.send_cmd('EN_VDD')

    def disable_vdd(self):
        self.comm.send_cmd('DISAB_VDD')

    def enable_bias(self):
        self.comm.send_cmd('EN_BIAS')

    def disable_bias(self):
        self.comm.send_cmd('DISAB_BIAS')

    def led_on(self):
        self.comm.send_cmd('LED_ON')

    def led_off(self):
        self.comm.send_cmd('LED_OFF')

    def read_adc(self):
        self.comm.send_cmd('READ_ADC', write=False)
        res = self.comm.get_data(size=2)
        return ''.join( ['%02x' % int(r) for r in res[::-1]] )

    # === RESET ===
    def global_reset(self):
        self.comm.send_cmd('GLOBAL_RESET')

    def data_reset(self):
        self.comm.send_cmd('DATA_RESET')

    # === OMR ===
    def read_omr(self):
        self.comm.send_cmd('READ_OMR', write=False)
        res = self.comm.get_data(size=3)
        return ''.join( ['%02x' % int(r) for r in res] )

    def write_omr(self, data):
        self.comm.send_cmd('WRITE_OMR')
        self.comm.send_data_binary(data)

    def set_pc_mode(self):
        omr_code = '%04x' % (
            (int(self.dpx.omr, 16) & ~((0b11) << 22)) | (0b10 << 22))
        self.write_omr(omr_code)
        return omr_code

    def set_dosi_mode(self):
        omr_code = '%06x' % (int(self.dpx.omr, 16) & ~((0b11) << 22))
        self.write_omr(omr_code)
        return omr_code

    # === PERIPHERY ====
    def read_periphery(self):
        self.comm.send_cmd('READ_PERIPHERY', write=False)
        res = self.comm.get_data(size=16)
        return ''.join( ['%02x' % r for r in res] )

    def write_periphery(self, data):
        self.comm.send_cmd('WRITE_PERIPHERY')
        self.comm.send_data_binary(data)

    # === PIXEL DAC ===
    def read_pixel_dacs(self):
        self.comm.send_cmd('READ_PIXEL_DAC', write=False)
        res = self.comm.get_data(size=256)
        return ''.join( ['%02x' % r for r in res] )

    def write_pixel_dacs(self, data):
        self.comm.send_cmd('WRITE_PIXEL_DAC')

        # Split in chunks
        for split in range(4):
            data_split = data[split*128:(split+1)*128]
            self.comm.send_data_binary( data_split )

    # === CONF BITS ===
    def read_conf_bits(self):
        self.comm.send_cmd('READ_CONFBITS', write=False)
        res = self.comm.get_data(size=256)
        return ''.join( ['%02x' % r for r in res] )

    def write_conf_bits(self, data):
        self.comm.send_cmd('WRITE_CONFBITS')
        self.comm.send_data_binary(data)

    # === COLUMN SELECT ===
    def read_column_select(self):
        self.comm.send_cmd('READ_COLSEL', write=False)
        res = self.comm.get_data(size=1)
        return res[0]

    def write_column_select(self, data):
        self.comm.send_cmd('WRITE_COLSEL')
        self.comm.send_data_binary(data)

    # === DATA ===
    def read_pc(self):
        self.comm.send_cmd('READ_PC', write=False)
        res = self.comm.get_data(size=256)
        return list(res)

    def read_tot(self):
        self.comm.send_cmd('READ_TOT', write=False)
        res = self.comm.get_data(size=512)
        return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]

    def read_dosi(self):
        self.comm.send_cmd('READ_BIN', write=False)
        res = self.comm.get_data(size=512)
        return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]

    # === FUNCTIONS ===
    def measure_tot(self):
        # Activate dosi mode
        self.dpx.omr = self.set_dosi_mode()

        # Data reset
        self.data_reset()

        print('Starting ToT Measurement!')
        print('=========================')
        try:
            start_time = time.time()
            while True:
                data = self.read_tot()

        except (KeyboardInterrupt, SystemExit):
            return

    def threshold_equalization(self,
            thl_step=1,
            noise_limit=3,
            thl_offset=0,
            use_gui=False
        ):
        # Get THL range
        thl_low, thl_high = 5100, 5700
        if (self.dpx.thl_edges is None) or (len(self.dpx.thl_edges) == 0):
            thl_range = np.arange(thl_low, thl_high, thl_step)
        else:
            thl_range = np.asarray(self.dpx.thl_edges)
            thl_range = np.around(
                    thl_range[np.logical_and(thl_range >= thl_low, thl_range <= thl_high)]
                )

        print('== Threshold equalization ==')
        if use_gui:
            yield {'stage': 'Init'}

        # Set PC Mode in OMR in order to read kVp values
        self.dpx.omr = self.set_pc_mode()
        print('OMR set to:', self.dpx.omr)

        # Linear dependence: start and end points are sufficient
        pixel_dacs = ['00', '3f']

        # Return status to GUI
        if use_gui:
            yield {'stage': 'THL_pre_start'}
            counts_dict_gen = self.get_thl_level(
                thl_range, pixel_dacs, use_gui=True)
            for res in counts_dict_gen:
                if 'status' in res.keys():
                    yield {'stage': 'THL_pre', 'status': np.round(res['status'], 4)}
                elif 'DAC' in res.keys():
                    yield {'stage': 'THL_pre_loop_start', 'status': res['DAC']}
                else:
                    counts_dict = res['countsDict']
        else:
            counts_dict_gen = self.get_thl_level(
                thl_range, pixel_dacs, use_gui=False)
            counts_dict = deque(counts_dict_gen, maxlen=1).pop()
        gauss_dict, noise_thl = support.get_noise_level(
            counts_dict, thl_range, pixel_dacs, noise_limit)

        # Transform values to indices and get mean_dict
        mean_dict, noise_thl = support.val_to_idx(
            pixel_dacs, gauss_dict, noise_thl,
            self.dpx.thl_edges_low,
            self.dpx.thl_edges_high,
            self.dpx.thl_fit_params)

        # Calculate slope, offset and mean
        slope = (noise_thl['00'] - noise_thl['3f']) / 64.
        offset = noise_thl['00']
        mean = 0.5 * (mean_dict['00'] + mean_dict['3f'])

        # Get adjustment value for each pixel
        adjust = np.asarray((offset - mean) / slope + 0.5)

        # Consider extreme values
        adjust[np.isnan(adjust)] = 0
        adjust[adjust > 63] = 63
        adjust[adjust < 0] = 0

        # Convert to integer
        adjust = adjust.astype(dtype=int)

        # Set new pixel_dac values, concert to hex
        pixel_dac_new = ''.join(['%02x' % entry for entry in adjust.flatten()])
        print('New pixel dac', pixel_dac_new)

        # Repeat procedure to get noise levels
        if use_gui:
            yield {'stage': 'THL_start'}
            counts_dict_gen = self.get_thl_level(
                thl_range, pixel_dac_new, use_gui=True)
            for res in counts_dict_gen:
                if 'status' in res.keys():
                    yield {'stage': 'THL', 'status': np.round(res['status'], 4)}
                elif 'DAC' in res.keys():
                    yield {'stage': 'THL_loop_start', 'status': res['DAC']}
                else:
                    counts_dict_new = res['countsDict']
        else:
            counts_dict_new_gen = self.get_thl_level(thl_range, [pixel_dac_new])
            counts_dict_new = deque(counts_dict_new_gen, maxlen=1).pop()

        gauss_dict_new, noise_thl_new = support.get_noise_level(
            counts_dict_new, thl_range, [pixel_dac_new], noise_limit)

        # Transform values to indices
        _, noise_thl_new = support.val_to_idx(
            [pixel_dac_new], gauss_dict_new, noise_thl_new,
            self.dpx.thl_edges_low,
            self.dpx.thl_edges_high,
            self.dpx.thl_fit_params)

        # Plot the results of the equalization
        if use_gui:
            yield {'stage': 'conf_bits'}

        # Create conf_bits
        conf_mask = np.zeros(256).astype(str)
        conf_mask.fill('00')

        # Check for noisy pixels after equalization. If there are still any left,
        # reduce THL even further. If a pixel is really noisy, it shouldn't change
        # its state even when THL is lowered. Therefore, if pixels don't change their
        # behavior after 5 decrements of THL, switch them off
        if use_gui:
            yield {'stage': 'noise'}

        thl_new = int(np.mean(gauss_dict_new[pixel_dac_new]))

        if self.dpx.thl_edges is not None:
            print('Getting rid of noisy pixels...')
            self.write_periphery(
                self.dpx.periphery_dacs[:-4] + ('%04x' % int(thl_new)))

            pc_noisy_last = []
            noisy_count = 0
            while True:
                pc_data = np.zeros((16, 16))
                self.data_reset()
                for _ in range(30):
                    pc_data += np.asarray(self.read_pc())
                    self.data_reset()
                pc_sum = pc_data.flatten()

                # Noisy pixels
                pc_noisy = np.argwhere(pc_sum > 0).flatten()
                print('THL: %d' % thl_new)
                print('Noisy pixels index:')
                print(pc_noisy)
                # Compare with previous read-out
                noisy_common = sorted(list(set(pc_noisy_last) & set(pc_noisy)))
                print(noisy_common)
                print(len(pc_noisy), len(noisy_common))
                print(noisy_count)
                print()

                # If noisy pixels don't change, increase counter
                # if len(list(set(noisy_common) & set(pc_noisy))) > 0:
                if len(pc_noisy) == len(noisy_common) and len(
                        pc_noisy) == len(pc_noisy_last):
                    noisy_count += 1
                pc_noisy_last = np.array(pc_noisy, copy=True)

                # If noisy pixels don't change for 5 succeeding steps,
                # interrupt
                if noisy_count == 3 or len(pc_noisy) > 0:
                    break
                # Reduce THL by 1
                thl_new = self.dpx.thl_edges[list(self.dpx.thl_edges).index(thl_new) - 1]
                self.write_periphery(
                    self.dpx.periphery_dacs[:-4] + ('%04x' % int(thl_new)))

            # Subtract additional offset to THL
            thl_new = self.dpx.thl_edges[list(self.dpx.thl_edges).index(thl_new) - thl_offset]

            # Switch off noisy pixels
            conf_mask[(pc_sum > 10).reshape((16, 16))] = '%02x' % (0b1 << 2)
        else:
            thl_new = int(np.mean(gauss_dict_new[pixel_dac_new]) - thl_offset)
            conf_mask[abs(noise_thl_new[pixel_dac_new] - mean) > 10] = '%02x' % (0b1 << 2)

        # Transform into string
        conf_mask = ''.join(conf_mask.flatten())

        print()
        print('Summary:')
        print('pixel_DACs:', pixel_dac_new)
        print('confMask:', conf_mask)
        print('THL:', '%04x' % int(thl_new))
        print('Bad pixels:', np.argwhere(
            (abs(noise_thl_new[pixel_dac_new] - mean) > 10)).flatten())

        # Restore OMR values
        self.write_omr(self.dpx.omr)

        if use_gui:
            yield {'stage': 'finished',
                    'pixel_DAC': pixel_dac_new,
                    'THL': '%04x' % int(thl_new),
                    'confMask': conf_mask}
        else:
            yield pixel_dac_new, '%04x' % int(thl_new), conf_mask

    def get_thl_level(
            self,
            thl_range,
            pixel_dacs=['00', '3f'],
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
                pc = self.read_pc()
                counts_list.append( pc )
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
                self.data_reset()
                # time.sleep(0.1)

                # Read ToT values into matrix
                counts = self.read_pc()
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

    def measure_adc(
            self,
            analog_out='v_tha',
            perc=False,
            adc_high=8191,
            adc_low=0,
            adc_step=1,
            n_meas=1,
            out_fn=None,
            plot=False,
            use_gui=False):
        # Display execution time at the end
        start_time = time.time()

        # Select analog out
        omr_code = int(self.dpx.omr, 16)
        omr_code &= ~(0b11111 << 12)
        omr_code |= getattr(dpx_support.omr_analog_out_sel, analog_out)
        self.dpx.omr = '%06x' % omr_code
        print('OMR set to:', self.dpx.omr)
        self.write_omr(self.dpx.omr)

        # Get peripherys
        d_peripherys = support.split_perihpery_dacs(
            self.dpx.periphery_dacs, perc=perc)
        print(d_peripherys)

        if analog_out == 'v_cascode_bias':
            analog_out = 'v_casc_reset'
        elif analog_out == 'v_tpref_fine':
            analog_out = 'v_tpref_fine'
        elif analog_out == 'v_per_bias':
            analog_out = 'v_casc_preamp'

        adc_list = np.arange(adc_low, adc_high, adc_step)
        adc_volt_mean = []
        adc_volt_err = []
        print('Measuring ADC!')

        if not use_gui:
            loop_range = tqdm(adc_list)
        else:
            loop_range = adc_list
        for adc in loop_range:
            # Set value in peripherys
            d_peripherys[analog_out] = adc
            code = support.perihery_dacs_dict_to_code(d_peripherys, perc=perc)
            self.dpx.periphery_dacs = code
            self.write_periphery(code)

            # Measure multiple times
            adc_val_list = []
            for _ in range(n_meas):
                adc_val = self.read_adc()
                adc_val_list.append( float(int(adc_val, 16)) )

            adc_volt_mean.append(np.mean(adc_val_list))
            adc_volt_err.append(np.std(adc_val_list) / np.sqrt(n_meas))

            if use_gui:
                yield {'Volt': adc_volt_mean[-1], 'ADC': int(adc)}

        if plot and not use_gui:
            plt.errorbar(adc_list, adc_volt_mean, yerr=adc_volt_err, marker='x')
            plt.show()

        adc_volt_mean_sort, adc_list_sort = zip(
            *sorted(zip(adc_volt_mean, adc_list)))
        out_dict = {'Volt': adc_volt_mean_sort, 'ADC': adc_list}
        if not use_gui:
            if plot:
                plt.plot(adc_volt_mean_sort, adc_list_sort)
                plt.show()

            print(
                'Execution time: %.2f min' %
                ((time.time() - start_time) / 60.))
        yield out_dict

    def measure_thl(self, out_fn=None, plot=False, use_gui=False):
        gen = self.measure_adc(
            analog_out='v_tha',
            perc=False,
            adc_high=8191,
            adc_low=0,
            adc_step=1,
            n_meas=1,
            out_fn=out_fn,
            plot=plot,
            use_gui=use_gui)

        if use_gui:
            return gen
        *_, out_dict = gen
        return out_dict

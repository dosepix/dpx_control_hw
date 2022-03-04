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
        # Split in chunks
        for split in range(len(data) // 128):
            data_split = data[split*128:(split+1)*128]
            self.comm.send_data_binary( data_split )

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
    def measure_tot(self,
        save_frames=None,
        out_dir='tot_measurement/',
        int_plot=False,
        meas_time=None,
        make_hist=False,
        use_gui=False
    ):
        # Activate dosi mode
        self.dpx.omr = self.set_dosi_mode()

        if int_plot:
            plt.ion()
            fig, ax = plt.subplots()
            bins = np.arange(400)

            # Create empty axis
            ax.set_ylabel('Counts')
            line = ax.plot(np.nan, np.nan, color='k')[-1]
            ax.set_xlim(min(bins), max(bins))
            ax.set_xlabel('ToT')
            ax.grid()

        # Check if output directory exists
        if not use_gui:
            out_dir = support.make_directory(out_dir)
            if out_dir.endswith('/'):
                out_dir_ = out_dir[:-1]
            if '/' in out_dir_:
                out_fn = out_dir_.split('/')[-1] + '.json'
            else:
                out_fn = out_dir_ + '.json'

        # Data reset
        self.data_reset()

        print('Starting ToT Measurement!')
        print('=========================')
        try:
            start_time = time.time()
            frame_last = np.zeros(256)
            frame_num = 0

            if make_hist:
                frame_list = np.zeros((256, 8192))
            else:
                frame_list = []
            plot_hist = np.zeros(4096)
            time_list = []

            self.data_reset()
            while True:
                if meas_time is not None:
                    if time.time() - start_time > meas_time:
                        break

                frame = np.asarray( self.read_tot() )
                time_list.append(time.time() - start_time)
                frame[np.argwhere(frame - frame_last == 0)] = 0
                frame_last = np.array(frame, copy=True)

                plot_hist[frame] += 1
                if int_plot and not (frame_num % 100):
                    print(frame_num)
                    line.set_ydata(plot_hist)
                    fig.canvas.draw()

                if not (frame_num % 100):
                    print( frame_num / (time.time() - start_time))

                if make_hist:
                    frame_list += frame
                else:
                    frame_list.append( frame.tolist() )

                if (save_frames is not None) and (frame_num <= save_frames):
                    # Only save if gui is not used
                    if not use_gui:
                        self.measure_tot_save(frame_list, time_list,
                            out_dir, out_fn, start_time)

                        # Reset for next save
                        frame_list, time_list = [], []

                    frame_num = 0
                frame_num += 1

            self.measure_tot_save(frame_list, time_list,
                out_dir, out_fn, start_time)
            yield frame_list

        except (KeyboardInterrupt, SystemExit):
            if not use_gui:
                self.measure_tot_save(frame_list, time_list,
                    out_dir, out_fn, start_time)
            yield frame_list

    @classmethod
    def measure_tot_save(cls,
            frame_list, time_list,
            out_dir, out_fn,
            start_time=None
        ):
        support.json_dump({'tot': frame_list, 'time': time_list},\
            '%s/%s' % (out_dir, out_fn))
        if start_time is not None:
            print('Registered %d events in %.2f minutes' %\
                (np.count_nonzero(frame_list), (time.time() - start_time) / 60.))

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

"""
This class file contains functions to control the Dosepix detector
and perform measurements.
"""
# pylint: disable=missing-function-docstring
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import communicate
from . import support
from . import dpx_support

class DPXFunctions():
    """Control and measurement functions""" 
    def __init__(self, dpx, comm: communicate.Communicate):
        self.dpx = dpx
        self.comm = comm

    # === HARDWARE ===
    def enable_vdd(self):
        """Enable VDD""" 
        self.comm.send_cmd('EN_VDD')

    def disable_vdd(self):
        """Disable VDD"""
        self.comm.send_cmd('DISAB_VDD')

    def enable_bias(self):
        """Enable bias-voltage"""
        self.comm.send_cmd('EN_BIAS')

    def disable_bias(self):
        """Disable bias-voltage"""
        self.comm.send_cmd('DISAB_BIAS')

    def led_on(self):
        """Turn flash-LED on"""
        self.comm.send_cmd('LED_ON')

    def led_off(self):
        """Turn flash-LED off"""
        self.comm.send_cmd('LED_OFF')

    def read_adc(self):
        """Read DPX-ADC"""
        self.comm.send_cmd('READ_ADC', write=False)
        res = self.comm.get_data(size=2)
        return ''.join( ['%02x' % int(r) for r in res[::-1]] )

    # === RESET ===
    def global_reset(self):
        """Global reset"""
        self.comm.send_cmd('GLOBAL_RESET')

    def data_reset(self):
        """Data reset"""
        self.comm.send_cmd('DATA_RESET')

    # === OMR ===
    def read_omr(self):
        """Read OMR-register"""
        self.comm.send_cmd('READ_OMR', write=False)
        res = self.comm.get_data(size=3)
        return ''.join( ['%02x' % int(r) for r in res] )

    def write_omr(self, data):
        """Write OMR-register"""
        self.comm.send_cmd('WRITE_OMR')
        self.comm.send_data_binary(data)

    def set_pc_mode(self):
        """Set photon counting mode"""
        omr_code = '%04x' % (
            (int(self.dpx.omr, 16) & ~((0b11) << 22)) | (0b10 << 22))
        self.write_omr(omr_code)
        return omr_code

    def set_dosi_mode(self):
        """Set dosi mode"""
        omr_code = '%06x' % (int(self.dpx.omr, 16) & ~((0b11) << 22))
        self.write_omr(omr_code)
        return omr_code

    # === PERIPHERY ====
    def read_periphery(self):
        """Read periphery dacs"""
        self.comm.send_cmd('READ_PERIPHERY', write=False)
        res = self.comm.get_data(size=16)
        return ''.join( ['%02x' % r for r in res] )

    def write_periphery(self, data):
        """Write periphery dacs"""
        self.comm.send_cmd('WRITE_PERIPHERY')
        self.comm.send_data_binary(data)

    # === PIXEL DAC ===
    def read_pixel_dacs(self):
        """Read pixel dacs"""
        self.comm.send_cmd('READ_PIXEL_DAC', write=False)
        res = self.comm.get_data(size=256)
        return ''.join( ['%02x' % r for r in res] )

    def write_pixel_dacs(self, data):
        """Write pixel dacs"""
        self.comm.send_cmd('WRITE_PIXEL_DAC')

        # Split in chunks
        for split in range(4):
            data_split = data[split*128:(split+1)*128]
            self.comm.send_data_binary( data_split )

    # === CONF BITS ===
    def read_conf_bits(self):
        """Read configuration bits"""
        self.comm.send_cmd('READ_CONFBITS', write=False)
        res = self.comm.get_data(size=256)
        return ''.join( ['%02x' % r for r in res] )

    def write_conf_bits(self, data):
        """Write configuration bits"""
        self.comm.send_cmd('WRITE_CONFBITS')
        # Split in chunks
        for split in range(len(data) // 128):
            data_split = data[split*128:(split+1)*128]
            self.comm.send_data_binary( data_split )

    # === SINGLE THRESHOLD ===
    def write_single_threshold(self, data):
        """Write single thresholds for dosi-mode"""
        self.comm.send_cmd('WRITE_DIGITHLS')
        for split in range(len(data) // 128):
            data_split = data[split*128:(split+1)*128]
            self.comm.send_data_binary( data_split )

    # === COLUMN SELECT ===
    def read_column_select(self):
        """Read selected column"""
        self.comm.send_cmd('READ_COLSEL', write=False)
        res = self.comm.get_data(size=1)
        return res[0]

    def write_column_select(self, column):
        """Select column"""
        self.comm.send_cmd('WRITE_COLSEL')
        self.comm.send_data_binary('%02x' % column)

    # === DATA ===
    def read_pc(self):
        """Read data in photon counting mode"""
        self.comm.send_cmd('READ_PC', write=False)
        res = self.comm.get_data(size=256)
        return list(res)

    def read_tot(self):
        """Read data in ToT-mode"""
        self.comm.send_cmd('READ_TOT', write=False)
        res = self.comm.get_data(size=512)
        if res:
            return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]
        return np.zeros(256)

    def read_dosi(self):
        """Read data in dosi-mode"""
        self.comm.send_cmd('READ_DOSI', write=False)
        res = self.comm.get_data(size=512)
        return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]

    # === CLEAR BINS ===
    def clear_bins(self):
        """Clear bins of dosi-mode"""
        self.data_reset()
        self.read_dosi()

        for col in range(16):
            self.write_column_select(16 - col)
            self.read_dosi()

    # === FUNCTIONS ===
    def measure_tot(self,
        frame_time=0,
        save_frames=None,
        out_dir='tot_measurement/',
        meas_time=None,
        make_hist=True,
        use_gui=False
    ):
        """Wrapper for generator measure_tot_gen"""
        gen = self.measure_tot_gen(
            frame_time=frame_time,
            save_frames=save_frames,
            out_dir=out_dir,
            meas_time=meas_time,
            make_hist=make_hist,
            use_gui=use_gui
        )

        if use_gui:
            return gen

        # Return last value of generator
        *_, last = gen
        return last

    def measure_tot_gen(self,
        frame_time=1,
        save_frames=None,
        out_dir='tot_measurement/',
        meas_time=None,
        make_hist=True,
        use_gui=False
    ):
        """Perform measurement in ToT-mode. Implemented as a generator to
        yield intermediate results when gui is used
        Parameters
        ----------
        save_frames : int or None
            ToT is measured in an endless loop. The data is written to file
            after `save_frames` frames were processed. Additionally, Keyboard
            Interrupts are caught in order to store data afterwards.
            If `save_frames` is None, no intermediate files are created.
        out_dir : str or None
            Output directory in which the measurement files are stored.
            A file is written after `save_frames` frames. If file already exists,
            a number is appended to the filename and incremented. If `out_dir`
            is `None`, no files are created.
        meas_time : float, optional
            If set, the measurement finishes after the given time in seconds.
        make_hist : bool
            If 'True', results are stored in a histogram per pixel. Otherwise,
            results are on a frame by frame basis.
        use_gui : bool
            Set to `True` if module is used with the GUI application

        Returns
        -------
        out : dict
            Dictionary containing results according to `make_hist`. The returned
            format also depends on `use_gui`.

        Notes
        -----
        The function is optimized for long term ToT measurements. It runs in
        an endless loop and waits for a keyboard interrupt. After `save_frames`
        frames were processed, data is written to a file in the directory specified
        via `out_dir`. Using these single files allows for a memory sufficient
        method to store the data in histograms without loosing information and
        without the necessecity to load the whole dataset at once. If no frame
        information is required, `make_hist` should be used which directly returns
        histogrammed ToT-spectra per pixel.
        """

        # Activate dosi mode
        self.dpx.omr = self.set_dosi_mode()

        # Check if output directory exists
        if not use_gui and out_dir is not None:
            out_dir = support.make_directory(out_dir)
            if out_dir.endswith('/'):
                out_dir_ = out_dir[:-1]
            if '/' in out_dir_:
                out_fn = out_dir_.split('/')[-1] + '.json'
            else:
                out_fn = out_dir_ + '.json'
        else:
            out_fn = None

        # Data reset
        self.data_reset()

        print('Starting ToT Measurement!')
        print('=========================')
        try:
            start_time = time.time()
            frame_last = np.zeros(256)
            frame_num = 0

            if make_hist:
                frame_hist = np.zeros( (256, 4095) )
            else:
                frame_list, time_list = [], []

            self.data_reset()
            while True:
                if meas_time is not None:
                    if time.time() - start_time > meas_time:
                        break

                # Frame readout
                frame = np.asarray( self.read_tot() )
                if not make_hist:
                    time_list.append(time.time() - start_time)
                frame_filt = np.argwhere(frame - frame_last == 0)
                frame_last = np.array(frame, copy=True)
                frame[frame_filt] = 0

                # Wait
                time.sleep( frame_time )

                if use_gui:
                    yield frame

                # Show readout speed
                if (frame_num > 0) and not (frame_num % 10):
                    print( '%.2f Hz' % (frame_num / (time.time() - start_time)))

                if make_hist:
                    px_idx = np.dstack([np.arange(256), frame])[0]
                    px_idx[px_idx > 4095] = 0
                    px_idx = px_idx[px_idx[:,1] > 0]

                    frame_hist[px_idx[:,0], px_idx[:,1]] += 1
                else:
                    frame_list.append( frame.tolist() )
                frame_num += 1

                if (save_frames is not None) and (frame_num <= save_frames):
                    # Only save if gui is not used
                    if not use_gui:
                        if make_hist:
                            data_save = frame_hist
                        else:
                            data_save = [frame_list, time_list]
                        self.measure_save(
                            data_save, out_dir + out_fn,
                            make_hist, start_time
                        )

                        # Reset for next save
                        if make_hist:
                            frame_hist = np.zeros((256, 4095))
                        else:
                            frame_list = []

            if out_dir is not None and out_fn is not None:
                if make_hist:
                    data_save = frame_hist
                else:
                    data_save = [frame_list, time_list]
                    self.measure_save(
                        data_save, out_dir + out_fn,
                        make_hist, start_time
                    )

        except (KeyboardInterrupt, SystemExit):
            print('Measurement interrupted!')
        finally:
            if not use_gui:
                if make_hist:
                    data_save = frame_hist
                else:
                    data_save = [frame_list, time_list]
                if (out_dir is not None) and (out_fn is not None):
                    self.measure_save(
                        data_save, out_dir + out_fn,
                        make_hist, start_time
                    )
                yield data_save
            else:
                if make_hist:
                    yield {'Slot1': frame_hist}
                else:
                    yield {'Slot1': frame_list}


    def measure_dosi(self,
        frame_time=10,
        frames=10,
        freq=False,
        out_fn='dose_measurement.json',
        use_gui=False
    ):
        """Wrapper for generator measure_dosi_gen"""
        gen = self.measure_dosi_gen(
            frame_time=frame_time,
            frames=frames,
            freq=freq,
            out_fn=out_fn,
            use_gui=use_gui
        )

        if use_gui:
            return gen

        # Return last value of generator
        *_, last = gen
        return last

    def measure_dosi_gen(self,
        frame_time=10,
        frames=10,
        freq=False,
        out_fn='dose_measurement.json',
        use_gui=False):
        """Perform measurement in dosi-mode. Implemented as a generator to
        yield intermediate results when gui is used
        Parameters
        ----------
        frame_time : int
            Integration time per frame. The specified time is implemented
            as a pause between frame aqcuisitions
        frames : int or None
            Number of frames to record. If `None` is specified, an infinite
            measurement is started
        freq : bool
            If `True`, data is stored as frequency, i.e. number of registered
            events per frame are normalized with the acquisition duration
        out_fn : str or None
            Name of the output file to store results in. If file already exists,
            a number is appended to the filename and incremented. If `out_fn`
            is `None`, no file is created.
        use_gui : bool
            Set to `True` if module is used with the GUI application

        Returns
        -------
        out : dict
            Dictionary containing results. The returned
            format also depends on `use_gui`.
        """

        # Activate dosi mode
        self.dpx.omr = self.set_dosi_mode()

        # Data reset
        self.data_reset()
        self.clear_bins()

        print('Starting Dosi-Measurement!')
        print('=========================')

        # Specify frame range
        if frames is not None:
            frame_range = tqdm(range(frames))
        else:
            frame_range = support.infinite_for()

        frame_list, time_list = [], []
        try:
            meas_start = time.time()
            for _ in frame_range:
                start_time = time.time()

                # Wait
                time.sleep(frame_time)

                col_list = []
                for col in range(16):
                    self.write_column_select(15 - col)
                    mat = np.asarray( self.read_dosi(), dtype=float )
                    if freq:
                        mat = mat / float(time.time() - start_time)
                    col_list.append( mat )

                frame_list.append( col_list )
                time_list.append( time.time() - meas_start )
                if use_gui:
                    yield np.asarray( frame_list )
        except (KeyboardInterrupt, SystemExit):
            print('Measurement interrupted!')
        finally:
            if out_fn is not None:
                yield self.measure_save(
                    [frame_list, time_list],
                    out_fn, False, start_time
                )

    @classmethod
    def measure_save(cls,
            data,
            out_fn,
            make_hist=False,
            start_time=None
        ):

        if make_hist:
            out_dict = {'hist': data}
        else:
            out_dict = {'frames': data[0], 'time': data[1]}
        support.json_dump(out_dict, out_fn)
        if start_time is not None:
            if make_hist:
                events = np.count_nonzero(data)
            else:
                events = np.count_nonzero(data[0])
            print('Registered %d events in %.2f minutes' %\
                (events, (time.time() - start_time) / 60.))
        return out_dict

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

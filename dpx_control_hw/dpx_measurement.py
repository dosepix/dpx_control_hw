"""
This class file contains functions to perform measurements
with the Dosepix detector
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import support
from . import dpx_support
from . import dpx_functions

class DPXMeasurement:
    def __init__(self,
            dpx,
            dpf: dpx_functions.DPXFunctions
        ):
        self.dpx = dpx
        self.dpf = dpf

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
        self.dpx.omr = self.dpf.set_dosi_mode()

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

        # Init containers
        start_time = time.time()
        frame_last = np.zeros(256)
        frame_num = 0

        if make_hist:
            frame_list = np.zeros( (256, 4095) )
        else:
            frame_list, time_list = [], []

        print('Starting ToT Measurement!')
        print('=========================')
        print_time = start_time
        try:
            self.dpf.data_reset()
            while True:
                if meas_time is not None:
                    if time.time() - start_time > meas_time:
                        break

                # Frame readout
                frame = np.asarray( self.dpf.read_tot() )
                if not make_hist:
                    time_list.append(time.time() - start_time)
                frame_filt = np.argwhere(frame - frame_last == 0)
                frame_last = np.array(frame, copy=True)
                frame[frame_filt] = 0

                # Wait
                time.sleep( frame_time )

                # Show readout speed
                if not use_gui and time.time() - print_time > 1:
                    print( '%.2f Hz' % (frame_num / (time.time() - start_time)))
                    print_time = time.time()

                if make_hist:
                    px_idx = np.dstack([np.arange(256), frame])[0]
                    px_idx[px_idx > 4095] = 0
                    px_idx = px_idx[px_idx[:,1] > 0]

                    frame_list[px_idx[:,0], px_idx[:,1]] += 1
                else:
                    frame_list.append( frame.tolist() )

                if use_gui:
                    yield frame_list
                frame_num += 1

                if (save_frames is not None) and (frame_num <= save_frames):
                    # Only save if gui is not used
                    if not use_gui:
                        data_save = {'frames': frame_list}
                        if not make_hist:
                            data_save['time'] = time_list
                        self.measure_save(
                            data_save, out_dir + out_fn, start_time
                        )

                        # Reset for next save
                        if make_hist:
                            frame_list = np.zeros((256, 4095))
                        else:
                            frame_list = []

        except (KeyboardInterrupt, SystemExit):
            print('Measurement interrupted!')
        finally:
            if not use_gui:
                data_save = {'frames': frame_list}
                if not make_hist:
                    data_save['times'] = time_list
                if (out_dir is not None) and (out_fn is not None):
                    self.measure_save(
                        data_save, out_dir + out_fn, start_time
                    )
                yield frame_list
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
        use_gui=False
    ):
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
        self.dpx.omr = self.dpf.set_dosi_mode()

        # Specify frame range
        if frames is not None:
            frame_range = tqdm(range(frames))
        else:
            frame_range = support.infinite_for()

        frame_list, time_list = [], []
        meas_start = time.time()

        print('Starting Dosi-Measurement!')
        print('=========================')
        try:
            # Data reset
            self.dpf.data_reset()
            self.dpf.clear_bins()
            for _ in frame_range:
                start_time = time.time()

                # Wait
                time.sleep(frame_time)

                col_list = []
                for col in range(16):
                    self.dpf.write_column_select(15 - col)
                    mat = np.asarray( self.dpf.read_dosi(), dtype=float )
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
            out_dict = {'frames': frame_list, 'times': time_list}
            if out_fn is not None:
                yield self.measure_save(
                    out_dict,
                    out_fn, start_time
                )
            else:
                yield out_dict

    def measure_integration(self,
        out_fn='integration_measurement.json',
        meas_time=None,
        frame_time=1,
        use_gui=False
    ):
        """Wrapper for generator measure_integration_gen"""
        gen = self.measure_integration_gen(
            out_fn = out_fn,
            meas_time = meas_time,
            frame_time = frame_time,
            use_gui = use_gui
        )

        if use_gui:
            return gen

        # Return last value of generator
        *_, last = gen
        return last

    def measure_integration_gen(self,
        out_fn='integration_measurement.json',
        meas_time=None,
        frame_time=1,
        use_gui=False
    ):
        """Perform measurement in integration-mode. Implemented as a generator to
        yield intermediate results when gui is used
        Parameters
        ----------
        frame_time : int
            Integration time per frame. The specified time is implemented
            as a pause between frame aqcuisitions
        meas_time : float, optional
            If set, the measurement finishes after the given time in seconds.
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
        self.dpx.omr = self.dpf.set_integration_mode()

        # Init containers
        frame_list, time_list = [], []
        start_time = time.time()
        print_time = start_time
        frame_num = 0

        print('Starting Integration-Measurement!')
        print('=================================')
        try:
            self.dpf.data_reset()
            while True:
                # Break if meas_time is surpassed
                if meas_time is not None:
                    if time.time() - start_time > meas_time:
                        break

                # Frame readout
                frame = np.asarray( self.dpf.read_integration() )
                frame_list.append( frame )

                # Wait
                time.sleep( frame_time )

                if use_gui:
                    yield frame

                # Show readout speed
                if not use_gui and time.time() - print_time > 1:
                    print( '%.2f Hz' % (frame_num / (time.time() - start_time)))
                    print_time = time.time()

                frame_num += 1
        except (KeyboardInterrupt, SystemExit):
            print('Measurement interrupted!')
        finally:
            if not use_gui:
                data_save = {
                    'frames': frame_list,
                    'times': time_list
                }
                if out_fn is not None:
                    self.measure_save(
                        data_save, out_fn, start_time
                    )
                yield data_save
            else:
                yield {'Slot1': frame_list}

    def measure_adc(self,
        analog_out='v_tha',
        perc=False,
        adc_high=8191,
        adc_low=0,
        adc_step=1,
        n_meas=1,
        out_fn=None,
        plot=False,
        use_gui=False
    ):
        """Wrapper for generator measure_adc_gen"""
        gen = self.measure_adc_gen(
            analog_out=analog_out,
            perc=perc,
            adc_high=adc_high,
            adc_low=adc_low,
            adc_step=adc_step,
            n_meas=n_meas,
            out_fn=out_fn,
            plot=plot,
            use_gui=use_gui
        )

        if use_gui:
            return gen

        # Return last value of generator
        *_, last = gen
        return last

    def select_adc(
        self,
        analog_out='v_tha'
    ):
        """Set OMR code according to selected analog_out"""
        omr_code = int(self.dpx.omr, 16)
        omr_code &= ~(0b11111 << 12)
        omr_code |= getattr(dpx_support._omr_analog_out_sel, analog_out)
        self.dpx.omr = '%06x' % omr_code
        print('OMR set to:', self.dpx.omr)
        self.dpf.write_omr(self.dpx.omr)

    def measure_adc_gen(
            self,
            analog_out='v_tha',
            perc=False,
            adc_high=8191,
            adc_low=0,
            adc_step=1,
            n_meas=1,
            out_fn=None,
            plot=False,
            use_gui=False
        ):
        """Take measurements of the analog voltages of Dosepix by utilizing
        the ADC of the microcontroller
        Parameters
        ----------
        analog_out : str
            Specify the analog output of the detector by chosing the name
            according to the datasheet
        perc : bool
            If set to `True`, results are stored as percentage instead of
            integer DAC values
        adc_high : int
            High end of the scanning range of the ADC. Maximum number is
            `8191`
        adc_low : int
            Low end of the scanning range of the ADC. Minimum number is `0`
        adc_step : int
            Step with when the ADC is scanned
        n_meas : int
            Number of measurements per step. Increasing this number reduces
            the noise in the measurements. Can remain low as precision of the
            ADC is great
        out_fn : str
            Name of the output file to store results in. If file already exists,
            a number is appended to the filename and incremented. If `out_fn`
            is `None`, no file is created
        plot : bool
            If set to `True`, results are plotted after the scan
        use_gui : bool
            Set to `True` if module is used with the GUI application

        Returns
        -------
        out : dict
            Dictionary containing results. The returned
            format also depends on `use_gui`.
        """

        # Display execution time at the end
        start_time = time.time()

        # Select analog out
        self.select_adc(analog_out=analog_out)

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
            self.dpf.write_periphery(code)

            # Measure multiple times
            adc_val_list = []
            for _ in range(n_meas):
                adc_val = self.dpf.read_adc()
                adc_val_list.append( float(int(adc_val, 16)) )

            adc_volt_mean.append(np.mean(adc_val_list))
            adc_volt_err.append(np.std(adc_val_list) / np.sqrt(n_meas))

            if use_gui:
                yield {'Volt': adc_volt_mean[-1], 'ADC': int(adc)}

        if plot and not use_gui:
            plt.errorbar(adc_list, adc_volt_mean, yerr=adc_volt_err, marker='x')
            plt.show()

        out_dict = {'Volt': adc_volt_mean, 'ADC': adc_list.tolist()}
        if not use_gui:
            if plot:
                plt.plot(adc_volt_mean, adc_list)
                plt.show()

            print(
                'Execution time: %.2f min' %
                ((time.time() - start_time) / 60.))

        if out_fn is not None:
            self.measure_save(out_dict, out_fn, start_time=None)
        yield out_dict

    def measure_thl(self, out_fn=None, plot=False, use_gui=False):
        """See `measure_adc_gen` for documentation. Wrapper for threshold"""
        gen = self.measure_adc_gen(
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

    @classmethod
    def measure_save(cls,
            data,
            out_fn,
            start_time=None
        ):
        """Store `data` in json-file `out_fn`
        Parameters
        ----------
        data : dict
            Dictionary containing the data to be stored in `out_fn`
        out_fn : str
            Output json-file
        start_time : float or None
            If provided, print the number of registered events per time
        """

        support.json_dump(data, out_fn)

        if start_time is not None:
            events = np.count_nonzero(data['frames'])
            print('Registered %d events in %.2f minutes' %\
                (events, (time.time() - start_time) / 60.))
        return data

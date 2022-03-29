# pylint: disable=missing-function-docstring
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

    # === SINGLE THRESHOLD ===
    def write_single_threshold(self, data):
        self.comm.send_cmd('WRITE_DIGITHLS')
        for split in range(len(data) // 128):
            data_split = data[split*128:(split+1)*128]
            self.comm.send_data_binary( data_split )

    # === COLUMN SELECT ===
    def read_column_select(self):
        self.comm.send_cmd('READ_COLSEL', write=False)
        res = self.comm.get_data(size=1)
        return res[0]

    def write_column_select(self, column):
        self.comm.send_cmd('WRITE_COLSEL')
        self.comm.send_data_binary('%02x' % column)

    # === DATA ===
    def read_pc(self):
        self.comm.send_cmd('READ_PC', write=False)
        res = self.comm.get_data(size=256)
        return list(res)

    def read_tot(self):
        self.comm.send_cmd('READ_TOT', write=False)
        res = self.comm.get_data(size=512)
        if res:
            return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]
        else:
            return np.zeros(256)
            
    def read_dosi(self):
        self.comm.send_cmd('READ_DOSI', write=False)
        res = self.comm.get_data(size=512)
        return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]

    # === CLEAR BINS ===
    def clear_bins(self):
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
            
            if out_fn is not None:
                yield self.measure_save(
                    frame_list, time_list,
                    out_fn, start_time
                )

        except (KeyboardInterrupt, SystemExit):
            if out_fn is not None:
                yield self.measure_save(
                    frame_list, time_list,
                    out_fn, start_time
                )

    
    def measure_integration(self,
        out_fn='integration_measurement.json',
        meas_time = None,
        frame_time = 1,
        integration = True,
        singlevalues = False,
        use_gui = False
    ):

        """Wrapper for generator measure_Integration_gen"""
        gen = self.measure_integration_gen(
            out_fn = out_fn,
            meas_time = meas_time,
            frame_time = frame_time,
            integration = integration,
            singlevalues = singlevalues,
            use_gui = use_gui
        )

        if use_gui:
            return gen
        
        *_, last = gen
        return last

    def measure_integration_gen(self,
        out_fn='integration_measurement.json',
        meas_time = None,
        frame_time = 1,
        integration = True,
        singlevalues = False,
        use_gui = False
    ):

            # Activate dosi mode
            self.dpx.omr = self.set_dosi_mode()

            # Data reset
            self.data_reset()
            self.clear_bins()

            print('Starting Integration-Measurement!')
            print('=========================')

            if singlevalues:
                Int_list = []
                for pix in range(256):
                    empty_pixel_list = []
                    Int_list.append(empty_pixel_list)
            else:
                Int_list = np.zeros(256)
            try:
                start_time = time.time()
                frame_last = np.zeros(256)
                frame_num = 0

                # if make_hist:
                #    frame_list = np.zeros((256, 8192))
                # else:
                #frame_list = []
                plot_hist = np.zeros(4096)
                #pixel_list = []
                #for pix in range(256):
                #    empty_pixel_list = []
                #    pixel_list.append(empty_pixel_list)
                #time_list = []

                self.data_reset()
                while True:
                    if meas_time is not None:
                        if time.time() - start_time > meas_time:
                            break

                    # Frame readout
                    frame = np.asarray( self.read_tot() )
                    #time_list.append(time.time() - start_time)
                    frame_filt = np.argwhere(frame - frame_last == 0)
                    frame_last = np.array(frame, copy=True)
                    frame[frame_filt] = 0

                    # Wait
                    time.sleep( frame_time )

                    if use_gui:
                        yield frame

                    # plot_hist[frame] += 1
                    # Show readout speed
                    if (frame_num > 0) and not (frame_num % 10):
                        print( '%.2f Hz' % (frame_num / (time.time() - start_time)))

                    if singlevalues:
                        for pix in range(256):
                            if not frame.tolist()[pix] == 0:
                                Int_list[pix].append(frame.tolist()[pix]) 
                    else:
                        for pix in range(256):
                            if not frame.tolist()[pix] == 0:
                                Int_list[pix] = Int_list[pix] + frame.tolist()[pix]
                    
                    frame_num += 1
                
                if out_fn is not None:
                    if singlevalues:
                        yield self.measure_save(
                            Int_list,
                            out_fn, integration, singlevalues, False, start_time 
                        )
                    else:
                        yield self.measure_save(
                            Int_list,
                            out_fn, integration, singlevalues, False, start_time
                        )


            except (KeyboardInterrupt, SystemExit):
                if out_fn is not None:
                    if singlevalues:
                        yield self.measure_save(
                            Int_list,
                            out_fn, integration, singlevalues, False, start_time
                        )
                    else:
                        yield self.measure_save(
                            Int_list,
                            out_fn, integration, singlevalues, False, start_time
                        )


    @classmethod
    def measure_save(cls,
            data,
            out_fn,
            integration = False,
            singlevalues = False,
            make_hist=False,
            start_time=None
        ):

        if make_hist:
            out_dict = {'hist': data}
        elif integration:
            if singlevalues:
                out_dict = {'pixel': data}
            else:
                out_dict = {'integration': data}
        else:
            out_dict = {'frames': data[0], 'time': data[1]}
        support.json_dump(out_dict, out_fn)
        if start_time is not None and (integration == False or (integration and singlevalues == True)):
            if make_hist:
                events = np.count_nonzero(data)
            elif integration and singlevalues:
                events = 0
                for pix in data:
                    events = events + len(pix)
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
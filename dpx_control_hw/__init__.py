import serial
import time
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from . import config
from . import support
from . import communicate
from . import dpx_functions
from . import equalization

class Dosepix():
    def __init__(self,
            port_name,
            config_fn=None,
            thl_calib_fn=None):
        self.port_name = port_name
        self.config_fn = config_fn

        # Detector settings, standard values
        self.periphery_dacs = 'dc3c0bc86450823877808032006414c0'
        self.thl = None
        self.omr = '381fc0'
        self.conf_bits = None
        self.pixel_dacs = None
        self.bin_edges = None

        # THL calibration
        self.thl_edges = None
        self.thl_edges_low = None
        self.thl_edges_high = None
        self.thl_fit_params = None

        # Set detector settings from config
        self.load_config(config_fn, thl_calib_fn)

        # Serial connection
        self.ser = None
        self.comm = None
        self.dpf = None

        # Init connection
        self.init_dpx()

    def load_config(self, config_fn, thl_calib_fn):
        if config_fn is not None:
            conf_d = config.read_config(config_fn)
            self.periphery_dacs = conf_d['peripherys']
            self.omr = conf_d['omr']
            self.conf_bits = conf_d['conf_bits']
            self.pixel_dacs = conf_d['pixel_dacs']
            self.bin_edges = conf_d['bin_edges']

        # Load thl calibration
        if thl_calib_fn is not None:
            thl_low, thl_high, thl_fit_params, thl_edges = config.read_thl_calibration(thl_calib_fn)
            print(thl_low)
            print(thl_high)
            self.thl_edges_low = thl_low
            self.thl_edges_high = thl_high
            self.thl_fit_params = thl_fit_params
            self.thl_edges = thl_edges

    def equalization(self, config_fn):
        equal = equalization.Equalization(self, self.comm)
        *_, last = equal.threshold_equalization(thl_offset=0)
        pixel_dacs, thl_new, conf_mask = last
        periphery_dacs = self.periphery_dacs[:-4] + thl_new

        config_d = {
            'peripherys': periphery_dacs,
            'omr': self.omr,
            'pixel_dacs': pixel_dacs,
            'conf_bits': conf_mask,
        }
        config.write_config(config_fn, config_d)

    def set_config(self):
        self.dpf.write_omr(self.omr)
        self.dpf.write_periphery(self.periphery_dacs)
        self.dpf.write_conf_bits(self.conf_bits)
        self.dpf.write_pixel_dacs(self.pixel_dacs)

    def init_dpx(self):
        self.connect()
        self.comm = communicate.Communicate(self.ser, debug=False)
        self.dpf = dpx_functions.DPXFunctions(self, self.comm)

        # Enable voltages
        self.dpf.enable_vdd()
        self.dpf.enable_bias()

        # Loads standard values if no config was specified
        self.set_config()
        self.dpf.led_off()

        self.dpf.global_reset()
        return

        # Change periphery dac settings
        d_periphery = support.split_perihpery_dacs(self.periphery_dacs, perc=False, show=True)
        d_periphery['i_krum'] = 10
        d_periphery['i_pixeldac'] = 40
        self.periphery_dacs = support.perihery_dacs_dict_to_code(d_periphery, perc=False)

        # Enable voltages
        self.dpf.enable_vdd()
        self.dpf.enable_bias()
        time.sleep(0.3)

        # == GLOBAL RESET ==
        self.dpf.global_reset()

        # == Start settings ==
        self.dpf.write_omr(self.omr)
        print( 'OMR: ' + self.dpf.read_omr() )
        self.dpf.write_periphery(self.periphery_dacs)

        # == Threshold equalization ==
        # *_, last = self.dpf.measure_thl(plot=True)
        if True:
            equal = equalization.Equalization(self, self.comm)
            *_, last = equal.threshold_equalization(thl_offset=0)

            pixel_dac_new, thl_new, conf_mask = last
            print()

            self.periphery_dacs = self.periphery_dacs[:-4] + thl_new
            self.dpf.write_periphery(self.periphery_dacs)
            print('New periphery:', self.dpf.read_periphery())

            self.pixel_dacs = pixel_dac_new
            self.dpf.write_pixel_dacs(self.pixel_dacs)
            print('New pixel dacs:', self.dpf.read_pixel_dacs())

            print(conf_mask)
            self.dpf.write_conf_bits(conf_mask)

        # Set dosi mode to measure ToT
        self.omr = self.dpf.set_dosi_mode()
        print( 'OMR: ' + self.dpf.read_omr() )

        # == CONFBITS ==
        # conf_bits = ''.join([np.random.choice(list('0123456789abcdef')) for n in range(512)])
        # print(conf_bits)
        # self.dpf.write_conf_bits(conf_bits)
        # print(self.dpf.read_conf_bits())

        # == COLSEL ==
        # self.dpf.write_column_select('04')
        # print( 'Colsel: ' + str(self.dpf.read_column_select()))

        start_time = time.time()
        cnt = 0
        self.dpf.write_periphery(self.periphery_dacs)
        print( 'Periphery: ' + self.dpf.read_periphery() )
        self.dpf.data_reset()

        tot = []
        tot_last = np.zeros(256)
        self.dpf.led_on()
        *_, last = self.dpf.measure_tot(out_dir='tot_measurement/')

        try:
            while True:
                # == ToT Benchmark ==
                frame = np.asarray( self.dpf.read_tot() )
                frame[np.argwhere(frame - tot_last == 0)] = 0
                tot_last = np.array(frame, copy=True)
                tot += list( frame )

                # == Periphery Benchmark ==
                '''
                # print('%032d' % cnt)
                self.dpf.write_periphery('%032d' % (cnt))
                self.dpf.read_periphery()
                '''

                # == OMR Benchmark ==
                '''
                # time.sleep(0.1)
                self.dpf.write_omr('%06d' % (cnt))
                # print('OMR send: ' + '%06d' % (cnt))
                res = self.dpf.read_omr()
                print('OMR read: ' + res)
                # print()
                '''

                cnt += 1
                if not cnt % 1000:
                    print(cnt / (time.time() - start_time))

        except KeyboardInterrupt:
            print( len(tot) )
            plt.hist(np.asarray(tot)[np.asarray(tot) > 0], bins=np.arange(2000))
            plt.show()

            self.dpf.led_off()
            self.dpf.disable_bias()
            self.dpf.disable_vdd()

            return

    def get_serial(self):
        return self.ser

    def connect(self, reconnect=False):
        if self.ser is None:
            self.ser = serial.Serial(self.port_name)
            assert self.ser.is_open, 'Error: Could not establish serial connection!'
        elif reconnect:
            self.disconnect()
            self.connect()

    def disconnect(self):
        if self.ser is None:
            return

        self.ser.close()
        self.ser = None

        self.dpf.led_off()
        self.dpf.disable_bias()
        self.dpf.disable_vdd()

    def __del__(self):
        # self.comm.send_cmd('DISAB_VDD')
        self.disconnect()

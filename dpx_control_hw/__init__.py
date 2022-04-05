import json
import serial
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from . import config
from . import support
from . import communicate
from . import dpx_functions
from . import dpx_functions_dummy
from . import dpx_measurement
from . import equalization

class Dosepix():
    def __init__(self,
            port_name,
            config_fn=None,
            params_fn=None,
            thl_calib_fn=None,
            bin_edges_fn=None,
            use_gui=False):
        self.port_name = port_name
        self.config_fn = config_fn

        self.use_gui = use_gui
        self.params_dict = None
        self.bin_edges = None
        self.single_thresholds = None

        # Detector settings, standard values
        self.periphery_dacs = 'dc3c0bc86450823877808032006414c0'
        self.omr = '381fc0'
        self.conf_bits = '00' * 256
        self.pixel_dacs = '3f' * 256

        # THL calibration
        self.thl_edges = None
        self.thl_edges_low = None
        self.thl_edges_high = None
        self.thl_fit_params = None

        # Set detector settings from config
        self.load_config(
            config_fn,
            thl_calib_fn,
            params_fn,
            bin_edges_fn
        )

        # Serial connection
        self.ser = None
        self.comm = None
        self.dpf = None
        self.dpm = None
        self.equal = None
        self.support = support

        # Init connection
        self.init_dpx()

    def load_config(self,
        config_fn,
        thl_calib_fn,
        params_fn,
        bin_edges_fn):
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
            self.thl_edges_low = thl_low
            self.thl_edges_high = thl_high
            self.thl_fit_params = thl_fit_params
            self.thl_edges = thl_edges

        # Load calibration parameters
        if params_fn is not None:
            self.params_dict = self.get_calib_params(params_fn)

        # Load bin edges
        if bin_edges_fn is not None:
            self.single_thresholds = config.load_bin_edges(
                bin_edges_fn=bin_edges_fn,
                params_d=self.params_dict
            )

    def equalization(self, config_fn,
        thl_step=1,
        noise_limit=10,
        n_evals=3,
        num_dacs=20,
        i_pixeldac=60,
        thl_offset=0,
        plot=True
    ):
        *_, last = self.equal.threshold_equalization(
            thl_step=thl_step,
            noise_limit=noise_limit,
            n_evals=n_evals,
            num_dacs=num_dacs,
            i_pixeldac=i_pixeldac,
            thl_offset=thl_offset,
            use_gui=False,
            plot=plot
        )
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

        # Load thresholds for dosi-mode
        if self.single_thresholds is not None:
            for thres in self.single_thresholds:
                self.dpf.write_single_threshold(thres)

    def get_calib_params(self, params_fn):
        if params_fn is not None:
            params_d = {}
            if params_fn.endswith('.json'):
                with open(params_fn, 'r') as params_file:
                    params_d = json.load(params_file)
                    params_d = {int(key): params_d[key] for key in params_d.keys()}
                    return params_d
            else:
                return None
        return None

    def set_thl_calib(self, thl_calib_d):
        thl_low, thl_high, thl_fit_params, thl_edges = config.load_thl_edges(thl_calib_d)
        self.thl_edges_low = thl_low
        self.thl_edges_high = thl_high
        self.thl_fit_params = thl_fit_params
        self.thl_edges = thl_edges
        print(self.thl_edges_low, self.thl_edges_high)

    def init_dpx(self):
        if self.port_name is not None:
            self.connect()

        self.comm = communicate.Communicate(self.ser, debug=False)

        # Functions to control the Dosepix detector
        # If port_name is None, the dummy generator is used
        if self.port_name is None:
            self.dpf = dpx_functions_dummy.DPXFunctionsDummy(self, self.comm)
        else:
            self.dpf = dpx_functions.DPXFunctions(self, self.comm)
        # Functions to perform measurements with DPX
        self.dpm = dpx_measurement.DPXMeasurement(self, self.dpf)
        # Functions to equalize DPX
        self.equal = equalization.Equalization(self, self.comm)

        # Enable voltages
        self.dpf.enable_vdd()
        self.dpf.enable_bias()

        # Loads standard values if no config was specified
        self.set_config()
        self.dpf.led_off()

        self.dpf.global_reset()
        return

    def set_config_gui(self, config):
        self.periphery_dacs = self.periphery_dacs[:-4] + '%04x' % config['v_tha']
        self.conf_bits = config['confbits']
        self.pixel_dacs = config['pixeldac']

    def get_serial(self):
        return self.ser

    def connect(self, reconnect=False):
        if self.ser is None:
            self.ser = serial.Serial(self.port_name)
            assert self.ser.is_open, 'Error: Could not establish serial connection!'
        elif reconnect:
            self.close()
            self.connect()

    def close(self):
        if self.ser is None:
            return

        self.dpf.led_off()
        self.dpf.disable_bias()
        self.dpf.disable_vdd()

        self.ser.close()
        self.ser = None

    def __del__(self):
        self.close()

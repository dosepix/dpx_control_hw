import serial

# Local imports
from . import config
from . import communicate
from . import dpx_functions

class Dosepix():
    def __init__(self,
            port_name,
            config_fn=None):
        self.port_name = port_name
        self.config_fn = config_fn

        # Detector settings
        self.periphery_dacs = None
        self.thl = None
        self.omr = None
        self.conf_bits = None
        self.pixel_dacs = None
        self.bin_edges = None

        # Set detector settings from config
        if config_fn is not None:
            self.load_config(config_fn)

        # Serial connection
        self.ser = None
        self.comm = None
        self.dpf = None

        # Init connection
        self.init_DPX()

    def load_config(self, config_fn):
        periphery_dacs, thl, omr, conf_bits, pixel_dacs, bin_edges = config.read_config(config_fn)
        self.periphery_dacs = periphery_dacs
        self.thl = thl
        self.omr = omr
        self.conf_bits = conf_bits
        self.pixel_dacs = pixel_dacs
        self.bin_edges = bin_edges

    def init_DPX(self):
        self.connect()
        self.comm = communicate.Communicate(self.ser, debug=True)
        self.dpf = dpx_functions.DPXFunctions(self, self.comm)

        while True:
            # res = self.comm.get_data()
            s = self.ser.read(size=512)
            s = [int.from_bytes(s[i:i+2], 'big') for i in range(0, len(s), 2)]
            print( s )

        self.dpf.enable_vdd()

        self.dpf.write_omr('123456')
        res = self.dpf.read_omr()
        print(['%02x' % r for r in res])

        self.dpf.write_periphery('12345678123456781234567812345612')
        res = self.dpf.read_periphery()
        print(['%02x' % r for r in res])

        self.dpf.disable_vdd()

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

    def __del__(self):
        # self.comm.send_cmd('DISAB_VDD')
        self.disconnect()

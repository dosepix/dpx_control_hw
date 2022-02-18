import serial
import time

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

        # Detector settings, standard values
        self.periphery_dacs = None
        self.thl = None
        self.omr = '01ffc0'
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
        self.comm = communicate.Communicate(self.ser, debug=False)
        self.dpf = dpx_functions.DPXFunctions(self, self.comm)

        self.dpf.enable_vdd()
        self.dpf.global_reset()
        # self.dpf.write_omr('123456')

        self.dpf.set_dosi_mode()
        print( self.dpf.read_omr() )

        self.dpf.write_periphery('12345678123456781234567812345612')
        print( self.dpf.read_periphery() )

        start_time = time.time()
        cnt = 0
        try:
            while True:
                self.dpf.data_reset()
                s = self.dpf.read_tot()
                print( s )

                # == Periphery Benchmark ==
                '''
                # print('%032d' % cnt)
                self.dpf.write_periphery('%032d' % (cnt))
                self.dpf.read_periphery()
                '''

                # == OMR Benchmark ==
                '''
                time.sleep(0.1)
                # self.dpf.write_omr('%06d' % (cnt))
                # print('OMR send: ' + '%06d' % (cnt))
                res = self.dpf.read_omr()
                print('OMR read: ' + res)
                # print()
                '''
                
                cnt += 1
                if not cnt % 1000:
                    print(cnt / (time.time() - start_time))
        except KeyboardInterrupt:
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

    def __del__(self):
        # self.comm.send_cmd('DISAB_VDD')
        self.disconnect()

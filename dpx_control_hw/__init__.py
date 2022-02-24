import serial
import time
import numpy as np
import matplotlib.pyplot as plt

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
        self.periphery_dacs = 'dc3c0bc86450823877808032006414c4'
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
        self.dpf.enable_bias()
        time.sleep(0.3)

        # == GLOBAL RESET ==
        self.dpf.global_reset()

        *_, last = self.dpf.measure_thl(plot=True)
        return

        # == PIXEL DACS ==
        # pixel_dacs = 'ff' * 256
        pixel_dacs = '19153f15163f3f173f3f181811143f173f3f1816141615183f163f17\
                        13133f16173f1813323f3f1717323f163f1612161319173f3f3f33\
                        151313003f3f3f163215143f18173f183f1713192c3f161a3f1317\
                        17133f311718171516193f163f3f183f173f17173f11183f173f18\
                        133f3f1737193f1814143f163f13183f1817181716173f34143f14\
                        3f183f17183f183f183314183f1714143f183f1815343f3f2e1815\
                        3f3f3f19171616183f3f3f3f16183f183514131718171816311813\
                        15170000130000001700000000003f003f001b3f133f3f18301812\
                        13123f3f15123f151615163f3f14143f171615143f3f3f16171714\
                        14143f153f3f143f163f1816'
        # pixel_dacs = ''.join([np.random.choice(list('0123456789abcdef')) for n in range(512)])
        print(pixel_dacs)
        self.dpf.write_pixel_dacs(pixel_dacs)
        print( self.dpf.read_pixel_dacs() )

        # == OMR ==
        self.dpf.write_omr('39ffc0')
        self.dpf.set_dosi_mode()
        print( 'OMR: ' + self.dpf.read_omr() )

        # *_, last = self.dpf.threshold_equalization(thl_offset=10, use_gui=False)
        # return

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
        self.dpf.led_off()

        tot = []
        tot_last = np.zeros(256)
        try:
            while True:
                '''
                print(thl)
                p = 'dc3c0bc864508238778080320064' + '%04x' % thl
                self.dpf.write_periphery(p)
                print( 'Periphery: ' + self.dpf.read_periphery() )

                thl += 100
                if thl > 8000:
                    thl = 5000
                '''

                # == ToT Benchmark ==
                s = np.asarray( self.dpf.read_tot() )
                s[np.argwhere(s - tot_last == 0)] = 0
                tot_last = np.array(s, copy=True)
                tot += list( s )

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

    def __del__(self):
        # self.comm.send_cmd('DISAB_VDD')
        self.disconnect()

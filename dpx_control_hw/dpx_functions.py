"""
This class file contains functions to control the Dosepix detector.
"""
# pylint: disable=missing-function-docstring
import numpy as np

from . import communicate

class DPXFunctions():
    """Control and measurement functions"""
    def __init__(self, 
        dpx,
        comm: communicate.Communicate
    ):
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

    def read_bias(self):
        self.comm.send_cmd('READ_BIAS', write=False)
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
        omr_code = int(self.dpx.omr, 16)
        omr_code ^= (0b10) << 22
        omr_code = '%06x' % omr_code
        self.write_omr(omr_code)
        return omr_code

    def set_dosi_mode(self):
        """Set dosi mode"""
        omr_code = int(self.dpx.omr, 16)
        omr_code ^= (0b00) << 22
        omr_code = '%06x' % omr_code
        self.write_omr(omr_code)
        return omr_code

    def set_integration_mode(self):
        """Set integration mode"""
        omr_code = int(self.dpx.omr, 16)
        omr_code |= (0b11) << 22
        omr_code = '%06x' % omr_code
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
    def read_single_threshold(self):
        """Read configuration bits"""
        self.comm.send_cmd('READ_DIGITHLS', write=False)

        res_tot = []
        for _ in range(16):
            res = self.comm.get_data(size=512)
            res_tot.append( ''.join( ['%02x' % r for r in res] ) )
        return res_tot

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

    # === TESTPULSE ===
    def generate_test_pulse(self):
        """Generate test pulse"""
        self.comm.send_cmd('TEST_PULSE')

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
        return np.zeros(256, dtype=int).tolist()

    def read_dosi(self):
        """Read data in dosi-mode"""
        self.comm.send_cmd('READ_DOSI', write=False)
        res = self.comm.get_data(size=512)
        if res:
            return [int.from_bytes(res[i:i+2], 'big') for i in range(0, len(res), 2)]
        return np.zeros(256, dtype=int).tolist()

    def read_integration(self):
        """Read data in integration-mode"""
        self.comm.send_cmd('READ_INT', write=False)
        res = self.comm.get_data(size=768)
        if res:
            return [int.from_bytes(res[i:i+3], 'big') for i in range(0, len(res), 3)]
        return np.zeros(256, dtype=int).tolist()

    # === CLEAR BINS ===
    def clear_bins(self):
        """Clear bins of dosi-mode"""
        self.data_reset()
        self.read_dosi()

        for col in range(16):
            self.write_column_select(16 - col)
            self.read_dosi()

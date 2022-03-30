"""
This class file contains functions to control the Dosepix detector.
This is the dummy implementation of the functions, i.e. the detector
is not accessed, but (randomly) generated values are returned instead.
This module is used for testing purposes
"""
import numpy as np

from . import communicate

class DPXFunctionsDummy():
    """Control and measurement functions"""
    def __init__(self, dpx, comm: communicate.Communicate):
        self.dpx = dpx
        self.comm = comm

    # === HARDWARE ===
    def enable_vdd(self):
        """Enable VDD"""
        pass

    def disable_vdd(self):
        """Disable VDD"""
        pass

    def enable_bias(self):
        """Enable bias-voltage"""
        pass

    def disable_bias(self):
        """Disable bias-voltage"""
        pass

    def led_on(self):
        """Turn flash-LED on"""
        pass

    def led_off(self):
        """Turn flash-LED off"""
        pass

    def read_adc(self):
        """Read DPX-ADC"""
        return np.random.randint(0, 255)

    # === RESET ===
    def global_reset(self):
        """Global reset"""
        pass

    def data_reset(self):
        """Data reset"""
        pass

    # === OMR ===
    def read_omr(self):
        """Read OMR-register"""
        return self.dpx.omr

    def write_omr(self, data):
        """Write OMR-register"""
        pass

    def set_pc_mode(self):
        """Set photon counting mode"""
        omr_code = int(self.dpx.omr, 16)
        omr_code ^= (0b10) << 22
        omr_code = '%06x' % omr_code
        return omr_code

    def set_dosi_mode(self):
        """Set dosi mode"""
        omr_code = int(self.dpx.omr, 16)
        omr_code &= (0b00) << 22
        omr_code = '%06x' % omr_code
        return omr_code

    def set_integration_mode(self):
        """Set integration mode"""
        omr_code = int(self.dpx.omr, 16)
        omr_code |= (0b11) << 22
        omr_code = '%06x' % omr_code
        return omr_code

    # === PERIPHERY ====
    def read_periphery(self):
        """Read periphery dacs"""
        return self.dpx.periphery_dacs

    def write_periphery(self, data):
        """Write periphery dacs"""
        pass

    # === PIXEL DAC ===
    def read_pixel_dacs(self):
        """Read pixel dacs"""
        self.comm.send_cmd('READ_PIXEL_DAC', write=False)
        res = self.comm.get_data(size=256)
        return ''.join( ['%02x' % r for r in res] )

    def write_pixel_dacs(self, data):
        """Write pixel dacs"""
        pass

    # === CONF BITS ===
    def read_conf_bits(self):
        """Read configuration bits"""
        return self.dpx.conf_bits

    def write_conf_bits(self, data):
        """Write configuration bits"""
        pass

    # === SINGLE THRESHOLD ===
    def write_single_threshold(self, data):
        """Write single thresholds for dosi-mode"""
        pass

    # === COLUMN SELECT ===
    def read_column_select(self):
        """Read selected column"""
        return np.random.randint(0, 16)

    def write_column_select(self, column):
        """Select column"""
        pass

    # === DATA ===
    def read_pc(self):
        """Read data in photon counting mode"""
        data = np.random.randint(0, 255, 256)
        data[data < 0] = 0
        return data.tolist()

    def read_tot(self):
        """Read data in ToT-mode"""
        data = np.random.normal(100, 10, 256)
        data[data < 0] = 0
        return data.tolist()

    def read_dosi(self):
        """Read data in dosi-mode"""
        data = np.random.normal(100, 10, 256)
        data[data < 0] = 0
        return data.tolist()

    # === CLEAR BINS ===
    def clear_bins(self):
        """Clear bins of dosi-mode"""
        pass

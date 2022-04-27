from textwrap import wrap
import numpy as np

from . import dpx_support
from . import dpx_functions

class DPXTestpulse:
    def __init__(self,
        dpx,
        dpf: dpx_functions.DPXFunctions
    ):
        self.dpx = dpx
        self.dpf = dpf

    def start(self, columns=None):
        """Start using test pulses. Sets confbits to use analog
        test pulses. Use `columns` parameter if only specific columns
        shall be used. Columns are provided as list of integers. If
        set to `None`, all columns are used
        """

        if columns is None:
            column_range = range(16)
        else:
            column_range = columns

        # Confbits from config
        conf_bits = [int(conf_bit, 16) for conf_bit in wrap(self.dpx.conf_bits, 2)]
        conf_bits = np.asarray( conf_bits ).reshape( (16, 16) )

        # Set confbits to analog test pulses for selected columns
        for column in column_range:
            conf_bits[column] = [getattr(dpx_support._conf_bits, 'testbit_analog')] * 16

        # Write to DPX
        conf_bits_str = ''.join( ['%02x' % conf_bit for conf_bit in conf_bits.flatten()] )
        self.dpf.write_conf_bits( conf_bits_str )

    def stop(self):
        """Stop using test pulses. Resets confbits to their
        initial state
        """
        self.dpf.write_conf_bits( self.dpx.conf_bits )

    def get_test_pulse_voltage(self, dac):
        """Get peripherydac with adjusted fine voltage of testpulses"""

        # Check limits of input dac
        assert dac >= 0, 'Minimum THL value must be at least 0'
        assert dac <= 0x1ff, 'Maximum THL value mustn\'t be greater than %d' % 0x1ff

        # Get periphery dac
        periphery_dacs = int(self.dpx.periphery_dacs, 16)

        # Delete current values
        periphery_dacs &= ~(0xff << 32)   # coarse
        periphery_dacs &= ~(0x1ff << 16)  # fine

        # Adjust fine voltage only
        periphery_dacs |= (dac << 16)
        periphery_dacs |= (0xff << 32)

        return '%032x' % periphery_dacs

    def set_test_pulse_voltage(self, dac):
        """Set peripherydac with adjusted fine voltage of testpulses"""
        periphery_dacs = self.get_test_pulse_voltage(dac)
        self.dpf.write_periphery( periphery_dacs )

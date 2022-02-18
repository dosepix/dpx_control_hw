import configparser

def read_config(config_fn):
    """Load config file and return corresponding DPX codes

    Parameters
    ----------
    config_fn : str
        Path of the config file

    Returns
    -------
    peripherys :
    OMR : 
    conf_bits : 
    pixel_DACs :
    bin_edges :
    """

    config = configparser.ConfigParser()
    config.read(config_fn)

    # Mandatory sections
    section_list = ['Peripherydac', 'OMR', 'Equalisation']

    # Check if set, else throw error
    for section in config.sections():
        assert section in section_list, \
            'Config: %s is a mandatory section and has to be specified' % section

    # Read Peripherydac
    peripherys = None
    THL = None
    if 'code' in config['Peripherydac']:
        peripherys_str = config['Peripherydac']
        peripherys = peripherys_str[:-4]
        THL = peripherys_str[-4:]
    else:
        periphery_dac_dict = {}
        periphery_dac_codes = [
            'V_ThA',
            'V_tpref_fine',
            'V_tpref_coarse',
            'I_tpbufout',
            'I_tpbufin',
            'I_disc1',
            'I_disc2',
            'V_casc_preamp',
            'V_gnd',
            'I_preamp',
            'V_fbk',
            'I_krum',
            'I_pixeldac',
            'V_casc_reset']
        for periphery_dac_code in periphery_dac_codes:
            assert periphery_dac_code in config['Peripherydac'], \
                'Config: %s has to be specified in OMR section!' % periphery_dac_code

            periphery_dac_dict[periphery_dac_code] = int(
                float(config['Peripherydac'][periphery_dac_code]))
        
        peripherys = perihery_DACs_dict_to_code(periphery_dac_dict)[:-4]
        THL = '%04x' % periphery_dac_dict['V_ThA']

    # Read OMR
    OMR = None
    if 'code' in config['OMR']:
        OMR = config['OMR']['code']
    else:
        OMR_list = []
        OMR_codes = [
            'OperationMode',
            'GlobalShutter',
            'PLL',
            'Polarity',
            'AnalogOutSel',
            'AnalogInSel',
            'OMRDisableColClkGate']
        for OMR_code in OMR_codes:
            assert OMR_code in config['OMR'], \
                'Config: %s has to be specified in OMR section!' % OMR_code
            OMR_list.append(config['OMR'][OMR_code])

        OMR = OMR_list

    # Equalisation
    # conf_bits - optional field
    conf_bits = None
    if 'conf_bits' in config['Equalisation']:
        conf_bits = config['Equalisation']['conf_bits']
    else:
        # Use all pixels
        conf_bits = '00' * 256

    # pixel_DAC
    assert 'pixel_DAC' in config['Equalisation'], \
        'Config: pixel_DAC has to be specified in Equalisation section!'
    pixel_DACs = config['Equalisation']['pixel_DAC']

    # bin_edges - optional field
    bin_edges = None
    if 'bin_edges' in config['Equalisation']:
        bin_edges = config['Equalisation']['bin_edges']

    return peripherys, THL, OMR, conf_bits, pixel_DACs, bin_edges

def write_config(self, config_fn, 
    peripherys, THL, OMR, conf_bits, pixel_DACs, bin_edges):
    config = configparser.ConfigParser()

    # Get periphery DAC code and split to dictionary
    periphery_dict = self.split_perihpery_DACs(peripherys + THL)
    periphery_dict['V_ThA'] = int(THL, 16)
    config['Peripherydac'] = periphery_dict

    if not isinstance(OMR, str):
        OMR_code_list = [
            'OperationMode',
            'GlobalShutter',
            'PLL',
            'Polarity',
            'AnalogOutSel',
            'AnalogInSel',
            'OMRDisableColClkGate']
        config['OMR'] = {OMR_code: OMR[i]
                            for i, OMR_code in enumerate(OMR_code_list)}
    else:
        config['OMR'] = {'code': OMR}

    config['Equalisation'] = {
        'pixel_DAC': pixel_DACs,
        'conf_bits': conf_bits,
        'bin_edges': ''.join(bin_edges)
    }

    with open(config_fn, 'w') as config_file:
        config.write(config_file)

def perihery_DACs_dict_to_code(d, perc=False):
    """Convert dictionary containing the periphery DACs to code
    that is understood by DPX

    Parameters
    ----------
    d : dict
        Dictionary containing the periphery DACs

    Returns
    -------
    out : str
        Hexidecimal string representing the periphery DAC code
    """

    if perc:
        perc_eight_bit = float(2**8)
        perc_nine_bit = float(2**9)
        perc_thirteen_bit = float(2**13)
    else:
        perc_eight_bit, perc_nine_bit, perc_thirteen_bit = 1, 1, 1

    code = 0
    code |= int(d['V_ThA'] * perc_thirteen_bit)
    code |= (int(d['V_tpref_fine'] * perc_nine_bit) << 25 - 9)
    code |= (int(d['V_tpref_coarse'] * perc_eight_bit) << 40 - 8)
    code |= (int(d['I_tpbufout'] * perc_eight_bit) << 48 - 8)
    code |= (int(d['I_tpbufin'] * perc_eight_bit) << 56 - 8)
    code |= (int(d['I_disc2'] * perc_eight_bit) << 64 - 8)
    code |= (int(d['I_disc1'] * perc_eight_bit) << 72 - 8)
    code |= (int(d['V_casc_preamp'] * perc_eight_bit) << 80 - 8)
    code |= (int(d['V_gnd'] * perc_eight_bit) << 88 - 8)
    code |= (int(d['I_preamp'] * perc_eight_bit) << 96 - 8)
    code |= (int(d['V_fbk'] * perc_eight_bit) << 104 - 8)
    code |= (int(d['I_krum'] * perc_eight_bit) << 112 - 8)
    code |= (int(d['I_pixeldac'] * perc_eight_bit) << 120 - 8)
    code |= (int(d['V_casc_reset'] * perc_eight_bit) << 128 - 8)

    return '%04x' % code

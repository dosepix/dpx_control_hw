# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=cell-var-from-loop
import os
import json
import configparser

import numpy as np
import scipy.optimize

from . import support

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

    peripherys = read_periphery_config(config)
    omr = read_omr_config(config)

    # Equalisation
    # conf_bits - optional field
    conf_bits = None
    if 'conf_bits' in config['Equalisation']:
        conf_bits = config['Equalisation']['conf_bits']
    else:
        # Use all pixels
        conf_bits = '00' * 256

    # pixel_DAC
    assert 'pixel_dac' in config['Equalisation'], \
        'Config: pixel_dac has to be specified in Equalisation section!'
    pixel_dacs = config['Equalisation']['pixel_dac']

    # bin_edges - optional field
    bin_edges = None
    if 'bin_edges' in config['Equalisation']:
        bin_edges = config['Equalisation']['bin_edges']

    return {
        'peripherys': peripherys,
        'omr': omr,
        'conf_bits': conf_bits,
        'pixel_dacs': pixel_dacs,
        'bin_edges': bin_edges
    }

def read_periphery_config(config : configparser.ConfigParser):
    # Read Peripherydac
    peripherys = None
    if 'code' in config['Peripherydac']:
        peripherys = config['Peripherydac']
    else:
        periphery_dac_dict = {}
        periphery_dac_codes = [
            'v_tha',
            'v_tpref_fine',
            'v_tpref_coarse',
            'i_tpbufout',
            'i_tpbufin',
            'i_disc1',
            'i_disc2',
            'v_casc_preamp',
            'v_gnd',
            'i_preamp',
            'v_fbk',
            'i_krum',
            'i_pixeldac',
            'v_casc_reset']
        for periphery_dac_code in periphery_dac_codes:
            assert periphery_dac_code in config['Peripherydac'], \
                'Config: %s has to be specified in OMR section!' % periphery_dac_code

            periphery_dac_dict[periphery_dac_code] = int(
                float(config['Peripherydac'][periphery_dac_code]))

        peripherys = support.perihery_dacs_dict_to_code(periphery_dac_dict)
    return peripherys

def read_omr_config(config : configparser.ConfigParser):
    # Read OMR
    omr = None
    if 'code' in config['OMR']:
        omr = config['OMR']['code']
    else:
        omr_list = []
        omr_codes = [
            'OperationMode',
            'GlobalShutter',
            'PLL',
            'Polarity',
            'AnalogOutSel',
            'AnalogInSel',
            'OMRDisableColClkGate']
        for omr_code in omr_codes:
            assert omr_code in config['OMR'], \
                'Config: %s has to be specified in OMR section!' % omr_code
            omr_list.append(config['OMR'][omr_code])

        omr = omr_list
    return omr

def write_config(config_fn, params_d):
    config = configparser.ConfigParser()

    # Get periphery DAC code and split to dictionary
    periphery_dict = support.split_perihpery_dacs(params_d['peripherys'])
    config['Peripherydac'] = periphery_dict

    if not isinstance(params_d['omr'], str):
        omr_code_list = [
            'OperationMode',
            'GlobalShutter',
            'PLL',
            'Polarity',
            'AnalogOutSel',
            'AnalogInSel',
            'OMRDisableColClkGate']
        config['OMR'] = {omr_code: params_d['omr'][i]
                            for i, omr_code in enumerate(omr_code_list)}
    else:
        config['OMR'] = {'code': params_d['omr']}

    config['Equalisation'] = {
        'pixel_DAC': params_d['pixel_dacs'],
        'conf_bits': params_d['conf_bits']
        # 'bin_edges': ''.join(params_d['bin_edges'])
    }

    with open(config_fn, 'w') as config_file:
        config.write(config_file)

def read_thl_calibration(thl_calib_file):
    # Load THL calibration data
    if (thl_calib_file is None) or\
        (not os.path.isfile(thl_calib_file)) or\
        (not thl_calib_file.endswith('.json')):
        print('Provide a thl calibration file in json-format!' % thl_calib_file)
        return None, None, None, None

    with open(thl_calib_file, 'r') as file:
        thl_calib_d = json.load(file)
        thl_low, thl_high, thl_fit_params, thl_edges = load_thl_edges(thl_calib_d)
    return thl_low, thl_high, thl_fit_params, thl_edges

def load_thl_edges(thl_calib_d):
    thl_low, thl_high, thl_fit_params = thl_calib_to_edges(thl_calib_d)

    # Rounding
    thl_low = np.ceil(thl_low)
    thl_high = np.floor(thl_high)

    thl_edges = []
    for idx, thl_low_elm in enumerate(thl_low):
        thl_edges += list( np.arange(thl_low_elm, thl_high[idx] + 1) )

    return thl_low, thl_high, thl_fit_params, thl_edges

def thl_calib_to_edges(thl_calib_d, thres=100):
    volt, thl = thl_calib_d['Volt'], thl_calib_d['ADC']

    # Sort by THL
    thl, volt = zip(*sorted(zip(thl, volt)))

    # Find edges by taking derivative
    edges = np.argwhere(abs(np.diff(volt)) > thres).flatten() + 1

    # Store fit results in dict
    edge_d = {}

    edges = edges.tolist()
    edges.insert(0, 0)
    edges.append( 8190 )

    thl_edges_low, thl_edges_high = [0], []

    # First edge is described by error function
    thl_range = np.asarray( thl[edges[0]:edges[1]] )
    volt_range = np.asarray( volt[edges[0]:edges[1]] )
    popt_zero, _ = scipy.optimize.curve_fit(support.erf_std_fit, thl_range, volt_range)
    edge_d[0] = popt_zero

    for idx in range(1, len(edges) - 2):
        # Succeeding section
        thl_range = np.asarray( thl[edges[idx]:edges[idx+1]] )
        volt_range = np.asarray( volt[edges[idx]:edges[idx+1]] )

        popt, _ = scipy.optimize.curve_fit(support.linear_fit, thl_range, volt_range)
        slope1, slope2, intercept1, intercept2 = popt_zero[0], popt[0], popt_zero[1], popt[1]
        edge_d[idx] = popt

        # Get central position
        # Calculate intersection to get edges
        if idx == 1:
            volt_center = 0.5 * (support.erf_std_fit(edges[idx], *popt_zero) +\
                support.linear_fit(edges[idx], slope2, intercept2))
            thl_edges_high.append(
                scipy.optimize.fsolve(
                    lambda x: support.erf_std_fit(x, *popt_zero) - volt_center, 100)[0] )
        else:
            volt_center = 1. / (slope1 + slope2) *\
                (2 * edges[idx] * slope1 * slope2 + intercept1 * slope1 + intercept2 * slope2)
            thl_edges_high.append( (volt_center - intercept1) / slope1 )

        thl_edges_low.append( (volt_center - intercept2) / slope2 )
        popt_zero = popt

    thl_edges_high.append( 8190 )
    return thl_edges_low, thl_edges_high, edge_d

def load_bin_edges(
        bin_edges_fn=None,
        params_d=None
    ):
    # Check if bin edges are given as list or file
    if bin_edges_fn is not None:
        # List
        if isinstance(bin_edges_fn, list):
            bin_edges = bin_edges_fn
        else:
            if bin_edges_fn.endswith('.json'):
                with open(bin_edges_fn, 'r') as bin_edges_file:
                    bin_edges = json.load( bin_edges_file )
            else:
                return None

        # If same bin edges are used for all pixels
        if len(bin_edges) < 256:
            bin_edges = np.asarray([bin_edges] * 256).T

        if params_d is None:
            return get_single_thresholds(bin_edges)
        return get_single_thresholds_energy(params_d, bin_edges)

    # Standard bin edges
    gray_idx = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]
    cmd_total = []
    for bin_edge in range(16):
        cmd = ('%01x' % gray_idx[bin_edge] + '%03x' % (20 * bin_edge + 15)) * 256
        cmd_total.append( cmd )
    return cmd_total

def get_single_thresholds(bin_edges):
    # The indices of the bins are specified via the following gray code
    gray_idx = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]

    cmd_total = []
    for idx, gray in enumerate(gray_idx):
        # Construct command
        cmd = ''.join( [
            ('%01x' % gray) + ('%03x' % int(bin_edge)) for bin_edge in bin_edges[idx]
            ] )
        cmd_total.append( cmd )
    return cmd_total

def get_single_thresholds_energy(
        params_d,
        bin_edges
    ):
    # Fill missing parameters with mean values of all pixels
    if len(params_d) != 256:
        params_d = support.fill_param_dict(params_d)

    bin_edges = np.asarray( bin_edges ).T
    bin_edges_list = []
    for pixel in sorted(params_d.keys()):
        params = params_d[pixel]

        bin_edges_energy = bin_edges[pixel]
        bin_edges_tot = support.energy_to_tot(
            bin_edges_energy, params['a'], params['b'], params['c'], params['t']
        )

        # Round the values - do not use floor function as this leads to bias
        bin_edges_tot = np.around( bin_edges_tot )
        bin_edges_tot = np.nan_to_num( bin_edges_tot )
        bin_edges_tot[bin_edges_tot < 0] = 0
        bin_edges_tot[bin_edges_tot > 4095] = 4095
        bin_edges_list.append( bin_edges_tot )

    # Transpose matrix to get pixel values
    bin_edges_list = np.asarray( bin_edges_list ).T
    return get_single_thresholds( bin_edges_list )

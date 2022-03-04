import os
import json
import numpy as np

def val_to_idx(pixel_dacs, gauss_dict, noise_thl,
    thl_edges_low=None, thl_edges_high=None, thl_fit_params=None):
    # Transform values to indices
    mean_dict = {}
    for pixel_dac in pixel_dacs:
        idxs = np.asarray(
            [get_volt_from_thl_fit(thl_edges_low, thl_edges_high, thl_fit_params, elm) if elm \
            else np.nan for elm in gauss_dict[pixel_dac] ], dtype=np.float)
        mean_dict[pixel_dac] = np.nanmean(idxs)

        for pixel in range(256):
            elm = noise_thl[pixel_dac][pixel]
            if elm:
                noise_thl[pixel_dac][pixel] = get_volt_from_thl_fit(
                    thl_edges_low, thl_edges_high, thl_fit_params, elm)
            else:
                noise_thl[pixel_dac][pixel] = np.nan

    return mean_dict, noise_thl

def get_volt_from_thl_fit(thl_edges_low, thl_edges_high, thl_fit_params, thl):
    if (thl_edges_low is None) or (len(thl_edges_low) == 0):
        return thl

    edges = zip(thl_edges_low, thl_edges_high)
    for idx, edge in enumerate(edges):
        if edge[1] > thl >= edge[0]:
            params = thl_fit_params[idx]
            if idx == 0:
                return erf_std_fit(thl, *params)
            return linear_fit(thl, *params)

def get_noise_level(counts_dict, thl_range, pixel_dacs=['00', '3f'], noise_limt=0):    
    # Get noise THL for each pixel
    noise_thl = {key: np.zeros(256) for key in pixel_dacs}
    gauss_dict = {key: [] for key in pixel_dacs}

    # Loop over each pixel in countsDict
    for pixel_dac in pixel_dacs:
        for thl in thl_range:
            if not thl in counts_dict[pixel_dac].keys():
                continue

            for pixel in range(256):
                if counts_dict[pixel_dac][thl][pixel] > noise_limt:
                    noise_thl[pixel_dac][pixel] = thl
                    gauss_dict[pixel_dac].append(thl)

    return gauss_dict, noise_thl

# === Periphyery DACs ===
def split_perihpery_dacs(code, perc=False, show=False):
    if perc:
        perc_eight_bit = float(2**8)
        perc_nine_bit = float(2**9)
        perc_thirteen_bit = float(2**13)
    else:
        perc_eight_bit, perc_nine_bit, perc_thirteen_bit = 1, 1, 1

    if show:
        print('Periphery DAC code:')
        print(code)
        print()
    code = format(int(code, 16), 'b')
    d_periphery = {'v_tha': int(code[115:], 2) / perc_thirteen_bit,
            'v_tpref_fine': int(code[103:112], 2) / perc_nine_bit,
            'v_tpref_coarse': int(code[88:96], 2) / perc_eight_bit,
            'i_tpbufout': int(code[80:88], 2) / perc_eight_bit,
            'i_tpbufin': int(code[72:80], 2) / perc_eight_bit,
            'i_disc2': int(code[64:72], 2) / perc_eight_bit,
            'i_disc1': int(code[56:64], 2) / perc_eight_bit,
            'v_casc_preamp': int(code[48:56], 2) / perc_eight_bit,
            'v_gnd': int(code[40:48], 2) / perc_eight_bit,
            'i_preamp': int(code[32:40], 2) / perc_eight_bit,
            'v_fbk': int(code[24:32], 2) / perc_eight_bit,
            'i_krum': int(code[16:24], 2) / perc_eight_bit,
            'i_pixeldac': int(code[8:16], 2) / perc_eight_bit,
            'v_casc_reset': int(code[:8], 2) / perc_eight_bit}

    if show:
        print('PeripheryDAC values in ', end='')
        if perc:
            print('percent:')
        else:
            print('DAC:')

        for key, value in d_periphery.items():
            if perc:
                print(key, value * 100)
            else:
                print(key, int( value ))
        print()

    return d_periphery

def perihery_dacs_dict_to_code(d_periphery, perc=False):
    if perc:
        perc_eight_bit = float(2**8)
        perc_nine_bit = float(2**9)
        perc_thirteen_bit = float(2**13)
    else:
        perc_eight_bit, perc_nine_bit, perc_thirteen_bit = 1, 1, 1

    code = 0
    code |= int(d_periphery['v_tha'] * perc_thirteen_bit)
    code |= (int(d_periphery['v_tpref_fine'] * perc_nine_bit) << 25 - 9)
    code |= (int(d_periphery['v_tpref_coarse'] * perc_eight_bit) << 40 - 8)
    code |= (int(d_periphery['i_tpbufout'] * perc_eight_bit) << 48 - 8)
    code |= (int(d_periphery['i_tpbufin'] * perc_eight_bit) << 56 - 8)
    code |= (int(d_periphery['i_disc2'] * perc_eight_bit) << 64 - 8)
    code |= (int(d_periphery['i_disc1'] * perc_eight_bit) << 72 - 8)
    code |= (int(d_periphery['v_casc_preamp'] * perc_eight_bit) << 80 - 8)
    code |= (int(d_periphery['v_gnd'] * perc_eight_bit) << 88 - 8)
    code |= (int(d_periphery['i_preamp'] * perc_eight_bit) << 96 - 8)
    code |= (int(d_periphery['v_fbk'] * perc_eight_bit) << 104 - 8)
    code |= (int(d_periphery['i_krum'] * perc_eight_bit) << 112 - 8)
    code |= (int(d_periphery['i_pixeldac'] * perc_eight_bit) << 120 - 8)
    code |= (int(d_periphery['v_casc_reset'] * perc_eight_bit) << 128 - 8)

    return '%04x' % code

# === File functions ===
def make_directory(directory):
    while os.path.isdir(directory):
        dir_front = directory.split('/')[0]
        dir_front_split = dir_front.split('_')
        if len(dir_front_split) >= 2:
            if dir_front_split[-1].isdigit():
                dir_num = int(dir_front_split[-1]) + 1
                directory = ''.join(dir_front_split[:-1]) + '_' + str(dir_num) + '/'
            else:
                directory = dir_front + '_1/'
        else:
            directory = dir_front + '_1/'

    os.makedirs(directory)
    return directory

def json_dump(out_dict, out_fn, overwrite=False):
    file_ending = '.json'

    # Check if file already exists
    if not overwrite:
        while os.path.isfile(out_fn):
            # Split dir and file
            out_fn_split = out_fn.split('/')
            if len(out_fn_split) >= 2:
                directory, out_fn = out_fn_split[:-1], out_fn_split[-1]
            else:
                directory = None

            out_fn_front = out_fn.split('.')[0]
            out_fn_front_split = out_fn_front.split('_')
            if len(out_fn_front_split) >= 2:
                if out_fn_front_split[-1].isdigit():
                    fn_num = int( out_fn_front_split[-1] ) + 1
                    out_fn = ''.join(out_fn_front_split[:-1]) + '_' + str(fn_num) + file_ending
                else:
                    out_fn = out_fn_front + '_1' + file_ending
            else:
                out_fn = out_fn_front + '_1' + file_ending

            # Reattach directory
            if directory:
                out_fn = '/'.join(directory + [out_fn])

    with open(out_fn, 'w') as out_file:
        json.dump(out_dict, out_file, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# === Fit functions ===
def linear_fit(x, m, t):
    return m * x + t

def erf_std_fit(x, a, b, c, d):
    from scipy.special import erf
    return a * (erf((x - b) / c) + 1) + d

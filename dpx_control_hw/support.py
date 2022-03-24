import os
import json
import numpy as np

def fill_param_dict(param_d):
    '''
    Fill missing entries with mean parameters
    '''
    # Get mean factors
    mean_dict = {}
    key_list = ['a', 'c', 'b', 't']

    for key in key_list:
        val_list = []
        for pixel in param_d.keys():
            val_list.append( param_d[pixel][key] )
        val_list = np.asarray(val_list)
        val_list[abs(val_list) == np.inf] = np.nan
        mean_dict[key] = np.nanmean(val_list)

    # Find non existent entries and loop
    new_param_d = {}
    for pixel in set(np.arange(256)) - set(param_d.keys()):
        new_param_d[pixel] = {val: mean_dict[val] for val in key_list}

    return new_param_d

# === Periphyery DACs ===
def split_perihpery_dacs(
        code,
        perc=False,
        show=False
    ):
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

def perihery_dacs_dict_to_code(
        d_periphery,
        perc=False
    ):
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
                directory = '_'.join(dir_front_split[:-1]) + '_' + str(dir_num) + '/'
            else:
                directory = dir_front + '_1/'
        else:
            directory = dir_front + '_1/'

    os.makedirs(directory)
    return directory

def json_dump(
        out_dict,
        out_fn,
        overwrite=False
    ):
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
                    out_fn = '_'.join(out_fn_front_split[:-1]) + '_' + str(fn_num) + file_ending
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

# === Conversion functions ===
def get_thl(p_a, p_b, p_c, p_t):
    return 1./(2*p_a) *\
        ( p_t*p_a - p_b + np.sqrt((p_b + p_t*p_a)**2 - 4*p_a*p_c) )

def energy_to_tot(val, p_a, p_b, p_c, p_t):
    return np.where(val >= get_thl(p_a, p_b, p_c, p_t),
        p_a*val + p_b + float(p_c)/(val - p_t), 0)

def tot_to_energy(val, p_a, p_b, p_c, p_t):
    return 1./(2*p_a) *\
        ( p_t*p_a + val - p_b + np.sqrt((p_b + p_t*p_a - val)**2 - 4*p_a*p_c) )

# === Fit functions ===
def linear_fit(x, m, t):
    return m * x + t

def erf_std_fit(x, a, b, c, d):
    from scipy.special import erf
    return a * (erf((x - b) / c) + 1) + d

# === ETC ===
def infinite_for():
    idx = 0
    while True:
        yield idx
        idx += 1

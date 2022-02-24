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

def get_noise_level(counts_dict, thl_range, pixel_dacs=['00', '3f'], noise_limt=3):
    # Get noise THL for each pixel
    noise_thl = {key: np.zeros(256) for key in pixel_dacs}
    gauss_dict = {key: [] for key in pixel_dacs}

    # Loop over each pixel in countsDict
    for pixel_dac in pixel_dacs:
        for thl in thl_range:
            if not thl in counts_dict[pixel_dac].keys():
                continue

            for pixel in range(256):
                if counts_dict[pixel_dac][thl][pixel] >= noise_limt and \
                    noise_thl[pixel_dac][pixel] == 0:
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
        print(code)
    code = format(int(code, 16), 'b')
    if show:
        print(code)
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
        print('PeripheryDAC values in', end='')
        if perc:
            print('percent:')
        else:
            print('DAC:')

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

# === Fit functions ===
def linear_fit(x, m, t):
    return m * x + t

def erf_std_fit(x, a, b, c, d):
    from scipy.special import erf
    return a * (erf((x - b) / c) + 1) + d

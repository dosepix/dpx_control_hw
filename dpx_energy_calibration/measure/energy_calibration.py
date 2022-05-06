#!/usr/bin/env python
import json

import dpx_control_hw as dch
import dpx_energy_calibration

# Settings
MODEL_FILE = '../models/calibration_large.h5'
PARAMETERS_FILE = '../models/parameters.json'
CONFIG = '../../measure/config_38.conf'
STOP_CONDITION = 0.01
PARAMS_FILE = 'params_file.json'

# Find port
port = dch.find_port()
if port is None:
    port = '/dev/ttyACM0'

# Connect Dosepix
dpx = dch.Dosepix(
    port_name=port,
    config_fn=CONFIG,
    thl_calib_fn=None,
    params_fn=None,
    bin_edges_fn=None
)

# Load energy calibration model
dec = dpx_energy_calibration.DPXEnergyCalibration(
    dpx,
    MODEL_FILE,
    PARAMETERS_FILE
)

# Start measurement
params, hist = dec.calibrate(
    frame_time=0,
    eval_after_frames=10,
    stop_condition=STOP_CONDITION,
    stop_condition_range=10
)

with open(PARAMS_FILE, 'w') as f:
    json.dump( dec.reformat_params(params), f)
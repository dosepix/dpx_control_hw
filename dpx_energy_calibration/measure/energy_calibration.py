#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
import dpx_control_hw as dch
import dpx_energy_calibration

# Settings
MODEL_FILE = '../models/calibration_large.h5'
CONFIG = '../../measure/config.conf'

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
    MODEL_FILE
)

dec.measure()

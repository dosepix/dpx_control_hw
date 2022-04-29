import json
import numpy as np
import matplotlib.pyplot as plt

import dpx_control_hw as dch
import tensorflow as tf
from tensorflow import keras

class DPXEnergyCalibration():
    def __init__(self,
        dpx: dch.Dosepix,
        model_file,
        parameters_file
    ):
        self.dpx = dpx

        # Model
        self.parameters_file = parameters_file
        self.p_norm = None
        self.model_file = model_file
        self.model = None
        assert self.load_model(),\
            "Failed to load model"

    def load_model(self):
        try:
            self.model = keras.models.load_model( self.model_file )
            with open(self.parameters_file, 'r') as f:
                self.p_norm = json.load(f)['p_norm']
            return True
        except:
            return False

    def predict(self, meas):
        return self.model.predict( meas )

    def measure(self,
        frame_time=0,
        eval_after_frames=10
    ):
        # Create ToT generator
        tot_gen = self.dpx.dpm.measure_tot(
            frame_time=frame_time,
            save_frames=None,
            out_dir=None,
            meas_time=None,
            make_hist=True,
            use_gui=True
        )

        assert eval_after_frames >= 1,\
            "eval_after_frames has to be greater than 0!"

        tot_range = np.arange( 400 )
        while True:
            for _ in range(eval_after_frames):
                hist = next(tot_gen)

            # Reshape measurement
            hist = np.asarray( hist )
            hist = hist[:,:400]
            hist = np.expand_dims(hist, -1)

            # Predict conversion factors for all pixels
            pred = self.predict( hist )

            # Transform to energy
            tot_x = []
            tot_y = []
            for pixel in [8]: # range(256):
                cbd = hist[pixel].flatten().tolist()
                energy = dch.support.tot_to_energy(tot_range, *pred[pixel] * self.p_norm).tolist()
                tot = (np.asarray(cbd) * np.diff([0] + energy)).tolist()
                plt.plot(energy, tot)
                plt.show()

                tot_x += energy
                tot_y += tot
            yield np.asarray(tot_x), np.asarray(tot_y)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import scipy.interpolate

import plotly.graph_objects as go
import plotly.subplots
import plotly.express as px

import dpx_control_hw as dch
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

        # Plots
        self.fig_hist = None
        self.fig_params = None

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
        eval_after_frames=10,
        stop_condition=0.001,
        stop_range=10
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

        last_params = np.zeros(4)
        deviation_history = []
        while True:
            for _ in range(eval_after_frames):
                hist_orig = next(tot_gen)

            # Reshape measurement
            hist = np.array(hist_orig, dtype=float, copy=True)
            hist = hist[:,:400]
            hist = hist.T / np.nanmax(hist, axis=1)
            hist = np.expand_dims(hist.T, -1)
            hist = np.nan_to_num( hist )

            # Predict conversion factors for all pixels
            pred = self.predict( hist )

            # Transform to energy
            tot_x, tot_y = [], []
            for pixel in range(256):
                cbd = hist[pixel].flatten().tolist()
                # Set number of ToT events with 0 to 0
                cbd[0] = 0

                # Transform ToT to energy
                energy = dch.support.tot_to_energy(tot_range, *pred[pixel] * self.p_norm)

                # Transform histogram of ToT events to energy axis
                num_tot_events = np.sum( cbd )
                if num_tot_events > 0:
                    events = (np.asarray(cbd) / np.diff([0] + energy.tolist()))
                    events = events / np.sum(events) * num_tot_events
                else:
                    events = np.zeros(len(energy))

                # Filter weird values
                energy, events = np.nan_to_num(energy), np.nan_to_num(events)
                energy_filt = energy < 70
                energy, events = energy[energy_filt], events[energy_filt]

                tot_x.append( energy )
                tot_y.append( events )

            params_d = {
                'mean': np.median(pred, axis=0),
                'std': np.std(pred, axis=0)
            }

            # Add deviation to history
            deviation = np.abs(last_params - params_d['mean'])
            deviation_history.append( deviation )
            last_params = params_d['mean']

            if len(deviation_history) >= stop_range:
                # Check for stop condition
                dev = np.mean(np.mean(deviation_history[-stop_range:], axis=0))
                if dev < stop_condition:
                    break

            yield tot_x, tot_y, params_d
        return pred * self.p_norm, hist_orig

    def calibrate(self,
        frame_time=0.05,
        eval_after_frames=100,
        stop_condition=0.01,
        stop_range=10
    ):
        # Create measurement generator
        gen = self.measure(
            frame_time=frame_time,
            eval_after_frames=eval_after_frames,
            stop_condition=stop_condition,
            stop_range=stop_range
        )

        readouts = 0
        params_history = []
        large_pixels = [pixel for pixel in range(256) if pixel % 16 not in [0, 1, 14, 15]]
        while True:
            try:
                tot_x, tot_y, params_d = next(gen)
            except StopIteration as res:
                return res.value

            # Single spectra
            for data_idx, pixel in enumerate( range(2, 14) ):
                self.fig_hist.data[data_idx]['x'] = tot_x[pixel]
                self.fig_hist.data[data_idx]['y'] = tot_y[pixel]

            # Sum spectrum
            energy_range = np.linspace(10, 70, 100)
            sum_spectrum = np.zeros(len(energy_range))

            num_pixels = 0
            for pixel in large_pixels:
                try:
                    f_intp = scipy.interpolate.interp1d(
                        tot_x[pixel],
                        tot_y[pixel],
                        bounds_error=False,
                        fill_value=0
                    )
                except ValueError:
                    continue
                num_pixels += 1
                sum_spectrum += f_intp(energy_range)
            sum_spectrum = sum_spectrum / num_pixels
            self.fig_hist.data[-1]['x'] = energy_range
            self.fig_hist.data[-1]['y'] = sum_spectrum

            # Parameters
            params_history.append( params_d )
            params = np.asarray(
                [params_history[idx]['mean'].tolist() for idx in range(len(params_history))]
            ).T

            if readouts > 0:
                for p_idx in range(4):
                    self.fig_params.data[p_idx]['x'] = np.arange(readouts - 1)
                    self.fig_params.data[p_idx]['y'] = np.abs(np.diff(params[p_idx]))

                self.fig_params.data[-1]['x'] = np.arange(readouts - 1)
                self.fig_params.data[-1]['y'] = np.mean([np.abs(np.diff(params[p_idx])) for p_idx in range(4)], axis=0)
            readouts += 1

    # === PLOTS ===
    def create_plots(self):
        # Histograms
        self.fig_hist = go.FigureWidget()
        for pixel in range(2, 14):
            self.fig_hist.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    name='Pixel %d' % pixel
                )
            )

        # Sum spectrum curve
        self.fig_hist.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                line=dict(
                    color='black',
                    dash='dash'
                ),
                name='Average'
            )
        )

        # Axes titles
        self.fig_hist.update_layout(
            xaxis_title='Deposited energy (keV)',
            yaxis_title='Normalized registered events'
        )

        # == Parameters ==
        self.fig_params = go.FigureWidget()
        p_titles = ['a', 'b', 'c', 't']
        for p_idx in range(4):
            self.fig_params.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    name='%s' % p_titles[p_idx],
                )
            )

        self.fig_params.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                line=dict(
                    color='black',
                    dash='dash'
                ),
                name='average',
            )
        )

        # Axes titles
        self.fig_params.update_layout(
            xaxis_title='Iterations',
            yaxis_title='Deviation'
        )
        self.fig_params.update_yaxes(type="log")

        return self.fig_hist, self.fig_params

    def get_plots(self):
        return self.fig_hist, self.fig_params

    @classmethod
    def thl_distribution(cls, params):
        thls = [dch.support.get_thl(*params[pixel]) for pixel in range(256)]
        fig = px.histogram(
            x=thls,
            nbins=50,
            marginal="rug"
        )
        fig.add_vline(
            x=np.nanmedian(thls),
            line=dict(
                color='black',
                dash='dash'
            )
        )

        fig.update_layout(
            xaxis_title='Threshold (deposited energy)',
            yaxis_title='Number of pixels'
        )

        return fig

    @classmethod
    def calib_curve_distribution(cls, params):
        thls = [dch.support.get_thl(*params[pixel]) for pixel in range(256)]
        thls = np.asarray( thls )

        energy_range = np.linspace(0, 60, 1000)
        energies, tots = [], []
        for pixel in range(256):
            tot = dch.support.energy_to_tot(energy_range, *params[pixel])
            filt = tot >= thls[pixel]

            energies += energy_range[filt].tolist()
            tots += tot[filt].tolist()
        
        fig = px.density_heatmap(
            x=energies,
            y=tots,
            nbinsx=200,
            nbinsy=300,
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            xaxis_title='Deposited energy (keV)',
            yaxis_title='ToT (10 ns)'
        )

        return fig

    @classmethod
    def params_distribution(cls, params):
        fig = plotly.subplots.make_subplots(rows=2, cols=2)

        titles = ['a', 'b', 'c', 't']
        for p_idx in range(4):
            trace = go.Histogram(
                x=params[:,p_idx],
                nbinsx=50,
                name=titles[p_idx]
            )

            fig.append_trace(
                trace,
                row=p_idx % 2 + 1,
                col=p_idx // 2 + 1
            )
        return fig

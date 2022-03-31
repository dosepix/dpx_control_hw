#!/usr/bin/env python
import unittest
import pathlib as pl
import os
import shutil
import dpx_control_hw as dch

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except:
            pass

class TestCaseBase(unittest.TestCase):
    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

class TestMeasurementFunctions(TestCaseBase):
    def setUp(self):
        self.dpx = dch.Dosepix(
            port_name=None,
            config_fn=None,
            thl_calib_fn=None,
            params_fn=None,
            bin_edges_fn=None
        )

        # Clean directory for test files
        self.output_dir = './test_output/'
        clear_directory(self.output_dir)

    def test_tot_measurement(self):
        print('=== Testing ToT-measurement ===')
        # Run twice to create two output files
        make_hist = [True, False, True]
        for idx in range(3):
            self.dpx.dpm.measure_tot(
                frame_time=1,
                save_frames=None,
                out_dir=self.output_dir + 'tot_measurement/',
                meas_time=0.1,
                make_hist=make_hist[idx],
                use_gui=False
            )
            print()
        print()

        # Check if outputs exist
        self.assertIsFile(self.output_dir + 'tot_measurement/tot_measurement.json')
        self.assertIsFile(self.output_dir + 'tot_measurement_1/tot_measurement_1.json')
        self.assertIsFile(self.output_dir + 'tot_measurement_2/tot_measurement_2.json')

    def test_dosi_measurement(self):
        print('=== Testing Dosi-measurement ===')
        for _ in range(3):
            self.dpx.dpm.measure_dosi(
                frame_time=0.1,
                frames=3,
                freq=False,
                out_fn=self.output_dir + 'dose_measurement.json',
                use_gui=False
            )
            print()
        print()

        self.assertIsFile(self.output_dir + 'dose_measurement.json')
        self.assertIsFile(self.output_dir + 'dose_measurement_1.json')
        self.assertIsFile(self.output_dir + 'dose_measurement_2.json')

    def test_integration_measurement(self):
        print('=== Testing Integration-measurement ===')
        for _ in range(3):
            self.dpx.dpm.measure_integration(
                meas_time=1,
                frame_time=0.1,
                integration=True,
                single_values=False,
                use_gui=False
            )
            print()
        print()

        self.assertIsFile(self.output_dir + 'integration_measurement.json')
        self.assertIsFile(self.output_dir + 'integration_measurement_1.json')
        self.assertIsFile(self.output_dir + 'integration_measurement_1.json')

if __name__ == '__main__':
    unittest.main()

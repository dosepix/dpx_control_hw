#!/usr/bin/env python
import unittest
import pathlib as pl
import os
import glob

import dpx_control_hw as dch

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
        files = glob.glob(self.output_dir)
        print(files)
        for file in files:
            if file == self.output_dir:
                continue
            os.remove(self.output_dir + file)

    def test_tot_measurement(self):
        # Run twice to create two output files
        for _ in range(2):
            self.dpx.dpm.measure_tot(
                frame_time=1,
                save_frames=None,
                out_dir=self.output_dir + 'tot_measurement/',
                meas_time=3,
                make_hist=True,
                use_gui=False
            )

        # Check if outputs exist
        self.assertIsFile(self.output_dir + 'tot_measurement/tot_measurement.json')
        self.assertIsFile(self.output_dir + 'tot_measurement_1/tot_measurement_1.json')

if __name__ == '__main__':
    unittest.main()

"""Tests for auxiliary dataset handling."""

import numpy as np
import xarray as xr

from pyturb.auxiliary import infer_sample_rate


class TestInferSampleRate:
    def test_regular_one_hz(self):
        t = np.datetime64("2026-01-01T00:00:00") + np.arange(120) * np.timedelta64(
            1, "s"
        )
        ds = xr.Dataset(coords={"time": t})
        np.testing.assert_allclose(infer_sample_rate(ds), 1.0)

    def test_regular_quarter_hz(self):
        # A glider-style CTD sampled every 4 seconds.
        t = np.datetime64("2026-01-01T00:00:00") + np.arange(60) * np.timedelta64(
            4, "s"
        )
        ds = xr.Dataset(coords={"time": t})
        np.testing.assert_allclose(infer_sample_rate(ds), 0.25)

    def test_robust_to_a_single_gap(self):
        # One dropped sample must not skew the median rate.
        t = np.datetime64("2026-01-01T00:00:00") + np.arange(100) * np.timedelta64(
            1, "s"
        )
        t = np.delete(t, 50)  # a 2-second gap where sample 50 was dropped
        ds = xr.Dataset(coords={"time": t})
        np.testing.assert_allclose(infer_sample_rate(ds), 1.0)

    def test_custom_time_variable_name(self):
        t = np.datetime64("2026-01-01T00:00:00") + np.arange(30) * np.timedelta64(
            2, "s"
        )
        ds = xr.Dataset(coords={"sample_time": t})
        np.testing.assert_allclose(infer_sample_rate(ds, time_var="sample_time"), 0.5)

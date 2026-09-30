"""Tests for QC-aware depth binning: each bin averages its best windows."""

import numpy as np
import xarray as xr

from pyturb.processing import _bin_var_group

BINS = np.array([0.0, 10.0, 20.0])


def _bin(eps, qc, depth):
    ds = xr.Dataset(
        {
            "pressure": ("time", np.asarray(depth, dtype="f8")),
            "eps": ("time", np.asarray(eps, dtype="f8")),
            "eps_qc": ("time", np.asarray(qc, dtype="i1")),
        }
    )
    return _bin_var_group(
        ds, "time", "pressure", ["eps", "eps_qc"], {}, BINS, 45.0, "depth", False
    )


class TestBestWindowBinning:
    def test_bad_windows_dropped_when_usable_exist(self):
        out = _bin(eps=[1e-9, 3e-9, 1e-6], qc=[1, 3, 4], depth=[5, 5, 5])
        np.testing.assert_allclose(out["eps"].values[0], 2e-9)
        assert out["eps_qc"].values[0] == 3
        assert out["eps_n"].values[0] == 2

    def test_falls_back_to_bad_windows(self):
        out = _bin(eps=[1e-10, 3e-10], qc=[4, 4], depth=[5, 5])
        np.testing.assert_allclose(out["eps"].values[0], 2e-10)
        assert out["eps_qc"].values[0] == 4
        assert out["eps_n"].values[0] == 2

    def test_bins_selected_independently(self):
        out = _bin(eps=[1e-9, 1e-6, 5e-10], qc=[1, 4, 4], depth=[5, 5, 15])
        np.testing.assert_allclose(out["eps"].values, [1e-9, 5e-10])
        np.testing.assert_array_equal(out["eps_qc"].values, [1, 4])
        np.testing.assert_array_equal(out["eps_n"].values, [1, 1])

    def test_bad_without_value_stays_bad(self):
        out = _bin(eps=[np.nan, np.nan, 1e-9], qc=[4, 9, 1], depth=[5, 5, 15])
        assert np.isnan(out["eps"].values[0])
        np.testing.assert_array_equal(out["eps_qc"].values, [4, 1])
        np.testing.assert_array_equal(out["eps_n"].values, [0, 1])

    def test_count_attrs(self):
        attrs = _bin(eps=[1e-9], qc=[1], depth=[5])["eps_n"].attrs
        assert attrs["long_name"] == "Number of windows averaged into eps"
        assert attrs["units"] == "1"
        assert "comment" in attrs

    def test_empty_bin_is_missing(self):
        out = _bin(eps=[1e-9], qc=[1], depth=[5])
        assert np.isnan(out["eps"].values[1])
        assert out["eps_qc"].values[1] == 9
        assert out["eps_n"].values[1] == 0

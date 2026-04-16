"""Tests for deconvolution (deconvolve.py)."""

import numpy as np

from pyturb._pfile import deconvolve


class TestDeconvolve:
    """Test deconvolution of pre-emphasized signals."""

    def test_output_length_matches_input(self):
        fs = 512.0
        diff_gain = 0.5
        x = np.random.randn(1000)
        result = deconvolve(x, fs, diff_gain)
        assert len(result) == len(x)

    def test_output_is_finite(self):
        fs = 512.0
        diff_gain = 0.5
        x = np.random.randn(1000)
        result = deconvolve(x, fs, diff_gain)
        assert np.all(np.isfinite(result))

    def test_with_low_res_signal(self):
        fs = 512.0
        diff_gain = 0.5
        n_fast = 1000
        n_slow = 125  # 8:1 ratio

        X = np.sin(np.linspace(0, 2 * np.pi, n_slow))
        X_dX = np.sin(np.linspace(0, 2 * np.pi, n_fast))

        result = deconvolve(X_dX, fs, diff_gain, X)
        assert len(result) == n_fast
        assert np.all(np.isfinite(result))

    def test_with_real_data(self, raw_data):
        """Test deconvolution on actual T1_dT1 with T1."""
        cfg = raw_data["cfgobj"]
        params = cfg.get_channel_params("T1_dT1")
        diff_gain = float(params["diff_gain"])

        result = deconvolve(
            raw_data["T1_dT1"].astype(np.float64),
            raw_data["fs_fast"],
            diff_gain,
            raw_data["T1"].astype(np.float64),
        )
        assert len(result) == len(raw_data["T1_dT1"])
        assert np.all(np.isfinite(result))

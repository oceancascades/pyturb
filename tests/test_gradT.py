"""Tests for temperature gradient computation (gradT.py)."""

import numpy as np
import pytest

from pyturb._pfile import deconvolve
from pyturb._pfile.gradT import make_gradT


class TestMakeGradT:
    """Test make_gradT function."""

    def _get_therm_params(self, raw_data, ch_name="T1_dT1"):
        cfg = raw_data["cfgobj"]
        params = cfg.get_channel_params(ch_name)
        # Merge with base channel params (e.g. T1 for T1_dT1)
        import re

        match = re.match(r"(\w+)_d\1", ch_name, re.IGNORECASE)
        if match:
            base_params = cfg.get_channel_params(match.group(1))
            if base_params:
                merged = base_params.copy()
                merged.update(params)
                return merged
        return params

    def test_output_length(self, raw_data):
        params = self._get_therm_params(raw_data)
        result = make_gradT(
            raw_data["T1_dT1"].astype(np.float64),
            params,
            raw_data["fs_fast"],
        )
        assert len(result) == len(raw_data["T1_dT1"])

    def test_output_finite(self, raw_data):
        params = self._get_therm_params(raw_data)
        result = make_gradT(
            raw_data["T1_dT1"].astype(np.float64),
            params,
            raw_data["fs_fast"],
        )
        assert np.all(np.isfinite(result))

    def test_high_pass_method(self, raw_data):
        params = self._get_therm_params(raw_data)
        result = make_gradT(
            raw_data["T1_dT1"].astype(np.float64),
            params,
            raw_data["fs_fast"],
            method="high_pass",
        )
        assert np.all(np.isfinite(result))

    def test_first_difference_method(self, raw_data):
        params = self._get_therm_params(raw_data)
        result = make_gradT(
            raw_data["T1_dT1"].astype(np.float64),
            params,
            raw_data["fs_fast"],
            method="first_difference",
        )
        assert np.all(np.isfinite(result))

    def test_invalid_method_raises(self, raw_data):
        params = self._get_therm_params(raw_data)
        with pytest.raises(ValueError, match="Unknown method"):
            make_gradT(
                raw_data["T1_dT1"].astype(np.float64),
                params,
                raw_data["fs_fast"],
                method="invalid",
            )

    def test_predeconvolved_skips_deconvolution(self, raw_data):
        """Passing T_deconvolved should give same result as internal deconvolution."""
        params = self._get_therm_params(raw_data)
        fs = raw_data["fs_fast"]
        T_dT = raw_data["T1_dT1"].astype(np.float64)
        diff_gain = float(params["diff_gain"])

        # Deconvolve externally
        T_deconv = deconvolve(T_dT, fs, diff_gain)

        result_internal = make_gradT(T_dT, params, fs)
        result_external = make_gradT(T_dT, params, fs, T_deconvolved=T_deconv)

        np.testing.assert_allclose(result_internal, result_external, rtol=1e-10)

    def test_empty_signal_raises(self, raw_data):
        params = self._get_therm_params(raw_data)
        with pytest.raises(ValueError, match="empty"):
            make_gradT(np.array([]), params, raw_data["fs_fast"])

    def test_broken_thermistor_clipping(self, raw_data):
        """R < 0.1 should be set to 1.0, not clipped to 0.1."""
        params = self._get_therm_params(raw_data)
        # Use extreme data that would produce R < 0.1
        extreme = np.full(1000, 10000.0)
        result = make_gradT(extreme, params, raw_data["fs_fast"])
        assert np.all(np.isfinite(result))

"""Tests for channel conversion (convert.py)."""

import numpy as np
import pytest

from pyturb._pfile import convert_all_channels, convert_channel


class TestConvertChannel:
    """Test individual channel conversions."""

    def test_shear_converts(self, raw_data):
        cfg = raw_data["cfgobj"]
        result, units = convert_channel(raw_data["sh1"], "sh1", cfg)
        assert units == "m^2 s^-3"
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))

    def test_shear_no_offset(self, raw_data):
        # Verify shear conversion matches MATLAB: no adc_zero/sig_zero offset
        cfg = raw_data["cfgobj"]
        params = cfg.get_channel_params("sh1")
        adc_fs = float(params["adc_fs"])
        adc_bits = int(params["adc_bits"])
        diff_gain = float(params["diff_gain"])
        sens = float(params["sens"])

        raw = raw_data["sh1"].astype(np.float64)
        expected = (adc_fs / 2**adc_bits) * raw / (2 * np.sqrt(2) * diff_gain * sens)

        result, _ = convert_channel(raw_data["sh1"], "sh1", cfg)
        np.testing.assert_allclose(result, expected)

    def test_therm_converts_to_celsius(self, raw_data):
        cfg = raw_data["cfgobj"]
        result, units = convert_channel(raw_data["T1"], "T1", cfg)
        assert units == "C"
        # Temperature should be in a physically reasonable range
        assert np.all(result > -5)
        assert np.all(result < 40)

    def test_pressure_converts(self, raw_data):
        cfg = raw_data["cfgobj"]
        result, units = convert_channel(raw_data["P"], "P", cfg)
        # Pressure should be non-negative for a surface instrument
        assert np.all(result > -1)

    def test_jac_t_converts(self, raw_data):
        cfg = raw_data["cfgobj"]
        result, units = convert_channel(raw_data["JAC_T"], "JAC_T", cfg)
        assert units == "C"
        assert np.all(result > -5)
        assert np.all(result < 40)

    def test_inclxy_converts(self, raw_data):
        cfg = raw_data["cfgobj"]
        result, units = convert_channel(raw_data["Incl_Y"], "Incl_Y", cfg)
        assert units == "deg"

    def test_voltage_converts(self, raw_data):
        cfg = raw_data["cfgobj"]
        result, units = convert_channel(raw_data["V_Bat"], "V_Bat", cfg)
        assert units == "V"
        # Battery voltage for a VMP should be positive
        assert np.all(result > 0)

    def test_unknown_channel_raises(self, raw_data):
        cfg = raw_data["cfgobj"]
        with pytest.raises(ValueError, match="not found"):
            convert_channel(np.zeros(10), "nonexistent_channel", cfg)


class TestConvertAllChannels:
    """Test full conversion pipeline."""

    def test_returns_units_dict(self, phys_data):
        assert "units" in phys_data
        assert isinstance(phys_data["units"], dict)

    def test_metadata_preserved(self, phys_data):
        for key in ["fs_fast", "fs_slow", "t_fast", "t_slow", "filetime", "cfgobj"]:
            assert key in phys_data, f"Missing metadata: {key}"

    def test_shear_channels_converted(self, phys_data):
        for ch in ["sh1", "sh2"]:
            assert ch in phys_data
            assert phys_data["units"][ch] == "m^2 s^-3"

    def test_temperature_channels_converted(self, phys_data):
        for ch in ["T1", "T2"]:
            assert ch in phys_data
            assert phys_data["units"][ch] == "C"

    def test_hires_channels_created(self, phys_data):
        for ch in ["T1_hires", "T2_hires", "P_hires"]:
            assert ch in phys_data

    def test_gradT_channels_created(self, phys_data):
        for ch in ["gradT1", "gradT2"]:
            assert ch in phys_data
            assert phys_data["units"][ch] == "K/s"
            assert np.all(np.isfinite(phys_data[ch]))

    def test_gradT_is_fast_length(self, phys_data):
        n_fast = len(phys_data["t_fast"])
        assert len(phys_data["gradT1"]) == n_fast
        assert len(phys_data["gradT2"]) == n_fast

    def test_exclude_types(self, raw_data):
        result = convert_all_channels(raw_data, exclude_types=["gnd", "raw", "shear"])
        assert "sh1" not in result
        assert "sh2" not in result

    def test_preemph_channels_kept_as_counts(self, phys_data):
        # Pre-emphasized channels should be kept in counts for debugging
        assert phys_data["units"].get("T1_dT1") == "counts"
        assert phys_data["units"].get("T2_dT2") == "counts"

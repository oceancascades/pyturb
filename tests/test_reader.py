"""Tests for P-file reading (reader.py)."""

import numpy as np
import pytest

from pyturb._pfile import SetupConfig, read_pfile


class TestReadPfile:
    """Test read_pfile returns correct metadata and channel data."""

    def test_sampling_rates(self, raw_data):
        assert raw_data["fs_fast"] == pytest.approx(512.03275)
        assert raw_data["fs_slow"] == pytest.approx(64.00409375)

    def test_time_vectors_length(self, raw_data):
        assert len(raw_data["t_fast"]) == 40960
        assert len(raw_data["t_slow"]) == 5120

    def test_time_vectors_start_at_zero(self, raw_data):
        assert raw_data["t_fast"][0] == 0.0
        assert raw_data["t_slow"][0] == 0.0

    def test_time_vectors_monotonic(self, raw_data):
        assert np.all(np.diff(raw_data["t_fast"]) > 0)
        assert np.all(np.diff(raw_data["t_slow"]) > 0)

    def test_time_vectors_consistent_with_fs(self, raw_data):
        dt_fast = np.diff(raw_data["t_fast"])
        dt_slow = np.diff(raw_data["t_slow"])
        assert dt_fast[0] == pytest.approx(1.0 / raw_data["fs_fast"])
        assert dt_slow[0] == pytest.approx(1.0 / raw_data["fs_slow"])

    def test_filetime(self, raw_data):
        ft = raw_data["filetime"]
        assert ft.year == 2026
        assert ft.month == 3
        assert ft.day == 15

    def test_header_version(self, raw_data):
        assert raw_data["header_version"] == pytest.approx(6.001)

    def test_expected_channels_present(self, raw_data):
        expected = [
            "Ax", "Ay", "T1", "T1_dT1", "T2", "T2_dT2",
            "sh1", "sh2", "P", "P_dP", "Incl_X", "Incl_Y",
            "Incl_T", "JAC_T", "JAC_C", "V_Bat", "PV", "Gnd",
        ]
        for ch in expected:
            assert ch in raw_data, f"Missing channel: {ch}"

    def test_fast_channels_have_fast_length(self, raw_data):
        n_fast = len(raw_data["t_fast"])
        for ch in ["Ax", "Ay", "T1_dT1", "T2_dT2", "sh1", "sh2"]:
            assert len(raw_data[ch]) == n_fast, f"{ch} length mismatch"

    def test_slow_channels_have_slow_length(self, raw_data):
        n_slow = len(raw_data["t_slow"])
        for ch in ["T1", "T2", "P", "P_dP", "Incl_X", "Incl_Y", "JAC_T"]:
            assert len(raw_data[ch]) == n_slow, f"{ch} length mismatch"

    def test_raw_data_is_integer(self, raw_data):
        for ch in ["sh1", "sh2", "T1", "P"]:
            assert np.issubdtype(raw_data[ch].dtype, np.integer)

    def test_config_object_present(self, raw_data):
        assert "cfgobj" in raw_data
        assert isinstance(raw_data["cfgobj"], SetupConfig)

    def test_setupfilestr_present(self, raw_data):
        assert "setupfilestr" in raw_data
        assert len(raw_data["setupfilestr"]) > 0


class TestReadPfileEdgeCases:
    """Test error handling in read_pfile."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_pfile("/nonexistent/file.p")

    def test_auto_extension(self, pfile_path):
        # Should work without .p extension
        data = read_pfile(pfile_path.with_suffix(""))
        assert "fs_fast" in data


class TestSetupConfig:
    """Test configuration parsing."""

    def test_channel_sections(self, raw_data):
        cfg = raw_data["cfgobj"]
        channels = cfg.get_section_dicts("channel")
        assert len(channels) > 0

    def test_get_channel_params(self, raw_data):
        cfg = raw_data["cfgobj"]
        params = cfg.get_channel_params("sh1")
        assert params is not None
        assert params["type"] == "shear"

    def test_get_value(self, raw_data):
        cfg = raw_data["cfgobj"]
        vehicle = cfg.get_value("instrument_info", "vehicle")
        # VMP instrument should have vehicle info
        assert vehicle is not None

    def test_get_value_default(self, raw_data):
        cfg = raw_data["cfgobj"]
        val = cfg.get_value("nonexistent", "param", default="fallback")
        assert val == "fallback"

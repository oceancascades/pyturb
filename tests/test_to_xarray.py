"""Tests for xarray/NetCDF conversion (to_xarray.py)."""

import numpy as np
import pytest

from pyturb._pfile import to_xarray


class TestToXarray:
    """Test xarray Dataset creation."""

    def test_returns_dataset(self, phys_data):
        ds = to_xarray(phys_data)
        import xarray as xr

        assert isinstance(ds, xr.Dataset)

    def test_has_time_coordinates(self, phys_data):
        ds = to_xarray(phys_data)
        assert "t_fast" in ds.coords
        assert "t_slow" in ds.coords

    def test_default_variables(self, phys_data):
        ds = to_xarray(phys_data)
        # Should include at least some of the standard variables
        found = [v for v in ["P", "sh1", "sh2", "gradT1", "gradT2"] if v in ds]
        assert len(found) > 0

    def test_custom_variables(self, phys_data):
        ds = to_xarray(phys_data, variables=["P", "sh1"])
        assert "P" in ds.data_vars
        assert "sh1" in ds.data_vars
        assert "sh2" not in ds.data_vars

    def test_fast_vars_on_fast_dim(self, phys_data):
        ds = to_xarray(phys_data, variables=["sh1"])
        assert ds["sh1"].dims == ("t_fast",)

    def test_slow_vars_on_slow_dim(self, phys_data):
        ds = to_xarray(phys_data, variables=["P"])
        assert ds["P"].dims == ("t_slow",)

    def test_global_attributes(self, phys_data):
        ds = to_xarray(phys_data)
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "fs_fast" in ds.attrs
        assert "fs_slow" in ds.attrs

    def test_config_string_stored(self, phys_data):
        ds = to_xarray(phys_data)
        assert "pfile_configuration" in ds.attrs
        assert len(ds.attrs["pfile_configuration"]) > 0

    def test_no_valid_variables_raises(self, phys_data):
        with pytest.raises(ValueError, match="No requested variables"):
            to_xarray(phys_data, variables=["nonexistent_var"])

    def test_data_is_float32(self, phys_data):
        ds = to_xarray(phys_data, variables=["sh1"])
        assert ds["sh1"].dtype == np.float32

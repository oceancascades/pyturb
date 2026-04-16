"""Tests for NetCDF save/load round-trip."""

import numpy as np
import pytest
import xarray as xr

from pyturb.pfile import save_netcdf


class TestSaveNetcdf:
    """Test saving to NetCDF."""

    def test_creates_file(self, phys_data, tmp_path):
        out = tmp_path / "test.nc"
        save_netcdf(phys_data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_roundtrip_variables(self, phys_data, tmp_path):
        out = tmp_path / "test.nc"
        save_netcdf(phys_data, out, variables=["P", "sh1", "sh2"])
        ds = xr.open_dataset(out)
        assert "P" in ds
        assert "sh1" in ds
        assert "sh2" in ds
        ds.close()

    def test_compressed_output(self, phys_data, tmp_path):
        out_plain = tmp_path / "plain.nc"
        out_compressed = tmp_path / "compressed.nc"
        save_netcdf(phys_data, out_plain)
        save_netcdf(phys_data, out_compressed, compress=True, compression_level=4)
        # Compressed file should be smaller
        assert out_compressed.stat().st_size < out_plain.stat().st_size

    def test_no_overwrite_by_default(self, phys_data, tmp_path):
        out = tmp_path / "test.nc"
        save_netcdf(phys_data, out)
        with pytest.raises(FileExistsError):
            save_netcdf(phys_data, out)

    def test_overwrite_flag(self, phys_data, tmp_path):
        out = tmp_path / "test.nc"
        save_netcdf(phys_data, out)
        save_netcdf(phys_data, out, overwrite=True)  # should not raise

    def test_time_coordinates_preserved(self, phys_data, tmp_path):
        out = tmp_path / "test.nc"
        save_netcdf(phys_data, out, variables=["P", "sh1"])
        # decode_times=False to get raw float values back
        ds = xr.open_dataset(out, decode_times=False)
        assert len(ds.t_fast) == len(phys_data["t_fast"])
        assert len(ds.t_slow) == len(phys_data["t_slow"])
        np.testing.assert_allclose(ds.t_fast.values, phys_data["t_fast"], atol=1e-4)
        ds.close()

    def test_data_values_preserved(self, phys_data, tmp_path):
        out = tmp_path / "test.nc"
        save_netcdf(phys_data, out, variables=["P"])
        ds = xr.open_dataset(out)
        # float32 in file, float64 in memory — allow some tolerance
        np.testing.assert_allclose(
            ds["P"].values, phys_data["P"].astype(np.float32), rtol=1e-5
        )
        ds.close()

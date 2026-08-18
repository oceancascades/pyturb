"""Tests for bin_profiles sourcing CTD variables from the highest-resolution
data available, and the separate --ctd-bin-width grid.
"""

from pathlib import Path

import pytest
import xarray as xr

from pyturb._pfile import to_xarray
from pyturb.pfile import load_pfile_phys
from pyturb.processing import _write_epsilon_profile, bin_profiles
from pyturb.profile import ProfileConfig, process_profile

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"
DEPTH_MIN, DEPTH_MAX = 90.0, 130.0


@pytest.fixture(scope="module")
def eps_file(tmp_path_factory):
    """One real, processed profile (with hires CTD vars) written to disk."""
    raw = to_xarray(load_pfile_phys(PFILE))
    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
    )
    result = process_profile(raw.copy(deep=True), config)
    out_dir = tmp_path_factory.mktemp("eps_hires")
    out_file = out_dir / "profile_p0000.nc"
    _write_epsilon_profile(result, raw, out_file, PFILE.name, 0, config)
    return out_file


@pytest.fixture(scope="module")
def eps_file_no_hires(eps_file, tmp_path_factory):
    """The same profile with ctd_time/*_hires stripped, for backward-compat checks."""
    ds = xr.load_dataset(eps_file, decode_times=False)
    hires_vars = [v for v in ds.data_vars if "ctd_time" in ds[v].dims]
    ds = ds.drop_vars(hires_vars).drop_dims("ctd_time", errors="ignore")
    out_file = tmp_path_factory.mktemp("eps_no_hires") / "profile_p0000.nc"
    ds.to_netcdf(out_file)
    return out_file


class TestBinPrefersHighestResolution:
    def test_main_grid_uses_hires_source_when_available(self, eps_file, tmp_path):
        binned = bin_profiles(
            [eps_file],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert "temperature" in binned
        assert binned["temperature"].dims == ("profile", "depth")
        assert not binned["temperature"].isnull().all()

    def test_backward_compatible_without_hires_vars(self, eps_file_no_hires, tmp_path):
        binned = bin_profiles(
            [eps_file_no_hires],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert binned is not None
        assert "temperature" in binned
        assert not any("_hires" in v for v in binned.data_vars)
        assert "ctd_depth" not in binned.dims


class TestCtdBinWidth:
    def test_adds_separate_finer_grid(self, eps_file, tmp_path):
        binned = bin_profiles(
            [eps_file],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
            ctd_bin_width=0.5,
        )
        assert "ctd_depth" in binned.dims
        assert binned.sizes["ctd_depth"] > binned.sizes["depth"]
        assert "temperature_hires" in binned
        assert binned["temperature_hires"].dims == ("profile", "ctd_depth")
        # Main-grid var must still be present and unrenamed.
        assert "temperature" in binned
        assert binned["temperature"].dims == ("profile", "depth")

    def test_omitted_by_default(self, eps_file, tmp_path):
        binned = bin_profiles(
            [eps_file],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert "ctd_depth" not in binned.dims
        assert "temperature_hires" not in binned

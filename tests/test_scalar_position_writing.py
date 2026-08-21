"""Tests that scalar lat/lon (stationary platforms), z, and N2 survive the
netCDF round-trip through _write_epsilon_profile, and flow through bin_profiles.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyturb._pfile import to_xarray
from pyturb.pfile import load_pfile_phys
from pyturb.processing import _write_epsilon_profile, bin_profiles
from pyturb.profile import ProfileConfig, process_profile

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"
DEPTH_MIN, DEPTH_MAX = 90.0, 130.0


@pytest.fixture(scope="module")
def eps_file_with_position(tmp_path_factory):
    """A real, processed VMP profile with a synthetic drifting aux track,
    written to disk. instrument_vehicle="VMP" -> stationary -> scalar lat/lon.
    """
    raw = to_xarray(load_pfile_phys(PFILE))
    assert raw.attrs.get("instrument_vehicle") == "VMP"

    n = raw.sizes["t_slow"]
    t = np.arange(n) / float(raw.fs_slow)
    raw["aux_latitude"] = ("t_slow", -45.0 + 0.001 * t)
    raw["aux_longitude"] = ("t_slow", 170.0 + 0.002 * t)

    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
        compute_thermo=True,
    )
    result = process_profile(raw.copy(deep=True), config)
    out_dir = tmp_path_factory.mktemp("eps_scalar_pos")
    out_file = out_dir / "profile_p0000.nc"
    _write_epsilon_profile(result, raw, out_file, PFILE.name, 0, config)
    return out_file


class TestScalarPositionSurvivesWriting:
    def test_lat_lon_are_scalar_after_write(self, eps_file_with_position):
        written = xr.load_dataset(eps_file_with_position, decode_times=False)
        assert written["lat"].dims == ()
        assert written["lon"].dims == ()

    def test_no_hires_position_vars(self, eps_file_with_position):
        written = xr.load_dataset(eps_file_with_position, decode_times=False)
        assert "lat_hires" not in written
        assert "lon_hires" not in written

    def test_z_and_n2_present_after_write(self, eps_file_with_position):
        written = xr.load_dataset(eps_file_with_position, decode_times=False)
        assert "z" in written
        assert written["z"].dims == ("time",)
        assert "z_hires" in written
        assert "N2" in written
        assert written["N2"].dims == ("time",)
        assert "N2_hires" in written
        assert written["N2_hires"].dims == ("ctd_time",)


class TestBinWithScalarPosition:
    def test_bin_profiles_handles_scalar_lat_lon(
        self, eps_file_with_position, tmp_path
    ):
        binned = bin_profiles(
            [eps_file_with_position],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert binned is not None
        assert "lat" in binned
        # A single position broadcasts across all depth bins for that profile.
        lat_vals = binned["lat"].values.ravel()
        finite = lat_vals[np.isfinite(lat_vals)]
        assert finite.size > 0
        assert np.unique(finite).size == 1

    def test_bin_profiles_includes_z_and_n2_by_default(
        self, eps_file_with_position, tmp_path
    ):
        binned = bin_profiles(
            [eps_file_with_position],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert "z" in binned
        assert "N2" in binned

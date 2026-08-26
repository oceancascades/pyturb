"""Tests for bin_profiles sourcing CTD variables from the highest-resolution
data available, and the separate --ctd-bin-width grid.
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


class TestBinIncludesFP07Thermistors:
    def test_t1_t2_on_main_grid(self, eps_file, tmp_path):
        binned = bin_profiles(
            [eps_file],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        for name in ["T1", "T2"]:
            assert name in binned, name
            assert binned[name].dims == ("profile", "depth")
            assert not binned[name].isnull().all()

    def test_t1_t2_on_ctd_depth_grid(self, eps_file, tmp_path):
        binned = bin_profiles(
            [eps_file],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
            ctd_bin_width=0.5,
        )
        for name in ["T1_hires", "T2_hires"]:
            assert name in binned, name
            assert binned[name].dims == ("profile", "ctd_depth")
            assert not binned[name].isnull().all()
        # Main-grid vars must still be present and unrenamed.
        assert binned["T1"].dims == ("profile", "depth")


@pytest.fixture(scope="module")
def eps_file_with_turbidity(tmp_path_factory):
    """The same profile, but with a synthetic turbidity channel added to the
    raw (t_slow) input before processing -- simulating an instrument that
    carries a turbidity sensor.
    """
    raw = to_xarray(load_pfile_phys(PFILE))
    n = raw.sizes["t_slow"]
    raw["turbidity"] = ("t_slow", np.linspace(0.1, 1.0, n))
    raw["turbidity"].attrs = {"long_name": "turbidity", "units": "FTU"}
    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
    )
    result = process_profile(raw.copy(deep=True), config)
    out_dir = tmp_path_factory.mktemp("eps_turbidity")
    out_file = out_dir / "profile_turb_p0000.nc"
    _write_epsilon_profile(result, raw, out_file, "with_turbidity.p", 0, config)
    return out_file


class TestBinIncludesOpticalVars:
    def test_optical_var_present_when_instrument_has_it(
        self, eps_file_with_turbidity, tmp_path
    ):
        binned = bin_profiles(
            [eps_file_with_turbidity],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert "turbidity" in binned
        assert binned["turbidity"].dims == ("profile", "depth")
        assert not binned["turbidity"].isnull().all()

    def test_missing_on_instruments_without_it_is_nan_not_dropped(
        self, eps_file, eps_file_with_turbidity, tmp_path
    ):
        # Binning across one instrument with turbidity and one without must
        # fill the profile lacking it with NaN, not drop the variable or
        # error out.
        binned = bin_profiles(
            [eps_file, eps_file_with_turbidity],
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert "turbidity" in binned
        assert binned["turbidity"].dims == ("profile", "depth")
        # Both source files share the same underlying cast/time, so profile
        # order after chronological sorting isn't guaranteed -- check by
        # per-profile all-NaN-ness instead of a fixed index.
        all_nan_per_profile = binned["turbidity"].isnull().all(dim="depth").values
        assert all_nan_per_profile.sum() == 1
        assert (~all_nan_per_profile).sum() == 1
        # Other variables must be unaffected by the missing sensor.
        assert not binned["temperature"].isnull().all().item()

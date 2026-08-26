"""Tests for the high-resolution CTD output (ctd_time / *_hires variables)."""

import gsw
import numpy as np
import xarray as xr

from pyturb.profile import (
    ProfileConfig,
    _attach_hires_ctd_vars,
    _attach_window_scalars,
)

FS_SLOW = 64.0

# Slow-channel-only windowing: n_fft/n_diss already in slow-channel samples,
# sampling_ratio=1 so _window_mean_slow uses them directly.
_PARAMS = {
    "n_fft": 64,
    "n_diss": 256,
    "fft_overlap": 32,
    "diss_overlap": 32,
    "sampling_ratio": 1,
}


def _make_ds(n=3000, **extra_vars) -> xr.Dataset:
    t = np.arange(n) / FS_SLOW
    data = {
        "P_smooth": ("t_slow", 10.0 + 0.5 * t),
        "W_smooth": ("t_slow", np.full(n, 0.5)),
        "JAC_T": ("t_slow", 10.0 + np.sin(2 * np.pi * 0.02 * t)),
        "JAC_C": ("t_slow", np.full(n, 37.0)),  # mS/cm, seawater range
        "fs_slow": FS_SLOW,
    }
    for name, values in extra_vars.items():
        data[name] = ("t_slow", np.asarray(values))
    return xr.Dataset(data, coords={"t_slow": t})


class TestHiresCtdVars:
    def test_hires_vars_present_on_ctd_time(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert "ctd_time" in out.dims
        for name in [
            "pressure_hires",
            "temperature_hires",
            "salinity_hires",
            "conductivity_hires",
            "density_hires",
            "z_hires",
        ]:
            assert name in out, name
            assert out[name].dims == ("ctd_time",)

    def test_hires_excludes_kinematics(self):
        # W and nu are only meaningful at dissipation-window resolution.
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert "W_hires" not in out
        assert "nu_hires" not in out
        assert "kappa_T_hires" not in out
        # Still present at the main (dissipation-window) resolution.
        assert "W" in out
        assert "nu" in out
        assert "kappa_T" in out

    def test_ctd_time_much_finer_than_time(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert out.sizes["ctd_time"] > 10 * out.sizes["time"]

    def test_dissipation_bin_vars_still_present(self):
        # The coarser, dissipation-window-resolution vars must still exist.
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        for name in ["pressure", "temperature", "salinity", "conductivity", "density"]:
            assert name in out
            assert out[name].dims == ("time",)

    def test_disabled_with_zero_bin_sec(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False, ctd_bin_sec=0.0)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert "ctd_time" not in out.dims
        assert "temperature_hires" not in out

    def test_bin_width_controls_resolution(self):
        ds = _make_ds()
        fine = _attach_window_scalars(
            ds, _PARAMS, ProfileConfig(match_conductivity=False, ctd_bin_sec=0.25)
        )
        coarse = _attach_window_scalars(
            ds, _PARAMS, ProfileConfig(match_conductivity=False, ctd_bin_sec=1.0)
        )
        assert fine.sizes["ctd_time"] > coarse.sizes["ctd_time"]

    def test_compute_thermo_hires_vars(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        for name in [
            "absolute_salinity_hires",
            "conservative_temperature_hires",
            "potential_density_hires",
            "N2_hires",
        ]:
            assert name in out
            assert out[name].dims == ("ctd_time",)

    def test_works_when_fs_slow_is_a_global_attr(self):
        # Real converted files store fs_slow as a plain Dataset attribute,
        # not a data variable -- must not be silently skipped.
        ds = _make_ds()
        ds = ds.drop_vars("fs_slow")
        ds.attrs["fs_slow"] = FS_SLOW
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "ctd_time" in out.dims
        assert "temperature_hires" in out

    def test_noop_without_temperature_or_conductivity(self):
        ds = xr.Dataset(
            {
                "P_smooth": ("t_slow", 10.0 + 0.5 * np.arange(3000) / FS_SLOW),
                "W_smooth": ("t_slow", np.full(3000, 0.5)),
                "fs_slow": FS_SLOW,
            },
            coords={"t_slow": np.arange(3000) / FS_SLOW},
        )
        config = ProfileConfig(match_conductivity=False, temperature="JAC_T")
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "ctd_time" not in out.dims

    def test_too_short_for_one_bin_is_noop(self):
        # _attach_hires_ctd_vars directly: too little data for even one bin
        # at the requested ctd_bin_sec must not crash, just skip.
        ds = _make_ds(n=200)  # 200 samples / 64 Hz ~= 3.1 s of data
        config = ProfileConfig(match_conductivity=False, ctd_bin_sec=10.0)
        out = _attach_hires_ctd_vars(ds, config)
        assert "ctd_time" not in out.dims


class TestDepth:
    def test_z_matches_gsw_z_from_p_at_default_latitude(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        expected = gsw.z_from_p(out["pressure"].values, config.default_latitude)
        np.testing.assert_allclose(out["z"].values, expected, rtol=1e-4)

    def test_z_present_at_both_resolutions(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert out["z"].dims == ("time",)
        assert out["z_hires"].dims == ("ctd_time",)

    def test_z_is_negative_below_surface(self):
        ds = _make_ds()  # P_smooth = 10 + 0.5*t, always positive pressure
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert np.all(out["z"].values < 0)

    def test_z_uses_aux_latitude_when_present(self):
        ds = _make_ds(aux_latitude=np.full(3000, -60.0))
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        expected = gsw.z_from_p(out["pressure"].values, -60.0)
        np.testing.assert_allclose(out["z"].values, expected, rtol=1e-4)


class TestFP07Thermistors:
    def test_t1_t2_present_at_both_resolutions(self):
        n = 3000
        t = np.arange(n) / FS_SLOW
        ds = _make_ds(T1=10.0 + 0.01 * t, T2=10.1 + 0.01 * t)
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert out["T1"].dims == ("time",)
        assert out["T2"].dims == ("time",)
        assert out["T1_hires"].dims == ("ctd_time",)
        assert out["T2_hires"].dims == ("ctd_time",)

    def test_t1_t2_are_window_means_not_overwritten_by_each_other(self):
        # T1/T2 share a name with their own raw input; the coarse and hires
        # outputs must each reflect a real window mean of the raw signal, not
        # get corrupted by the other resolution's pass (see profile.py's
        # ordering comment in _attach_window_scalars).
        n = 3000
        t = np.arange(n) / FS_SLOW
        ds = _make_ds(T1=10.0 + 0.01 * t, T2=20.0 + 0.02 * t)
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert out.sizes["ctd_time"] > 10 * out.sizes["time"]
        assert np.all((out["T1"].values > 9.5) & (out["T1"].values < 11.5))
        assert np.all((out["T2"].values > 19.5) & (out["T2"].values < 21.5))
        assert np.all((out["T1_hires"].values > 9.5) & (out["T1_hires"].values < 11.5))
        assert np.all((out["T2_hires"].values > 19.5) & (out["T2_hires"].values < 21.5))

    def test_absent_without_t1_t2_in_input(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "T1" not in out
        assert "T2" not in out
        assert "T1_hires" not in out
        assert "T2_hires" not in out


class TestOpticalVars:
    def test_present_at_both_resolutions_when_in_input(self):
        n = 3000
        t = np.arange(n) / FS_SLOW
        ds = _make_ds(turbidity=0.5 + 0.01 * t, chlorophyll=1.0 + 0.02 * t)
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        for name in ["turbidity", "chlorophyll"]:
            assert out[name].dims == ("time",)
            assert out[f"{name}_hires"].dims == ("ctd_time",)
            assert not out[name].isnull().all()
            assert not out[f"{name}_hires"].isnull().all()

    def test_absent_without_optical_vars_in_input(self):
        # "if they exist": instruments without these sensors must not error
        # or fabricate the variables.
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "turbidity" not in out
        assert "chlorophyll" not in out
        assert "turbidity_hires" not in out
        assert "chlorophyll_hires" not in out

    def test_only_present_var_is_attached(self):
        # One sensor present, the other absent -- only the present one shows up.
        n = 3000
        t = np.arange(n) / FS_SLOW
        ds = _make_ds(turbidity=0.5 + 0.01 * t)
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "turbidity" in out
        assert "chlorophyll" not in out

    def test_configurable_var_names(self):
        # A differently-named channel (e.g. lowercase in this instrument's
        # setup string) works when passed via optical_vars.
        n = 3000
        t = np.arange(n) / FS_SLOW
        ds = _make_ds(turb=0.5 + 0.01 * t)
        config = ProfileConfig(match_conductivity=False, optical_vars=("turb",))
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "turb" in out
        assert "turb_hires" in out

    def test_fast_sampled_optical_var_aligns_with_ctd_time(self):
        # Regression: on real instruments the turbidity/chlorophyll
        # fluorometer is wired into the p-file's fast channel matrix, not
        # the slow one like JAC_T/JAC_C/T1/T2. _attach_hires_ctd_vars must
        # scale the block size for a t_fast var to match the ctd_time grid
        # built from t_slow, or block_mean() produces a mismatched length
        # and xarray raises "conflicting sizes for dimension 'ctd_time'".
        ratio = 8
        n_slow = 3000
        t_slow = np.arange(n_slow) / FS_SLOW
        fs_fast = FS_SLOW * ratio
        n_fast = n_slow * ratio
        t_fast = np.arange(n_fast) / fs_fast

        ds = xr.Dataset(
            {
                "P_smooth": ("t_slow", 10.0 + 0.5 * t_slow),
                "W_smooth": ("t_slow", np.full(n_slow, 0.5)),
                "JAC_T": ("t_slow", 10.0 + np.sin(2 * np.pi * 0.02 * t_slow)),
                "JAC_C": ("t_slow", np.full(n_slow, 37.0)),
                "turbidity": ("t_fast", 0.5 + 0.001 * t_fast),
                "fs_slow": FS_SLOW,
                "fs_fast": fs_fast,
            },
            coords={"t_slow": t_slow, "t_fast": t_fast},
        )
        config = ProfileConfig(match_conductivity=False)
        out = _attach_hires_ctd_vars(ds, config)

        assert "turbidity_hires" in out
        assert out["turbidity_hires"].dims == ("ctd_time",)
        assert (
            out["turbidity_hires"].sizes["ctd_time"]
            == out["temperature_hires"].sizes["ctd_time"]
        )
        assert not out["turbidity_hires"].isnull().all()

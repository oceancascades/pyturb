"""Tests for the high-resolution CTD output (ctd_time / *_hires variables)."""

import gsw
import numpy as np
import xarray as xr

from pyturb.profile import ProfileConfig, _attach_hires_ctd_vars, _attach_window_scalars

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

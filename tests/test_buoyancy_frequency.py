"""Tests for N2 (buoyancy frequency squared)."""

import gsw
import numpy as np
import xarray as xr

from pyturb.profile import ProfileConfig, _attach_window_scalars

FS_SLOW = 64.0

_PARAMS = {
    "n_fft": 64,
    "n_diss": 256,
    "fft_overlap": 32,
    "diss_overlap": 32,
    "sampling_ratio": 1,
}


def _make_stratified_ds(n=3000) -> xr.Dataset:
    t = np.arange(n) / FS_SLOW
    # Increasing pressure, decreasing temperature -> stably stratified,
    # positive N2 expected.
    return xr.Dataset(
        {
            "P_smooth": ("t_slow", 10.0 + 0.5 * t),
            "W_smooth": ("t_slow", np.full(n, 0.5)),
            "JAC_T": ("t_slow", 12.0 - 0.002 * t),
            "JAC_C": ("t_slow", np.full(n, 37.0)),
            "fs_slow": FS_SLOW,
        },
        coords={"t_slow": t},
    )


class TestBuoyancyFrequency:
    def test_n2_present_when_compute_thermo(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert "N2" in out
        assert out["N2"].dims == ("time",)
        assert np.isfinite(out["N2"].values).sum() >= 2

    def test_n2_absent_without_compute_thermo(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "N2" not in out

    def test_n2_matches_interpolated_gsw_nsquared(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        SA = out["absolute_salinity"].values.astype(float)
        CT = out["conservative_temperature"].values.astype(float)
        P = out["pressure"].values.astype(float)

        N2_mid, P_mid = gsw.Nsquared(SA, CT, P, lat=config.default_latitude)
        expected = np.interp(P, P_mid, N2_mid, left=np.nan, right=np.nan)

        actual = out["N2"].values.astype(float)
        finite = np.isfinite(expected)
        assert finite.sum() >= 2
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-2)

    def test_n2_indicates_stable_stratification(self):
        # Increasing density with pressure -> positive N2 (stable).
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        n2 = out["N2"].values
        finite = n2[np.isfinite(n2)]
        assert finite.size >= 2
        assert np.all(finite > 0)

    def test_n2_absent_with_too_few_windows(self):
        # A single-window profile can't form any gsw.Nsquared midpoint.
        n = 300
        t = np.arange(n) / FS_SLOW
        ds = xr.Dataset(
            {
                "P_smooth": ("t_slow", 10.0 + 0.5 * t),
                "W_smooth": ("t_slow", np.full(n, 0.5)),
                "JAC_T": ("t_slow", np.full(n, 10.0)),
                "JAC_C": ("t_slow", np.full(n, 37.0)),
                "fs_slow": FS_SLOW,
            },
            coords={"t_slow": t},
        )
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert out.sizes["time"] == 1
        assert "N2" not in out


class TestBuoyancyFrequencyHires:
    def test_n2_hires_present_when_compute_thermo(self):
        # Computed from the ctd_bin_sec-averaged (not raw) profile, at the
        # finer ctd_time resolution, alongside the dissipation-window N2.
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert "N2_hires" in out
        assert out["N2_hires"].dims == ("ctd_time",)
        assert np.isfinite(out["N2_hires"].values).sum() >= 2

    def test_n2_hires_absent_without_compute_thermo(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "N2_hires" not in out

    def test_n2_hires_matches_interpolated_gsw_nsquared(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        SA = out["absolute_salinity_hires"].values.astype(float)
        CT = out["conservative_temperature_hires"].values.astype(float)
        P = out["pressure_hires"].values.astype(float)

        N2_mid, P_mid = gsw.Nsquared(SA, CT, P, lat=config.default_latitude)
        expected = np.interp(P, P_mid, N2_mid, left=np.nan, right=np.nan)

        actual = out["N2_hires"].values.astype(float)
        finite = np.isfinite(expected)
        assert finite.sum() >= 2
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-2)

    def test_n2_hires_indicates_stable_stratification(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(match_conductivity=False, compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)

        n2 = out["N2_hires"].values
        finite = n2[np.isfinite(n2)]
        assert finite.size >= 2
        assert np.all(finite > 0)

    def test_n2_hires_disabled_with_zero_ctd_bin_sec(self):
        ds = _make_stratified_ds()
        config = ProfileConfig(
            match_conductivity=False, compute_thermo=True, ctd_bin_sec=0.0
        )
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "N2_hires" not in out
        # Dissipation-window N2 is unaffected.
        assert "N2" in out

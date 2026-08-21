"""Tests for per-platform lat/lon handling (stationary vs moving)."""

import numpy as np
import xarray as xr

from pyturb.profile import ProfileConfig, _attach_window_scalars, _first_valid

FS_SLOW = 64.0

_PARAMS = {
    "n_fft": 64,
    "n_diss": 256,
    "fft_overlap": 32,
    "diss_overlap": 32,
    "sampling_ratio": 1,
}


def _make_ds(vehicle: str | None = None, n=3000) -> xr.Dataset:
    t = np.arange(n) / FS_SLOW
    # Drifting position over the profile: a stationary platform must collapse
    # this to a single value; a moving platform should track it per-window.
    lat = -45.0 + 0.01 * t
    lon = 170.0 + 0.02 * t
    ds = xr.Dataset(
        {
            "P_smooth": ("t_slow", 10.0 + 0.5 * t),
            "W_smooth": ("t_slow", np.full(n, 0.5)),
            "JAC_T": ("t_slow", 10.0 + np.sin(2 * np.pi * 0.02 * t)),
            "JAC_C": ("t_slow", np.full(n, 37.0)),
            "aux_latitude": ("t_slow", lat),
            "aux_longitude": ("t_slow", lon),
            "fs_slow": FS_SLOW,
        },
        coords={"t_slow": t},
    )
    if vehicle is not None:
        ds.attrs["instrument_vehicle"] = vehicle
    return ds


class TestStationaryPlatformDetection:
    def test_vmp_gets_scalar_lat_lon(self):
        ds = _make_ds(vehicle="VMP")
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert out["lat"].dims == ()
        assert out["lon"].dims == ()
        # A single scalar per profile replaces the hires version entirely.
        assert "lat_hires" not in out
        assert "lon_hires" not in out

    def test_vehicle_name_is_case_insensitive(self):
        ds = _make_ds(vehicle="vmp")
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert out["lat"].dims == ()

    def test_glider_keeps_interpolated_lat_lon(self):
        ds = _make_ds(vehicle="slocum_glider")
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert out["lat"].dims == ("time",)
        assert out["lon"].dims == ("time",)
        assert np.unique(out["lat"].values).size > 1
        assert np.unique(out["lon"].values).size > 1
        assert "lat_hires" in out
        assert np.unique(out["lat_hires"].values).size > 1

    def test_unknown_vehicle_defaults_to_moving(self):
        ds = _make_ds(vehicle=None)
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert out["lat"].dims == ("time",)
        assert np.unique(out["lat"].values).size > 1

    def test_explicit_override_forces_stationary(self):
        ds = _make_ds(vehicle="slocum_glider")
        config = ProfileConfig(match_conductivity=False, stationary_platform=True)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert out["lat"].dims == ()

    def test_explicit_override_forces_moving(self):
        ds = _make_ds(vehicle="VMP")
        config = ProfileConfig(match_conductivity=False, stationary_platform=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert out["lat"].dims == ("time",)
        assert np.unique(out["lat"].values).size > 1

    def test_vmp_uses_position_at_first_timestamp(self):
        # Not the mean of the (already-interpolated) track -- the value at
        # the profile's first sample.
        ds = _make_ds(vehicle="VMP")
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        expected_lat = float(ds["aux_latitude"].values[0])
        expected_lon = float(ds["aux_longitude"].values[0])
        assert abs(float(out["lat"].values) - expected_lat) < 1e-9
        assert abs(float(out["lon"].values) - expected_lon) < 1e-9

        mean_lat = float(np.nanmean(ds["aux_latitude"].values))
        assert abs(expected_lat - mean_lat) > 1e-3  # sanity: track really drifts

    def test_skips_leading_nan_for_first_valid_position(self):
        ds = _make_ds(vehicle="VMP")
        lat = ds["aux_latitude"].values.copy()
        lat[:5] = np.nan
        ds["aux_latitude"] = ("t_slow", lat)
        config = ProfileConfig(match_conductivity=False)
        out = _attach_window_scalars(ds, _PARAMS, config)

        assert abs(float(out["lat"].values) - float(lat[5])) < 1e-9


class TestFirstValid:
    def test_returns_first_element_when_all_finite(self):
        assert _first_valid(np.array([1.0, 2.0, 3.0])) == 1.0

    def test_skips_leading_nan(self):
        assert _first_valid(np.array([np.nan, np.nan, 5.0, 6.0])) == 5.0

    def test_falls_back_to_first_element_when_all_nan(self):
        assert np.isnan(_first_valid(np.array([np.nan, np.nan])))

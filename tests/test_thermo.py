"""Tests for Conservative Temperature / Absolute Salinity / potential density."""

import numpy as np
import xarray as xr

from pyturb.profile import ProfileConfig, _attach_window_scalars

# One dissipation window, one FFT window, no fast/slow oversampling.
_PARAMS = {
    "n_fft": 4,
    "n_diss": 4,
    "fft_overlap": 2,
    "diss_overlap": 2,
    "sampling_ratio": 1,
}


def _make_ds(**extra_vars):
    n = 4
    data = {
        "P_smooth": ("t_slow", np.full(n, 10.0)),
        "W_smooth": ("t_slow", np.full(n, 0.5)),
        "JAC_T": ("t_slow", np.full(n, 10.0)),
        "JAC_C": ("t_slow", np.full(n, 35.0)),  # mS/cm, seawater range
    }
    for name, values in extra_vars.items():
        data[name] = ("t_slow", np.full(n, values))
    return xr.Dataset(data, coords={"t_slow": np.arange(n, dtype=float)})


class TestComputeThermo:
    def test_off_by_default(self):
        ds = _make_ds()
        config = ProfileConfig(compute_thermo=False)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "absolute_salinity" not in out
        assert "conservative_temperature" not in out
        assert "potential_density" not in out

    def test_computed_when_salinity_from_jac(self):
        ds = _make_ds()
        config = ProfileConfig(compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "absolute_salinity" in out
        assert "conservative_temperature" in out
        assert "potential_density" in out
        # Sanity: SA slightly above SP=salinity for typical seawater, CT close
        # to in-situ T at these T/S/P values, potential density near 1026-1027.
        assert np.all(out["absolute_salinity"].values > out["salinity"].values)
        assert np.all(np.abs(out["conservative_temperature"].values - 10.0) < 1.0)
        assert np.all(
            (out["potential_density"].values > 1020.0)
            & (out["potential_density"].values < 1030.0)
        )

    def test_skipped_without_real_salinity(self):
        # JAC_C below the seawater-range guard -> no real salinity -> no thermo.
        ds = _make_ds()
        ds["JAC_C"] = ("t_slow", np.full(4, 1.0))
        config = ProfileConfig(compute_thermo=True)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "salinity" not in out
        assert "absolute_salinity" not in out

    def test_uses_aux_lat_lon_when_present(self):
        ds = _make_ds(aux_latitude=-45.0, aux_longitude=170.0)
        config = ProfileConfig(compute_thermo=True, default_latitude=45.0)
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "absolute_salinity" in out
        # Different hemisphere should not silently fall back to the default.
        assert np.all(out["lat"].values < 0)

    def test_falls_back_to_default_position(self):
        ds = _make_ds()
        config = ProfileConfig(
            compute_thermo=True, default_latitude=60.0, default_longitude=-150.0
        )
        out = _attach_window_scalars(ds, _PARAMS, config)
        assert "absolute_salinity" in out
        assert "lat" not in out
        assert "lon" not in out

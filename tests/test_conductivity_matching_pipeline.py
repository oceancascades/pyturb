"""Tests for _apply_conductivity_matching's integration into the profile pipeline."""

import numpy as np
import xarray as xr

from pyturb.conductivity import match_conductivity_to_temperature
from pyturb.profile import (
    ProfileConfig,
    _apply_conductivity_matching,
    _despike_conductivity,
)

FS_SLOW = 64.0


def _make_ds(**extra_vars) -> xr.Dataset:
    n = 3000
    t = np.arange(n) / FS_SLOW
    data = {
        "JAC_C": ("t_slow", 35.0 + np.sin(2 * np.pi * 0.05 * t)),
        "JAC_T": ("t_slow", 10.0 + np.cos(2 * np.pi * 0.05 * t)),
        "W_smooth": ("t_slow", np.full(n, 0.5)),
        "fs_slow": FS_SLOW,
    }
    for name, values in extra_vars.items():
        data[name] = ("t_slow", np.asarray(values))
    return xr.Dataset(data, coords={"t_slow": t})


class TestApplyConductivityMatching:
    def test_on_by_default(self):
        ds = _make_ds()
        config = ProfileConfig()
        out = _apply_conductivity_matching(ds, config)
        assert not np.array_equal(out["JAC_C"].values, ds["JAC_C"].values)

    def test_disabled_when_off(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=False)
        out = _apply_conductivity_matching(ds, config)
        np.testing.assert_array_equal(out["JAC_C"].values, ds["JAC_C"].values)

    def test_modifies_jac_c_when_enabled(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=True)
        out = _apply_conductivity_matching(ds, config)
        assert not np.array_equal(out["JAC_C"].values, ds["JAC_C"].values)

    def test_never_modifies_temperature(self):
        ds = _make_ds()
        config = ProfileConfig(match_conductivity=True)
        out = _apply_conductivity_matching(ds, config)
        np.testing.assert_array_equal(out["JAC_T"].values, ds["JAC_T"].values)

    def test_noop_without_jac_c(self):
        ds = _make_ds().drop_vars("JAC_C")
        config = ProfileConfig(match_conductivity=True)
        out = _apply_conductivity_matching(ds, config)
        assert "JAC_C" not in out

    def test_noop_without_temperature(self):
        ds = _make_ds().drop_vars("JAC_T")
        config = ProfileConfig(match_conductivity=True, temperature="JAC_T")
        out = _apply_conductivity_matching(ds, config)
        np.testing.assert_array_equal(out["JAC_C"].values, ds["JAC_C"].values)

    def test_auxiliary_variables_untouched(self):
        aux_temp = np.full(3000, 11.0)
        aux_sal = np.full(3000, 34.5)
        ds = _make_ds(aux_temperature=aux_temp, aux_salinity=aux_sal)
        config = ProfileConfig(match_conductivity=True)
        out = _apply_conductivity_matching(ds, config)
        np.testing.assert_array_equal(out["aux_temperature"].values, aux_temp)
        np.testing.assert_array_equal(out["aux_salinity"].values, aux_sal)

    def test_fitted_response_overrides_speed_scaled_defaults(self):
        t = np.arange(3000) / FS_SLOW
        ds = _make_ds(JAC_C=np.where(t < 10, 35.0, 36.0))
        ds["JAC_C"].attrs = {"JAC_C_lag": 0.05, "JAC_C_tau": 0.15}
        out = _apply_conductivity_matching(ds, ProfileConfig())["JAC_C"].values
        expected = match_conductivity_to_temperature(
            ds["JAC_C"].values,
            FS_SLOW,
            0.5,
            lag=0.05,
            f_tc=1 / (2 * np.pi * 0.15),
            reference_speed=0.5,
        )
        np.testing.assert_allclose(out, expected)


class TestDespikeConductivity:
    def test_removes_spike_and_records_mask(self):
        ds = _make_ds()
        clean = ds["JAC_C"].values.copy()
        ds["JAC_C"].values[1500] -= 5.0
        out = _despike_conductivity(ds, ProfileConfig())
        assert abs(out["JAC_C"].values[1500] - clean[1500]) < 0.01
        mask = out["JAC_C_despike_mask"].values
        assert mask[1500] and mask.sum() < 10

    def test_noop_without_jac_c(self):
        ds = _make_ds().drop_vars("JAC_C")
        assert "JAC_C_despike_mask" not in _despike_conductivity(ds, ProfileConfig())

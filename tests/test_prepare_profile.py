"""Tests for prepare_profile's speed-source resolution (onboard vs.
auxiliary vs. pressure-derivative fallback)."""

import numpy as np
import pytest
import xarray as xr

from pyturb.profile import ProfileConfig, prepare_profile

FS_SLOW = 64.0


def _make_raw_ds(n: int = 2000, **extra_vars) -> xr.Dataset:
    t = np.arange(n) / FS_SLOW
    data = {"P": ("t_slow", 10.0 + 0.5 * t), "fs_slow": FS_SLOW}
    for name, values in extra_vars.items():
        data[name] = ("t_slow", np.asarray(values))
    return xr.Dataset(data, coords={"t_slow": t})


class TestSpeedSourcePriority:
    def test_onboard_speed_used_when_present(self):
        ds = _make_raw_ds(W=np.full(2000, 1.2))
        out = prepare_profile(ds, ProfileConfig())
        np.testing.assert_allclose(out["W_smooth"].values, 1.2, atol=0.05)

    def test_pressure_fallback_when_neither_present(self):
        ds = _make_raw_ds()
        out = prepare_profile(ds, ProfileConfig())
        assert "W_smooth" in out
        assert np.all(np.isfinite(out["W_smooth"].values))

    def test_aux_speed_used_when_onboard_absent(self):
        ds = _make_raw_ds(aux_speed=np.full(2000, 0.8))
        out = prepare_profile(ds, ProfileConfig())
        np.testing.assert_allclose(out["W_smooth"].values, 0.8, atol=0.05)

    def test_both_present_raises(self):
        ds = _make_raw_ds(W=np.full(2000, 1.2), aux_speed=np.full(2000, 0.8))
        with pytest.raises(ValueError, match="[Bb]oth"):
            prepare_profile(ds, ProfileConfig())

    def test_aux_speed_smoothing_matches_onboard_code_path(self):
        # Identical raw values through either source must smooth to the
        # same result -- confirms aux_speed reuses the same gap-aware
        # filtering as the onboard branch, not a divergent implementation.
        t = np.arange(2000) / FS_SLOW
        values = 0.9 + 0.1 * np.sin(2 * np.pi * 0.05 * t)

        onboard_out = prepare_profile(_make_raw_ds(W=values), ProfileConfig())
        aux_out = prepare_profile(_make_raw_ds(aux_speed=values), ProfileConfig())

        np.testing.assert_allclose(
            onboard_out["W_smooth"].values, aux_out["W_smooth"].values
        )

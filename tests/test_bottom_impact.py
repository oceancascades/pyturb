"""Tests for trimming profiles at a bottom impact."""

import numpy as np
import xarray as xr

from pyturb.io import load_profile_nc
from pyturb.pfile import load_pfile_phys, save_netcdf
from pyturb.processing import _write_epsilon_profile, bin_profiles
from pyturb.profile import (
    ProfileConfig,
    find_bottom_impact,
    process_profile,
    split_into_profiles,
    trim_bottom_impacts,
)

from .conftest import PFILE

FS_FAST, FS_SLOW = 512.0, 64.0
RATIO = int(FS_FAST / FS_SLOW)
DURATION = 60.0
CONFIG = ProfileConfig(trim_bottom_impact=True)


def _make_ds(impact_time: float | None = None, down: bool = True) -> xr.Dataset:
    rng = np.random.default_rng(0)
    t_fast = np.arange(int(DURATION * FS_FAST)) / FS_FAST
    t_slow = np.arange(int(DURATION * FS_SLOW)) / FS_SLOW
    accel = rng.normal(0, 50.0, (2, t_fast.size))
    if impact_time is not None:
        burst = (t_fast >= impact_time) & (t_fast < impact_time + 0.2)
        accel[:, burst] *= 100
    pressure = 5.0 + t_slow if down else 65.0 - t_slow
    return xr.Dataset(
        {
            "P_smooth": ("t_slow", pressure),
            "Ax": ("t_fast", accel[0]),
            "Ay": ("t_fast", accel[1]),
        },
        coords={"t_slow": t_slow, "t_fast": t_fast},
        attrs={"fs_fast": FS_FAST, "fs_slow": FS_SLOW},
    )


N_SLOW = int(DURATION * FS_SLOW)
SEGMENT = (0, N_SLOW - 1)


class TestFindBottomImpact:
    def test_cuts_just_before_impact(self):
        last_good = find_bottom_impact(_make_ds(impact_time=59.5), *SEGMENT, CONFIG)
        assert 59.4 < last_good / FS_SLOW <= 59.5

    def test_impact_just_after_profile_end_keeps_end(self):
        ds = _make_ds(impact_time=59.5)
        end = int(59.3 * FS_SLOW)
        assert find_bottom_impact(ds, 0, end, CONFIG) == end

    def test_none_without_impact(self):
        assert find_bottom_impact(_make_ds(), *SEGMENT, CONFIG) is None

    def test_burst_mid_profile_ignored(self):
        assert find_bottom_impact(_make_ds(impact_time=30.0), *SEGMENT, CONFIG) is None

    def test_none_below_threshold(self):
        config = ProfileConfig(trim_bottom_impact=True, impact_thresh=500.0)
        assert find_bottom_impact(_make_ds(impact_time=59.5), *SEGMENT, config) is None

    def test_none_without_accelerometers(self):
        ds = _make_ds(impact_time=59.5).drop_vars(["Ax", "Ay"])
        assert find_bottom_impact(ds, *SEGMENT, CONFIG) is None


class TestTrimBottomImpacts:
    def test_off_by_default(self):
        segments, impacted = trim_bottom_impacts(
            _make_ds(impact_time=59.5), [SEGMENT], ProfileConfig()
        )
        assert segments == [SEGMENT] and impacted == [False]

    def test_trims_down_profile(self):
        segments, impacted = trim_bottom_impacts(
            _make_ds(impact_time=59.5), [SEGMENT], CONFIG
        )
        assert impacted == [True]
        assert segments[0][0] == 0 and segments[0][1] < SEGMENT[1]

    def test_up_profile_untouched(self):
        segments, impacted = trim_bottom_impacts(
            _make_ds(impact_time=59.5, down=False), [SEGMENT], CONFIG
        )
        assert segments == [SEGMENT] and impacted == [False]

    def test_no_accelerometers_untouched(self):
        ds = _make_ds(impact_time=59.5).drop_vars(["Ax", "Ay"])
        segments, impacted = trim_bottom_impacts(ds, [SEGMENT], CONFIG)
        assert segments == [SEGMENT] and impacted == [False]


class TestSplitIntoProfiles:
    def _split(self, monkeypatch, config):
        monkeypatch.setattr(
            "pyturb.profile.find_all_profiles", lambda ds, config: [SEGMENT]
        )
        return next(split_into_profiles(_make_ds(impact_time=59.5), config))[1]

    def test_profile_ends_before_impact_and_is_flagged(self, monkeypatch):
        profile = self._split(monkeypatch, CONFIG)
        assert profile.attrs["bottom_impact"] == 1
        assert profile.t_fast.values[-1] <= 59.5
        assert profile.t_slow.values[-1] <= 59.5

    def test_unflagged_when_option_off(self, monkeypatch):
        profile = self._split(monkeypatch, ProfileConfig())
        assert "bottom_impact" not in profile.attrs
        assert profile.sizes["t_slow"] == N_SLOW


class TestBottomImpactOutput:
    def test_flag_written_and_binned(self, tmp_path):
        raw = save_and_reload(tmp_path)
        raw.attrs["bottom_impact"] = 1
        config = ProfileConfig(
            diss_len_sec=4.0, fft_len_sec=1.0, trim_bottom_impact=True
        )
        result = process_profile(raw.copy(deep=True), config)
        assert result["bottom_impact"].values == 1

        eps_dir = tmp_path / "eps"
        eps_dir.mkdir()
        out_file = eps_dir / "profile_p0000.nc"
        _write_epsilon_profile(result, raw, out_file, PFILE.name, 0, config)
        assert xr.load_dataset(out_file)["bottom_impact"].values == 1

        binned = bin_profiles(
            [out_file], tmp_path / "bin.nc", depth_max=200, n_workers=1
        )
        assert binned["bottom_impact"].dims == ("profile",)
        assert binned["bottom_impact"].values[0] == 1

    def test_no_flag_without_option(self, tmp_path):
        raw = save_and_reload(tmp_path)
        result = process_profile(raw.copy(deep=True), ProfileConfig())
        assert "bottom_impact" not in result


def save_and_reload(tmp_path) -> xr.Dataset:
    path = tmp_path / "converted.nc"
    save_netcdf(load_pfile_phys(PFILE), path, despike_kwargs={})
    return load_profile_nc(path)


def test_profiles_keep_their_own_metadata(monkeypatch):
    half = N_SLOW // 2
    monkeypatch.setattr(
        "pyturb.profile.find_all_profiles",
        lambda ds, config: [(0, half), (half + 1, N_SLOW - 1)],
    )
    profiles = [p for _, p in split_into_profiles(_make_ds(impact_time=59.5), CONFIG)]
    assert [p.attrs["bottom_impact"] for p in profiles] == [0, 1]
    assert [p.attrs["profile_index"] for p in profiles] == [0, 1]

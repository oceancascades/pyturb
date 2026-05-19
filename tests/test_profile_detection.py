"""Tests for multi-profile detection (find_all_profiles).

Four scenarios are covered:

1. Single glider up-cast  – brief descent preamble then long ascent (peak-based
   detection path; no time gaps in dataset).
2. Single glider down-cast – long descent then brief ascent preamble (peak-based
   detection path; no time gaps in dataset).
3. Merged glider dataset   – multiple up-cast segments separated by large time
   gaps (gap-based detection path).
4. Continuous VMP dataset  – 5 complete dive/ascent cycles with no time gaps;
   direction='down' should extract only the 5 descending segments (peak-based
   detection path, signed-velocity fix).
"""

import numpy as np
import pytest
import xarray as xr

from pyturb.profile import ProfileConfig, find_all_profiles

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FS = 64.0  # slow sampling rate [Hz]
DT = 1.0 / FS


def _make_ds(t: np.ndarray, pressure: np.ndarray, speed: np.ndarray) -> xr.Dataset:
    """Build the minimal xarray Dataset expected by find_all_profiles."""
    return xr.Dataset(
        {
            "P_smooth": ("t_slow", pressure.astype(float)),
            "W_smooth": ("t_slow", speed.astype(float)),
            "fs_slow": float(FS),
        },
        coords={"t_slow": t.astype(float)},
    )


def _abs_speed(pressure: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Absolute speed from the pressure gradient, with a minimum floor."""
    return np.maximum(np.abs(np.gradient(pressure, DT)) * 1.005, floor)


def _base_config(**overrides) -> ProfileConfig:
    """ProfileConfig with fixed detection parameters, suitable for synthetic data."""
    return ProfileConfig(
        min_speed=0.05,
        gap_threshold=2.0,
        gap_factor=4.0,
        peaks_kwargs={"height": 25, "distance": 200, "width": 200, "prominence": 25},
        **overrides,
    )


# ---------------------------------------------------------------------------
# 1. Single glider up-cast
# ---------------------------------------------------------------------------


class TestSingleGliderUpCast:
    """One up-cast: brief 20 s descent preamble (0 → 100 dbar) then a 500 s
    ascent (100 → 0 dbar).

    The preamble creates a strict local pressure maximum that the peak-based
    detection code can find.  The long ascent is the segment of interest.
    """

    @pytest.fixture
    def ds(self):
        n_down = int(20 * FS)
        n_up = int(500 * FS)
        p_down = np.linspace(0.0, 100.0, n_down)
        p_up = np.linspace(100.0, 0.0, n_up)
        pressure = np.concatenate([p_down, p_up])
        # Nudge the turnaround sample so it is a strict local maximum
        pressure[n_down - 1] += 0.01
        t = np.arange(len(pressure)) * DT
        return _make_ds(t, pressure, _abs_speed(pressure))

    def test_up_direction_finds_one_profile(self, ds):
        """direction='up' should detect exactly one ascending segment."""
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        assert len(segs) == 1

    def test_up_segment_pressure_decreases(self, ds):
        """The ascending segment must have lower pressure at its end than its start."""
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        s, e = segs[0]
        p = ds["P_smooth"].values
        assert p[e] < p[s], (
            "ascending segment: pressure must decrease from start to end"
        )

    def test_down_direction_finds_preamble_descent(self, ds):
        """direction='down' should extract the short descent preamble."""
        segs = find_all_profiles(ds, _base_config(profile_direction="down"))
        assert len(segs) == 1
        s, e = segs[0]
        p = ds["P_smooth"].values
        assert p[e] > p[s], "descent preamble: pressure must increase from start to end"


# ---------------------------------------------------------------------------
# 2. Single glider down-cast
# ---------------------------------------------------------------------------


class TestSingleGliderDownCast:
    """One down-cast: 500 s descent (0 → 100 dbar) then a brief 20 s ascent
    preamble (100 → 0 dbar).

    The preamble creates a strict local maximum at the deepest point so that
    the peak-based detection code can identify the profile.
    """

    @pytest.fixture
    def ds(self):
        n_down = int(500 * FS)
        n_up = int(20 * FS)
        p_down = np.linspace(0.0, 100.0, n_down)
        p_up = np.linspace(100.0, 0.0, n_up)
        pressure = np.concatenate([p_down, p_up])
        # Nudge the turnaround sample so it is a strict local maximum
        pressure[n_down - 1] += 0.01
        t = np.arange(len(pressure)) * DT
        return _make_ds(t, pressure, _abs_speed(pressure))

    def test_down_direction_finds_one_profile(self, ds):
        """direction='down' should detect exactly one descending segment."""
        segs = find_all_profiles(ds, _base_config(profile_direction="down"))
        assert len(segs) == 1

    def test_down_segment_pressure_increases(self, ds):
        """The descending segment must have higher pressure at its end than its start."""
        segs = find_all_profiles(ds, _base_config(profile_direction="down"))
        s, e = segs[0]
        p = ds["P_smooth"].values
        assert p[e] > p[s], "descent segment: pressure must increase from start to end"

    def test_up_direction_finds_preamble_ascent(self, ds):
        """direction='up' should extract the short ascent preamble."""
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        assert len(segs) == 1
        s, e = segs[0]
        p = ds["P_smooth"].values
        assert p[e] < p[s], "ascent preamble: pressure must decrease from start to end"


# ---------------------------------------------------------------------------
# 3. Merged glider dataset (gap-based detection)
# ---------------------------------------------------------------------------


class TestMergedGliderUpCasts:
    """Five independent up-cast segments joined into one dataset with 2000 s
    time gaps between them.

    This exercises the gap-based detection path: the time jumps trigger gap
    detection and each contiguous segment is evaluated as a separate profile.
    """

    N_PROFILES = 5
    CAST_DURATION_S = 500.0
    MAX_DEPTH_DBAR = 100.0
    GAP_S = 2000.0

    @pytest.fixture
    def ds(self):
        n_cast = int(self.CAST_DURATION_S * FS)
        t_off = 0.0
        t_parts, p_parts = [], []
        for _ in range(self.N_PROFILES):
            t_seg = np.arange(n_cast) * DT + t_off
            p_seg = np.linspace(self.MAX_DEPTH_DBAR, 0.0, n_cast)
            t_parts.append(t_seg)
            p_parts.append(p_seg)
            t_off = t_seg[-1] + self.GAP_S
        t = np.concatenate(t_parts)
        pressure = np.concatenate(p_parts)
        return _make_ds(t, pressure, _abs_speed(pressure))

    def test_finds_all_up_cast_profiles(self, ds):
        """Gap-based detection should find one profile per gap-separated segment."""
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        assert len(segs) == self.N_PROFILES

    def test_finds_no_down_profiles(self, ds):
        """All segments are ascending; direction='down' should return nothing."""
        segs = find_all_profiles(ds, _base_config(profile_direction="down"))
        assert len(segs) == 0

    def test_all_detected_segments_are_ascending(self, ds):
        """Every detected segment must have decreasing pressure (ascending cast)."""
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        p = ds["P_smooth"].values
        for s, e in segs:
            assert p[e] < p[s], f"segment [{s},{e}] does not have decreasing pressure"

    def test_shallow_segments_are_filtered_out(self):
        """Segments whose maximum pressure is below peaks_kwargs['height'] (25 dbar)
        must be excluded even when they have the right direction."""
        n_cast = int(200 * FS)
        dt_offset = n_cast * DT + 2000.0  # gap of 2000 s between segments

        # Three segments: deep, shallow, deep
        t1 = np.arange(n_cast) * DT
        t2 = t1 + dt_offset
        t3 = t2 + dt_offset
        pressure = np.concatenate(
            [
                np.linspace(100.0, 0.0, n_cast),  # deep up-cast (passes)
                np.linspace(
                    10.0, 0.0, n_cast
                ),  # shallow (10 dbar < height=25; excluded)
                np.linspace(100.0, 0.0, n_cast),  # deep up-cast (passes)
            ]
        )
        t = np.concatenate([t1, t2, t3])
        ds = _make_ds(t, pressure, _abs_speed(pressure))
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        assert len(segs) == 2  # only the two deep casts survive


# ---------------------------------------------------------------------------
# 4. Continuous VMP dataset (peak-based detection, signed-velocity fix)
# ---------------------------------------------------------------------------


class TestContinuousVMPDownProfiles:
    """Five continuous dive/ascent cycles as produced by a free-fall VMP.

    The instrument records shear data only on the way down, so the caller
    passes direction='down'.  This exercises the signed-velocity fix: the
    `W_smooth` variable is always positive (|dP/dt|), but `find_all_profiles`
    internally computes a signed velocity from the smoothed pressure so that
    profinder can correctly distinguish descent (positive) from ascent
    (negative) and apply the speed threshold.
    """

    N_CYCLES = 5
    CYCLE_DURATION_S = 2000.0
    MAX_DEPTH_DBAR = 760.0

    @pytest.fixture
    def ds(self):
        n = int(self.N_CYCLES * self.CYCLE_DURATION_S * FS)
        t = np.arange(n) * DT
        # Smooth cosine cycles from 0 to MAX_DEPTH and back, N_CYCLES times
        pressure = (self.MAX_DEPTH_DBAR / 2) * (
            1 - np.cos(2 * np.pi * t / self.CYCLE_DURATION_S)
        )
        return _make_ds(t, pressure, _abs_speed(pressure))

    def test_down_direction_finds_five_profiles(self, ds):
        """direction='down' must extract exactly one segment per descent."""
        segs = find_all_profiles(ds, _base_config(profile_direction="down"))
        assert len(segs) == self.N_CYCLES

    def test_up_direction_finds_five_profiles(self, ds):
        """direction='up' must extract exactly one segment per ascent."""
        segs = find_all_profiles(ds, _base_config(profile_direction="up"))
        assert len(segs) == self.N_CYCLES

    def test_both_direction_finds_ten_profiles(self, ds):
        """direction='both' must return one descent + one ascent per cycle."""
        segs = find_all_profiles(ds, _base_config(profile_direction="both"))
        assert len(segs) == 2 * self.N_CYCLES

    def test_down_segments_have_increasing_pressure(self, ds):
        """Every descent segment must have higher pressure at its end than its start."""
        segs = find_all_profiles(ds, _base_config(profile_direction="down"))
        p = ds["P_smooth"].values
        for s, e in segs:
            assert p[e] > p[s], (
                f"down segment [{s},{e}]: pressure at end ({p[e]:.1f}) "
                f"must exceed pressure at start ({p[s]:.1f})"
            )

    def test_signed_velocity_not_required_in_w_smooth(self, ds):
        """W_smooth is always positive (absolute speed); the signed-velocity
        computation inside find_all_profiles must not rely on W_smooth having
        the correct sign for up/down discrimination."""
        p = ds["P_smooth"].values
        # Overwrite W_smooth with all-positive values (mimics estimate_speed_from_pressure)
        speed_positive = np.full_like(p, 0.5)
        ds_abs = _make_ds(ds["t_slow"].values, p, speed_positive)
        segs = find_all_profiles(ds_abs, _base_config(profile_direction="down"))
        assert len(segs) == self.N_CYCLES

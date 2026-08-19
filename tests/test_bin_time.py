"""Tests for time handling in bin_profiles across files with different
reference epochs (each eps file's ``time``/``ctd_time`` is stored as raw
seconds since that source p-file's own ``filetime``), and the ``profile_time``
coordinate.
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
DAY = 86400.0


def _make_eps_file(tmp_path_factory, label, epoch: np.datetime64):
    """Process the shared test profile and relabel its time coordinates'
    reference epoch, simulating a distinct cast recorded at ``epoch`` --
    without altering the underlying raw elapsed-time values.
    """
    raw = to_xarray(load_pfile_phys(PFILE))
    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
    )
    result = process_profile(raw.copy(deep=True), config)

    units = f"seconds since {str(epoch).replace('T', ' ')}"
    for tdim in ("time", "ctd_time"):
        if tdim in result.coords:
            result[tdim].attrs["units"] = units

    out_dir = tmp_path_factory.mktemp(f"eps_{label}")
    out_file = out_dir / f"profile_{label}.nc"
    _write_epsilon_profile(result, raw, out_file, f"{label}.p", 0, config)
    return out_file


@pytest.fixture(scope="module")
def two_day_apart_files(tmp_path_factory):
    """Two eps files whose real-world casts are exactly one day apart, but
    which were passed to bin_profiles in reverse chronological order.
    """
    day0 = np.datetime64("2026-01-01T00:00:00")
    day1 = day0 + np.timedelta64(int(DAY), "s")
    file_day0 = _make_eps_file(tmp_path_factory, "day0", day0)
    file_day1 = _make_eps_file(tmp_path_factory, "day1", day1)
    return file_day1, file_day0  # reverse order on purpose


class TestCrossFileTimeEpochs:
    def test_time_values_are_real_dates_not_1970(self, two_day_apart_files, tmp_path):
        binned = bin_profiles(
            list(two_day_apart_files),
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        t = binned["time"].values
        finite = t[np.isfinite(t)]
        assert finite.size > 0
        # Seconds since 1970 for a 2026 date is ~1.77e9; 1970 itself would be ~0.
        assert np.all(finite > 1_700_000_000)

    def test_profiles_sorted_chronologically(self, two_day_apart_files, tmp_path):
        binned = bin_profiles(
            list(two_day_apart_files),
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        assert "profile_time" in binned.coords
        assert binned["profile_time"].dims == ("profile",)
        pt = binned["profile_time"].values
        assert pt[0] < pt[1]

    def test_profile_time_gap_matches_real_epoch_offset(
        self, two_day_apart_files, tmp_path
    ):
        binned = bin_profiles(
            list(two_day_apart_files),
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        pt = binned["profile_time"].values
        np.testing.assert_allclose(pt[1] - pt[0], DAY, atol=0.01)

    def test_profile_time_round_trips_to_expected_calendar_dates(
        self, two_day_apart_files, tmp_path
    ):
        binned = bin_profiles(
            list(two_day_apart_files),
            output_file=tmp_path / "binned.nc",
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            bin_width=2.0,
        )
        decoded = xr.decode_cf(binned[["profile_time"]])["profile_time"].values
        assert str(decoded[0])[:10] == "2026-01-01"
        assert str(decoded[1])[:10] == "2026-01-02"

"""Tests for the profile index (build_profile_index, batch_index_profiles,
extract_profile).
"""

import numpy as np
import pytest
import xarray as xr

from pyturb.profile import ProfileConfig, prepare_profile
from pyturb.profile_index import (
    batch_index_profiles,
    build_profile_index,
    extract_profile,
)

FS_SLOW = 64.0
FS_FAST = 512.0
RATIO = int(FS_FAST / FS_SLOW)


def _base_config(**overrides) -> ProfileConfig:
    return ProfileConfig(
        min_speed=0.05,
        gap_threshold=2.0,
        gap_factor=4.0,
        peaks_kwargs={"height": 25, "distance": 200, "width": 200, "prominence": 25},
        **overrides,
    )


def _make_raw_ds(pressure: np.ndarray) -> xr.Dataset:
    """A raw (p2nc-style) dataset: raw P on t_slow, a noise probe on t_fast."""
    n_slow = len(pressure)
    n_fast = n_slow * RATIO
    t_slow = np.arange(n_slow) / FS_SLOW
    t_fast = np.arange(n_fast) / FS_FAST
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "P": ("t_slow", pressure.astype(float)),
            "sh1": ("t_fast", rng.normal(size=n_fast)),
            "fs_slow": FS_SLOW,
            "fs_fast": FS_FAST,
        },
        coords={"t_slow": t_slow, "t_fast": t_fast},
    )


@pytest.fixture
def down_up_ds():
    """20 s descent preamble then a 500 s ascent (mirrors test_profile_detection)."""
    n_down = int(20 * FS_SLOW)
    n_up = int(500 * FS_SLOW)
    p_down = np.linspace(0.0, 100.0, n_down)
    p_up = np.linspace(100.0, 0.0, n_up)
    pressure = np.concatenate([p_down, p_up])
    pressure[n_down - 1] += 0.01
    return _make_raw_ds(pressure)


class TestBuildProfileIndex:
    def test_finds_up_segment(self, down_up_ds):
        config = _base_config(profile_direction="up")
        prepared = prepare_profile(down_up_ds, config)
        idx_ds = build_profile_index(prepared, config)

        assert idx_ds.sizes["profile"] == 1
        assert idx_ds["direction"].values[0] == "up"
        assert idx_ds["start_time"].values[0] < idx_ds["end_time"].values[0]

    def test_indices_bracket_the_up_leg(self, down_up_ds):
        config = _base_config(profile_direction="up")
        prepared = prepare_profile(down_up_ds, config)
        idx_ds = build_profile_index(prepared, config)

        s = int(idx_ds["start_idx"].values[0])
        e = int(idx_ds["end_idx"].values[0])
        p = prepared["P_smooth"].values
        assert p[e] < p[s]  # ascending: pressure decreases start -> end

    def test_no_fast_channel_columns(self, down_up_ds):
        # Detection must never touch/copy the fast-channel probe data.
        config = _base_config(profile_direction="up")
        prepared = prepare_profile(down_up_ds, config)
        idx_ds = build_profile_index(prepared, config)
        assert "t_fast" not in idx_ds.dims
        assert "sh1" not in idx_ds


class TestBatchIndexAndExtract:
    def test_round_trip(self, down_up_ds, tmp_path):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)

        config = _base_config(profile_direction="up")
        results = batch_index_profiles(
            [raw_file], config=config, output_dir=tmp_path, n_workers=1
        )

        assert len(results) == 1
        assert results[0]["success"]
        indices_file = results[0]["output"]
        assert indices_file.name == "FILE001_profiles.nc"

        profile_ds = extract_profile(indices_file, raw_file, profile_index=0)

        assert "sh1" in profile_ds
        assert "t_fast" in profile_ds.dims
        # Fast-channel length should match the slow-channel span * RATIO.
        idx_ds = xr.load_dataset(indices_file)
        s = int(idx_ds["start_idx"].values[0])
        e = int(idx_ds["end_idx"].values[0])
        expected_slow = e - s + 1
        assert profile_ds.sizes["t_slow"] == expected_slow

    def test_extract_unknown_profile_index_raises(self, down_up_ds, tmp_path):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)
        config = _base_config(profile_direction="up")
        results = batch_index_profiles(
            [raw_file], config=config, output_dir=tmp_path, n_workers=1
        )
        indices_file = results[0]["output"]

        with pytest.raises(ValueError, match="profile_index"):
            extract_profile(indices_file, raw_file, profile_index=99)

    def test_skips_existing_by_default(self, down_up_ds, tmp_path):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)
        config = _base_config(profile_direction="up")

        batch_index_profiles(
            [raw_file], config=config, output_dir=tmp_path, n_workers=1
        )
        results = batch_index_profiles(
            [raw_file], config=config, output_dir=tmp_path, n_workers=1
        )

        assert results[0]["error"] == "skipped (exists)"


class TestMaterialize:
    def test_writes_hires_file_per_profile(self, down_up_ds, tmp_path):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)
        config = _base_config(profile_direction="up")

        results = batch_index_profiles(
            [raw_file],
            config=config,
            output_dir=tmp_path,
            n_workers=1,
            materialize=True,
        )

        assert results[0]["success"]
        hires = results[0]["materialized"]
        assert len(hires) == 1
        hires_file, error = hires[0]
        assert error is None
        assert hires_file.name == "FILE001_p0000_hires.nc"
        assert hires_file.exists()

        hires_ds = xr.load_dataset(hires_file)
        assert "sh1" in hires_ds
        assert "t_fast" in hires_ds.dims
        assert hires_ds.attrs["profile_index"] == 0
        assert hires_ds.attrs["profile_direction"] == "up"

    def test_no_hires_files_by_default(self, down_up_ds, tmp_path):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)
        config = _base_config(profile_direction="up")

        results = batch_index_profiles(
            [raw_file], config=config, output_dir=tmp_path, n_workers=1
        )

        assert results[0]["materialized"] == []
        assert not list(tmp_path.glob("*_hires.nc"))

    def test_materialize_reruns_detection_even_if_index_exists(
        self, down_up_ds, tmp_path
    ):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)
        config = _base_config(profile_direction="up")

        batch_index_profiles(
            [raw_file], config=config, output_dir=tmp_path, n_workers=1
        )
        results = batch_index_profiles(
            [raw_file],
            config=config,
            output_dir=tmp_path,
            n_workers=1,
            materialize=True,
        )

        assert results[0]["success"]
        assert len(results[0]["materialized"]) == 1
        assert results[0]["materialized"][0][1] is None

    def test_skips_existing_hires_file_without_overwrite(self, down_up_ds, tmp_path):
        raw_file = tmp_path / "FILE001.nc"
        down_up_ds.to_netcdf(raw_file)
        config = _base_config(profile_direction="up")

        batch_index_profiles(
            [raw_file],
            config=config,
            output_dir=tmp_path,
            n_workers=1,
            materialize=True,
        )
        results = batch_index_profiles(
            [raw_file],
            config=config,
            output_dir=tmp_path,
            n_workers=1,
            materialize=True,
        )

        assert results[0]["materialized"][0][1] == "skipped (exists)"

"""Tests for the eps --skip-existing fast path."""

from pyturb.processing import _has_existing_outputs, _process_file
from pyturb.profile import ProfileConfig


class TestHasExistingOutputs:
    def test_detects_prefix_match(self, tmp_path):
        (tmp_path / "FILE001_p0000.nc").touch()
        assert _has_existing_outputs("FILE001", tmp_path)

    def test_no_match(self, tmp_path):
        (tmp_path / "FILE001_p0000.nc").touch()
        assert not _has_existing_outputs("FILE002", tmp_path)

    def test_missing_output_dir(self, tmp_path):
        assert not _has_existing_outputs("FILE001", tmp_path / "missing")


class TestProcessFileSkipExisting:
    """input_file is never created: the skip path must short-circuit before
    any attempt to load it, so a load failure would surface as a different
    error string.
    """

    def test_skips_when_output_exists(self, tmp_path):
        (tmp_path / "FILE001_p0000.nc").touch()
        input_file = tmp_path / "FILE001.nc"

        results = _process_file(
            input_file, tmp_path, ProfileConfig(), overwrite=False, skip_existing=True
        )

        assert results == [(input_file, None, -1, "skipped (existing outputs found)")]

    def test_off_by_default(self, tmp_path):
        (tmp_path / "FILE001_p0000.nc").touch()
        input_file = tmp_path / "FILE001.nc"

        results = _process_file(input_file, tmp_path, ProfileConfig(), overwrite=False)

        assert results[0][3] != "skipped (existing outputs found)"

    def test_ignored_when_overwrite_is_set(self, tmp_path):
        (tmp_path / "FILE001_p0000.nc").touch()
        input_file = tmp_path / "FILE001.nc"

        results = _process_file(
            input_file, tmp_path, ProfileConfig(), overwrite=True, skip_existing=True
        )

        assert results[0][3] != "skipped (existing outputs found)"

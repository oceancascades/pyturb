"""Tests for extract_pfile_segment (cutp functionality)."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pyturb._pfile import extract_pfile_segment, read_pfile
from pyturb.cli import app

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0002.p"

runner = CliRunner()


class TestExtractPfileSegment:
    """Test the low-level extract function."""

    def test_output_file_created(self, tmp_path):
        out = tmp_path / "segment.p"
        result = extract_pfile_segment(PFILE, out, start_record=0, n_records=5)
        assert result == out
        assert out.exists()

    def test_output_smaller_than_input(self, tmp_path):
        out = tmp_path / "segment.p"
        extract_pfile_segment(PFILE, out, start_record=0, n_records=5)
        assert out.stat().st_size < PFILE.stat().st_size

    def test_round_trip_read(self, tmp_path):
        """Extracted file should be readable by read_pfile."""
        out = tmp_path / "segment.p"
        n = 10
        extract_pfile_segment(PFILE, out, start_record=0, n_records=n)
        data = read_pfile(out)
        # Should have the correct number of fast samples
        # (n_records * rows * fast_columns_per_row)
        assert len(data["t_fast"]) > 0
        assert len(data["t_slow"]) > 0

    def test_correct_sample_count(self, tmp_path):
        """Number of samples should scale with n_records."""
        out_5 = tmp_path / "seg5.p"
        out_10 = tmp_path / "seg10.p"
        extract_pfile_segment(PFILE, out_5, start_record=0, n_records=5)
        extract_pfile_segment(PFILE, out_10, start_record=0, n_records=10)
        data_5 = read_pfile(out_5)
        data_10 = read_pfile(out_10)
        assert len(data_10["t_fast"]) == 2 * len(data_5["t_fast"])

    def test_channels_preserved(self, tmp_path):
        """Extracted file should have all the same channels."""
        out = tmp_path / "segment.p"
        extract_pfile_segment(PFILE, out, start_record=0, n_records=5)
        orig = read_pfile(PFILE)
        seg = read_pfile(out)
        for ch in ["sh1", "sh2", "Ax", "Ay", "T1", "T2", "P"]:
            assert ch in seg, f"Missing channel: {ch}"

    def test_data_matches_original(self, tmp_path):
        """First records should have identical data to the original file."""
        out = tmp_path / "segment.p"
        n = 5
        extract_pfile_segment(PFILE, out, start_record=0, n_records=n)
        orig = read_pfile(PFILE)
        seg = read_pfile(out)
        n_fast = len(seg["t_fast"])
        np.testing.assert_array_equal(seg["sh1"], orig["sh1"][:n_fast])

    def test_start_record_offset(self, tmp_path):
        """Starting at a later record should skip early data."""
        out_0 = tmp_path / "from0.p"
        out_5 = tmp_path / "from5.p"
        extract_pfile_segment(PFILE, out_0, start_record=0, n_records=10)
        extract_pfile_segment(PFILE, out_5, start_record=5, n_records=5)
        data_0 = read_pfile(out_0)
        data_5 = read_pfile(out_5)
        # The from5 data should match the second half of from0
        n = len(data_5["sh1"])
        np.testing.assert_array_equal(data_5["sh1"], data_0["sh1"][-n:])

    def test_sampling_rates_preserved(self, tmp_path):
        """Sampling rates should be identical in extracted file."""
        out = tmp_path / "segment.p"
        extract_pfile_segment(PFILE, out, start_record=0, n_records=5)
        orig = read_pfile(PFILE)
        seg = read_pfile(out)
        assert seg["fs_fast"] == pytest.approx(orig["fs_fast"])
        assert seg["fs_slow"] == pytest.approx(orig["fs_slow"])

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "segment.p"
        extract_pfile_segment(PFILE, out, start_record=0, n_records=5)
        assert out.exists()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_pfile_segment(tmp_path / "nonexistent.p", tmp_path / "out.p")

    def test_start_record_out_of_range(self, tmp_path):
        with pytest.raises(ValueError, match="out of range"):
            extract_pfile_segment(PFILE, tmp_path / "out.p", start_record=9999)

    def test_negative_start_record(self, tmp_path):
        with pytest.raises(ValueError, match="out of range"):
            extract_pfile_segment(PFILE, tmp_path / "out.p", start_record=-1)

    def test_too_many_records(self, tmp_path):
        with pytest.raises(ValueError, match="only .* are available"):
            extract_pfile_segment(
                PFILE, tmp_path / "out.p", start_record=0, n_records=99999
            )


class TestCutpCommand:
    """Test the cutp CLI command."""

    def test_basic_extraction(self, tmp_path):
        out = tmp_path / "segment.p"
        result = runner.invoke(
            app,
            [
                "cutp",
                "--output",
                str(out),
                "--start",
                "0",
                "--n-records",
                "5",
                str(PFILE),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "Wrote 5 records" in result.output

    def test_output_is_valid_pfile(self, tmp_path):
        out = tmp_path / "segment.p"
        runner.invoke(
            app,
            ["cutp", "-o", str(out), "-n", "5", str(PFILE)],
        )
        data = read_pfile(out)
        assert "sh1" in data
        assert len(data["t_fast"]) > 0

    def test_no_input_file_errors(self):
        result = runner.invoke(app, ["cutp", "-o", "/tmp/out.p"])
        assert result.exit_code != 0

    def test_invalid_start_errors(self, tmp_path):
        out = tmp_path / "segment.p"
        result = runner.invoke(
            app,
            ["cutp", "-o", str(out), "-s", "99999", str(PFILE)],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_too_many_records_errors(self, tmp_path):
        out = tmp_path / "segment.p"
        result = runner.invoke(
            app,
            ["cutp", "-o", str(out), "-n", "99999", str(PFILE)],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

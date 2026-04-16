"""Tests for the CLI (typer app)."""

from pathlib import Path

import pytest
import xarray as xr
from typer.testing import CliRunner

from pyturb.cli import app

runner = CliRunner()

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0002.p"


class TestP2ncCommand:
    """Test the p2nc CLI command."""

    def test_converts_single_file(self, tmp_path):
        result = runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), str(PFILE)],
        )
        assert result.exit_code == 0, result.output
        nc_file = tmp_path / "RIOTSHAKE_VMP142_0002.nc"
        assert nc_file.exists()

    def test_output_is_valid_netcdf(self, tmp_path):
        runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), str(PFILE)],
        )
        nc_file = tmp_path / "RIOTSHAKE_VMP142_0002.nc"
        ds = xr.open_dataset(nc_file)
        assert "t_fast" in ds.coords
        assert "t_slow" in ds.coords
        assert len(ds.data_vars) > 0
        ds.close()

    def test_no_overwrite_by_default(self, tmp_path):
        # First convert
        runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), str(PFILE)],
        )
        # Second convert should skip (not error)
        result = runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), str(PFILE)],
        )
        assert result.exit_code == 0

    def test_overwrite_flag(self, tmp_path):
        runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), str(PFILE)],
        )
        nc_file = tmp_path / "RIOTSHAKE_VMP142_0002.nc"
        mtime1 = nc_file.stat().st_mtime

        result = runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), "--overwrite", str(PFILE)],
        )
        assert result.exit_code == 0
        mtime2 = nc_file.stat().st_mtime
        assert mtime2 > mtime1

    def test_compress_flag(self, tmp_path):
        result = runner.invoke(
            app,
            ["p2nc", "--output", str(tmp_path), "--compress", str(PFILE)],
        )
        assert result.exit_code == 0
        nc_file = tmp_path / "RIOTSHAKE_VMP142_0002.nc"
        assert nc_file.exists()

    def test_no_input_files_errors(self):
        result = runner.invoke(app, ["p2nc"])
        assert result.exit_code != 0

    def test_log_level_option(self, tmp_path):
        result = runner.invoke(
            app,
            ["--log-level", "debug", "p2nc", "--output", str(tmp_path), str(PFILE)],
        )
        assert result.exit_code == 0


class TestVersionFlag:
    """Test --version flag."""

    def test_version_output(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "pyturb version" in result.output

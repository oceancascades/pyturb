"""Tests for the CLI (typer app)."""

from pathlib import Path

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


class TestCalibrateFp07Commands:
    """Test the calibrate-fp07 fit/apply/auto CLI commands."""

    # A different test p-file: this one is pre-trimmed to a single clean
    # detectable profile under the default ProfileConfig, unlike PFILE above.
    CAL_PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"

    def _convert(self, tmp_path: Path, name: str = "converted.nc") -> Path:
        runner.invoke(app, ["p2nc", "--output", str(tmp_path), str(self.CAL_PFILE)])
        src = tmp_path / f"{self.CAL_PFILE.stem}.nc"
        dst = tmp_path / name
        if dst != src:
            dst.write_bytes(src.read_bytes())
        return dst

    def test_fit_writes_report(self, tmp_path):
        converted = self._convert(tmp_path)
        report = tmp_path / "cal.yaml"
        result = runner.invoke(
            app,
            [
                "calibrate-fp07",
                "fit",
                str(converted),
                "--profile",
                "0",
                "--order",
                "1",
                "-o",
                str(report),
            ],
        )
        assert result.exit_code == 0, result.output
        assert report.exists()
        assert "T1" in report.read_text()

    def test_fit_then_apply_changes_gradT(self, tmp_path):
        converted = self._convert(tmp_path)
        report = tmp_path / "cal.yaml"
        runner.invoke(
            app,
            [
                "calibrate-fp07",
                "fit",
                str(converted),
                "--profile",
                "0",
                "--order",
                "1",
                "-o",
                str(report),
            ],
        )
        before = xr.load_dataset(converted, decode_times=False)["gradT1"].values.copy()

        result = runner.invoke(
            app,
            [
                "calibrate-fp07",
                "apply",
                str(report),
                str(converted),
                "--overwrite",
            ],
        )
        assert result.exit_code == 0, result.output
        after = xr.load_dataset(converted, decode_times=False)["gradT1"].values
        assert not (before == after).all()

    def test_apply_requires_overwrite_or_output(self, tmp_path):
        converted = self._convert(tmp_path)
        report = tmp_path / "cal.yaml"
        runner.invoke(
            app,
            [
                "calibrate-fp07",
                "fit",
                str(converted),
                "--profile",
                "0",
                "--order",
                "1",
                "-o",
                str(report),
            ],
        )
        result = runner.invoke(
            app, ["calibrate-fp07", "apply", str(report), str(converted)]
        )
        assert result.exit_code != 0

    def test_auto_groups_and_applies_across_files(self, tmp_path):
        file_a = self._convert(tmp_path, "a.nc")
        file_b = self._convert(tmp_path, "b.nc")
        before_a = xr.load_dataset(file_a, decode_times=False)["gradT1"].values.copy()
        before_b = xr.load_dataset(file_b, decode_times=False)["gradT1"].values.copy()

        report = tmp_path / "auto_cal.yaml"
        result = runner.invoke(
            app,
            [
                "calibrate-fp07",
                "auto",
                str(file_a),
                str(file_b),
                "--order",
                "1",
                "--overwrite",
                "-r",
                str(report),
            ],
        )
        assert result.exit_code == 0, result.output
        assert report.exists()
        assert "T1" in report.read_text() and "T2" in report.read_text()

        after_a = xr.load_dataset(file_a, decode_times=False)["gradT1"].values
        after_b = xr.load_dataset(file_b, decode_times=False)["gradT1"].values
        assert not (before_a == after_a).all()
        assert not (before_b == after_b).all()

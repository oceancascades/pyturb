"""Tests for the CLI (typer app)."""

from pathlib import Path

import numpy as np
import xarray as xr
from typer.testing import CliRunner

from pyturb.cli import (
    _fit_from_middle_of_group,
    _is_railed,
    app,
)
from pyturb.fp07_calibration import ProbeCalibrationFit
from pyturb.profile import ProfileConfig

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


class TestIsRailed:
    """A probe channel that dies mid-deployment (broken connection, shorted
    or open thermistor) rails its raw counts to the ADC's saturation limit
    -- must never be picked as a fit source."""

    def test_flags_counts_pinned_near_negative_rail(self):
        counts = np.full(1000, -32378.0)
        assert _is_railed(counts, adc_bits=16)

    def test_flags_counts_pinned_near_positive_rail(self):
        counts = np.full(1000, 32700.0)
        assert _is_railed(counts, adc_bits=16)

    def test_does_not_flag_normal_varying_counts(self):
        rng = np.random.default_rng(0)
        counts = rng.normal(0, 2000, size=1000)
        assert not _is_railed(counts, adc_bits=16)

    def test_partial_railing_below_threshold_not_flagged(self):
        rng = np.random.default_rng(0)
        counts = rng.normal(0, 2000, size=1000)
        counts[:100] = -32700.0  # 10% railed, below _RAIL_FRACTION_THRESHOLD
        assert not _is_railed(counts, adc_bits=16)

    def test_majority_railing_flagged(self):
        rng = np.random.default_rng(0)
        counts = rng.normal(0, 2000, size=1000)
        counts[:500] = -32700.0  # 50% railed, above threshold
        assert _is_railed(counts, adc_bits=16)


class TestFitFromMiddleOfGroupSkipsRailedCandidates:
    """Integration test: a railed candidate profile must be skipped in
    favor of a healthy one elsewhere in the group, not accepted just
    because it happens to have a confident lag/low residual on its own
    (collapsed) segment."""

    CAL_PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"

    def _convert(self, tmp_path: Path, name: str) -> Path:
        runner.invoke(app, ["p2nc", "--output", str(tmp_path), str(self.CAL_PFILE)])
        src = tmp_path / f"{self.CAL_PFILE.stem}.nc"
        dst = tmp_path / name
        dst.write_bytes(src.read_bytes())
        return dst

    def _rail(self, path: Path) -> None:
        ds = xr.load_dataset(path, decode_times=False)
        ds["T1_counts"] = (
            ds["T1_counts"].dims,
            np.full_like(ds["T1_counts"].values, -32378.0),
        )
        ds.to_netcdf(path)

    def test_skips_railed_middle_file_and_uses_a_healthy_one(self, tmp_path):
        file_a = self._convert(tmp_path, "a.nc")
        file_b = self._convert(tmp_path, "b.nc")  # sorts to the "middle" of 3
        file_c = self._convert(tmp_path, "c.nc")
        self._rail(file_b)

        config = ProfileConfig()
        fit = _fit_from_middle_of_group(
            [file_a, file_b, file_c], "T1", config, "JAC_T", order=1, min_range=1.0
        )
        assert fit is not None
        assert fit.fit_file != "b.nc"


def _make_fit(**overrides) -> ProbeCalibrationFit:
    """A synthetic fit with sane defaults; pass e.g. new_T_0=713.0 to make
    one field implausible for a specific test."""
    fields = dict(
        probe="T1",
        sn="SYN1",
        instrument_sn="142",
        fit_file="synthetic",
        profile_index=0,
        reference="JAC_T",
        order=1,
        old_T_0=289.301,
        old_beta_1=3143.55,
        old_beta_2=None,
        new_T_0=289.0,
        new_beta_1=3100.0,
        new_beta_2=None,
        lag_s=0.0,
        lag_corr=0.95,
        n_points=100,
        temperature_range_c=10.0,
        rms_diff_old_c=0.0,
        rms_diff_new_c=0.0,
        max_abs_diff_old_c=0.0,
        max_abs_diff_new_c=0.0,
        mean_bias_old_c=0.0,
        mean_bias_new_c=0.0,
        residual_std_c=0.0,
        fit_date="2026-01-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return ProbeCalibrationFit(**fields)


class TestFitFromMiddleOfGroupSkipsImplausibleFits:
    """A confident lag and a low residual don't guarantee the fitted
    coefficients generalize -- _fit_from_middle_of_group must skip a
    plausibility-failing candidate even when it would otherwise be
    accepted immediately, and prefer a plausible-but-unconfident fit over
    an implausible one. _fit_from_middle_of_group now fits exclusively via
    fit_probe_calibration_multi (see TestFitFromMiddleOfGroupUsesAggregateOnly
    for why), so these monkeypatch that rather than fit_probe_calibration."""

    def test_prefers_plausible_unconfident_fit_over_implausible_confident_one(
        self, tmp_path, monkeypatch
    ):
        file_a = tmp_path / "a.nc"
        file_b = tmp_path / "b.nc"  # sorts to the "middle" of 2 -> tried first
        file_a.touch()
        file_b.touch()

        fits_by_file = {
            # tried first (middle of 2 files): confident lag, but T_0 way off
            "b.nc": _make_fit(
                fit_file="b.nc",
                profile_index=-1,
                new_T_0=713.7,
                new_beta_1=1484.5,
                lag_corr=0.92,
            ),
            # tried second: plausible coefficients, but a weak lag
            "a.nc": _make_fit(
                fit_file="a.nc",
                profile_index=-1,
                new_T_0=288.0,
                new_beta_1=2900.0,
                lag_corr=0.3,
            ),
        }

        monkeypatch.setattr("pyturb.cli.prepare_profile", lambda ds, config: ds)
        monkeypatch.setattr(
            "pyturb.cli.split_into_profiles", lambda ds, config: [(0, ds)]
        )
        monkeypatch.setattr(
            "pyturb.cli.load_profile_nc",
            lambda f: xr.Dataset({"JAC_T": ("t_slow", np.linspace(8.0, 18.0, 100))}),
        )
        monkeypatch.setattr(
            "pyturb.cli.fit_probe_calibration_multi",
            lambda profile_list, probe, config, fit_file, **kw: fits_by_file[fit_file],
        )

        config = ProfileConfig()
        fit = _fit_from_middle_of_group(
            [file_a, file_b], "T1", config, "JAC_T", order=2, min_range=8.0
        )
        assert fit is not None
        assert fit.fit_file == "a.nc"
        assert fit.new_T_0 == 288.0


class TestFitFromMiddleOfGroupUsesAggregateOnly:
    """calibrate-fp07 auto's candidate search fits exclusively via
    fit_probe_calibration_multi, which matches or beats the best
    single-profile fit and rescues a probe whose per-profile lag search is
    individually too noisy to trust. There's no single-profile fallback
    path left to exercise."""

    def test_accepts_a_confident_plausible_aggregate_fit(self, tmp_path, monkeypatch):
        file_a = tmp_path / "a.nc"
        file_a.touch()

        confident_fit = _make_fit(
            fit_file="a.nc", profile_index=-1, new_T_0=289.5, lag_corr=0.9
        )

        monkeypatch.setattr("pyturb.cli.prepare_profile", lambda ds, config: ds)
        monkeypatch.setattr(
            "pyturb.cli.split_into_profiles", lambda ds, config: [(0, ds)]
        )
        monkeypatch.setattr(
            "pyturb.cli.load_profile_nc",
            lambda f: xr.Dataset({"JAC_T": ("t_slow", np.linspace(8.0, 18.0, 100))}),
        )
        monkeypatch.setattr(
            "pyturb.cli.fit_probe_calibration_multi",
            lambda profile_list, probe, config, fit_file, **kw: confident_fit,
        )

        config = ProfileConfig()
        fit = _fit_from_middle_of_group(
            [file_a], "T1", config, "JAC_T", order=2, min_range=8.0
        )
        assert fit is confident_fit

    def test_moves_to_next_file_when_aggregate_fit_raises(self, tmp_path, monkeypatch):
        file_a = tmp_path / "a.nc"
        file_b = tmp_path / "b.nc"  # sorts to the "middle" of 2 -> tried first
        file_a.touch()
        file_b.touch()

        good_fit = _make_fit(fit_file="a.nc", profile_index=-1, lag_corr=0.9)

        def fake_multi(profile_list, probe, config, fit_file, **kw):
            if fit_file == "b.nc":
                raise ValueError("boom")
            return good_fit

        monkeypatch.setattr("pyturb.cli.prepare_profile", lambda ds, config: ds)
        monkeypatch.setattr(
            "pyturb.cli.split_into_profiles", lambda ds, config: [(0, ds)]
        )
        monkeypatch.setattr(
            "pyturb.cli.load_profile_nc",
            lambda f: xr.Dataset({"JAC_T": ("t_slow", np.linspace(8.0, 18.0, 100))}),
        )
        monkeypatch.setattr("pyturb.cli.fit_probe_calibration_multi", fake_multi)

        config = ProfileConfig()
        fit = _fit_from_middle_of_group(
            [file_a, file_b], "T1", config, "JAC_T", order=2, min_range=8.0
        )
        assert fit is good_fit

"""Global attributes at each stage: explicit sets, lineage, history, user attrs."""

import logging

import numpy as np
import pytest
import xarray as xr
import yaml
from typer.testing import CliRunner

from pyturb.cli import app
from pyturb.merge import merge_netcdf
from pyturb.metadata import load_global_attrs, validate_global_attrs
from pyturb.pfile import load_pfile_phys, save_netcdf
from pyturb.processing import (
    _write_epsilon_profile,
    batch_compute_epsilon,
    bin_profiles,
)
from pyturb.profile import ProfileConfig

from .conftest import PFILE

runner = CliRunner()

USER_ATTRS = {"title": "Test cruise", "institution": "OSU", "creator_name": "A. Person"}
INSTRUMENT = {"instrument_vehicle", "instrument_model", "instrument_sn"}
P2NC_ATTRS = {
    "Conventions",
    "title",
    "source_pfile",
    "pfile_start_time",
    "header_version",
    "fs_fast",
    "fs_slow",
    "pfile_configuration",
    "pyturb_version",
    "date_created",
    "history",
} | INSTRUMENT
EPS_ATTRS = (
    {
        "Conventions",
        "title",
        "source_pfile",
        "source_file",
        "profile_index",
        "profile_direction",
        "fs_fast",
        "fs_slow",
        "n_fft",
        "n_diss",
        "pyturb_version",
        "date_created",
        "pyturb_config",
        "history",
        "pyturb_user_attrs",
    }
    | INSTRUMENT
    | set(USER_ATTRS)
)
BIN_ATTRS = {
    "Conventions",
    "title",
    "instrument_vehicle",
    "instrument_model",
    "pyturb_version",
    "date_created",
    "pyturb_bin_config",
    "pyturb_eps_config",
    "history",
    "pyturb_user_attrs",
} | set(USER_ATTRS)


def _load(path):
    return xr.load_dataset(path, decode_times=False)


def _bin(files, out, **kwargs):
    return bin_profiles(files, out, depth_max=200, n_workers=4, **kwargs)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("metadata")
    p2nc = tmp / "RIOTSHAKE_VMP142_0002.nc"
    save_netcdf(load_pfile_phys(PFILE), p2nc, despike_kwargs={})
    result = runner.invoke(
        app, ["calibrate-jac-c", "142", str(p2nc), "--offset", "-0.05", "--overwrite"]
    )
    assert result.exit_code == 0, result.output
    batch_compute_epsilon(
        [p2nc], output_dir=tmp / "eps", n_workers=1, global_attrs=USER_ATTRS
    )
    eps = sorted((tmp / "eps").glob("*.nc"))
    _bin(eps, tmp / "bin.nc")
    return {"p2nc": p2nc, "eps": eps[0], "bin": tmp / "bin.nc", "tmp": tmp}


class TestGlobalSets:
    def test_p2nc(self, outputs):
        assert set(_load(outputs["p2nc"]).attrs) == P2NC_ATTRS

    def test_eps(self, outputs):
        assert set(_load(outputs["eps"]).attrs) == EPS_ATTRS

    def test_bin(self, outputs):
        assert set(_load(outputs["bin"]).attrs) == BIN_ATTRS

    def test_history_grows_one_line_per_step(self, outputs):
        p2nc = _load(outputs["p2nc"]).attrs["history"].splitlines()
        eps = _load(outputs["eps"]).attrs["history"].splitlines()
        assert ["p2nc:" in p2nc[0], "calibrate-jac-c:" in p2nc[1]] == [True, True]
        assert eps[:2] == p2nc and "eps:" in eps[2]
        assert "bin: 1 profiles" in _load(outputs["bin"]).attrs["history"]

    def test_lineage(self, outputs):
        eps = _load(outputs["eps"])
        assert eps.attrs["source_pfile"] == PFILE.name
        assert eps.attrs["source_file"] == outputs["p2nc"].name
        binned = _load(outputs["bin"])
        assert binned["source_pfile"].values[0] == PFILE.name
        assert binned["source_file"].values[0] == outputs["eps"].name
        assert binned["profile_direction"].values[0] in ("down", "up")

    def test_user_attrs_reach_eps_and_bin(self, outputs):
        for stage in ("eps", "bin"):
            attrs = _load(outputs[stage]).attrs
            assert {k: attrs[k] for k in USER_ATTRS} == USER_ATTRS
            assert attrs["pyturb_user_attrs"] == "title institution creator_name"


class TestVariableProvenance:
    def test_despike_params_from_p2nc(self, outputs):
        attrs = _load(outputs["eps"])["sh1_despike_frac"].attrs
        assert attrs["despike_source"] == "p2nc"
        assert attrs["despike_thresh"] == 8.0

    def test_despike_params_when_redone_in_eps(self, outputs, tmp_path):
        config = ProfileConfig(force_despike=True, despike_thresh=6.0)
        batch_compute_epsilon(
            [outputs["p2nc"]], config=config, output_dir=tmp_path, n_workers=1
        )
        attrs = _load(next(tmp_path.glob("*.nc")))["sh1_despike_frac"].attrs
        assert attrs["despike_source"] == "eps"
        assert attrs["despike_thresh"] == 6.0

    def test_calibration_provenance_carried(self, outputs):
        eps = _load(outputs["eps"])
        for name in ("conductivity", "conductivity_hires"):
            assert float(eps[name].attrs["JAC_C_offset_applied"]) == pytest.approx(
                -0.05
            )
        assert not any(k.startswith("cal_") for k in eps["T1"].attrs)


def test_eps_direction_is_actual_cast(tmp_path):
    result = xr.Dataset(
        {
            "eps": ("time", [1e-9, 1e-9]),
            "k": (("time", "frequency"), np.ones((2, 3))),
            "P_smooth": ("t_slow", [10.0, 5.0, 1.0]),
        },
        coords={"time": [0.0, 1.0], "frequency": [1.0, 2.0, 3.0]},
    )
    out = tmp_path / "eps.nc"
    _write_epsilon_profile(result, xr.Dataset(), out, "x.nc", 0, ProfileConfig())
    assert _load(out).attrs["profile_direction"] == "up"


class TestBinInheritance:
    def _variant(self, outputs, tmp_path, name, **attrs):
        ds = _load(outputs["eps"])
        ds.attrs.update(attrs)
        path = tmp_path / name
        ds.to_netcdf(path)
        return path

    def test_mismatch_warns_and_omits(self, outputs, tmp_path, caplog):
        other = self._variant(outputs, tmp_path, "b.nc", institution="Elsewhere")
        with caplog.at_level(logging.WARNING, logger="pyturb.metadata"):
            _bin([outputs["eps"], other], tmp_path / "bin.nc")
        attrs = _load(tmp_path / "bin.nc").attrs
        assert "institution" not in attrs
        assert attrs["creator_name"] == "A. Person"
        assert "institution" in caplog.text

    def test_own_attrs_override_without_warning(self, outputs, tmp_path, caplog):
        other = self._variant(outputs, tmp_path, "b.nc", institution="Elsewhere")
        with caplog.at_level(logging.WARNING, logger="pyturb.metadata"):
            _bin(
                [outputs["eps"], other],
                tmp_path / "bin.nc",
                global_attrs={"institution": "Chosen"},
            )
        assert _load(tmp_path / "bin.nc").attrs["institution"] == "Chosen"
        assert "institution" not in caplog.text

    def test_instrument_specific_var_attrs_removed(self, outputs):
        binned = _load(outputs["bin"])
        assert "JAC_C_offset_applied" not in binned["conductivity"].attrs
        assert binned["conductivity"].attrs["long_name"] == "Conductivity"

    def test_despike_frac_not_binned(self, outputs):
        assert not any("despike_frac" in v for v in _load(outputs["bin"]).variables)

    def test_floats_saved_as_float32(self, outputs):
        binned = _load(outputs["bin"])
        float64 = {v for v in binned.data_vars if binned[v].dtype == np.float64}
        assert float64 == {"time"}
        assert binned["eps"].dtype == np.float32

    def test_returned_dataset_stays_float64(self, outputs, tmp_path):
        assert _bin([outputs["eps"]], tmp_path / "bin.nc")["eps"].dtype == np.float64

    def test_vehicle_is_scalar(self, outputs):
        vehicle = _load(outputs["bin"])["instrument_vehicle"]
        assert vehicle.dims == ()
        assert str(vehicle.values) == "VMP"

    def test_mixed_vehicles_rejected(self, outputs, tmp_path):
        self._variant(outputs, tmp_path, "b.nc", instrument_vehicle="slocum_glider")
        with pytest.raises(ValueError, match="different vehicles"):
            _bin([outputs["eps"], tmp_path / "b.nc"], tmp_path / "bin.nc")
        assert not (tmp_path / "bin.nc").exists()

    def test_differing_var_attrs_dropped(self, outputs, tmp_path, caplog):
        ds = _load(outputs["eps"])
        ds["eps_qc"].attrs["comment"] = "different config"
        ds.to_netcdf(tmp_path / "b.nc")
        with caplog.at_level(logging.WARNING, logger="pyturb.processing"):
            _bin([outputs["eps"], tmp_path / "b.nc"], tmp_path / "bin.nc")
        attrs = _load(tmp_path / "bin.nc")["eps_qc"].attrs
        assert "comment" not in attrs
        assert "flag_values" in attrs
        assert "eps_qc:comment" in caplog.text

    def test_cli_eps_attrs_inherited_by_bin(self, outputs, tmp_path):
        attrs_file = tmp_path / "attrs.yml"
        attrs_file.write_text(yaml.safe_dump({"project": "CLI test"}))
        eps_dir = tmp_path / "eps"
        result = runner.invoke(
            app,
            [
                "eps",
                "-o",
                str(eps_dir),
                "-n",
                "1",
                "--attrs",
                str(attrs_file),
                str(outputs["p2nc"]),
            ],
        )
        assert result.exit_code == 0, result.output
        binned = tmp_path / "bin.nc"
        result = runner.invoke(
            app,
            [
                "bin",
                "-o",
                str(binned),
                "--dmax",
                "200",
                *map(str, eps_dir.glob("*.nc")),
            ],
        )
        assert result.exit_code == 0, result.output
        assert _load(binned).attrs["project"] == "CLI test"


class TestUserAttrValidation:
    @pytest.mark.parametrize(
        "attrs",
        [
            {"history": "x"},
            {"instrument_sn": "1"},
            {"pyturb_config": "x"},
            {"nested": {"a": 1}},
            {"flag": True},
            {"mixed": [1, "a"]},
        ],
    )
    def test_rejected(self, attrs):
        with pytest.raises(ValueError, match=next(iter(attrs))):
            validate_global_attrs(attrs)

    def test_load(self, tmp_path):
        path = tmp_path / "attrs.yml"
        path.write_text("title: T\nlicense: CC-BY-4.0\nversion: 2\nbounds: [1, 2.5]\n")
        assert load_global_attrs(path) == {
            "title": "T",
            "license": "CC-BY-4.0",
            "version": 2,
            "bounds": [1, 2.5],
        }


class TestMerge:
    def _converted(self, tmp_path, name):
        path = tmp_path / name
        save_netcdf(load_pfile_phys(PFILE), path)
        return path

    def test_records_sources_and_history(self, tmp_path):
        files = [self._converted(tmp_path, n) for n in ("a.nc", "b.nc")]
        merged = merge_netcdf(files, tmp_path / "merged.nc")
        attrs = _load(merged).attrs
        assert attrs["source_pfiles"] == f"{PFILE.name}, {PFILE.name}"
        assert "source_pfile" not in attrs
        assert "merge: merged 2 files" in attrs["history"]

    def test_refuses_different_calibration(self, tmp_path):
        a = self._converted(tmp_path, "a.nc")
        ds = _load(self._converted(tmp_path, "b.nc"))
        ds["T1"].attrs["cal_beta_1"] = "1234.5"
        ds.to_netcdf(tmp_path / "b2.nc")
        with pytest.raises(ValueError, match="T1:cal_beta_1"):
            merge_netcdf([a, tmp_path / "b2.nc"], tmp_path / "merged.nc")

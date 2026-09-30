"""Tests for manual QC override rules."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyturb._pfile import to_xarray
from pyturb.pfile import load_pfile_phys
from pyturb.profile import ProfileConfig, process_profile
from pyturb.qc import load_overrides

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"


def _write(tmp_path, text):
    path = tmp_path / "overrides.yaml"
    path.write_text(text)
    return path


class TestLoadOverrides:
    def test_defaults_and_time_normalisation(self, tmp_path):
        rules = load_overrides(
            _write(
                tmp_path,
                "- instrument_sn: 194\n"
                "  probes: sh1\n"
                "  start: 2026-08-06T19:40:00\n"
                "  reason: destroyed\n",
            )
        )
        assert rules == [
            {
                "instrument_sn": "194",
                "probes": ["sh1"],
                "flag": 4,
                "reason": "destroyed",
                "start": "2026-08-06T19:40:00",
            }
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "- instrument_sn: 194\n  probes: [sh1]\n",
            "- instrument_sn: 194\n  probes: [sh3]\n  reason: x\n",
            "- instrument_sn: 194\n  probes: [sh1]\n  flag: 1\n  reason: x\n",
            "instrument_sn: 194\n",
        ],
    )
    def test_invalid_rules_rejected(self, tmp_path, text):
        with pytest.raises(ValueError):
            load_overrides(_write(tmp_path, text))


@pytest.fixture(scope="module")
def raw():
    return to_xarray(load_pfile_phys(PFILE))


def _process(raw, rules):
    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
        qc_overrides=rules,
    )
    return process_profile(raw.copy(deep=True), config)


def _times(ds):
    return xr.decode_cf(ds[["time"]])["time"].values.astype("datetime64[s]")


class TestApplyOverrides:
    def test_flags_only_matching_probe_and_time(self, raw):
        base = _process(raw, [])
        t = _times(base)
        start = str(t[len(t) // 2])
        rule = {
            "instrument_sn": "142",
            "probes": ["sh1", "T2"],
            "flag": 4,
            "reason": "probe destroyed",
            "start": start,
        }
        out = _process(raw, [rule])
        hit = t >= np.datetime64(start)
        for v in ("eps_1_qc", "T2_qc", "chi_2_qc"):
            assert (out[v].values[hit] >= 4).all(), v
            np.testing.assert_array_equal(out[v].values[~hit], base[v].values[~hit])
            assert "probe destroyed" in out[v].attrs["comment"]
        for v in ("eps_2_qc", "T1_qc"):
            np.testing.assert_array_equal(out[v].values, base[v].values)

    def test_both_shear_probes_make_combined_eps_bad(self, raw):
        rule = {
            "instrument_sn": "142",
            "probes": ["sh1", "sh2"],
            "flag": 4,
            "reason": "x",
        }
        out = _process(raw, [rule])
        assert np.isnan(out["eps"].values).all()
        assert (out["eps_qc"].values >= 4).all()
        assert (out["chi_1_qc"].values >= 4).all()

    def test_other_instrument_untouched(self, raw):
        base = _process(raw, [])
        rule = {"instrument_sn": "194", "probes": ["sh1"], "flag": 4, "reason": "x"}
        out = _process(raw, [rule])
        np.testing.assert_array_equal(out["eps_1_qc"].values, base["eps_1_qc"].values)

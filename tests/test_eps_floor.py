"""Tests for the platform epsilon noise floor QC."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyturb import qc
from pyturb._pfile import to_xarray
from pyturb.pfile import load_pfile_phys
from pyturb.profile import ProfileConfig, _resolve_eps_floor, process_profile

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"


class TestResolveEpsFloor:
    @pytest.mark.parametrize("vehicle", ["vmp", "RVMP", " xmp "])
    def test_vmp_style(self, vehicle):
        ds = xr.Dataset(attrs={"instrument_vehicle": vehicle})
        assert _resolve_eps_floor(ds, ProfileConfig()) == qc.VMP_EPS_FLOOR

    @pytest.mark.parametrize("attrs", [{"instrument_vehicle": "slocum_glider"}, {}])
    def test_other_platforms(self, attrs):
        ds = xr.Dataset(attrs=attrs)
        assert _resolve_eps_floor(ds, ProfileConfig()) == qc.DEFAULT_EPS_FLOOR

    def test_config_overrides(self):
        ds = xr.Dataset(attrs={"instrument_vehicle": "vmp"})
        assert _resolve_eps_floor(ds, ProfileConfig(eps_floor=1e-9)) == 1e-9


def test_eps_below_floor_flagged_bad():
    raw = to_xarray(load_pfile_phys(PFILE))
    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
        compute_chi=False,
        eps_floor=1e-8,
    )
    result = process_profile(raw, config)
    for probe in ("1", "2"):
        eps = result[f"eps_{probe}"].values
        flags = result[f"eps_{probe}_qc"].values
        assert (eps < 1e-8).any()
        assert (flags[eps < 1e-8] == 4).all()
    assert (result["eps_qc"].values[np.isnan(result["eps"].values)] >= 4).all()

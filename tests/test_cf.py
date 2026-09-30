"""Every written variable carries CF attributes from the registry."""

import numpy as np
import pytest
import xarray as xr

from pyturb.cf import apply_cf, lookup, missing_cf, normalize_units
from pyturb.pfile import load_pfile_phys, save_netcdf
from pyturb.processing import batch_compute_epsilon, bin_profiles
from pyturb.profile import ProfileConfig
from pyturb.profile_index import batch_index_profiles

from .conftest import PFILE


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cf")
    p2nc = tmp / "p2nc.nc"
    save_netcdf(load_pfile_phys(PFILE), p2nc, despike_kwargs={})
    config = ProfileConfig(diss_len_sec=4.0, fft_len_sec=1.0)
    index = batch_index_profiles(
        [p2nc], config=config, output_dir=tmp / "idx", n_workers=1, materialize=True
    )
    batch_compute_epsilon([p2nc], config=config, output_dir=tmp / "eps", n_workers=1)
    eps = sorted((tmp / "eps").glob("*.nc"))
    bin_profiles(eps, tmp / "bin.nc", depth_max=200, ctd_bin_width=0.5, n_workers=1)
    hires = sorted((tmp / "idx").glob("*_p*.nc"))
    return {
        "p2nc": p2nc,
        "index": index[0]["output"],
        "hires": hires[0],
        "eps": eps[0],
        "bin": tmp / "bin.nc",
    }


@pytest.mark.parametrize("stage", ["p2nc", "index", "hires", "eps", "bin"])
def test_no_variable_missing_cf(outputs, stage):
    ds = xr.open_dataset(outputs[stage], decode_times=False)
    assert missing_cf(ds) == []


def test_binned_attrs(outputs):
    ds = xr.open_dataset(outputs["bin"], decode_times=False)
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "source_files" in ds.attrs
    assert ds["depth"].attrs["positive"] == "down"
    assert ds["eps"].attrs["cell_methods"] == "depth: mean"
    assert ds["eps"].attrs["ancillary_variables"] == "eps_qc eps_n"
    assert ds["eps_1_qc"].attrs["cell_methods"] == "depth: maximum"
    assert "flag_values" in ds["eps_1_qc"].attrs


def test_lookup_exact_pattern_and_suffix():
    assert lookup("P")["standard_name"] == "sea_water_pressure"
    assert lookup("eps_2")["long_name"] == "Epsilon 2"
    assert lookup("T1_dT1")["long_name"] == "ADC counts pre-emphasized FP07 1"
    assert lookup("T1_dT2") is None
    assert lookup("gradT2_despike_frac")["long_name"] == (
        "Fraction of gradT2 samples modified by despiking"
    )
    assert lookup("pressure_hires") == lookup("pressure")
    assert lookup("sh1_clean")["long_name"] == "Velocity time derivative 1 (despiked)"
    assert lookup("W_smooth")["long_name"] == "Platform speed (smoothed)"
    assert lookup("not_a_variable") is None


def test_apply_cf_keeps_code_attrs_and_skips_qc():
    ds = xr.Dataset(
        {
            "eps_1": ("time", np.ones(2), {"comment": "kept", "long_name": "old"}),
            "eps_1_qc": ("time", np.ones(2, "i1"), {"long_name": "QC flag"}),
        }
    )
    apply_cf(ds)
    assert ds["eps_1"].attrs["long_name"] == "Epsilon 1"
    assert ds["eps_1"].attrs["comment"] == "kept"
    assert ds["eps_1"].attrs["ancillary_variables"] == "eps_1_qc"
    assert ds["eps_1_qc"].attrs == {"long_name": "QC flag"}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("C", "degree_C"),
        ("mS/cm", "mS cm-1"),
        ("m/s", "m s-1"),
        ("m^2 s^-3", "m2 s-3"),
        ("K/s", "K s-1"),
        ("counts", "1"),
        ("deg", "degree"),
        ("dBar", "dbar"),
        ("µT", "uT"),
        ("[FTU]", "FTU"),
        ("ug/L", "ug L-1"),
        ("m/s^2", "m s-2"),
        ("V", "V"),
    ],
)
def test_normalize_units(raw, expected):
    assert normalize_units(raw) == expected

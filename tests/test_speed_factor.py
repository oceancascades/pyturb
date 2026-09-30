"""--speed-factor scales the speed used for W, the spectra and epsilon."""

import numpy as np
import xarray as xr
import yaml
from typer.testing import CliRunner

from pyturb._pfile.to_xarray import to_xarray
from pyturb.cli import app
from pyturb.pfile import load_pfile_phys, save_netcdf
from pyturb.profile import ProfileConfig, process_profile

from .conftest import PFILE


def test_scaled_speed_propagates_to_outputs():
    raw = to_xarray(load_pfile_phys(PFILE))
    base = process_profile(raw.copy(deep=True), ProfileConfig())
    scaled = process_profile(raw.copy(deep=True), ProfileConfig(speed_factor=0.9))
    np.testing.assert_allclose(scaled["W"].values, 0.9 * base["W"].values, rtol=1e-5)
    np.testing.assert_allclose(scaled["k"].values, base["k"].values / 0.9, rtol=1e-5)
    assert not np.allclose(scaled["eps_1"].values, base["eps_1"].values, equal_nan=True)


def test_cli_records_speed_factor(tmp_path):
    converted = tmp_path / "raw.nc"
    save_netcdf(load_pfile_phys(PFILE), converted)
    result = CliRunner().invoke(
        app,
        [
            "eps",
            "-o",
            str(tmp_path / "eps"),
            "-n",
            "1",
            "--speed-factor",
            "0.9",
            str(converted),
        ],
    )
    assert result.exit_code == 0, result.output
    out = xr.load_dataset(next((tmp_path / "eps").glob("*.nc")), decode_times=False)
    assert yaml.safe_load(out.attrs["pyturb_config"])["speed_factor"] == 0.9

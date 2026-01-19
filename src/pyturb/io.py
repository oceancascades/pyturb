# Functions for reading and writing data

import h5py  # type: ignore[import]
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Union
from scipy.io import loadmat

__all__ = ["load_rockland_mat", "load_profile_nc"]


_variable_map = dict(
    sh1=["t_fast"],
    sh2=["t_fast"],
    gradT1=["t_fast"],
    gradT2=["t_fast"],
    U_EM=["t_slow"],
    P_slow=["t_slow"],
    fs_fast=[],
    fs_slow=[],
)


def load_profile_nc(filename: Union[str, Path]) -> xr.Dataset:
    return xr.load_dataset(filename, decode_times=False)


def load_rockland_mat(filename: str) -> xr.Dataset:
    store = "mat"

    try:
        dat: Any = loadmat(filename, squeeze_me=True)
    except NotImplementedError:
        try:
            dat = h5py.File(filename, "r")
            store = "hdf5"
        except OSError:
            raise RuntimeError("File cannot be opened using scipy.loadmat or h5py.")

    coordinates = [
        _variable_map[key] for key in _variable_map if len(_variable_map[key]) > 0
    ]
    unique_coordinates = np.unique(coordinates)

    if store == "mat":
        data_vars = {v: (c, dat[v]) for v, c in _variable_map.items()}
        coords = {c: (c, dat[c]) for c in unique_coordinates}
    elif store == "hdf5":
        data_vars = {v: (c, dat[v][:].squeeze()) for v, c in _variable_map.items()}
        coords = {c: (c, dat[c][:].squeeze()) for c in unique_coordinates}

    if store == "hdf5":
        dat.close()

    return xr.Dataset(data_vars, coords)

"""
Private implementation modules for P-file reading and processing.

This subpackage contains the internal implementation details for reading
RSI P-files. Users should use the public API in pyturb.pfile instead.
"""

from .config import SetupConfig
from .convert import convert_all_channels, convert_channel
from .deconvolve import deconvolve
from .gradT import compute_gradT_channels, make_gradT
from .reader import open_pfile, read_pfile
from .sensors import adis_extract
from .to_xarray import to_xarray

__all__ = [
    "SetupConfig",
    "read_pfile",
    "open_pfile",
    "convert_channel",
    "convert_all_channels",
    "deconvolve",
    "adis_extract",
    "make_gradT",
    "compute_gradT_channels",
    "to_xarray",
]

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

from ..cf import normalize_units
from ..metadata import stamp_globals

# Suffixes stripped (in order) to find a variable's underlying channel name
# in the setup config, e.g. "T1_counts" and "T1_raw" both look up "T1".
_CHANNEL_SUFFIXES = ("_counts", "_raw", "_hires")


def _channel_calibration_attrs(cfg, var_name: str) -> Dict[str, str]:
    """The variable's channel-config parameters, as ``cal_<key>`` attrs.

    Looks up ``var_name`` directly, then its base channel name (stripping
    ``_counts``/``_raw``/``_hires``) if there's no direct match. Empty if no
    matching channel section exists (e.g. for variables with no calibration,
    such as accelerometers converted with fixed constants).
    """
    if cfg is None:
        return {}
    names = [var_name]
    for suffix in _CHANNEL_SUFFIXES:
        if var_name.endswith(suffix):
            names.append(var_name[: -len(suffix)])
    for name in names:
        params = cfg.get_channel_params(name)
        if params:
            return {f"cal_{k}": v for k, v in params.items()}
    return {}


# Default variables to save (in order of priority)
_DEFAULT_VARIABLES = [
    "P",
    "sh1",
    "sh2",
    "gradT1",
    "gradT2",
    "U_EM",
    "EMC_Cur",
    "EM_Cur",
    "JAC_T",
    "JAC_C",
    "T1",
    "T2",
    "T1_counts",
    "T2_counts",
    "T1_dT1",
    "T2_dT2",
    "Ax",
    "Ay",
    "Az",
    "Incl_X",
    "Incl_Y",
    "Incl_T",
    "Turbidity",
    "Chlorophyll",
]

# Source channel names (as they appear in the setup string) renamed on
# output, e.g. to this package's lowercase convention for non-CTD-standard
# sensor names.
_OUTPUT_RENAME = {
    "Turbidity": "turbidity",
    "Chlorophyll": "chlorophyll",
}


def to_xarray(data: Dict, variables: Optional[list] = None) -> xr.Dataset:
    """
    Convert P-file data to xarray Dataset.

    Parameters
    ----------
    data : dict
        Data dictionary from load_pfile_phys().
    variables : list, optional
        Variable names to include. Defaults to standard microstructure variables.

    Returns
    -------
    xr.Dataset
        CF-1.8 compliant dataset with t_fast and t_slow dimensions.
    """

    # Determine which variables to save
    if variables is None:
        variables = _DEFAULT_VARIABLES

    # Filter to only variables that exist in data
    available_vars = [v for v in variables if v in data]

    if not available_vars:
        raise ValueError(
            f"No requested variables found in data. "
            f"Available: {[k for k in data.keys() if isinstance(data[k], np.ndarray)]}"
        )

    # Get time vectors and sampling rates
    t_fast = data.get("t_fast")
    t_slow = data.get("t_slow")
    fs_fast = data.get("fs_fast")
    fs_slow = data.get("fs_slow")

    if t_fast is None or t_slow is None:
        raise ValueError("Data must contain t_fast and t_slow time vectors")

    # Determine which variables go on which time dimension
    n_fast = len(t_fast)
    n_slow = len(t_slow)

    # Build xarray Dataset
    data_vars = {}
    units_dict = data.get("units", {})
    cfg = data.get("cfgobj")

    for var_name in available_vars:
        var_data = data[var_name]

        if not isinstance(var_data, np.ndarray):
            continue

        # Determine dimension based on length
        if len(var_data) == n_fast:
            dims = ["t_fast"]
        elif len(var_data) == n_slow:
            dims = ["t_slow"]
        else:
            # Skip variables that don't match either dimension
            continue

        # Convert to float32 for space efficiency
        var_data = var_data.astype(np.float32)

        out_name = _OUTPUT_RENAME.get(var_name, var_name)

        attrs = {"long_name": out_name}
        if var_name in units_dict:
            attrs["units"] = normalize_units(units_dict[var_name])
        attrs.update(_channel_calibration_attrs(cfg, var_name))

        data_vars[out_name] = (dims, var_data, attrs)

    # Create coordinate variables
    # Use reference time from file
    filetime = data.get("filetime")
    if filetime:
        time_units = f"seconds since {filetime.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        time_units = "seconds since 1970-01-01 00:00:00"

    coords = {
        "t_fast": (
            "t_fast",
            t_fast.astype(np.float64),
            {"units": time_units},
        ),
        "t_slow": (
            "t_slow",
            t_slow.astype(np.float64),
            {"units": time_units},
        ),
    }

    source = {}
    if data.get("fullPath"):
        source["source_pfile"] = Path(data["fullPath"]).name
    if "cfgobj" in data:
        for key in ("vehicle", "model", "sn"):
            value = data["cfgobj"].get_value("instrument_info", key, default="")
            if value:
                source[f"instrument_{key}"] = value

    extra = {}
    if filetime:
        extra["pfile_start_time"] = filetime.isoformat(timespec="seconds")
    extra["header_version"] = float(data.get("header_version", 0))
    if fs_fast:
        extra["fs_fast"] = float(fs_fast)
    if fs_slow:
        extra["fs_slow"] = float(fs_slow)
    if "setupfilestr" in data:
        extra["pfile_configuration"] = data["setupfilestr"]

    global_attrs = stamp_globals(
        source,
        step="p2nc",
        title="RSI microstructure p-file converted to NetCDF",
        detail=f"converted {source.get('source_pfile', 'p-file')}",
        **extra,
    )

    # Create Dataset
    ds = xr.Dataset(data_vars, coords=coords, attrs=global_attrs)

    return ds

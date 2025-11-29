from typing import Dict, Optional

import numpy as np
import xarray as xr

# CF-compliant variable metadata
# Maps variable names to (standard_name, long_name, units)
# standard_name follows CF conventions where applicable
_CF_VARIABLE_METADATA = {
    # Pressure
    "P": ("sea_water_pressure", "Pressure", "dbar"),
    # Shear probes
    "sh1": (None, "Velocity time derivative from probe 1", "m2 s-3"),
    "sh2": (None, "Velocity time derivative from probe 2", "m2 s-3"),
    # Temperature gradients
    "gradT1": (None, "Temperature time derivative from thermistor 1", "K s-1"),
    "gradT2": (None, "Temperature time derivative from thermistor 2", "K s-1"),
    # EM current meter
    "U_EM": (None, "EM current meter velocity", "m s-1"),
    # JAC CT sensor
    "JAC_T": ("sea_water_temperature", "JAC CT temperature", "degree_C"),
    "JAC_C": ("sea_water_electrical_conductivity", "JAC CT conductivity", "S m-1"),
    # FP07 thermistors
    "T1": ("sea_water_temperature", "FP07 thermistor 1 temperature", "degree_C"),
    "T2": ("sea_water_temperature", "FP07 thermistor 2 temperature", "degree_C"),
    # Pre-emphasized thermistor signals
    # "T1_dT1": (None, "Pre-emphasized thermistor 1 signal", "counts"),
    # "T2_dT2": (None, "Pre-emphasized thermistor 2 signal", "counts"),
    # Accelerometers
    "Ax": (None, "Acceleration X", "m s-2"),
    "Ay": (None, "Acceleration Y", "m s-2"),
    "Az": (None, "Acceleration Z", "m s-2"),
    # Inclinometers
    "Incl_X": (None, "Inclinometer X angle", "degree"),
    "Incl_Y": (None, "Inclinometer Y angle", "degree"),
    # "Incl_T": (None, "Inclinometer temperature", "degree_C"),
}

# Default variables to save (in order of priority)
_DEFAULT_VARIABLES = [
    "P",
    "sh1",
    "sh2",
    "gradT1",
    "gradT2",
    "U_EM",
    "JAC_T",
    "JAC_C",
    "T1",
    "T2",
    "Ax",
    "Ay",
    "Az",
    "Incl_X",
    "Incl_Y",
    "Incl_T",
]


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

        # Build attributes
        attrs = {}

        # Get CF metadata if available
        if var_name in _CF_VARIABLE_METADATA:
            standard_name, long_name, cf_units = _CF_VARIABLE_METADATA[var_name]
            if standard_name:
                attrs["standard_name"] = standard_name
            if long_name:
                attrs["long_name"] = long_name
            # Use CF units, falling back to data units
            attrs["units"] = cf_units
        else:
            # Use units from data if available
            if var_name in units_dict:
                attrs["units"] = units_dict[var_name]
            attrs["long_name"] = var_name

        data_vars[var_name] = (dims, var_data, attrs)

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
            {
                "long_name": "Time (fast sampling)",
                "units": time_units,
                "axis": "T",
            },
        ),
        "t_slow": (
            "t_slow",
            t_slow.astype(np.float64),
            {
                "long_name": "Time (slow sampling)",
                "units": time_units,
                "axis": "T",
            },
        ),
    }

    # Build global attributes
    global_attrs = {
        "Conventions": "CF-1.8",
        "title": "",
        "institution": "",
        "source": "",
        "history": f"p file created {data.get('filetime', '').strftime('%Y-%m-%d %H:%M:%S') if data.get('filetime') else ''}",
        "fs_fast": float(fs_fast) if fs_fast else "not found",
        "fs_slow": float(fs_slow) if fs_slow else "not found",
        "source_file": data.get("fullPath", ""),
        "date": data.get("date", ""),
        "time": data.get("time", ""),
        "header_version": float(data.get("header_version", 0)),
    }

    # Add configuration string as global attribute
    if "setupfilestr" in data:
        global_attrs["pfile_configuration"] = data["setupfilestr"]

    # Create Dataset
    ds = xr.Dataset(data_vars, coords=coords, attrs=global_attrs)

    return ds

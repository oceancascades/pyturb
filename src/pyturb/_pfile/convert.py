"""
Convert RSI P-file data from raw counts to physical units.

Translated from MATLAB ODAS library v4.5.1 (convert_odas.m)
"""

import re
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .deconvolve import deconvolve
from .gradT import make_gradT
from .sensors import adis_extract


def _adc_to_voltage(data: np.ndarray, params: Dict) -> np.ndarray:
    """Common ADC counts to voltage conversion."""
    adc_zero = float(params.get("adc_zero", 0))
    adc_fs = float(params["adc_fs"])
    adc_bits = int(params["adc_bits"])
    return data * adc_fs / 2**adc_bits + adc_zero


def _poly(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Polynomial conversion."""
    coeffs = []
    for i in range(10):
        coef_name = f"coef{i}"
        if coef_name in params:
            coeffs.insert(0, float(params[coef_name]))
        else:
            break

    if not coeffs:
        raise ValueError("No polynomial coefficients found")

    units = params.get("units", "")
    if units.startswith("[") and units.endswith("]"):
        units = units[1:-1]

    return np.polyval(coeffs, data), units


def _shear(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Shear probe conversion to m^2 s^-3."""
    adc_zero = float(params.get("adc_zero", 0))
    sig_zero = float(params.get("sig_zero", 0))
    adc_fs = float(params["adc_fs"])
    adc_bits = int(params["adc_bits"])
    diff_gain = float(params["diff_gain"])
    sens = float(params["sens"])

    voltage = (adc_fs / 2**adc_bits) * data + (adc_zero - sig_zero)
    physical = voltage / (2 * np.sqrt(2) * diff_gain * sens)

    return physical, "m^2 s^-3"


def _piezo(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Piezo-electric shear probe - remove offset only."""
    a_0 = float(params.get("a_0", 0))
    return data - a_0, "counts"


def _therm(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """FP07 thermistor conversion to degrees C using Steinhart-Hart equation."""
    a = float(params["a"])
    b = float(params["b"])
    adc_fs = float(params["adc_fs"])
    adc_bits = int(params["adc_bits"])
    g = float(params["g"])
    e_b = float(params["e_b"])
    t_0 = float(params["t_0"])

    Z = ((data - a) / b) * (adc_fs / 2**adc_bits) * 2 / (g * e_b)
    Z = np.clip(Z, -0.6, 0.6)
    R_ratio = (1 - Z) / (1 + Z)
    Log_R = np.log(R_ratio)

    if "beta" in params:
        T_inv = 1 / t_0 + (1 / float(params["beta"])) * Log_R
    elif "beta_1" in params:
        T_inv = 1 / t_0 + (1 / float(params["beta_1"])) * Log_R
    else:
        raise ValueError("No beta or beta_1 parameter for thermistor")

    if "beta_2" in params:
        T_inv += (1 / float(params["beta_2"])) * Log_R**2
        if "beta_3" in params:
            T_inv += (1 / float(params["beta_3"])) * Log_R**3

    return 1 / T_inv - 273.15, "C"


def _ucond(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Micro-conductivity conversion to mS/cm."""
    adc_zero = float(params.get("adc_zero", 0))
    adc_fs = float(params.get("adc_fs", 1))
    adc_bits = int(params.get("adc_bits", 0))
    a = float(params["a"])
    b = float(params["b"])
    k = float(params["k"])

    voltage = (adc_fs / 2**adc_bits) * data + adc_zero
    conductance = (voltage - a) / b
    conductivity = conductance / k

    return conductivity * 10, "mS/cm"


def _accel(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Accelerometer conversion to m/s^2."""
    g = 9.81
    sig_zero = float(params.get("sig_zero", 0))
    coef0 = float(params["coef0"])
    coef1 = float(params["coef1"])

    voltage = _adc_to_voltage(data, params) - sig_zero
    physical = g * (voltage - coef0) / coef1

    return physical, "m/s^2"


def _voltage(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Generic voltage conversion."""
    gain = float(params.get("g", 1))
    return _adc_to_voltage(data, params) / gain, "V"


def _magn(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Magnetometer conversion to micro-Tesla."""
    coef0 = float(params["coef0"])
    coef1 = float(params["coef1"])
    return (data - coef0) / coef1, "µT"


def _jac_t(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """JAC thermistor conversion to degrees C."""
    data = data.copy()
    data[data < 0] += 2**16

    coeffs = [float(params[n]) for n in ["a", "b", "c", "d", "e", "f"] if n in params]
    coeffs.reverse()

    return np.polyval(coeffs, data), "C"


def _jac_c(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """JAC conductivity: split 32-bit into period ratio, apply polynomial."""
    i = np.floor(data / 2**16).astype(int)
    v = (data % 2**16).astype(int)
    v[v == 0] = 1
    periods = i / v

    coeffs = [float(params["c"]), float(params["b"]), float(params["a"])]
    return np.polyval(coeffs, periods), "s"


def _sbt(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Sea-Bird thermistor conversion to degrees C."""
    f = params["coef6"] * params["coef5"] / data.astype(float)
    c = np.log(params["coef4"] / f)
    coeffs = [params["coef3"], params["coef2"], params["coef1"], params["coef0"]]
    return 1 / np.polyval(coeffs, c) - 273.15, "C"


def _sbc(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Sea-Bird conductivity conversion to mS/cm."""
    f = params["coef6"] * params["coef5"] / data.astype(float) / 1000
    coeffs = [
        params["coef4"],
        params["coef3"],
        params["coef2"],
        params["coef1"],
        params["coef0"],
    ]
    return np.polyval(coeffs, f), "mS/cm"


def _raw(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    return data, "counts"


def _gnd(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    return data, "counts"


def _inclxy(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """ADIS16209 inclinometer X/Y conversion to degrees."""
    raw, _, _ = adis_extract(data, "xy")
    coef0 = float(params.get("coef0", 0))
    coef1 = float(params.get("coef1", 1))
    return coef1 * raw + coef0, "deg"


def _inclt(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """ADIS16209 inclinometer temperature conversion to degrees C."""
    raw, _, _ = adis_extract(data, "t")
    coef0 = float(params.get("coef0", 624))
    coef1 = float(params.get("coef1", -0.47))
    return coef1 * raw + coef0, "C"


def _alec_emc(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Alec Electronics EM current meter to m/s."""
    coef0 = float(params["coef0"])
    coef1 = float(params["coef1"])
    return np.polyval([coef1, coef0], data), "m/s"


def _vector(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Nortek Vector to m/s."""
    bias = float(params.get("bias", 0))
    offset = float(params["offset"])
    sens = float(params["sens"])

    V = _adc_to_voltage(data, params)
    velocity = (V - offset) * sens

    return velocity - bias, "m/s"


def _aem1g_a(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """AEM1-G analog output to m/s."""
    bias = float(params.get("bias", 0))
    a = float(params["a"]) / 100  # cm/s to m/s
    b = float(params["b"]) / 100

    V = _adc_to_voltage(data, params)
    velocity = a + b * V

    if b < 1:
        warnings.warn("A, B coefficients may not be correct for analog output.")

    return velocity - bias, "m/s"


def _aem1g_d(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """AEM1-G digital output to m/s."""
    data = data.copy()
    data[data < 0] += 2**16  # Convert signed to unsigned

    a = float(params["a"]) / 100  # cm/s to m/s
    b = float(params["b"]) / 100
    physical = a + b * data

    if b > 1:
        warnings.warn("A, B coefficients may not be correct for digital output.")

    return physical, "m/s"


def _jac_emc(data: np.ndarray, params: Dict) -> Tuple[np.ndarray, str]:
    """Deprecated: use aem1g_a instead."""
    warnings.warn('Type "jac_emc" is deprecated. Use "aem1g_a" instead.')
    return _aem1g_a(data, params)


_TYPE_CONVERTERS: Dict[str, Callable] = {
    "poly": _poly,
    "shear": _shear,
    "piezo": _piezo,
    "therm": _therm,
    "ucond": _ucond,
    "accel": _accel,
    "voltage": _voltage,
    "magn": _magn,
    "jac_t": _jac_t,
    "jac_c": _jac_c,
    "sbt": _sbt,
    "sbc": _sbc,
    "raw": _raw,
    "gnd": _gnd,
    "inclxy": _inclxy,
    "inclt": _inclt,
    "alec_emc": _alec_emc,
    "vector": _vector,
    "aem1g_a": _aem1g_a,
    "aem1g_d": _aem1g_d,
    "jac_emc": _jac_emc,
    # Aliases
    "xmp_shear": _shear,
    "xmp_pres": _poly,
    "t_ms": _therm,
    "c_ms": _ucond,
    "xmp_therm": _therm,
    "xmp_volt": _poly,
}


def convert_channel(data: np.ndarray, channel_name: str, cfg) -> Tuple[np.ndarray, str]:
    """
    Convert channel data from raw counts to physical units.

    Parameters
    ----------
    data : ndarray
        Raw data in counts
    channel_name : str
        Channel name matching config
    cfg : SetupConfig
        Configuration with calibration coefficients

    Returns
    -------
    tuple
        (converted_data, units_string)
    """
    params = None
    for section in cfg.get_section_dicts("channel"):
        if section["params"].get("name", "").lower() == channel_name.lower():
            params = section["params"]
            break

    if params is None:
        raise ValueError(f"Channel '{channel_name}' not found in configuration")

    ch_type = params.get("type", "").lower()
    if not ch_type:
        raise ValueError(f"No type specified for channel '{channel_name}'")

    # Get the conversion function
    converter = _TYPE_CONVERTERS.get(ch_type)
    if converter is None:
        raise ValueError(f"Unknown channel type: '{ch_type}'")

    data = data.astype(np.float64)
    return converter(data, params)


def convert_all_channels(
    data: Dict,
    exclude_types: Optional[list] = None,
    gradT_method: str = "high_pass",
) -> Dict:
    """
    Convert all channels to physical units, including deconvolution and gradT.

    This function performs all conversions needed to produce calibrated physical
    data from raw P-file counts:
    - Standard channel conversions (temperature, pressure, conductivity, etc.)
    - Deconvolution of pre-emphasized signals to high-resolution versions
    - Temperature gradient (gradT) computation from pre-emphasized thermistors
    - Shear probe conversion

    Parameters
    ----------
    data : dict
        Dictionary from read_pfile()
    exclude_types : list, optional
        Channel types to skip (default: ['gnd', 'raw'])
    gradT_method : str, optional
        Method for gradT computation: 'high_pass' (default) or 'first_difference'

    Returns
    -------
    dict
        Converted channels with 'units' dict and metadata preserved.
        Includes gradT1, gradT2, etc. (in K/s) for pre-emphasized thermistors.

    Notes
    -----
    gradT outputs are time derivatives (K/s). To convert to spatial gradients
    (K/m), divide by fall speed in the processing pipeline, which is a
    platform-dependent calculation.
    """
    if exclude_types is None:
        exclude_types = ["gnd", "raw"]
    exclude_types = [t.lower() for t in exclude_types]

    result = {"units": {}}

    # Preserve metadata
    metadata_keys = [
        "fs_fast",
        "fs_slow",
        "t_fast",
        "t_slow",
        "filetime",
        "date",
        "time",
        "header_version",
        "fullPath",
        "cfgobj",
        "setupfilestr",
    ]
    for key in metadata_keys:
        if key in data:
            result[key] = data[key]

    cfg = data["cfgobj"]
    channel_sections = cfg.get_section_dicts("channel")

    # First pass: deconvolve pre-emphasized channels (except shear)
    deconvolved = {}
    for section in channel_sections:
        params = section["params"]
        ch_name = params.get("name")
        ch_type = params.get("type", "").lower()

        if not ch_name or ch_type in exclude_types or ch_name not in data:
            continue

        if "diff_gain" in params and ch_type not in ["shear", "xmp_shear"]:
            match = re.match(r"(\w+)_d\1", ch_name)
            if match:
                base_name = match.group(1)
                try:
                    X_hires = deconvolve(
                        data[ch_name],
                        data["fs_fast"],
                        float(params["diff_gain"]),
                        data.get(base_name),
                    )
                    deconvolved[f"{base_name}_hires"] = X_hires
                except Exception as e:
                    warnings.warn(f"Failed to deconvolve '{ch_name}': {e}")

    data.update(deconvolved)

    # Second pass: convert all channels
    for section in channel_sections:
        params = section["params"]
        ch_name = params.get("name")
        ch_type = params.get("type", "").lower()

        if not ch_name or ch_type in exclude_types or ch_name not in data:
            continue

        # Keep pre-emphasized signals as counts (needed for gradT)
        if "diff_gain" in params and ch_type not in ["shear", "xmp_shear"]:
            result[ch_name] = data[ch_name]
            result["units"][ch_name] = "counts"
            continue

        try:
            result[ch_name], result["units"][ch_name] = convert_channel(
                data[ch_name], ch_name, cfg
            )
        except Exception as e:
            warnings.warn(f"Failed to convert '{ch_name}': {e}")
            result[ch_name] = data[ch_name]
            result["units"][ch_name] = "counts"

    # Convert deconvolved high-res channels using base channel calibration
    for hires_name, hires_data in deconvolved.items():
        base_name = hires_name.replace("_hires", "")
        for section in channel_sections:
            if section["params"].get("name", "").lower() == base_name.lower():
                try:
                    result[hires_name], result["units"][hires_name] = convert_channel(
                        hires_data, base_name, cfg
                    )
                except Exception as e:
                    warnings.warn(f"Failed to convert '{hires_name}': {e}")
                    result[hires_name] = hires_data
                    result["units"][hires_name] = "counts"
                break

    # Compute temperature gradients from pre-emphasized thermistor channels
    fs_fast = data["fs_fast"]
    therm_types = ["therm", "t_ms", "xmp_therm"]

    for section in channel_sections:
        params = section["params"]
        ch_type = params.get("type", "").lower()

        # Only process pre-emphasized thermistor types
        if ch_type not in therm_types or "diff_gain" not in params:
            continue

        ch_name = params.get("name", "")
        if not ch_name or ch_name not in data:
            continue

        # Determine output name: gradT1 from T1_dT1
        match = re.match(r"(\w+)_d\1", ch_name, re.IGNORECASE)
        if match:
            base_name = match.group(1)
            grad_name = f"grad{base_name}"

            # Merge base channel params with pre-emphasis params
            base_params = cfg.get_channel_params(base_name)
            if base_params:
                merged_params = base_params.copy()
                merged_params.update(params)
                params = merged_params
        else:
            grad_name = f"grad{ch_name}"

        try:
            gradT_result = make_gradT(
                data[ch_name], params, fs_fast, gradT_method
            )
            result[grad_name] = gradT_result
            result["units"][grad_name] = "K/s"
        except Exception as e:
            warnings.warn(f"Could not compute {grad_name} from {ch_name}: {e}")

    return result

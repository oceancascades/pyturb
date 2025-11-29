"""
Compute temperature gradient from pre-emphasized thermistor signals.

Implements the algorithm from ODAS library v4.5.1 (make_gradT_odas.m)
"""

import re
import warnings
from typing import Dict

import numpy as np
from scipy import signal

from .deconvolve import deconvolve


def make_gradT(
    T_dT: np.ndarray,
    params: Dict,
    fs: float,
    method: str = "high_pass",
) -> np.ndarray:
    """
    Compute temperature time-derivative from pre-emphasized thermistor signal.

    Parameters
    ----------
    T_dT : ndarray
        Pre-emphasized temperature signal (e.g., T1_dT1)
    params : dict
        Thermistor parameters including diff_gain, beta, t_0, etc.
    fs : float
        Sampling rate in Hz
    method : str
        'high_pass' (recommended) or 'first_difference'

    Returns
    -------
    ndarray
        Temperature time derivative dT/dt in K/s

    Notes
    -----
    To convert to spatial gradient dT/dz, divide by fall speed in the
    processing pipeline (platform-dependent calculation).
    """
    if len(T_dT) == 0:
        raise ValueError("Temperature signal is empty")

    # Get diff_gain
    if "diff_gain" not in params:
        raise ValueError("diff_gain parameter is required")
    diff_gain = float(params["diff_gain"])

    # Get thermistor type
    therm_type = params.get("type", "therm").lower()

    # Extract parameters based on type
    if therm_type == "xmp_therm":
        a = 0.0
        b = 1.0
        adc_fs = 4.096
        adc_bits = 16
        g = 1.0
        e_b = 4.096
        T_0 = float(params["coef0"])
        beta_1 = float(params["coef1"])
        beta_2 = np.inf
        beta_3 = np.inf
    else:
        a = float(params["a"])
        b = float(params["b"])
        adc_fs = float(params["adc_fs"])
        adc_bits = int(params["adc_bits"])
        g = float(params["g"])
        e_b = float(params["e_b"])
        T_0 = float(params["t_0"])

        # Beta can be 'beta' or 'beta_1'
        if "beta" in params:
            beta_1 = float(params["beta"])
        elif "beta_1" in params:
            beta_1 = float(params["beta_1"])
        else:
            raise ValueError("No beta or beta_1 parameter for thermistor")

        beta_2 = float(params["beta_2"]) if "beta_2" in params else np.inf
        beta_3 = float(params["beta_3"]) if "beta_3" in params else np.inf

    # Deconvolve to get temperature
    T = deconvolve(T_dT, fs, diff_gain)

    # High-pass filter to get time derivative
    fc = 1 / (2 * np.pi * diff_gain)
    b_hp, a_hp = signal.butter(1, fc / (fs / 2), btype="high")

    # Calculate initial conditions
    zi = signal.lfiltic(b_hp, a_hp, [0], [T_dT[0]])
    dT_dt = signal.lfilter(b_hp, a_hp, T_dT, zi=zi)[0]
    dT_dt = dT_dt / diff_gain  # Now a time derivative

    # Compute resistance ratio
    if therm_type in ["therm", "xmp_therm"]:
        Z = ((T - a) / b) * (adc_fs / 2**adc_bits) * 2 / (g * e_b)
    elif therm_type == "t_ms":
        zero = float(params.get("zero", 0))
        Z = T * (adc_fs / 2**adc_bits) + zero
        Z = ((Z - a) / b) * 2 / (g * e_b)
    else:
        # Default to therm behavior
        Z = ((T - a) / b) * (adc_fs / 2**adc_bits) * 2 / (g * e_b)

    R = (1 - Z) / (1 + Z)
    R = np.clip(R, 0.1, None)  # Prevent log of negative/zero for broken thermistor
    log_R = np.log(R)

    # Compute absolute temperature using Steinhart-Hart equation
    T_inv = 1 / T_0 + log_R / beta_1
    if np.isfinite(beta_2):
        T_inv = T_inv + log_R**2 / beta_2
    if np.isfinite(beta_3):
        T_inv = T_inv + log_R**3 / beta_3
    T_abs = 1 / T_inv  # Temperature in Kelvin

    # Calculate temperature gradient
    if method.lower() == "high_pass":
        eta = (b / 2) * (2**adc_bits) * g * e_b / adc_fs
        scale_factor = np.ones_like(T)
        if np.isfinite(beta_2):
            scale_factor = 1 + 2 * (beta_1 / beta_2) * log_R
        scale_factor = scale_factor * T_abs**2 * (1 + R) ** 2 / (2 * eta * beta_1 * R)
        gradT = scale_factor * dT_dt
    elif method.lower() == "first_difference":
        gradT = fs * np.diff(T_abs, append=T_abs[-1])
        gradT[-1] = gradT[-2]
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'high_pass' or 'first_difference'"
        )

    return gradT


def compute_gradT_channels(data: Dict, method: str = "high_pass") -> Dict:
    """
    Compute dT/dt for all pre-emphasized thermistor channels (T1_dT1, T2_dT2, etc.).

    Adds gradT1, gradT2, etc. to the data dictionary.

    Parameters
    ----------
    data : dict
        Data from load_pfile_phys() with 'cfgobj' and 'fs_fast'
    method : str
        'high_pass' (default) or 'first_difference'

    Returns
    -------
    dict
        Input dict with added gradT channels (dT/dt in K/s)

    Notes
    -----
    To convert to spatial gradient dT/dz, divide by fall speed in the
    processing pipeline (platform-dependent calculation).
    """
    cfg = data["cfgobj"]
    fs_fast = data["fs_fast"]

    # Find all thermistor channels with diff_gain (pre-emphasized)
    therm_types = ["therm", "t_ms", "xmp_therm"]

    for section in cfg.sections:
        if section["section"] != "channel":
            continue

        params = section["params"].copy()  # Copy to avoid modifying original
        ch_type = params.get("type", "").lower()

        # Only process thermistor types
        if ch_type not in therm_types:
            continue

        # Only process channels with diff_gain (pre-emphasized)
        if "diff_gain" not in params:
            continue

        ch_name = params.get("name", "")
        if not ch_name or ch_name not in data:
            continue

        # Determine output name and base name: gradT1 from T1_dT1
        # Pattern: (word)_d\1 -> extracts base name
        match = re.match(r"(\w+)_d\1", ch_name, re.IGNORECASE)
        if match:
            base_name = match.group(1)
            grad_name = f"grad{base_name}"

            # Get parameters from base channel and merge
            # (base channel has calibration coefficients)
            base_params = cfg.get_channel_params(base_name)
            if base_params:
                # Merge: base params first, then pre-emphasis params override
                merged_params = base_params.copy()
                merged_params.update(params)
                params = merged_params
        else:
            grad_name = f"grad{ch_name}"

        try:
            gradT_result = make_gradT(data[ch_name], params, fs_fast, method)
            data[grad_name] = gradT_result
            if "units" in data:
                data["units"][grad_name] = "K/s"
        except Exception as e:
            warnings.warn(f"Could not compute {grad_name} from {ch_name}: {e}")

    return data

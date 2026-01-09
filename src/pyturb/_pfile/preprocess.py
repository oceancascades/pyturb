"""
Preprocessing functions for shear probe data.

Applies despiking and high-pass filtering during p2nc conversion,
so these expensive operations only need to be performed once.
"""

import logging
from typing import Dict

import numpy as np
import scipy.signal as sig

from ..signal import despike

logger = logging.getLogger(__name__)


def preprocess_shear(
    data: Dict,
    despike_passes: int = 6,
    hp_cutoff_hz: float = 0.5,
    shear_probes: tuple[str, ...] = ("sh1", "sh2"),
) -> Dict:
    """
    Apply despiking and high-pass filtering to shear probe data.

    This preprocessing step removes spikes and low-frequency contamination
    from shear signals. When performed during p2nc conversion, it eliminates
    the need to repeat these expensive operations during epsilon processing.

    Parameters
    ----------
    data : dict
        Data dictionary from convert_all_channels(), containing shear probe
        signals and fs_fast sampling rate.
    despike_passes : int, optional
        Maximum number of despike iterations. Default 6.
    hp_cutoff_hz : float, optional
        High-pass filter cutoff frequency in Hz. Default 0.5.
        Set to 0 or negative to disable HP filtering.
    shear_probes : tuple of str, optional
        Names of shear probe variables. Default ('sh1', 'sh2').

    Returns
    -------
    dict
        Input dictionary with additional preprocessed variables:
        - sh1_hp, sh2_hp: despiked and high-pass filtered shear signals
        - Units and attributes are copied from original variables

    Notes
    -----
    The preprocessing matches the MATLAB ODAS approach:
    1. Despike using iterative HP/LP filtering and spike detection
    2. High-pass filter at ~0.5 Hz (or 0.5/fft_length) to remove
       low-frequency contamination from profiler motion

    The preprocessed variables have suffix '_hp' and can be automatically
    detected by pyturb's epsilon processing to skip redundant filtering.
    """
    fs_fast = data.get("fs_fast", 512.0)
    units = data.get("units", {})

    # Find which probes are available
    available_probes = [p for p in shear_probes if p in data]
    if not available_probes:
        logger.warning("No shear probes found for preprocessing")
        return data

    logger.info(
        f"Preprocessing shear: despike_passes={despike_passes}, "
        f"hp_cutoff={hp_cutoff_hz} Hz, probes={available_probes}"
    )

    for probe in available_probes:

        signal = data[probe]
        if not isinstance(signal, np.ndarray):
            continue

        # Step 1: Despike
        cleaned, _, _, _ = despike(signal, fs=fs_fast, max_passes=despike_passes)

        # Step 2: High-pass filter (if enabled)
        if hp_cutoff_hz > 0:
            # Design first-order Butterworth high-pass filter
            sos = sig.butter(
                1, hp_cutoff_hz / (fs_fast / 2), btype="high", output="sos"
            )
            # Forward-backward filtering (zero phase)
            cleaned = sig.sosfiltfilt(sos, cleaned).astype(np.float32)

        # Store preprocessed signal
        hp_name = f"{probe}_hp"
        data[hp_name] = cleaned

        # Copy units
        if probe in units:
            units[hp_name] = units[probe]

    # Store preprocessing parameters as metadata
    data["_preprocess_params"] = {
        "despike_passes": despike_passes,
        "hp_cutoff_hz": hp_cutoff_hz,
        "shear_probes": shear_probes,
    }

    return data

"""
Sensor-specific data extraction utilities.
"""

from typing import Tuple

import numpy as np


def adis_extract(
    data: np.ndarray, data_type: str = "xy"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data from ADIS16209 inclinometer packed words.

    Word format:
    - bit 15: new data flag
    - bit 14: error flag
    - bits[13:0]: X/Y inclination (14-bit signed) or bits[11:0]: temp (12-bit unsigned)

    Parameters
    ----------
    data : ndarray
        Raw ADIS16209 words
    data_type : str
        'xy' for inclination or 't' for temperature

    Returns
    -------
    tuple
        (extracted_data, old_data_indices, error_indices)
    """
    data = data.copy()

    # Find old data (MS bit not set)
    old_flag = np.where(data >= 0)[0]

    # Find new data (MS bit set) and clear it
    new_data = data < -(2**14)
    data[new_data] = data[new_data] + 2**15

    # Find error flags (bit 14 set) and clear them
    error_flag = np.where(data >= 2**14)[0]
    data[error_flag] = data[error_flag] - 2**14

    # Convert to 2's complement for inclination data (14-bit)
    # Temperature is unsigned 12-bit - mask to get lower 12 bits only
    if data_type == "xy":
        # Values >= 2^13 are negative in 14-bit 2's complement
        negative = data >= 2**13
        data[negative] = data[negative] - 2**14
    elif data_type == "t":
        # Temperature is unsigned 12-bit, mask to lower 12 bits
        # This handles the case where data was read as signed int16
        data = data.astype(np.int16) & 0x0FFF

    return data, old_flag, error_flag

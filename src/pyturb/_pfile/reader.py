"""
Low-level file reading for RSI P-files.
"""

import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from .config import SetupConfig

logger = logging.getLogger(__name__)

# Header field indices (0-based)
_BYTES_PER_WORD = 2
_HEADER_SIZE_0 = 64
_HEADER_VERSION_I = 10
_SETUPFILE_SIZE_I = 11
_HEADER_SIZE_I = 17
_BLOCK_SIZE_I = 18
_CLOCK_WHOLE_I = 20
_CLOCK_FRACTION_I = 21
_FAST_COLS_I = 28
_SLOW_COLS_I = 29
_ROWS_I = 30


def _detect_endianness(filename: Path) -> Tuple[str, str]:
    """Detect file endianness from header flag."""
    FLAG_BIG, FLAG_LITTLE = 2, 1
    FLAG_POS = 63

    with open(filename, "rb") as f:
        header_bytes = f.read(128)

    flag_big = np.frombuffer(header_bytes, dtype=">u2")[FLAG_POS]
    flag_little = np.frombuffer(header_bytes, dtype="<u2")[FLAG_POS]

    if flag_big == flag_little == 0:
        return "<", "Endian flag not found, assuming little endian"
    elif flag_big == FLAG_BIG:
        return ">", ""
    elif flag_little == FLAG_LITTLE:
        return "<", ""
    elif flag_little == FLAG_BIG:
        return "<", "Endian flag mismatch, assuming little endian"
    elif flag_big == FLAG_LITTLE:
        return ">", "Endian flag mismatch, assuming big endian"
    else:
        raise ValueError(f"Invalid endian flag: {flag_big}")


def open_pfile(filename: Union[str, Path], mode: str = "rb") -> Tuple[object, str, str]:
    """
    Open an RSI P-file with automatic endian detection.

    Parameters
    ----------
    filename : str or Path
        Path to the P-file
    mode : str
        File mode (default 'rb')

    Returns
    -------
    file : file object
        Open file handle
    endian : str
        Endian format ('<' for little, '>' for big)
    error_msg : str
        Warning message if endian mismatch detected
    """
    filename = Path(filename).expanduser()

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    endian, error_msg = _detect_endianness(filename)
    file_obj = open(filename, mode)

    return file_obj, endian, error_msg


def read_pfile(filename: Union[str, Path]) -> Tuple[list, Dict]:
    """
    Read an RSI P-file and return demultiplexed channel data.

    This function logs diagnostic information at the module logger's DEBUG
    level instead of using an explicit `verbose` flag.

    Parameters
    ----------
    filename : str or Path
        Path to the P-file (with or without .p extension)

    Returns
    -------
    ch_list : list
        Channel names found in the file
    data : dict
        Channel arrays plus metadata (fs_fast, fs_slow, t_fast, t_slow,
        filetime, cfgobj, setupfilestr, header, etc.)
    """
    filename = Path(filename).expanduser()

    if not filename.suffix:
        filename = filename.with_suffix(".p")
    elif filename.suffix.lower() != ".p" and not filename.exists():
        filename = filename.with_suffix(".p")

    if not filename.exists():
        raise FileNotFoundError(f"Unable to find file: {filename}")

    fid, endian, error_message = open_pfile(filename)
    if error_message:
        warnings.warn(error_message)

    try:
        data = _read_pfile_impl(fid, endian, filename)
    finally:
        fid.close()

    return data


def _read_pfile_impl(fid, endian: str, filename: Path) -> Tuple[list, Dict]:
    """Implementation of P-file reading (called with open file handle)."""
    data = {"fullPath": str(filename.absolute())}

    # Read first header
    fid.seek(0)
    HD = np.frombuffer(fid.read(_HEADER_SIZE_0 * _BYTES_PER_WORD), dtype=f"{endian}u2")

    # Extract header parameters
    header_size = int(HD[_HEADER_SIZE_I]) // _BYTES_PER_WORD
    setupfile_size = int(HD[_SETUPFILE_SIZE_I])
    record_size = int(HD[_BLOCK_SIZE_I]) // _BYTES_PER_WORD
    data_size = record_size - header_size
    fast_cols = int(HD[_FAST_COLS_I])
    slow_cols = int(HD[_SLOW_COLS_I])
    n_cols = fast_cols + slow_cols
    n_rows = int(HD[_ROWS_I])
    f_clock = float(HD[_CLOCK_WHOLE_I]) + float(HD[_CLOCK_FRACTION_I]) / 1000.0

    data["fs_fast"] = f_clock / n_cols
    data["fs_slow"] = data["fs_fast"] / n_rows

    fid.seek(0, 2)
    filesize = fid.tell()

    header_ver = int(HD[_HEADER_VERSION_I])
    data["header_version"] = (header_ver >> 8) + (header_ver & 255) / 1000.0

    first_record_size = header_size * _BYTES_PER_WORD + setupfile_size
    block_size = int(HD[_BLOCK_SIZE_I])
    n_records = (filesize - first_record_size) / block_size

    if n_records % 1 != 0:
        n_records = int(n_records)
        warnings.warn(f"File {filename} does not contain an integer number of records")
    else:
        n_records = int(n_records)

    if n_records <= 1:
        raise ValueError(f"File {filename} contains no data")

    filesize = first_record_size + n_records * block_size

    matrix_count = int(n_records * data_size / (n_rows * n_cols))
    data["t_slow"] = np.arange(matrix_count) / data["fs_slow"]
    data["t_fast"] = np.arange(matrix_count * n_rows) / data["fs_fast"]

    # Read setup file string
    fid.seek(header_size * _BYTES_PER_WORD)
    data["setupfilestr"] = fid.read(setupfile_size).decode("ascii", errors="ignore")

    if not data["setupfilestr"]:
        raise ValueError("Failed to extract setup file string from first record")

    # Parse configuration
    cfg = SetupConfig(data["setupfilestr"])
    data["cfgobj"] = cfg

    # Load address matrix from config
    ch_matrix = np.zeros((n_rows, n_cols), dtype=int)
    matrix_sections = cfg.get_section_dicts("matrix")

    if matrix_sections:
        matrix_params = matrix_sections[0]["params"]
        for i in range(n_rows):
            row_key = f"row{i + 1:02d}"
            if row_key in matrix_params:
                row_str = matrix_params[row_key]
                values = [
                    int(x.strip()) for x in re.split(r"[,\s]+", row_str) if x.strip()
                ]
                ch_matrix[i, : len(values)] = values

    logger.debug("Address Matrix:")
    for row in ch_matrix:
        logger.debug("".join(f"{x:4d}" for x in row))

    # Build channel list from config
    ch_nums, ch_names = [], []

    if 255 in ch_matrix:
        ch_nums.append(255)
        ch_names.append("ch255")

    for section_dict in cfg.get_section_dicts("channel"):
        params = section_dict["params"]
        ch_id, ch_type, ch_name = (
            params.get("id"),
            params.get("type"),
            params.get("name"),
        )

        if not (ch_id and ch_type and ch_name):
            continue

        try:
            ids = [int(x.strip()) for x in ch_id.split(",")]
        except ValueError:
            continue

        if len(ids) == 1:
            if ids[0] in ch_matrix:
                logger.debug(f"     channel: {ids[0]:2d} = {ch_name}")
                ch_nums.append(ids[0])
                ch_names.append(ch_name)
        elif len(ids) == 2:
            # Even/odd pair for 32-bit channels
            if ids[0] in ch_matrix:
                logger.debug(f"even channel: {ids[0]:2d} = {ch_name}")
                ch_nums.append(ids[0])
                ch_names.append(f"{ch_name}_E")
            if ids[1] in ch_matrix:
                logger.debug(f" odd channel: {ids[1]:2d} = {ch_name}")
                ch_nums.append(ids[1])
                ch_names.append(f"{ch_name}_O")

    # Read data records
    fid.seek(first_record_size)
    bytes_to_read = filesize - first_record_size
    file_data = np.frombuffer(fid.read(bytes_to_read), dtype=f"{endian}i2")
    file_data = file_data.reshape(-1, header_size + data_size)

    # Extract headers
    headers = file_data[:, :64].astype(np.int32)
    headers[headers < 0] += 2**16
    data["header"] = headers.astype(np.uint16)

    # Demultiplex channels from data portion
    file_data = file_data[:, 64:]
    ch_data = {}
    for i, name in enumerate(ch_names):
        indices = ch_matrix == ch_nums[i]
        reshaped_data = file_data.reshape(-1, n_rows * n_cols)
        ch_data[name] = reshaped_data[:, indices.ravel()].ravel()

    # Extract timestamp from header
    second = float(HD[8]) + float(HD[9]) / 1000.0
    data["filetime"] = datetime(
        int(HD[3]),
        int(HD[4]),
        int(HD[5]),
        int(HD[6]),
        int(HD[7]),
        int(second),
        int((second % 1) * 1e6),
    )
    data["date"] = data["filetime"].strftime("%Y-%m-%d")
    data["time"] = data["filetime"].strftime("%H:%M:%S.%f")[:-3]

    # Combine even/odd pairs into 32-bit values
    # Use int64 to avoid signed overflow when high bit is set
    for name in list(ch_data.keys()):
        if not name.endswith("_E"):
            continue
        odd_name = name[:-2] + "_O"
        if odd_name not in ch_data:
            continue

        ch_even = ch_data[name].astype(np.int64)
        ch_even[ch_even < 0] += 2**16
        ch_odd = ch_data[odd_name].astype(np.int64)
        ch_odd[ch_odd < 0] += 2**16

        base_name = name[:-2]
        ch_data[base_name] = ch_odd * 2**16 + ch_even
        del ch_data[name]
        del ch_data[odd_name]

    # Apply signed/unsigned conversions based on channel type
    for name in list(ch_data.keys()):
        sign = cfg.get_value(name, "sign", "signed")
        ch_type = cfg.get_value(name, "type", "")

        if ch_type and ch_type.lower() == "jac_t":
            sign = "unsigned"
        elif ch_type and ch_type.lower() in ["sbt", "sbc", "jac_c", "o2_43f"]:
            continue

        data_array = ch_data[name].astype(np.int64)

        if sign and sign.lower() == "unsigned":
            data_array = np.where(data_array < 0, data_array + 2**16, data_array)
        else:
            data_array = np.where(data_array >= 2**31, data_array - 2**32, data_array)

        ch_data[name] = data_array

    # ch_list = list(ch_data.keys())
    data.update(ch_data)

    return data

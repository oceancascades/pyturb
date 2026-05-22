"""Profile processing for microstructure data."""

import logging
from dataclasses import dataclass, field
from typing import Any, Generator, Literal, Optional

import gsw  # type: ignore[import]
import numpy as np
import scipy.signal as sig
import xarray as xr
from profinder import find_profiles  # type: ignore[import]

from .shear import clean_shear_spec, estimate_epsilon
from .signal import despike, window_mean, window_psd
from .viscosity import viscosity

_log = logging.getLogger(__name__)


@dataclass
class ProfileConfig:
    """Configuration for profile processing.

    This dataclass contains all settings needed for the complete profile
    processing pipeline, including preprocessing (smoothing, scaling) and
    epsilon estimation.
    """

    # === Processing window parameters ===
    diss_len_sec: float = 4.0  # Dissipation window length in seconds
    fft_len_sec: float = 1.0  # FFT segment length in seconds

    # === Variable names (raw input) ===
    pressure: str = "P"
    speed: str = "W"
    temperature: str = "JAC_T"
    pitch: str = "Incl_Y"  # Pitch angle variable (degrees, positive nose up)

    # === Speed estimation parameters ===
    use_pitch_correction: bool = False  # Whether to correct speed for pitch/AoA
    angle_of_attack: float = 3.0  # Angle of attack in degrees
    dbar_to_m: float = 1.005  # Conversion from dbar to meters

    # === Probe names ===
    shear_probes: tuple[str, ...] = ("sh1", "sh2")
    temperature_probes: tuple[str, ...] = ("gradT1", "gradT2")

    # === Preprocessing parameters ===
    pressure_smoothing_period: float = (
        0.5  # Cutoff period for pressure low-pass filter (seconds)
    )
    filter_order: int = 4
    gap_threshold: float = (
        2.0  # Minimum gap duration to treat as discontinuity (seconds)
    )
    gap_factor: float = 4.0  # Gap detected if dt > gap_factor * median(dt)
    hp_cutoff_hz: float = 0.0  # High-pass cutoff for shear before spectra (Hz)
    # 0 = auto (0.5/fft_len_sec), >0 = explicit value, <0 = disabled

    # === Thresholds ===
    min_speed: float = (
        0.2  # Speed below which a window's epsilon is QC-flagged questionable
    )

    # === Default values for missing data ===
    default_temperature: float = 10.0
    default_salinity: float = 35.0
    default_density: float = 1025.0

    # === Auxiliary dataset variable names ===
    aux_time: str = "time"  # Time variable in auxiliary dataset
    aux_latitude: str = "lat"  # Latitude variable in auxiliary dataset
    aux_longitude: str = "lon"  # Longitude variable in auxiliary dataset
    aux_temperature: Optional[str] = (
        None  # Temperature variable in auxiliary dataset (opt-in)
    )
    aux_salinity: Optional[str] = (
        None  # Salinity variable in auxiliary dataset (opt-in)
    )
    aux_density: Optional[str] = None  # Density variable in auxiliary dataset (opt-in)

    # === Processing options ===
    chop_start: bool = True
    scale_probes: bool = True
    despike_max_passes: int = 6  # Max despike iterations (1 = ~4x faster)
    accel_clean: bool = False  # Goodman coherent-noise removal using accelerometers
    emc_clean: bool = (
        True  # Goodman coherent-noise removal using EM current meter channels
    )
    accel_channels: tuple[str, ...] = ("Ax", "Ay", "Az")
    emc_channels: tuple[str, ...] = (
        "EMC_Cur",
        "EM_Cur",
    )  # EM current channels used as noise references

    # === Multi-profile detection settings ===
    profile_direction: Literal["down", "up", "both"] = "down"  # Which casts to process
    min_profile_pressure: float = 0.0  # Minimum pressure (dbar) for profile detection
    peaks_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "height": 25,
            "distance": 200,
            "width": 200,
            "prominence": 25,
        }
    )  # kwargs for scipy.signal.find_peaks

    @property
    def all_probes(self) -> tuple[str, ...]:
        """All probe names (shear + temperature)."""
        return self.shear_probes + self.temperature_probes

    @property
    def speed_smooth(self) -> str:
        """Name of smoothed speed variable."""
        return f"{self.speed}_smooth"

    @property
    def pressure_smooth(self) -> str:
        """Name of smoothed pressure variable."""
        return f"{self.pressure}_smooth"


def estimate_speed_from_pressure(
    pressure: np.ndarray,
    fs: float,
    pitch: Optional[np.ndarray] = None,
    angle_of_attack: float = 3.0,
    dbar_to_m: float = 1.005,
) -> np.ndarray:
    """
    Estimate fall speed from pressure rate of change. Optionally corrects for pitch.

    Parameters
    ----------
    pressure : ndarray
        Pressure in dbar (should be pre-smoothed)
    fs : float
        Sampling frequency in Hz
    pitch : ndarray, optional
        Pitch angle in degrees (positive = nose up). If None, assumes vertical.
    angle_of_attack : float
        Angle of attack in degrees (default: 3.0)
    dbar_to_m : float
        Conversion factor from dbar to meters (default: 1.005 = 1025 * 9.81 / 1e4)

    Returns
    -------
    ndarray
        Estimated speed along profiler path in m/s (positive = moving through water)
    """
    depth = pressure * dbar_to_m

    w = np.gradient(depth, 1 / fs)

    if pitch is not None:
        total_angle = np.abs(pitch) + angle_of_attack
        total_angle_rad = np.deg2rad(total_angle)
        speed = np.abs(w) / np.sin(total_angle_rad)
    else:
        # No pitch correction - assume vertical profiler
        speed = np.abs(w)

    return speed


def gap_aware_sosfiltfilt(
    sos: np.ndarray,
    data: np.ndarray,
    time: np.ndarray,
    gap_threshold: float = 5.0,
    gap_factor: float = 10.0,
    min_segment_length: int = 10,
) -> np.ndarray:
    """
    Apply sosfiltfilt independently to contiguous time segments.

    Detects gaps in the time series and applies the filter separately to each
    segment to avoid filter artifacts at discontinuities.

    Parameters
    ----------
    sos : ndarray
        Second-order sections representation of the filter.
    data : ndarray
        Input data to filter.
    time : ndarray
        Time vector (same length as data).
    gap_threshold : float, optional
        Minimum gap duration in seconds to treat as discontinuity. Default 5.0.
    gap_factor : float, optional
        Gap detected if dt > gap_factor * median(dt). Default 10.0.
    min_segment_length : int, optional
        Minimum segment length to apply filter. Shorter segments are
        returned unfiltered. Default 10.

    Returns
    -------
    ndarray
        Filtered data with same shape as input.
    """
    if len(data) < min_segment_length:
        return data.copy()

    # Compute time differences in seconds
    dt = np.diff(time)

    # Convert to float seconds if datetime64
    if np.issubdtype(dt.dtype, np.timedelta64):
        dt = dt.astype("timedelta64[ns]").astype(float) / 1e9
    elif np.issubdtype(dt.dtype, np.datetime64):
        # Shouldn't happen with diff, but handle just in case
        dt = dt.astype("datetime64[ns]").astype(float) / 1e9

    median_dt = np.median(dt)

    # Detect gaps where time jump exceeds threshold
    threshold = max(gap_threshold, gap_factor * median_dt)
    gap_mask = dt > threshold
    gap_indices = np.where(gap_mask)[0] + 1  # +1 because diff reduces length by 1

    n_gaps = len(gap_indices)

    if n_gaps > 0:
        _log.debug(
            f"Detected {n_gaps} time gap(s) in data "
            f"(threshold={threshold:.2f}s, median_dt={median_dt:.4f}s)"
        )
        for i, idx in enumerate(gap_indices):
            gap_size = dt[idx - 1]  # -1 because gap_indices is offset by 1
            _log.debug(f"  Gap {i + 1}: {gap_size:.2f}s at index {idx}")

    if n_gaps == 0:
        # No gaps, filter entire array
        return sig.sosfiltfilt(sos, data)

    # Split data at gap boundaries
    split_indices = gap_indices.tolist()
    segments = np.split(data, split_indices)

    # Filter each segment independently
    filtered_segments = []
    for seg in segments:
        if len(seg) >= min_segment_length:
            filtered_segments.append(sig.sosfiltfilt(sos, seg))
        else:
            # Segment too short for filtfilt, return unfiltered
            filtered_segments.append(seg.copy())

    return np.concatenate(filtered_segments)


def prepare_profile(
    ds: xr.Dataset,
    config: Optional[ProfileConfig] = None,
) -> xr.Dataset:
    """
    Prepare raw p2nc output for epsilon processing.

    This function performs the preprocessing needed for data converted via p2nc:
    1. Low-pass filters the pressure data
    2. Estimates or smooths the speed variable
    3. Scales shear by 1/U^2 to convert to du/dz
    4. Scales temperature gradient by 1/U to convert to dT/dz

    If the speed variable (default 'W') is not present in the dataset, speed
    is estimated from pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from p2nc conversion containing:
        - P on t_slow dimension
        - Optionally W (speed) on t_slow dimension
        - Optionally Incl_Y (pitch) on t_slow dimension
        - sh1, sh2, gradT1, gradT2 on t_fast dimension
        - fs_slow, fs_fast sampling rates as attributes or variables
    config : ProfileConfig, optional
        Configuration for preprocessing. If None, uses defaults.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - {speed}_smooth: smoothed or estimated speed
        - {pressure}_smooth: smoothed pressure
        - sh1, sh2: scaled by 1/speed^2 (overwrites original)
        - gradT1, gradT2: scaled by 1/speed (overwrites original)
    """
    if config is None:
        config = ProfileConfig()

    ds = ds.copy()

    # Get sampling rate for slow channels
    fs_slow = float(ds.fs_slow)

    # Design low-pass filter for pressure (and existing speed if present)
    # using a configurable cutoff period.
    cutoff = 1 / config.pressure_smoothing_period
    sos = sig.butter(config.filter_order, cutoff, btype="low", fs=fs_slow, output="sos")

    # Get time vector for gap detection
    t_slow = ds.t_slow.values

    # Smooth pressure with gap-aware filtering
    if config.pressure in ds:
        ds[config.pressure_smooth] = (
            "t_slow",
            gap_aware_sosfiltfilt(
                sos,
                ds[config.pressure].values,
                t_slow,
                gap_threshold=config.gap_threshold,
                gap_factor=config.gap_factor,
            ),
        )
    else:
        raise ValueError(f"Pressure variable '{config.pressure}' not found in dataset")

    if config.speed in ds:
        # Speed variable exists - smooth it with gap-aware filtering
        ds[config.speed_smooth] = (
            "t_slow",
            gap_aware_sosfiltfilt(
                sos,
                ds[config.speed].values,
                t_slow,
                gap_threshold=config.gap_threshold,
                gap_factor=config.gap_factor,
            ),
        )
    else:
        _log.info(
            f"Speed variable '{config.speed}' not found, "
            "estimating from pressure derivative"
        )

        pitch = None
        if config.use_pitch_correction and config.pitch in ds:
            pitch = ds[config.pitch].values
            _log.info(f"Using pitch correction with AoA={config.angle_of_attack}°")
        speed_est = estimate_speed_from_pressure(
            ds[config.pressure_smooth].values,
            fs_slow,
            pitch=pitch,
            angle_of_attack=config.angle_of_attack,
            dbar_to_m=config.dbar_to_m,
        )

        # Speed is already smoothed in estimate_speed_from_pressure
        ds[config.speed_smooth] = ("t_slow", speed_est)

    if not config.scale_probes:
        return ds

    # Interpolate smoothed speed to fast time for scaling
    interp_kwargs = dict(
        t_slow=ds.t_fast,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    U_fast = ds[config.speed_smooth].interp(**interp_kwargs)

    for probe in config.shear_probes:
        if probe in ds:
            ds[probe] = ds[probe] / U_fast**2

    for probe in config.temperature_probes:
        if probe in ds:
            ds[probe] = ds[probe] / U_fast

    return ds


def despike_variables(
    ds: xr.Dataset,
    variables: tuple[str, ...],
    suffix: str = "_clean",
    max_passes: int = 10,
) -> xr.Dataset:
    """Despike specified variables, creating new cleaned versions."""
    ds = ds.copy()

    for var in variables:
        if var not in ds:
            continue
        cleaned, _, _, _ = despike(ds[var].values, max_passes=max_passes)
        ds[var + suffix] = ("t_fast", cleaned)

    return ds


def highpass_filter(
    ds: xr.Dataset,
    variables: tuple[str, ...],
    fs: float,
    cutoff_hz: float,
    suffix: str = "_clean",
) -> xr.Dataset:
    """
    Apply high-pass filter to variables before spectral analysis.

    This removes low-frequency contamination (profiler motion, etc.) that
    would otherwise bias the spectral variance estimate. MATLAB ODAS recommends
    HP filtering at ~0.5 / fft_length_seconds before computing dissipation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with despiked variables.
    variables : tuple of str
        Variable names to filter (without suffix).
    fs : float
        Sampling frequency in Hz.
    cutoff_hz : float
        High-pass cutoff frequency in Hz.
    suffix : str
        Suffix for cleaned variables (default: "_clean").

    Returns
    -------
    xr.Dataset
        Dataset with high-pass filtered variables (overwrites *_clean).
    """
    ds = ds.copy()

    # Design first-order Butterworth high-pass filter
    sos = sig.butter(1, cutoff_hz / (fs / 2), btype="high", output="sos")

    for var in variables:
        var_clean = f"{var}{suffix}"
        if var_clean not in ds:
            continue
        filtered = sig.sosfiltfilt(sos, ds[var_clean].values)
        ds[var_clean] = ("t_fast", filtered.astype(ds[var_clean].dtype))

    return ds


def find_all_profiles(
    ds: xr.Dataset,
    config: ProfileConfig,
) -> list[tuple[int, int]]:
    """
    Find all profile segments in a dataset.

    A combination of strategies are used:

    1. Gap-based (merged / pre-segmented datasets): When the
       time series contains breaks larger than ``gap_threshold`` seconds (or
       ``gap_factor x median_dt``), and the pressure differences at each step
       are mostly in one direction then the data are treated as a single profile.

    2. Peak-based (multi-profile): When no
       gaps are found, ``profinder.find_profiles`` identifies dive/ascent
       cycles from pressure peaks and troughs.  Signed velocity (negative =
       ascending) is derived from the smoothed pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with smoothed pressure and speed variables (from prepare_profile).
    config : ProfileConfig
        Configuration specifying profile detection parameters.

    Returns
    -------
    list of tuple[int, int]
        List of (start_idx, end_idx) tuples for each detected profile segment
        on the t_slow dimension. Returns empty list if no profiles found.
    """
    pressure_var = config.pressure_smooth

    pressure = ds[pressure_var].values
    t_slow = ds.t_slow.values
    n = len(pressure)

    dt = np.diff(t_slow)
    if np.issubdtype(dt.dtype, np.timedelta64):
        dt = dt.astype("timedelta64[ns]").astype(float) / 1e9
    else:
        dt = dt.astype(float)

    median_dt = np.median(dt)
    threshold = max(config.gap_threshold, config.gap_factor * median_dt)
    gap_indices = (np.where(dt > threshold)[0] + 1).tolist()

    if gap_indices:
        min_height = config.peaks_kwargs.get(
            "height", max(config.min_profile_pressure, 1.0)
        )
        boundaries = [0] + gap_indices + [n]
        segments: list[tuple[int, int]] = []

        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1] - 1  # inclusive

            seg_p = pressure[seg_start : seg_end + 1]
            seg_n = len(seg_p)
            if seg_n < 2:
                continue

            # Skip segments that never reach the minimum depth
            if seg_p.max() < min_height:
                continue

            # Determine if this segment is monotonic. For a
            # single glider cast, most pressure steps will be in one direction.
            # For a VMP segment cycling between surface and depth the fraction
            # will be close to 0.5.
            seg_dp_steps = np.diff(seg_p)
            n_pos = int(np.sum(seg_dp_steps > 0))  # steps toward deeper
            n_neg = int(np.sum(seg_dp_steps < 0))  # steps toward shallower
            n_total = n_pos + n_neg
            dominant_frac = max(n_pos, n_neg) / n_total if n_total > 0 else 1.0
            monotonic = dominant_frac >= 0.8
            mostly_down = n_pos >= n_neg

            is_single = (
                (config.profile_direction == "down" and monotonic and mostly_down)
                or (config.profile_direction == "up" and monotonic and not mostly_down)
                or (config.profile_direction == "both" and monotonic)
            )

            if is_single:
                # Single monotonic profile: keep the entire segment; speed is
                # only used to QC-flag individual dissipation windows later.
                if seg_end > seg_start:
                    segments.append((seg_start, seg_end))
            else:
                try:
                    sub_profiles = find_profiles(
                        seg_p,
                        min_pressure=config.min_profile_pressure,
                        peaks_kwargs=config.peaks_kwargs,
                        apply_speed_threshold=False,
                        direction=config.profile_direction,
                    )
                except Exception:
                    continue
                for down_start, down_end, up_start, up_end in sub_profiles:
                    if config.profile_direction == "down":
                        s = seg_start + max(0, down_start)
                        e = seg_start + min(down_end, seg_n - 1)
                        if e > s:
                            segments.append((s, e))
                    elif config.profile_direction == "up":
                        s = seg_start + max(0, up_start)
                        e = seg_start + min(up_end, seg_n - 1)
                        if e > s:
                            segments.append((s, e))
                    else:  # "both"
                        d_s = seg_start + max(0, down_start)
                        d_e = seg_start + min(down_end, seg_n - 1)
                        if d_e > d_s:
                            segments.append((d_s, d_e))
                        u_s = seg_start + max(0, up_start)
                        u_e = seg_start + min(up_end, seg_n - 1)
                        if u_e > u_s:
                            segments.append((u_s, u_e))

        _log.info(f"Found {len(segments)} profile segment(s) via gap-based detection")
        return segments

    try:
        profiles = find_profiles(
            pressure,
            min_pressure=config.min_profile_pressure,
            peaks_kwargs=config.peaks_kwargs,
            apply_speed_threshold=False,
            direction=config.profile_direction,
        )
    except Exception as e:
        _log.warning(f"Profile detection failed: {e}")
        return []

    if not profiles:
        _log.info("No profiles detected in dataset")
        return []

    # Extract segments based on direction
    # profiles is list of (down_start, down_end, up_start, up_end)
    segments = []
    for down_start, down_end, up_start, up_end in profiles:
        if config.profile_direction == "down":
            start = max(0, down_start)
            end = min(down_end, n - 1)
            if end > start:
                segments.append((start, end))
        elif config.profile_direction == "up":
            start = max(0, up_start)
            end = min(up_end, n - 1)
            if end > start:
                segments.append((start, end))
        else:  # "both"
            d_start = max(0, down_start)
            d_end = min(down_end, n - 1)
            if d_end > d_start:
                segments.append((d_start, d_end))
            u_start = max(0, up_start)
            u_end = min(up_end, n - 1)
            if u_end > u_start:
                segments.append((u_start, u_end))

    _log.info(f"Found {len(segments)} profile segment(s)")

    return segments


def split_into_profiles(
    ds: xr.Dataset,
    config: ProfileConfig,
) -> Generator[tuple[int, xr.Dataset], None, None]:
    """
    Split a dataset into individual profile segments.

    This generator yields individual profile datasets suitable for processing
    with process_profile(). Each yielded dataset is a subset of the original
    containing data for one down or up cast.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with smoothed pressure and speed variables (from prepare_profile).
    config : ProfileConfig
        Configuration specifying profile detection parameters.

    Yields
    ------
    tuple[int, xr.Dataset]
        Tuple of (profile_index, profile_dataset) where profile_index is
        0-based and profile_dataset is the subset for that profile.

    Examples
    --------
    >>> ds = prepare_profile(raw_ds, config)
    >>> for i, profile_ds in split_into_profiles(ds, config):
    ...     result = process_profile(profile_ds, config)
    ...     result.to_netcdf(f'profile_{i:03d}.nc')
    """
    segments = find_all_profiles(ds, config)

    for i, (idx_start, idx_end) in enumerate(segments):
        # Get time bounds for slicing
        t0 = ds.t_slow.values[idx_start]
        t1 = ds.t_slow.values[idx_end]

        # Slice both time dimensions
        profile_ds = ds.sel(t_slow=slice(t0, t1), t_fast=slice(t0, t1))

        # Add profile metadata
        profile_ds.attrs["profile_index"] = i
        profile_ds.attrs["profile_start_idx"] = idx_start
        profile_ds.attrs["profile_end_idx"] = idx_end

        yield i, profile_ds


def compute_window_parameters(ds: xr.Dataset, config: ProfileConfig) -> dict:
    """Compute windowing parameters based on config and sampling rates."""
    fs_fast = float(ds.fs_fast)
    fs_slow = float(ds.fs_slow)

    n_fft = int(config.fft_len_sec * fs_fast)
    n_diss = int(config.diss_len_sec * fs_fast)

    return {
        "n_fft": n_fft,
        "n_diss": n_diss,
        "fft_overlap": n_fft // 2,
        "diss_overlap": n_fft // 2,
        "sampling_ratio": int(fs_fast / fs_slow),
    }


def trim_to_complete_windows(
    ds: xr.Dataset,
    params: dict,
    chop_start: bool = True,
) -> xr.Dataset:
    """Trim dataset to contain exactly n complete dissipation windows."""
    n_diss = params["n_diss"]
    diss_overlap = params["diss_overlap"]
    sampling_ratio = params["sampling_ratio"]
    diss_step = n_diss - diss_overlap

    # Calculate number of complete windows that fit
    n_windows = (ds.t_fast.size - n_diss) // diss_step + 1

    if n_windows < 1:
        raise ValueError("Insufficient data for even one dissipation window")

    # Exact number of fast samples needed: first window is n_diss,
    # each additional window adds diss_step samples
    n_fast = n_diss + (n_windows - 1) * diss_step

    # Ensure n_slow aligns with sampling ratio
    n_slow = n_fast // sampling_ratio
    # Adjust n_fast to be exact multiple of sampling_ratio
    n_fast = n_slow * sampling_ratio

    # Recalculate windows with adjusted n_fast
    n_windows = (n_fast - n_diss) // diss_step + 1
    n_fast = n_diss + (n_windows - 1) * diss_step
    n_slow = n_fast // sampling_ratio

    if chop_start:
        # Take from end (chop start of profile)
        fast_start = ds.t_fast.size - n_fast
        slow_start = ds.t_slow.size - n_slow
        return ds.isel(
            t_fast=slice(fast_start, fast_start + n_fast),
            t_slow=slice(slow_start, slow_start + n_slow),
        )
    else:
        # Take from start (chop end of profile)
        return ds.isel(
            t_fast=slice(0, n_fast),
            t_slow=slice(0, n_slow),
        )


def compute_window_means(
    ds: xr.Dataset,
    variables: list[str],
    params: dict,
) -> dict[str, np.ndarray]:
    """Compute window means, auto-detecting fast vs slow dimension."""
    n_fft = params["n_fft"]
    n_diss = params["n_diss"]
    ratio = params["sampling_ratio"]

    result = {}
    for var in variables:
        if var not in ds:
            continue
        if "t_slow" in ds[var].dims:
            result[var] = window_mean(ds[var].values, n_fft // ratio, n_diss // ratio)
        else:
            result[var] = window_mean(ds[var].values, n_fft, n_diss)
    return result


def compute_viscosity(
    temperature: np.ndarray,
    salinity: float = 35.0,
    density: float = 1025.0,
) -> np.ndarray:
    """Compute kinematic viscosity from temperature."""
    nu, _ = viscosity(salinity, temperature, density)
    return nu


def compute_spectra(
    ds: xr.Dataset,
    variables: tuple[str, ...],
    fs: float,
    n_fft: int,
    n_diss: int,
) -> tuple[Optional[np.ndarray], dict[str, np.ndarray]]:
    """Compute power spectra for cleaned variables."""
    spectra = {}
    freq = None

    for var in variables:
        clean_var = f"{var}_clean"
        if clean_var not in ds:
            continue
        freq, psd = window_psd(ds[clean_var].values, fs, n_fft, n_diss)
        spectra[var] = psd

    return freq, spectra


def compute_epsilon(
    frequency: np.ndarray,
    spectra: dict[str, np.ndarray],
    speed: np.ndarray,
    nu: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute epsilon for each shear probe spectrum."""
    results = {}

    for name, psd in spectra.items():
        if not name.startswith("sh"):
            continue

        n_windows = psd.shape[0]
        eps = np.full(n_windows, np.nan)
        k_max = np.full(n_windows, np.nan)

        for i in range(n_windows):
            eps[i], k_max[i] = estimate_epsilon(frequency, psd[i], W=speed[i], nu=nu[i])

        results[name] = (eps, k_max)

    return results


def _resolve_hp_cutoff(config: ProfileConfig) -> Optional[float]:
    """Resolve the shear high-pass cutoff (positive value, 0 = auto, <0 = off)."""
    if config.hp_cutoff_hz > 0:
        return config.hp_cutoff_hz
    if config.hp_cutoff_hz == 0:
        return 0.5 / config.fft_len_sec
    return None


def _window_mean_slow(x: np.ndarray, params: dict) -> np.ndarray:
    """Window-mean a slow-channel array using params from compute_window_parameters."""
    return window_mean(
        x,
        params["n_fft"] // params["sampling_ratio"],
        params["n_diss"] // params["sampling_ratio"],
    )


def _preprocess_for_spectra(
    ds: xr.Dataset, config: ProfileConfig
) -> tuple[xr.Dataset, dict]:
    """Despike, high-pass, segment, and window-align the profile.

    Returns the trimmed dataset and the window-parameter dict.
    """
    if config.speed_smooth not in ds:
        _log.debug("Smoothed speed not found, running prepare_profile")
        ds = prepare_profile(ds, config)

    ds = despike_variables(ds, config.all_probes, max_passes=config.despike_max_passes)

    hp_cutoff = _resolve_hp_cutoff(config)
    if hp_cutoff is not None and hp_cutoff > 0:
        ds = highpass_filter(ds, config.shear_probes, float(ds.fs_fast), hp_cutoff)

    params = compute_window_parameters(ds, config)
    ds = trim_to_complete_windows(ds, params, config.chop_start)
    for key, val in params.items():
        ds.attrs[key] = val
    return ds, params


def _derive_thermo(
    ds: xr.Dataset,
    means: dict,
    params: dict,
    config: ProfileConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Resolve window-mean T, S, density, and the temperature used for viscosity.

    Priority cascade:
      - temperature:  aux_temperature  > CT sensor (config.temperature) > default
      - salinity:     aux_salinity     > derived from JAC_C if valid     > default
      - density:      aux_density      > derived from JAC-derived S + T  > default
      - T for nu:     aux_temperature if present, else CT/default ``T_mean``

    Returns (T_mean, S_mean, rho_mean, T_visc, salinity_from_jac).
    """
    n_windows = len(means["t_slow"])
    pressure_var = config.pressure_smooth

    if config.temperature in means:
        T_mean = means[config.temperature]
    else:
        T_mean = np.full(n_windows, config.default_temperature)

    salinity_from_jac = False
    if "aux_salinity" in ds:
        S_mean = _window_mean_slow(ds["aux_salinity"].values, params)
    elif "JAC_C" in means and config.temperature in means:
        # JAC_C is in mS/cm (matching MATLAB ODAS output). Only trust values
        # in the seawater range.
        C_mScm = means["JAC_C"]
        if np.nanmedian(C_mScm) > 10.0:
            T_insitu = means[config.temperature]
            P_dbar = means.get(pressure_var, np.full(n_windows, 0.0))
            S_mean = gsw.SP_from_C(C_mScm, T_insitu, P_dbar)
            salinity_from_jac = True
        else:
            _log.warning(
                f"JAC_C values too low (median={np.nanmedian(C_mScm):.3f} mS/cm), "
                "skipping salinity calculation from CT sensor"
            )
            S_mean = np.full(n_windows, config.default_salinity)
    else:
        S_mean = np.full(n_windows, config.default_salinity)

    if "aux_density" in ds:
        rho_mean = _window_mean_slow(ds["aux_density"].values, params)
    elif salinity_from_jac:
        lon = (
            float(np.nanmean(ds["aux_longitude"].values))
            if "aux_longitude" in ds
            else 0.0
        )
        lat = (
            float(np.nanmean(ds["aux_latitude"].values))
            if "aux_latitude" in ds
            else 45.0
        )
        P_dbar = means.get(pressure_var, np.full(n_windows, 0.0))
        T_insitu = means[config.temperature]
        SA = gsw.SA_from_SP(S_mean, P_dbar, lon, lat)
        CT = gsw.CT_from_t(SA, T_insitu, P_dbar)
        rho_mean = gsw.rho(SA, CT, P_dbar)
    else:
        rho_mean = np.full(n_windows, config.default_density)

    if "aux_temperature" in ds:
        T_visc = _window_mean_slow(ds["aux_temperature"].values, params)
    else:
        T_visc = T_mean

    return T_mean, S_mean, rho_mean, T_visc, salinity_from_jac


def _attach_window_scalars(
    ds: xr.Dataset, params: dict, config: ProfileConfig
) -> xr.Dataset:
    """Compute window-mean scalars and attach them on the output ``time`` axis.

    Adds: ``time`` coord, ``pressure``, ``W``, ``temperature``, ``nu``;
    plus ``salinity``, ``density``, ``lat``, ``lon``, ``conductivity`` when
    the relevant inputs are available.
    """
    pressure_var = config.pressure_smooth
    speed_var = config.speed_smooth

    means = compute_window_means(
        ds,
        ["t_slow", pressure_var, speed_var, config.temperature, "JAC_C"],
        params,
    )

    n_windows = len(means["t_slow"])
    ds = ds.assign_coords(time=("time", means["t_slow"]))
    if "units" in ds.t_slow.attrs:
        ds.time.attrs["units"] = ds.t_slow.attrs["units"]
    if "long_name" in ds.t_slow.attrs:
        ds.time.attrs["long_name"] = "Time (dissipation windows)"

    ds["pressure"] = (
        "time",
        means.get(pressure_var, np.full(n_windows, np.nan)).astype("f4"),
    )
    ds["W"] = ("time", means.get(speed_var, np.full(n_windows, np.nan)).astype("f4"))

    T_mean, S_mean, rho_mean, T_visc, salinity_from_jac = _derive_thermo(
        ds, means, params, config
    )

    # Only attach S and rho if they come from a real source (not the default).
    if "aux_salinity" in ds or salinity_from_jac:
        ds["salinity"] = ("time", S_mean.astype("f4"))
    if "aux_density" in ds or salinity_from_jac:
        ds["density"] = ("time", rho_mean.astype("f4"))

    if "aux_temperature" in ds:
        ds["temperature"] = ("time", T_visc.astype("f4"))
    else:
        ds["temperature"] = ("time", T_mean.astype("f4"))

    ds["nu"] = ("time", compute_viscosity(T_visc, S_mean, rho_mean).astype("f4"))

    if "aux_latitude" in ds:
        ds["lat"] = (
            "time",
            _window_mean_slow(ds["aux_latitude"].values, params).astype("f4"),
        )
    if "aux_longitude" in ds:
        ds["lon"] = (
            "time",
            _window_mean_slow(ds["aux_longitude"].values, params).astype("f4"),
        )
    if "JAC_C" in means:
        ds["conductivity"] = ("time", means["JAC_C"].astype("f4"))

    return ds


def _compute_shear_spectra_with_cleaning(
    ds: xr.Dataset, params: dict, config: ProfileConfig
) -> tuple[xr.Dataset, np.ndarray, dict[str, np.ndarray]]:
    """Compute power spectra and optionally apply Goodman coherent-noise removal.

    Returns the dataset with ``frequency`` coord and ``S_*`` data vars
    attached, plus the frequency vector and spectra dict (for downstream
    epsilon estimation).
    """
    freq, spectra = compute_spectra(
        ds,
        config.all_probes,
        float(ds.fs_fast),
        params["n_fft"],
        params["n_diss"],
    )

    if config.accel_clean or config.emc_clean:
        avail_accel = (
            [ch for ch in config.accel_channels if ch in ds]
            if config.accel_clean
            else []
        )
        avail_emc = (
            [ch for ch in config.emc_channels if ch in ds] if config.emc_clean else []
        )
        all_noise_refs = avail_accel + avail_emc
        if all_noise_refs:
            avail_shear = [p for p in config.shear_probes if f"{p}_clean" in ds]
            if avail_shear:
                accel_data = np.column_stack([ds[ch].values for ch in all_noise_refs])
                shear_data = np.column_stack(
                    [ds[f"{p}_clean"].values for p in avail_shear]
                )
                freq_clean, clean_psd = clean_shear_spec(
                    shear_data,
                    accel_data,
                    params["n_fft"],
                    float(ds.fs_fast),
                    params["n_diss"],
                )
                # clean_psd shape: (n_windows, n_probes, n_freq) or (n_windows, n_freq)
                if clean_psd.ndim == 2:
                    spectra[avail_shear[0]] = clean_psd
                else:
                    for i, p in enumerate(avail_shear):
                        spectra[p] = clean_psd[:, i, :]
                freq = freq_clean
                _log.info(
                    "Applied Goodman cleaning using %s", ", ".join(all_noise_refs)
                )
        else:
            requested = (list(config.accel_channels) if config.accel_clean else []) + (
                list(config.emc_channels) if config.emc_clean else []
            )
            _log.warning(
                "Goodman cleaning requested but no noise reference channels (%s) "
                "found in dataset",
                ", ".join(requested),
            )

    ds = ds.assign_coords(frequency=("frequency", freq))
    for name, psd in spectra.items():
        ds[f"S_{name}"] = (("time", "frequency"), psd.astype("f4"))
    return ds, freq, spectra


_QC_FLAG_VALUES = np.array([0, 1, 2, 4, 9], dtype="i1")
_QC_FLAG_MEANINGS = "unknown good questionable bad missing"


def _attach_epsilon(
    ds: xr.Dataset,
    freq: np.ndarray,
    spectra: dict[str, np.ndarray],
    config: ProfileConfig,
) -> xr.Dataset:
    """Attach per-probe epsilon, k_max, wavenumber ``k``, and QC flags.

    QC convention (IODE): 0=unknown, 1=good, 2=questionable, 4=bad, 9=missing.
    Flags start at 0 and are raised to 2 where the window-mean speed ``W``
    falls below ``config.min_speed`` — epsilon is still computed in those
    windows, but at speeds below the probe's normal operating range.
    """
    epsilon_results = compute_epsilon(freq, spectra, ds["W"].values, ds["nu"].values)
    qc_base = np.zeros(ds["W"].size, dtype="i1")
    qc_base[ds["W"].values < config.min_speed] = 2

    for name, (eps, k_max) in epsilon_results.items():
        probe_num = name[-1]
        ds[f"eps_{probe_num}"] = ("time", eps.astype("f4"))
        ds[f"k_max_{probe_num}"] = ("time", k_max.astype("f4"))
        qc_var = f"eps_{probe_num}_qc"
        ds[qc_var] = ("time", qc_base.copy())
        ds[qc_var].attrs = {
            "long_name": f"QC flag for eps_{probe_num}",
            "flag_values": _QC_FLAG_VALUES,
            "flag_meanings": _QC_FLAG_MEANINGS,
            "valid_min": np.int8(0),
            "valid_max": np.int8(9),
        }

    ds["k"] = ds.frequency / ds.W
    return ds


def process_profile(
    ds: xr.Dataset,
    config: Optional[ProfileConfig] = None,
) -> xr.Dataset:
    """Process a microstructure profile to compute dissipation rates.

    If the dataset hasn't been prepared (no smoothed speed variable),
    :func:`prepare_profile` will be called automatically.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from p2nc conversion or after :func:`prepare_profile`.
    config : ProfileConfig, optional
        Configuration for processing. If None, uses defaults.

    Returns
    -------
    xr.Dataset
        Dataset with epsilon estimates, shear spectra, and supporting scalars
        on the ``time`` dimension.
    """
    if config is None:
        config = ProfileConfig()

    ds, params = _preprocess_for_spectra(ds, config)
    ds = _attach_window_scalars(ds, params, config)
    ds, freq, spectra = _compute_shear_spectra_with_cleaning(ds, params, config)
    return _attach_epsilon(ds, freq, spectra, config)

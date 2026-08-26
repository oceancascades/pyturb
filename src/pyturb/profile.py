"""Profile processing for microstructure data."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generator, Literal, Optional

import gsw  # type: ignore[import]
import numpy as np
import scipy.signal as sig
import xarray as xr
import yaml
from profinder import find_profiles  # type: ignore[import]

from .conductivity import match_conductivity_to_temperature
from .noise import _channel_calibration_params, thermistor_noise_phi
from .shear import estimate_epsilon, viscosity
from .shear import single_pole_correction as shear_response_correction
from .signal import (
    block_mean,
    clean_spec,
    despike_mask_name,
    despike_variables,
    window_mean,
    window_psd,
)
from .temperature import estimate_chi, thermal_diffusivity
from .temperature import single_pole_correction as gradT_response_correction

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
    # FM (figure of merit) thresholds for per-window QC.
    # FM = mad(log10(spectrum / Nasmyth)) * sqrt(dof_spec). Low FM = good
    # Nasmyth-shaped spectrum. FM <= fm_good -> qc=1 (good);
    # fm_good < FM <= fm_bad -> qc=2 (questionable); FM > fm_bad -> qc=4 (bad).
    # Speed-based and FM-based QC are combined by taking the higher flag.
    fm_good: float = 1.5
    fm_bad: float = 2.5
    # Despike fraction thresholds for per-window QC. The per-window fraction
    # of fast samples modified by despiking promotes the eps_N_qc flag for
    # the matching shear probe: fraction > despike_frac_questionable -> 2,
    # fraction > despike_frac_bad -> 4. Combined with the speed and FM
    # contributions via max.
    despike_frac_questionable: float = 0.1
    despike_frac_bad: float = 0.2

    # === Default values for missing data ===
    default_temperature: float = 10.0
    default_salinity: float = 35.0
    default_density: float = 1025.0
    # Fallback position for gsw calculations (SA_from_SP, in-situ density) when
    # no lat/lon is available from an auxiliary dataset.
    default_latitude: float = 45.0
    default_longitude: float = 0.0

    # === Conservative Temperature / Absolute Salinity / potential density ===
    # Requires real (non-default) temperature and salinity to be available.
    compute_thermo: bool = False

    # === Temperature variance dissipation (chi) ===
    compute_chi: bool = True
    # FP07 single-pole response: tau = fp07_tau0 * W^fp07_speed_exp.
    fp07_tau0: float = 0.010
    fp07_speed_exp: float = -0.5

    # === JAC-CT conductivity matching ===
    match_conductivity: bool = True  # lag/low-pass match JAC_C to temperature
    jac_lag: float = 0.0234  # seconds, at jac_reference_speed
    jac_f_tc: float = 0.73  # Hz, at jac_reference_speed
    jac_reference_speed: float = 0.62  # m/s

    # === High-resolution CTD output ===
    # CTD scalars (pressure, temperature, salinity, conductivity, density)
    # aren't limited by the FFT/dissipation window, so also attach them on a
    # finer ctd_time axis (suffix "_hires"), alongside the dissipation-bin
    # versions. Set <= 0 to disable.
    ctd_bin_sec: float = 0.25

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

    # VMP-style GPS: a vertical profiler tracked by a single ship/surface GPS
    # fix per profile (not a continuously-tracked position), so it gets one
    # lat/lon per profile instead of a per-window/bin interpolated position.
    # None = auto-detect from the instrument_vehicle attribute (see
    # _VMP_STYLE_VEHICLES); True/False overrides detection.
    vmp_style_gps: Optional[bool] = None

    # === Processing options ===
    chop_start: bool = True
    # === Despike parameters (see signal.despike) ===
    despike_max_passes: int = 6  # Max despike iterations (1 = ~4x faster)
    despike_thresh: float = (
        8.0  # Spike detection threshold (ratio of HP to LP envelope)
    )
    despike_smooth: float = 0.5  # Low-pass cutoff for the spike envelope (Hz)
    despike_replace_sec: float = 0.04  # Replacement window around each spike (seconds)
    # When True, embedded <probe>_clean / <probe>_despike_mask variables in the
    # input file are discarded and despiking is re-run with the params above.
    # CLI sets this only when the user explicitly passes --despike to `eps`.
    force_despike: bool = False

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

    def to_yaml(self) -> str:
        """Serialize this config to a human-readable YAML string.

        Tuple-typed fields (e.g. ``shear_probes``) round-trip as YAML lists.
        """
        return yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False)


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
    """Prepare raw p2nc output for epsilon processing.

    Performs only the slow-channel preprocessing:
      1. Low-pass filters the pressure data.
      2. Smooths the speed variable, or estimates it from pressure if absent.

    Probe signals (shear, gradT) are left in their raw calibrated units. The
    velocity normalisation that converts them to physical gradients is applied
    later in the spectral domain using the window-mean speed (see
    ``_compute_shear_spectra_with_cleaning``). Keeping the time series raw
    means despiking and high-pass filtering operate on stationary signals
    whose amplitude does not balloon near turnarounds.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from p2nc conversion containing:
        - ``P`` on ``t_slow``
        - Optionally ``W`` (speed) on ``t_slow``
        - Optionally ``Incl_Y`` (pitch) on ``t_slow``
        - ``sh1``, ``sh2``, ``gradT1``, ``gradT2`` on ``t_fast``
        - ``fs_slow``, ``fs_fast`` sampling rates as attributes or variables.
    config : ProfileConfig, optional
        Configuration for preprocessing. If None, uses defaults.

    Returns
    -------
    xr.Dataset
        Dataset with two added variables: ``{speed}_smooth`` and
        ``{pressure}_smooth``. Probe channels are unchanged.
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
        _log.warning(f"Peak-based profile detection failed: {e}")
        return []

    if not profiles:
        _log.info("Peak-based detection found no complete profiles.")
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

    _log.info(f"Peak-based detection found {len(segments)} profile segment(s)")

    return segments


def extract_profile_by_indices(
    ds: xr.Dataset, idx_start: int, idx_end: int
) -> xr.Dataset:
    """Slice ds to a single profile segment on t_slow/t_fast by t_slow index.

    ``idx_start``/``idx_end`` are inclusive indices into ``ds.t_slow``, as
    returned by :func:`find_all_profiles`.
    """
    t0 = ds.t_slow.values[idx_start]
    t1 = ds.t_slow.values[idx_end]
    return ds.sel(t_slow=slice(t0, t1), t_fast=slice(t0, t1))


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
        profile_ds = extract_profile_by_indices(ds, idx_start, idx_end)

        # Add profile metadata
        profile_ds.attrs["profile_index"] = i
        profile_ds.attrs["profile_start_idx"] = idx_start
        profile_ds.attrs["profile_end_idx"] = idx_end

        yield i, profile_ds


def compute_window_parameters(ds: xr.Dataset, config: ProfileConfig) -> dict:
    """Compute windowing parameters based on config and sampling rates."""
    fs_fast = float(ds.fs_fast)
    fs_slow = float(ds.fs_slow)

    n_fft = round(config.fft_len_sec * fs_fast)
    n_diss = round(config.diss_len_sec * fs_fast)

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
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute (epsilon, k_max, mad) for each shear probe spectrum."""
    results = {}

    for name, psd in spectra.items():
        if not name.startswith("sh"):
            continue

        n_windows = psd.shape[0]
        eps = np.full(n_windows, np.nan)
        k_max = np.full(n_windows, np.nan)
        mad = np.full(n_windows, np.nan)

        n_nan = int((~np.all(np.isfinite(psd), axis=1)).sum())
        if n_nan:
            _log.warning(f"{name}: {n_nan}/{n_windows} windows have a NaN spectrum")

        for i in range(n_windows):
            eps[i], k_max[i], mad[i] = estimate_epsilon(
                frequency,
                psd[i],
                W=speed[i],
                nu=nu[i],
                apply_single_pole_correction=False,
            )

        results[name] = (eps, k_max, mad)

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


def _drop_embedded_clean(ds: xr.Dataset, config: ProfileConfig) -> xr.Dataset:
    """Drop any ``<probe>_clean`` / ``<probe>_despike_mask`` variables.

    Used when ``config.force_despike`` is True so the subsequent
    ``despike_variables`` call recomputes from raw with the requested params.
    No-op (and no log) when nothing is embedded.
    """
    drop = [
        name
        for probe in config.all_probes
        for name in (f"{probe}_clean", despike_mask_name(probe))
        if name in ds
    ]
    if not drop:
        return ds
    _log.info(
        "Dropping embedded despike output (%s) to honour --despike re-clean request.",
        ", ".join(drop),
    )
    return ds.drop_vars(drop)


def _preprocess_for_spectra(
    ds: xr.Dataset, config: ProfileConfig
) -> tuple[xr.Dataset, dict]:
    """Despike, high-pass, segment, and window-align the profile.

    Returns the trimmed dataset and the window-parameter dict.
    """
    if config.speed_smooth not in ds:
        _log.debug("Smoothed speed not found, running prepare_profile")
        ds = prepare_profile(ds, config)

    if config.force_despike:
        ds = _drop_embedded_clean(ds, config)
    ds = despike_variables(
        ds,
        config.all_probes,
        fs=float(ds.fs_fast),
        max_passes=config.despike_max_passes,
        thresh=config.despike_thresh,
        smooth=config.despike_smooth,
        replace_sec=config.despike_replace_sec,
    )

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
    aux_mean: Callable[[np.ndarray], np.ndarray],
    config: ProfileConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Resolve window-mean T, S, density, and the temperature used for viscosity.

    Priority cascade:
      - temperature:  aux_temperature  > CT sensor (config.temperature) > default
      - salinity:     aux_salinity     > derived from JAC_C if valid     > default
      - density:      aux_density      > derived from JAC-derived S + T  > default
      - T for nu:     aux_temperature if present, else CT/default ``T_mean``

    ``aux_mean`` averages a raw aux_* array to the same resolution as ``means``.

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
        S_mean = aux_mean(ds["aux_salinity"].values)
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
        rho_mean = aux_mean(ds["aux_density"].values)
    elif salinity_from_jac:
        lon = (
            float(np.nanmean(ds["aux_longitude"].values))
            if "aux_longitude" in ds
            else config.default_longitude
        )
        lat = (
            float(np.nanmean(ds["aux_latitude"].values))
            if "aux_latitude" in ds
            else config.default_latitude
        )
        P_dbar = means.get(pressure_var, np.full(n_windows, 0.0))
        T_insitu = means[config.temperature]
        SA = gsw.SA_from_SP(S_mean, P_dbar, lon, lat)
        CT = gsw.CT_from_t(SA, T_insitu, P_dbar)
        rho_mean = gsw.rho(SA, CT, P_dbar)
    else:
        rho_mean = np.full(n_windows, config.default_density)

    if "aux_temperature" in ds:
        T_visc = aux_mean(ds["aux_temperature"].values)
    else:
        T_visc = T_mean

    return T_mean, S_mean, rho_mean, T_visc, salinity_from_jac


def _apply_conductivity_matching(ds: xr.Dataset, config: ProfileConfig) -> xr.Dataset:
    """Lag/low-pass match JAC_C to temperature, on the raw signal before window-averaging."""
    if not config.match_conductivity:
        return ds
    if "JAC_C" not in ds or config.temperature not in ds:
        return ds

    speed = float(np.nanmean(np.abs(ds[config.speed_smooth].values)))
    matched = match_conductivity_to_temperature(
        ds["JAC_C"].values,
        float(ds.fs_slow),
        speed,
        lag=config.jac_lag,
        f_tc=config.jac_f_tc,
        reference_speed=config.jac_reference_speed,
    )
    ds = ds.copy()
    ds["JAC_C"] = ("t_slow", matched.astype(ds["JAC_C"].values.dtype))
    return ds


def _first_valid(x: np.ndarray) -> float:
    """First finite value in x; falls back to x[0] if none are finite."""
    finite = np.flatnonzero(np.isfinite(x))
    idx = int(finite[0]) if finite.size else 0
    return float(x[idx])


# Vehicle types (from the p-file's [instrument_info] vehicle field, stored as
# ds.attrs["instrument_vehicle"]) tracked by a single ship/surface GPS fix
# per cast (VMP-style), rather than a continuously-tracked position (e.g. a
# glider's own navigation). Matches ODAS's own vmp/rvmp/xmp grouping in
# default_vehicle_attributes.ini.
_VMP_STYLE_VEHICLES = frozenset({"vmp", "rvmp", "xmp"})


def _is_vmp_style_gps(ds: xr.Dataset, config: ProfileConfig) -> bool:
    """Whether to use one lat/lon per profile instead of per-window/bin."""
    if config.vmp_style_gps is not None:
        return config.vmp_style_gps
    vehicle = str(ds.attrs.get("instrument_vehicle", "")).strip().lower()
    return vehicle in _VMP_STYLE_VEHICLES


def _build_ctd_vars(
    ds: xr.Dataset,
    means: dict,
    aux_mean: Callable[[np.ndarray], np.ndarray],
    config: ProfileConfig,
    include_kinematics: bool = True,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Compute pressure, z, temperature, salinity, density, conductivity,
    and (if config.compute_thermo) absolute_salinity, conservative_temperature,
    potential_density from window means. Also computes W and nu when
    ``include_kinematics`` is True.

    With VMP-style GPS (see :func:`_is_vmp_style_gps`), lat/lon are used
    internally (for z and the thermo calc) but not included in the returned
    dict -- they're attached once, as scalars, by
    :func:`_attach_scalar_position`. With a continuously-tracked position
    (e.g. a glider), "lat"/"lon" are included in the returned dict like any
    other per-window variable.

    ``means`` and ``aux_mean`` must be at the same time resolution. Returns
    ``{var_name: (array, attrs)}``; salinity/density/lat/lon are only
    included when derived from a real source.
    """
    pressure_var = config.pressure_smooth
    speed_var = config.speed_smooth
    n_out = len(means["t_slow"])

    out: dict[str, tuple[np.ndarray, dict]] = {}
    out["pressure"] = (means.get(pressure_var, np.full(n_out, np.nan)), {})
    if include_kinematics:
        out["W"] = (means.get(speed_var, np.full(n_out, np.nan)), {})

    T_mean, S_mean, rho_mean, T_visc, salinity_from_jac = _derive_thermo(
        ds, means, aux_mean, config
    )

    if "aux_salinity" in ds or salinity_from_jac:
        out["salinity"] = (S_mean, {})
    if "aux_density" in ds or salinity_from_jac:
        out["density"] = (rho_mean, {})

    out["temperature"] = (T_visc if "aux_temperature" in ds else T_mean, {})

    if include_kinematics:
        nu, _ = viscosity(S_mean, T_visc, rho_mean)
        out["nu"] = (nu, {})
        kappa_T = thermal_diffusivity(S_mean, T_visc, rho_mean, out["pressure"][0])
        out["kappa_T"] = (
            kappa_T,
            {
                "long_name": "Molecular thermal diffusivity",
                "units": "m2 s-1",
                "comment": "Caldwell (1974) conductivity / (rho * cp0)",
            },
        )

    vmp_style_gps = _is_vmp_style_gps(ds, config)
    lat_arr = lon_arr = None
    if "aux_latitude" in ds:
        if vmp_style_gps:
            lat_arr = np.full(n_out, _first_valid(ds["aux_latitude"].values))
        else:
            lat_arr = aux_mean(ds["aux_latitude"].values)
            out["lat"] = (lat_arr, {})
    if "aux_longitude" in ds:
        if vmp_style_gps:
            lon_arr = np.full(n_out, _first_valid(ds["aux_longitude"].values))
        else:
            lon_arr = aux_mean(ds["aux_longitude"].values)
            out["lon"] = (lon_arr, {})
    if "JAC_C" in means:
        out["conductivity"] = (means["JAC_C"], {})
    if "T1" in means:
        out["T1"] = (
            means["T1"],
            {
                "long_name": "FP07 thermistor 1 temperature",
                "standard_name": "sea_water_temperature",
                "units": "degree_C",
            },
        )
    if "T2" in means:
        out["T2"] = (
            means["T2"],
            {
                "long_name": "FP07 thermistor 2 temperature",
                "standard_name": "sea_water_temperature",
                "units": "degree_C",
            },
        )

    lat_for_gsw = (
        lat_arr if lat_arr is not None else np.full(n_out, config.default_latitude)
    )
    lon_for_gsw = (
        lon_arr if lon_arr is not None else np.full(n_out, config.default_longitude)
    )

    z = gsw.z_from_p(out["pressure"][0], lat_for_gsw)
    out["z"] = (
        z,
        {
            "long_name": "Height (negative below sea surface)",
            "standard_name": "height",
            "units": "m",
            "comment": "gsw.z_from_p(pressure, lat)",
        },
    )

    has_real_temperature = config.temperature in means or "aux_temperature" in ds
    has_real_salinity = "salinity" in out
    if config.compute_thermo and has_real_temperature and has_real_salinity:
        P_dbar = out["pressure"][0]
        SP = out["salinity"][0]
        T_insitu = out["temperature"][0]

        SA = gsw.SA_from_SP(SP, P_dbar, lon_for_gsw, lat_for_gsw)
        CT = gsw.CT_from_t(SA, T_insitu, P_dbar)
        potential_density = gsw.sigma0(SA, CT) + 1000.0

        out["absolute_salinity"] = (
            SA,
            {
                "long_name": "Absolute Salinity",
                "standard_name": "sea_water_absolute_salinity",
                "units": "g kg-1",
            },
        )
        out["conservative_temperature"] = (
            CT,
            {
                "long_name": "Conservative Temperature",
                "standard_name": "sea_water_conservative_temperature",
                "units": "degC",
            },
        )
        out["potential_density"] = (
            potential_density,
            {
                "long_name": "Potential density referenced to 0 dbar",
                "standard_name": "sea_water_potential_density",
                "units": "kg m-3",
                "comment": "gsw.sigma0(SA, CT) + 1000",
            },
        )

    return out


def _attach_scalar_position(ds: xr.Dataset, config: ProfileConfig) -> xr.Dataset:
    """Attach a single scalar lat/lon (no dimension) for VMP-style GPS.

    No-op with a continuously-tracked position (e.g. a glider), where
    lat/lon already vary per window/bin and are attached by
    :func:`_build_ctd_vars` instead.
    """
    if not _is_vmp_style_gps(ds, config):
        return ds
    if "aux_latitude" in ds:
        ds["lat"] = float(_first_valid(ds["aux_latitude"].values))
        ds["lat"].attrs = {"long_name": "Latitude", "units": "degree_north"}
    if "aux_longitude" in ds:
        ds["lon"] = float(_first_valid(ds["aux_longitude"].values))
        ds["lon"].attrs = {"long_name": "Longitude", "units": "degree_east"}
    return ds


def _attach_buoyancy_frequency(
    ds: xr.Dataset, config: ProfileConfig, suffix: str = ""
) -> xr.Dataset:
    """Attach N2 (buoyancy frequency squared) from the bin-averaged
    absolute_salinity/conservative_temperature/pressure at one resolution.

    ``suffix=""`` computes ``N2`` from the dissipation-window means (on
    ``time``); ``suffix="_hires"`` computes ``N2_hires`` from the CTD hires
    bins (on ``ctd_time``, see :func:`_attach_hires_ctd_vars`). gsw.Nsquared
    returns values at the midpoints between adjacent bins; computing it from
    the already bin-averaged (not raw) profile reduces noise. The result is
    then interpolated from that mid-pressure grid back onto each bin's own
    pressure value. No-op unless config.compute_thermo produced
    absolute_salinity/conservative_temperature at this resolution.
    """
    dim = "ctd_time" if suffix else "time"
    sa_name, ct_name, p_name = (
        f"{v}{suffix}"
        for v in ("absolute_salinity", "conservative_temperature", "pressure")
    )
    if not all(v in ds for v in (sa_name, ct_name, p_name)):
        return ds

    SA = np.asarray(ds[sa_name].values, dtype=float)
    CT = np.asarray(ds[ct_name].values, dtype=float)
    P = np.asarray(ds[p_name].values, dtype=float)
    valid = np.isfinite(SA) & np.isfinite(CT) & np.isfinite(P)
    if valid.sum() < 2:
        return ds

    lat_name = f"lat{suffix}"
    if lat_name in ds:
        lat_val = ds[lat_name].values
        lat_arg = float(lat_val) if lat_val.ndim == 0 else np.asarray(lat_val)[valid]
    elif "lat" in ds and ds["lat"].ndim == 0:
        lat_arg = float(ds["lat"].values)
    else:
        lat_arg = config.default_latitude

    N2_mid, P_mid = gsw.Nsquared(SA[valid], CT[valid], P[valid], lat=lat_arg)

    order = np.argsort(P_mid)
    N2 = np.full(len(P), np.nan)
    N2[valid] = np.interp(
        P[valid], P_mid[order], N2_mid[order], left=np.nan, right=np.nan
    )

    source = "CTD hires bins" if suffix else "dissipation-window means"
    ds[f"N2{suffix}"] = (dim, N2.astype("f4"))
    ds[f"N2{suffix}"].attrs = {
        "long_name": "Buoyancy frequency squared",
        "standard_name": "square_of_brunt_vaisala_frequency_in_sea_water",
        "units": "s-2",
        "comment": (
            "gsw.Nsquared(absolute_salinity, conservative_temperature, "
            f"pressure) from the {source}, interpolated from gsw's "
            "mid-pressure grid back onto pressure."
        ),
    }
    return ds


def _attach_hires_ctd_vars(ds: xr.Dataset, config: ProfileConfig) -> xr.Dataset:
    """Attach CTD scalars on a finer ``ctd_time`` axis (suffix ``_hires``).

    Excludes W and nu, which are only meaningful at dissipation-window
    resolution. Bin width is ``config.ctd_bin_sec``, independent of the
    FFT/dissipation window. No-op if disabled, ``fs_slow`` is unavailable, or
    there isn't a full bin's worth of data.
    """
    if config.ctd_bin_sec <= 0 or not hasattr(ds, "fs_slow"):
        return ds
    if config.temperature not in ds and "JAC_C" not in ds:
        return ds

    n_ctd = max(1, round(config.ctd_bin_sec * float(ds.fs_slow)))
    if ds.sizes["t_slow"] < n_ctd:
        return ds

    vars_to_mean = [
        "t_slow",
        config.pressure_smooth,
        config.speed_smooth,
        config.temperature,
        "JAC_C",
        "T1",
        "T2",
    ]
    means_ctd = {v: block_mean(ds[v].values, n_ctd) for v in vars_to_mean if v in ds}
    if len(means_ctd.get("t_slow", [])) == 0:
        return ds

    ctd_vars = _build_ctd_vars(
        ds,
        means_ctd,
        lambda x: block_mean(x, n_ctd),
        config,
        include_kinematics=False,
    )

    ds = ds.assign_coords(ctd_time=("ctd_time", means_ctd["t_slow"]))
    if "units" in ds.t_slow.attrs:
        ds.ctd_time.attrs["units"] = ds.t_slow.attrs["units"]
    ds.ctd_time.attrs["long_name"] = "Time (CTD bins)"

    for name, (arr, attrs) in ctd_vars.items():
        out_name = f"{name}_hires"
        ds[out_name] = ("ctd_time", arr.astype("f4"))
        if attrs:
            ds[out_name].attrs = attrs

    return ds


def _attach_window_scalars(
    ds: xr.Dataset,
    params: dict,
    config: ProfileConfig,
    range_masks: Optional[dict] = None,
    fit_confident: Optional[dict] = None,
) -> xr.Dataset:
    """Compute window-mean scalars and attach them on the output ``time`` axis.

    Adds: ``time`` coord, ``pressure``, ``z``, ``W``, ``temperature``, ``nu``;
    plus ``salinity``, ``density``, ``conductivity``, ``T1``, ``T2``, and
    (with a continuously-tracked position, e.g. a glider) ``lat``/``lon``
    when the relevant inputs are available; plus ``absolute_salinity``,
    ``conservative_temperature``, ``potential_density``, ``N2`` when
    ``config.compute_thermo`` is set and real temperature/salinity exist.
    With VMP-style GPS, a single scalar ``lat``/``lon`` is attached instead
    (see :func:`_attach_scalar_position`). Also attaches ``_hires`` versions
    of the CTD scalars (including ``N2_hires``) on a finer ``ctd_time`` axis
    (see :func:`_attach_hires_ctd_vars`).
    """
    pressure_var = config.pressure_smooth
    speed_var = config.speed_smooth

    ds = _apply_conductivity_matching(ds, config)

    means = compute_window_means(
        ds,
        ["t_slow", pressure_var, speed_var, config.temperature, "JAC_C", "T1", "T2"],
        params,
    )

    ds = ds.assign_coords(time=("time", means["t_slow"]))
    if "units" in ds.t_slow.attrs:
        ds.time.attrs["units"] = ds.t_slow.attrs["units"]
    if "long_name" in ds.t_slow.attrs:
        ds.time.attrs["long_name"] = "Time (dissipation windows)"

    # Must run before the coarse loop below, which overwrites "T1"/"T2" with
    # their window-mean values -- this reads them at full resolution first.
    ds = _attach_hires_ctd_vars(ds, config)

    ctd_vars = _build_ctd_vars(
        ds, means, lambda x: _window_mean_slow(x, params), config
    )
    for name, (arr, attrs) in ctd_vars.items():
        ds[name] = ("time", arr.astype("f4"))
        if attrs:
            ds[name].attrs = attrs

    ds = _attach_scalar_position(ds, config)
    ds = _attach_buoyancy_frequency(ds, config)
    ds = _attach_buoyancy_frequency(ds, config, suffix="_hires")

    n_fft = params["n_fft"]
    n_diss = params["n_diss"]
    for probe in config.all_probes:
        mask_name = despike_mask_name(probe)
        if mask_name not in ds:
            continue
        frac = window_mean(ds[mask_name].values.astype("f4"), n_fft, n_diss)
        out_name = f"{probe}_despike_frac"
        ds[out_name] = ("time", frac.astype("f4"))
        ds[out_name].attrs = {
            "long_name": f"Fraction of {probe} samples modified by despiking",
            "units": "1",
            "valid_min": np.float32(0.0),
            "valid_max": np.float32(1.0),
        }
        ds = ds.drop_vars(mask_name)

    for probe, mask_slow in (range_masks or {}).items():
        if probe not in ds:
            continue
        frac = _window_mean_slow(mask_slow.astype("f4"), params)
        ds[f"{probe}_range_frac"] = ("time", frac.astype("f4"))
        ds[f"{probe}_range_frac"].attrs = {
            "long_name": f"Fraction of {probe} raw samples outside "
            f"[{_MIN_SANE_TEMP_C}, {_MAX_SANE_TEMP_C}] C",
            "units": "1",
            "valid_min": np.float32(0.0),
            "valid_max": np.float32(1.0),
        }
        confident = (fit_confident or {}).get(probe)
        qc_var = f"{probe}_qc"
        ds[qc_var] = ("time", _compose_range_qc(frac, config, confident))
        ds[qc_var].attrs = {
            "long_name": f"QC flag for {probe}",
            "flag_values": _QC_FLAG_VALUES,
            "flag_meanings": _QC_FLAG_MEANINGS,
            "valid_min": np.int8(0),
            "valid_max": np.int8(9),
            "comment": (
                f"Composed from {probe}_range_frac -- the fraction of raw "
                f"{probe} samples outside a physically sane seawater "
                "temperature range in this window, e.g. from a calibration "
                "extrapolated beyond its fitted range "
                f"(questionable>{config.despike_frac_questionable}, "
                f"bad>{config.despike_frac_bad}) -- and floored to bad if "
                f"the calibration fit itself wasn't confident "
                f"({probe}_fp07_confident=0; see fp07_calibration."
                "fit_is_confident)."
            ),
        }

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
                freq_clean, clean_psd = clean_spec(
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

    # Convert spectra so that they represent shear variance [s-2/Hz]
    W = ds["W"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_W2 = 1 / W**2
        inv_W4 = inv_W2 * inv_W2
        k = freq[None, :] / W[:, None]
        # Shear probe spatial-averaging + anti-alias single-pole correction
        # (Rockland TN026), and the FP07 thermistor's single-pole frequency
        # response correction (Lueck) -- both applied here (once, per
        # window) rather than inside estimate_epsilon/estimate_chi, so the
        # saved S_sh1/S_sh2/S_gradT1/S_gradT2 are the corrected spectra
        # actually used downstream, not raw ones a reader would otherwise
        # have to correct themselves before comparing to a model spectrum.
        shear_corr = shear_response_correction(k)
        gradT_corr = gradT_response_correction(
            freq[None, :], W[:, None], config.fp07_tau0, config.fp07_speed_exp
        )
    for name in list(spectra):
        if name in config.shear_probes:
            spectra[name] = spectra[name] * inv_W4[:, None] * shear_corr
        elif name in config.temperature_probes:
            spectra[name] = spectra[name] * inv_W2[:, None] * gradT_corr

    ds = ds.assign_coords(frequency=("frequency", freq))
    for name, psd in spectra.items():
        ds[f"S_{name}"] = (("time", "frequency"), psd.astype("f4"))
        if name in config.temperature_probes:
            ds[f"S_{name}"].attrs = {
                "long_name": f"Power spectral density of {name}",
                "units": "K2 m-2 Hz-1",
                "comment": (
                    "Corrected for the FP07 thermistor's single-pole frequency "
                    "response (Lueck), tau = fp07_tau0 * W^fp07_speed_exp "
                    f"(fp07_tau0={config.fp07_tau0}, "
                    f"fp07_speed_exp={config.fp07_speed_exp}); W is the "
                    "per-window mean fall speed."
                ),
            }
        elif name in config.shear_probes:
            ds[f"S_{name}"].attrs = {
                "long_name": f"Power spectral density of {name}",
                "units": "s-2 Hz-1",
                "comment": (
                    "Corrected for the shear probe's spatial-averaging and "
                    "anti-alias response with a single-pole transfer function "
                    "(Macoun & Lueck; Rockland Technical Note 026): "
                    "H^-2 = 1 + (k/48)^2 for k <= 150 cpm, else 1, with "
                    "k = frequency / W (W = per-window mean fall speed)."
                ),
            }
    return ds, freq, spectra


_QC_FLAG_VALUES = np.array([0, 1, 2, 4, 9], dtype="i1")
_QC_FLAG_MEANINGS = "unknown good questionable bad missing"
_QC_MISSING = np.int8(9)
_EPS_AGREEMENT_FACTOR = 10.0

# Broad, globally-safe bounds on seawater temperature. A T1/T2 value outside
# this range usually indicates a calibration extrapolated beyond its fitted
# range (e.g. applied to an anomalous profile) rather than real data.
# calibrate-fp07's apply_probe_calibration applies the best available fit
# as-is and does not mask this -- flagged via QC here instead, at the eps
# step, per policy: don't destroy data, mark it untrustworthy and let the
# consumer decide.
_MIN_SANE_TEMP_C = -3.0
_MAX_SANE_TEMP_C = 40.0


def _temperature_range_mask(T: np.ndarray) -> np.ndarray:
    """True where T is missing or outside the physically sane seawater range."""
    return ~np.isfinite(T) | (T < _MIN_SANE_TEMP_C) | (T > _MAX_SANE_TEMP_C)


def _compose_range_qc(
    range_frac: np.ndarray,
    config: ProfileConfig,
    fit_confident: Optional[bool] = None,
) -> np.ndarray:
    """QC flag from the fraction of a temperature probe's raw samples that
    fell outside the physically sane range within a window, floored to
    "bad" if the calibration fit itself wasn't confident.

      * fit_confident is False                 -> 4 (bad), regardless of
        range_frac -- a fit that isn't confident (see fp07_calibration.
        fit_is_confident) can produce values that look physically plausible
        while still being substantially wrong; no per-sample range check
        can catch that, so it's flagged from the fit's own quality instead.
        fit_confident is None (never run through calibrate-fp07, or an
        older file predating this attr) applies no such floor.
      * range_frac > despike_frac_bad          -> 4 (bad)
      * range_frac > despike_frac_questionable -> 2 (questionable)
      * range_frac is NaN (no raw samples)     -> 9 (missing)
      * otherwise                              -> 1 (good)
    """
    qc = np.ones(range_frac.shape, dtype="i1")
    qc[range_frac > config.despike_frac_questionable] = 2
    qc[range_frac > config.despike_frac_bad] = 4
    qc[np.isnan(range_frac)] = 9
    if fit_confident is False:
        qc[qc != 9] = 4
    return qc


def _combine_eps_pair(
    eps1: np.ndarray, eps2: np.ndarray, qc1: np.ndarray, qc2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two probe estimates into (eps, eps_qc).

    eps:
      * both finite & within factor of ``_EPS_AGREEMENT_FACTOR`` → mean
      * both finite, disagreement larger → element-wise minimum
      * exactly one finite → that value
      * neither finite → NaN

    eps_qc:
      * both eps finite → max(qc1, qc2)
      * exactly one finite → the surviving probe's qc
      * neither finite → 9 (missing)
    """
    e1_ok = np.isfinite(eps1)
    e2_ok = np.isfinite(eps2)
    both = e1_ok & e2_ok

    hi = np.fmax(eps1, eps2)
    lo = np.fmin(eps1, eps2)
    within = both & (hi <= _EPS_AGREEMENT_FACTOR * lo)

    eps = np.where(
        within,
        0.5 * (eps1 + eps2),
        np.where(both, lo, np.where(e1_ok, eps1, np.where(e2_ok, eps2, np.nan))),
    )

    eps_qc = np.full(eps1.shape, _QC_MISSING, dtype="i1")
    only1 = e1_ok & ~e2_ok
    only2 = e2_ok & ~e1_ok
    eps_qc[only1] = qc1[only1]
    eps_qc[only2] = qc2[only2]
    eps_qc[both] = np.maximum(qc1[both], qc2[both])
    return eps.astype("f4"), eps_qc


def _dof_spec(params: dict) -> float:
    """Spectral degrees of freedom per Nuttall (1971) for 50%% FFT overlap."""
    n_fft = params["n_fft"]
    n_diss = params["n_diss"]
    num_of_ffts = 2 * (n_diss // n_fft) - 1
    return 1.9 * num_of_ffts


def _compose_qc(
    eps: np.ndarray,
    fm: np.ndarray,
    speed_bad: np.ndarray,
    despike_frac: np.ndarray,
    config: ProfileConfig,
) -> np.ndarray:
    """Combine speed, FM, and despike-fraction contributions into a per-window flag.

    Precedence (max wins across the three contributions, then NaN-eps overrides as 9):
      * FM <= fm_good                                           -> 1 (good)
      * fm_good < FM <= fm_bad                                  -> 2 (questionable)
      * FM > fm_bad                                             -> 4 (bad)
      * speed below min_speed                                   -> 2 (questionable)
      * despike_frac > despike_frac_questionable                -> 2 (questionable)
      * despike_frac > despike_frac_bad                         -> 4 (bad)
      * eps NaN                                                 -> 9 (missing, overrides)
    """
    qc_speed = np.zeros(eps.size, dtype="i1")
    qc_speed[speed_bad] = 2

    qc_fm = np.zeros(eps.size, dtype="i1")
    # Comparisons against NaN are False, so FM=NaN leaves qc_fm at 0 (unknown).
    qc_fm[fm <= config.fm_good] = 1
    qc_fm[(fm > config.fm_good) & (fm <= config.fm_bad)] = 2
    qc_fm[fm > config.fm_bad] = 4

    # Despike fraction: 0 (or absent → zeros from caller) leaves qc_dsp at 0.
    qc_dsp = np.zeros(eps.size, dtype="i1")
    qc_dsp[despike_frac > config.despike_frac_questionable] = 2
    qc_dsp[despike_frac > config.despike_frac_bad] = 4

    qc = np.maximum(np.maximum(qc_speed, qc_fm), qc_dsp)
    qc[np.isnan(eps)] = 9
    return qc


def _attach_epsilon(
    ds: xr.Dataset,
    freq: np.ndarray,
    spectra: dict[str, np.ndarray],
    config: ProfileConfig,
    params: dict,
) -> xr.Dataset:
    """Attach per-probe epsilon, k_max, FM, wavenumber ``k``, and QC flags.

    QC convention (IODE): 0=unknown, 1=good, 2=questionable, 4=bad, 9=missing.
    Three contributions are folded together via max (see ``_compose_qc``):

      * Window-mean speed below ``config.min_speed`` raises the flag to 2.
      * FM = mad * sqrt(dof_spec) is the spectrum-vs-Nasmyth fit residual
        (low = trustworthy). It promotes flags to 1/2/4 against the two
        ``config.fm_good`` / ``config.fm_bad`` thresholds.
      * Per-probe window despike fraction (``sh1_despike_frac`` for eps_1,
        ``sh2_despike_frac`` for eps_2) promotes flags to 2/4 against the
        two ``config.despike_frac_questionable`` / ``config.despike_frac_bad``
        thresholds. Treated as zero if the despike-fraction variable is
        absent (e.g., probe was never despiked).
    """
    epsilon_results = compute_epsilon(freq, spectra, ds["W"].values, ds["nu"].values)
    sqrt_dof = float(np.sqrt(_dof_spec(params)))
    W = ds["W"].values
    speed_bad = W < config.min_speed

    for name, (eps, k_max, mad) in epsilon_results.items():
        probe_num = name[-1]
        fm = mad * sqrt_dof
        ds[f"eps_{probe_num}"] = ("time", eps.astype("f4"))
        ds[f"k_max_{probe_num}"] = ("time", k_max.astype("f4"))
        ds[f"eps_{probe_num}_fm"] = ("time", fm.astype("f4"))
        ds[f"eps_{probe_num}_fm"].attrs = {
            "long_name": f"Figure of merit for eps_{probe_num} Nasmyth fit",
            "units": "1",
            "comment": (
                "FM = mean(|log10(P_sh / Nasmyth)|) * sqrt(dof_spec) over the "
                "fit wavenumber band. Lower is better; independent of "
                "fft/diss window length. NaN if the fit band was too narrow."
            ),
        }
        # The despike-fraction contribution: sh1 drives eps_1's flag, sh2
        # drives eps_2's. Absent variable -> zeros (no QC penalty).
        despike_frac_var = f"{name}_despike_frac"
        if despike_frac_var in ds:
            despike_frac = ds[despike_frac_var].values
        else:
            despike_frac = np.zeros(eps.size, dtype="f4")
        qc = _compose_qc(eps, fm, speed_bad, despike_frac, config)
        qc_var = f"eps_{probe_num}_qc"
        ds[qc_var] = ("time", qc)
        ds[qc_var].attrs = {
            "long_name": f"QC flag for eps_{probe_num}",
            "flag_values": _QC_FLAG_VALUES,
            "flag_meanings": _QC_FLAG_MEANINGS,
            "valid_min": np.int8(0),
            "valid_max": np.int8(9),
            "comment": (
                f"Composed from speed (min_speed={config.min_speed} m/s), "
                f"FM (fm_good={config.fm_good}, fm_bad={config.fm_bad}), "
                f"and {name}_despike_frac "
                f"(questionable>{config.despike_frac_questionable}, "
                f"bad>{config.despike_frac_bad})."
            ),
        }

    ds["k"] = ds.frequency / ds.W
    return ds


def _best_window_epsilon(ds: xr.Dataset) -> Optional[np.ndarray]:
    """Per-window best epsilon combined from eps_1/eps_2 via _combine_eps_pair.

    Falls back to the single available probe; None if neither exists.
    """
    have1 = "eps_1" in ds
    have2 = "eps_2" in ds
    if have1 and have2:
        eps, _ = _combine_eps_pair(
            ds["eps_1"].values,
            ds["eps_2"].values,
            ds["eps_1_qc"].values.astype("i1"),
            ds["eps_2_qc"].values.astype("i1"),
        )
        return eps
    if have1:
        return ds["eps_1"].values
    if have2:
        return ds["eps_2"].values
    return None


def _attach_chi(
    ds: xr.Dataset,
    freq: np.ndarray,
    spectra: dict[str, np.ndarray],
    config: ProfileConfig,
    params: dict,
    cal_params: dict,
    fit_confident: Optional[dict] = None,
) -> xr.Dataset:
    """Attach per-probe chi, chi_k_max, FM, and QC flags on the ``time`` dim.

    chi is estimated from each temperature gradient spectrum with epsilon
    taken as the per-window combined shear-probe estimate (see
    :func:`_best_window_epsilon` and :func:`pyturb.temperature.estimate_chi`).
    QC composition mirrors epsilon: speed, FM vs the Kraichnan model, and the
    matching gradT despike fraction.

    ``cal_params`` (from :func:`process_profile`, keyed by probe e.g.
    ``"T1"``) lets each window's predicted electronics noise floor (see
    :func:`pyturb.noise.thermistor_noise_phi`) cap chi's k_max in addition
    to the spectral-minimum search -- skipped (falls back to the
    spectral-minimum search alone) for a probe with no calibration attrs.
    """
    if not config.compute_chi:
        return ds

    eps_best = _best_window_epsilon(ds)
    if eps_best is None:
        _log.warning("compute_chi requested but no epsilon estimates available")
        return ds

    W = ds["W"].values
    nu = ds["nu"].values
    kappa_T = ds["kappa_T"].values
    n = W.size
    fs_fast = float(ds.fs_fast)

    sqrt_dof = float(np.sqrt(_dof_spec(params)))
    speed_bad = W < config.min_speed

    for name in config.temperature_probes:
        if name not in spectra:
            continue
        probe = name.removeprefix("grad")
        cal = cal_params.get(probe)
        T_probe = ds[probe].values if cal and probe in ds else None
        psd = spectra[name]
        chi = np.full(n, np.nan)
        k_max = np.full(n, np.nan)
        mad = np.full(n, np.nan)
        for i in range(n):
            phi_noise = None
            if T_probe is not None and np.isfinite(T_probe[i]) and W[i] > 0:
                phi_noise = thermistor_noise_phi(freq, W[i], T_probe[i], cal, fs_fast)
            chi[i], k_max[i], mad[i] = estimate_chi(
                freq,
                psd[i],
                W=W[i],
                eps=eps_best[i],
                nu=nu[i],
                kappa_T=kappa_T[i],
                phi_noise=phi_noise,
            )

        probe_num = name[-1]
        fm = mad * sqrt_dof
        ds[f"chi_{probe_num}"] = ("time", chi.astype("f4"))
        ds[f"chi_{probe_num}"].attrs = {
            "long_name": f"Temperature variance dissipation rate from {name}",
            "units": "K2 s-1",
            "comment": (
                "Integrated response-corrected temperature gradient spectrum, "
                "corrected for unresolved variance with the Kraichnan "
                "spectrum using the combined shear-probe epsilon."
            ),
        }
        ds[f"chi_k_max_{probe_num}"] = ("time", k_max.astype("f4"))
        ds[f"chi_{probe_num}_fm"] = ("time", fm.astype("f4"))
        ds[f"chi_{probe_num}_fm"].attrs = {
            "long_name": f"Figure of merit for chi_{probe_num} Kraichnan fit",
            "units": "1",
            "comment": (
                "FM = mean(|log10(P_gradT / Kraichnan)|) * sqrt(dof_spec) "
                "over the integration band. Lower is better."
            ),
        }
        despike_frac_var = f"{name}_despike_frac"
        if despike_frac_var in ds:
            despike_frac = ds[despike_frac_var].values
        else:
            despike_frac = np.zeros(n, dtype="f4")
        # Fold in the probe's own out-of-range fraction (see
        # _attach_window_scalars/_compose_range_qc) -- a calibration
        # extrapolated beyond its fitted range corrupts gradT the same way
        # a despiked-out transient does, so it's treated the same way here.
        range_frac_var = f"{probe}_range_frac"
        if range_frac_var in ds:
            range_frac = np.nan_to_num(ds[range_frac_var].values, nan=0.0)
        else:
            range_frac = np.zeros(n, dtype="f4")
        bad_frac = np.clip(despike_frac + range_frac, 0.0, 1.0)
        # Floor to bad if the calibration fit itself wasn't confident (see
        # _compose_range_qc's docstring) -- a bad-but-not-implausible fit
        # corrupts the gradient the same way, and no per-sample check can
        # catch it either.
        if (fit_confident or {}).get(probe) is False:
            bad_frac = np.ones_like(bad_frac)
        qc = _compose_qc(chi, fm, speed_bad, bad_frac, config)
        qc_var = f"chi_{probe_num}_qc"
        ds[qc_var] = ("time", qc)
        ds[qc_var].attrs = {
            "long_name": f"QC flag for chi_{probe_num}",
            "flag_values": _QC_FLAG_VALUES,
            "flag_meanings": _QC_FLAG_MEANINGS,
            "valid_min": np.int8(0),
            "valid_max": np.int8(9),
            "comment": (
                f"Composed from speed (min_speed={config.min_speed} m/s), "
                f"FM (fm_good={config.fm_good}, fm_bad={config.fm_bad}), "
                f"{name}_despike_frac + {probe}_range_frac "
                f"(questionable>{config.despike_frac_questionable}, "
                f"bad>{config.despike_frac_bad}), and floored to bad if the "
                f"calibration fit itself wasn't confident "
                f"({probe}_fp07_confident=0)."
            ),
        }

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

    # Grabbed before _attach_window_scalars overwrites T1/T2 with their
    # window-mean values (dropping the cal_* attrs and the raw samples
    # range_masks needs) -- see _attach_chi and _attach_window_scalars.
    cal_params = {
        probe: p
        for probe in ("T1", "T2")
        if (p := _channel_calibration_params(ds, probe))
    }
    range_masks = {
        probe: _temperature_range_mask(ds[probe].values)
        for probe in ("T1", "T2")
        if probe in ds
    }
    # None (key absent) means "never run through calibrate-fp07" -- no
    # penalty, unlike an explicit False (see _compose_range_qc).
    fit_confident = {
        probe: bool(ds[probe].attrs[f"{probe}_fp07_confident"])
        for probe in ("T1", "T2")
        if probe in ds and f"{probe}_fp07_confident" in ds[probe].attrs
    }

    ds, params = _preprocess_for_spectra(ds, config)
    ds = _attach_window_scalars(ds, params, config, range_masks, fit_confident)
    ds, freq, spectra = _compute_shear_spectra_with_cleaning(ds, params, config)
    ds = _attach_epsilon(ds, freq, spectra, config, params)
    return _attach_chi(ds, freq, spectra, config, params, cal_params, fit_confident)

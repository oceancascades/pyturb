"""Profile processing for microstructure data."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as sig
import xarray as xr
from profinder import find_segment

from .shear import estimate_epsilon
from .signal import despike, window_mean, window_psd
from .viscosity import viscosity

logger = logging.getLogger(__name__)


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
    dbar_to_m: float = 1.0197  # Conversion from dbar to meters

    # === Probe names ===
    shear_probes: tuple[str, ...] = ("sh1", "sh2")
    temperature_probes: tuple[str, ...] = ("gradT1", "gradT2")

    # === Preprocessing parameters ===
    smoothing_period: float = 0.25  # Cutoff period for low-pass filter (seconds)
    speed_smoothing_period: float = (
        2.0  # Cutoff period for estimated speed smoothing (seconds)
    )
    filter_order: int = 4

    # === Thresholds ===
    min_speed: float = 0.2  # Minimum speed for valid profile segment

    # === Default values for missing data ===
    default_temperature: float = 10.0
    default_salinity: float = 35.0
    default_density: float = 1025.0

    # === Auxiliary dataset variable names ===
    aux_time: str = "time"  # Time variable in auxiliary dataset
    aux_latitude: str = "lat"  # Latitude variable in auxiliary dataset
    aux_longitude: str = "lon"  # Longitude variable in auxiliary dataset
    aux_temperature: str = "temperature"  # Temperature variable in auxiliary dataset
    aux_salinity: str = "salinity"  # Salinity variable in auxiliary dataset
    aux_density: str = "density"  # Density variable in auxiliary dataset

    # === Processing options ===
    chop_start: bool = True
    verbose: bool = False
    scale_probes: bool = True

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


def merge_auxiliary_data(
    ds: xr.Dataset,
    aux_ds: xr.Dataset,
    config: Optional["ProfileConfig"] = None,
) -> xr.Dataset:
    """
    Merge auxiliary data (lat, lon, T, S, density) into profile dataset.

    Interpolates auxiliary time series data onto the profile's slow time coordinate.

    Parameters
    ----------
    ds : xr.Dataset
        Profile dataset with t_slow coordinate (should have decoded times)
    aux_ds : xr.Dataset
        Auxiliary dataset with time series of latitude, longitude, temperature,
        salinity, and/or density
    config : ProfileConfig, optional
        Configuration specifying variable names. If None, uses defaults.

    Returns
    -------
    xr.Dataset
        Profile dataset with interpolated auxiliary variables added:
        - aux_latitude, aux_longitude: position
        - aux_temperature, aux_salinity, aux_density: for viscosity calculation
    """
    if config is None:
        config = ProfileConfig()

    ds = ds.copy()

    # Get profile time coordinate
    profile_time = ds.t_slow

    # Get auxiliary time coordinate
    aux_time_var = config.aux_time
    if aux_time_var not in aux_ds.dims and aux_time_var not in aux_ds.coords:
        raise ValueError(
            f"Auxiliary time variable '{aux_time_var}' not found in auxiliary dataset"
        )

    # Variables to interpolate: (aux_name, output_name)
    var_mappings = [
        (config.aux_latitude, "aux_latitude"),
        (config.aux_longitude, "aux_longitude"),
        (config.aux_temperature, "aux_temperature"),
        (config.aux_salinity, "aux_salinity"),
        (config.aux_density, "aux_density"),
    ]

    for aux_var, output_var in var_mappings:
        if aux_var in aux_ds:
            # Interpolate onto profile time
            interp_data = aux_ds[aux_var].interp(
                {aux_time_var: profile_time},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            ds[output_var] = ("t_slow", interp_data.values)
            if config.verbose:
                logger.info(f"Interpolated {aux_var} -> {output_var}")
        elif config.verbose:
            logger.info(f"Auxiliary variable '{aux_var}' not found, skipping")

    return ds


def estimate_speed_from_pressure(
    pressure: np.ndarray,
    fs: float,
    pitch: Optional[np.ndarray] = None,
    angle_of_attack: float = 3.0,
    dbar_to_m: float = 1.0197,
) -> np.ndarray:
    """
    Estimate fall speed from pressure rate of change. Optionally corrects for pitch.

    Parameters
    ----------
    pressure : ndarray
        Pressure in dbar
    fs : float
        Sampling frequency in Hz
    pitch : ndarray, optional
        Pitch angle in degrees (positive = nose up). If None, assumes vertical.
    angle_of_attack : float
        Angle of attack in degrees (default: 3.0)
    dbar_to_m : float
        Conversion factor from dbar to meters (default: 1.0197)

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


def prepare_profile(
    ds: xr.Dataset,
    config: Optional[ProfileConfig] = None,
) -> xr.Dataset:
    """
    Prepare raw p2nc output for epsilon processing.

    This function performs the preprocessing needed for data converted via p2nc:
    1. Low-pass filters the pressure data
    2. Estimates or smooths the speed variable
    3. Scales shear probes by 1/U^2 to convert to du/dz
    4. Scales temperature gradient probes by 1/U to convert to dT/dz

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

    # Design low-pass filter
    cutoff = 1 / (2 * config.smoothing_period)
    sos = sig.butter(config.filter_order, cutoff, btype="low", fs=fs_slow, output="sos")

    # Smooth pressure
    if config.pressure in ds:
        ds[config.pressure_smooth] = (
            "t_slow",
            sig.sosfiltfilt(sos, ds[config.pressure].values),
        )
    else:
        raise ValueError(f"Pressure variable '{config.pressure}' not found in dataset")

    if config.speed in ds:
        # Speed variable exists - smooth it
        ds[config.speed_smooth] = (
            "t_slow",
            sig.sosfiltfilt(sos, ds[config.speed].values),
        )
    else:
        if config.verbose:
            logger.info(
                f"Speed variable '{config.speed}' not found, "
                "estimating from pressure derivative"
            )

        pitch = None
        if config.use_pitch_correction and config.pitch in ds:
            pitch = ds[config.pitch].values
            if config.verbose:
                logger.info(
                    f"Using pitch correction with AoA={config.angle_of_attack}°"
                )
        speed_est = estimate_speed_from_pressure(
            ds[config.pressure_smooth].values,
            fs_slow,
            pitch=pitch,
            angle_of_attack=config.angle_of_attack,
            dbar_to_m=config.dbar_to_m,
        )

        # Speed needs a longer smoothing period
        sos_speed = sig.butter(
            config.filter_order,
            1 / (2 * config.speed_smoothing_period),
            btype="low",
            fs=fs_slow,
            output="sos",
        )
        ds[config.speed_smooth] = ("t_slow", sig.sosfiltfilt(sos_speed, speed_est))

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
) -> xr.Dataset:
    """Despike specified variables, creating new cleaned versions."""
    ds = ds.copy()

    for var in variables:
        if var not in ds:
            continue
        cleaned, _, _, _ = despike(ds[var].values)
        ds[var + suffix] = ("t_fast", cleaned)

    return ds


def find_valid_segment(ds: xr.Dataset, config: ProfileConfig) -> xr.Dataset:
    """Extract valid profile segment based on speed threshold.

    Expects smoothed variables from prepare_profile.
    """

    pressure_var = config.pressure_smooth
    speed_var = config.speed_smooth

    idx = find_segment(
        ds[pressure_var],
        apply_speed_threshold=True,
        min_speed=config.min_speed,
        velocity=ds[speed_var],
    )

    # Clamp end index to valid range
    idx_start = idx[0]
    idx_end = min(idx[1], ds.t_slow.size - 1)

    t0, t1 = ds.t_slow[idx_start], ds.t_slow[idx_end]
    return ds.sel(t_slow=slice(t0, t1), t_fast=slice(t0, t1))


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
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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


def process_profile(
    ds: xr.Dataset,
    config: Optional[ProfileConfig] = None,
) -> xr.Dataset:
    """Process a microstructure profile to compute dissipation rates.

    If the dataset hasn't been prepared (no smoothed speed variable),
    prepare_profile will be called automatically.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from p2nc conversion or after prepare_profile
    config : ProfileConfig, optional
        Configuration for processing. If None, uses defaults.

    Returns
    -------
    xr.Dataset
        Dataset with epsilon estimates and spectra
    """
    if config is None:
        config = ProfileConfig()

    # Auto-prepare if smoothed speed doesn't exist
    if config.speed_smooth not in ds:
        if config.verbose:
            logger.info("Smoothed speed not found, running prepare_profile")
        ds = prepare_profile(ds, config)

    pressure_var = config.pressure_smooth
    speed_var = config.speed_smooth

    ds = despike_variables(ds, config.all_probes)
    ds = find_valid_segment(ds, config)

    params = compute_window_parameters(ds, config)
    ds = trim_to_complete_windows(ds, params, config.chop_start)

    for key, val in params.items():
        ds.attrs[key] = val

    means = compute_window_means(
        ds,
        ["t_slow", pressure_var, speed_var, config.temperature],
        params,
    )

    n_windows = len(means["t_slow"])
    ds = ds.assign_coords(time=("time", means["t_slow"]))
    ds["pressure"] = (
        "time",
        means.get(pressure_var, np.full(n_windows, np.nan)).astype("f4"),
    )
    ds["W"] = ("time", means.get(speed_var, np.full(n_windows, np.nan)).astype("f4"))

    if config.temperature in means:
        T_mean = means[config.temperature]
    else:
        T_mean = np.full(n_windows, config.default_temperature)

    # Get salinity and density - prefer auxiliary data if available
    if "aux_salinity" in ds:
        S_mean = window_mean(
            ds["aux_salinity"].values,
            params["n_fft"] // params["sampling_ratio"],
            params["n_diss"] // params["sampling_ratio"],
        )
        ds["salinity"] = ("time", S_mean.astype("f4"))
    else:
        S_mean = np.full(n_windows, config.default_salinity)

    if "aux_density" in ds:
        rho_mean = window_mean(
            ds["aux_density"].values,
            params["n_fft"] // params["sampling_ratio"],
            params["n_diss"] // params["sampling_ratio"],
        )
        ds["density"] = ("time", rho_mean.astype("f4"))
    else:
        rho_mean = np.full(n_windows, config.default_density)

    # Use auxiliary temperature if available, otherwise use MicroRider temp or default
    if "aux_temperature" in ds:
        T_visc = window_mean(
            ds["aux_temperature"].values,
            params["n_fft"] // params["sampling_ratio"],
            params["n_diss"] // params["sampling_ratio"],
        )
        ds["temperature"] = ("time", T_visc.astype("f4"))
    else:
        T_visc = T_mean
        ds["temperature"] = ("time", T_mean.astype("f4"))

    # Compute viscosity using best available T, S, rho
    ds["nu"] = (
        "time",
        compute_viscosity(T_visc, S_mean, rho_mean).astype("f4"),
    )

    # Add latitude/longitude if available
    if "aux_latitude" in ds:
        lat_mean = window_mean(
            ds["aux_latitude"].values,
            params["n_fft"] // params["sampling_ratio"],
            params["n_diss"] // params["sampling_ratio"],
        )
        ds["latitude"] = ("time", lat_mean.astype("f4"))

    if "aux_longitude" in ds:
        lon_mean = window_mean(
            ds["aux_longitude"].values,
            params["n_fft"] // params["sampling_ratio"],
            params["n_diss"] // params["sampling_ratio"],
        )
        ds["longitude"] = ("time", lon_mean.astype("f4"))

    freq, spectra = compute_spectra(
        ds,
        config.all_probes,
        float(ds.fs_fast),
        params["n_fft"],
        params["n_diss"],
    )

    ds = ds.assign_coords(frequency=("frequency", freq))
    for name, psd in spectra.items():
        ds[f"S_{name}"] = (("time", "frequency"), psd.astype("f4"))

    epsilon_results = compute_epsilon(freq, spectra, ds["W"].values, ds["nu"].values)

    for name, (eps, k_max) in epsilon_results.items():
        probe_num = name[-1]
        ds[f"eps_{probe_num}"] = ("time", eps.astype("f4"))
        ds[f"k_max_{probe_num}"] = ("time", k_max.astype("f4"))

    ds["k"] = ds.frequency / ds.W

    return ds

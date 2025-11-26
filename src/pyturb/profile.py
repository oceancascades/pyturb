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
    """Configuration for profile processing."""

    diss_len_sec: float = 4.0
    fft_len_sec: float = 1.0

    pressure: str = "P"
    speed: str = "U_EM"
    temperature: str = "JAC_T"
    shear_probes: tuple[str, ...] = ("sh1", "sh2")
    temperature_probes: tuple[str, ...] = ("gradT1", "gradT2")

    min_speed: float = 0.2

    default_temperature: float = 10.0
    default_salinity: float = 35.0
    default_density: float = 1025.0

    chop_start: bool = True
    verbose: bool = False

    @property
    def all_probes(self) -> tuple[str, ...]:
        return self.shear_probes + self.temperature_probes


@dataclass
class PrepareConfig:
    """Configuration for preparing raw p2nc output for processing.

    This handles the preprocessing needed for data converted via p2nc,
    including smoothing of slow variables and scaling of shear/gradT
    by fall speed.
    """

    # Smoothing parameters
    smoothing_period: float = 0.25  # seconds (cutoff period for low-pass filter)
    filter_order: int = 4

    # Variables to smooth (slow-sampled)
    speed_var: str = "U_EM"
    pressure_var: str = "P"

    # Output variable names for smoothed data
    speed_smooth: str = "U_EM_smooth"
    pressure_smooth: str = "P_smooth"

    # Probes to scale by fall speed
    shear_probes: tuple[str, ...] = ("sh1", "sh2")
    temperature_probes: tuple[str, ...] = ("gradT1", "gradT2")


def prepare_profile(
    ds: xr.Dataset,
    config: Optional[PrepareConfig] = None,
) -> xr.Dataset:
    """
    Prepare raw p2nc output for epsilon processing.

    This function performs the preprocessing needed for data converted via p2nc:
    1. Low-pass filters the speed and pressure data to remove high-frequency noise
    2. Scales shear probes by 1/U^2 to convert to du/dz
    3. Scales temperature gradient probes by 1/U to convert to dT/dz

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from p2nc conversion containing:
        - U_EM, P on t_slow dimension
        - sh1, sh2, gradT1, gradT2 on t_fast dimension
        - fs_slow, fs_fast sampling rates as attributes or variables
    config : PrepareConfig, optional
        Configuration for preprocessing. If None, uses defaults.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - U_EM_smooth, P_smooth: smoothed slow variables
        - sh1, sh2: scaled by 1/U_smooth^2 (overwrites original)
        - gradT1, gradT2: scaled by 1/U_smooth (overwrites original)
    """
    if config is None:
        config = PrepareConfig()

    ds = ds.copy()

    # Get sampling rate for slow channels
    fs_slow = float(ds.fs_slow)

    # Design low-pass filter
    cutoff = 1 / (2 * config.smoothing_period)
    sos = sig.butter(config.filter_order, cutoff, btype="low", fs=fs_slow, output="sos")

    # Smooth speed and pressure
    if config.speed_var in ds:
        ds[config.speed_smooth] = (
            "t_slow",
            sig.sosfiltfilt(sos, ds[config.speed_var].values),
        )
    else:
        raise ValueError(f"Speed variable '{config.speed_var}' not found in dataset")

    if config.pressure_var in ds:
        ds[config.pressure_smooth] = (
            "t_slow",
            sig.sosfiltfilt(sos, ds[config.pressure_var].values),
        )

    # Interpolate smoothed speed to fast time for scaling
    interp_kwargs = dict(
        t_slow=ds.t_fast,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    U_fast = ds[config.speed_smooth].interp(**interp_kwargs)

    # Scale shear probes by 1/U^2
    for probe in config.shear_probes:
        if probe in ds:
            ds[probe] = ds[probe] / U_fast**2

    # Scale temperature gradient probes by 1/U
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
    """Extract valid profile segment based on speed threshold."""
    idx = find_segment(
        ds[config.pressure],
        apply_speed_threshold=True,
        min_speed=config.min_speed,
        velocity=ds[config.speed],
    )

    # find_segment returns (start, end) where end is exclusive (like Python slice)
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
    """Process a microstructure profile to compute dissipation rates."""
    if config is None:
        config = ProfileConfig()

    ds = despike_variables(ds, config.all_probes)
    ds = find_valid_segment(ds, config)

    params = compute_window_parameters(ds, config)
    ds = trim_to_complete_windows(ds, params, config.chop_start)

    for key, val in params.items():
        ds.attrs[key] = val

    means = compute_window_means(
        ds,
        ["t_slow", config.pressure, config.speed, config.temperature],
        params,
    )

    n_windows = len(means["t_slow"])
    ds = ds.assign_coords(t_diss=("t_diss", means["t_slow"]))
    ds["P_diss"] = ("t_diss", means.get(config.pressure, np.full(n_windows, np.nan)))
    ds["U_diss"] = ("t_diss", means.get(config.speed, np.full(n_windows, np.nan)))

    if config.temperature in means:
        T_mean = means[config.temperature]
    else:
        T_mean = np.full(n_windows, config.default_temperature)

    ds["T_diss"] = ("t_diss", T_mean.astype("f4"))
    ds["nu"] = (
        "t_diss",
        compute_viscosity(
            T_mean, config.default_salinity, config.default_density
        ).astype("f4"),
    )

    freq, spectra = compute_spectra(
        ds,
        config.all_probes,
        float(ds.fs_fast),
        params["n_fft"],
        params["n_diss"],
    )

    ds = ds.assign_coords(frequency=("frequency", freq))
    for name, psd in spectra.items():
        ds[f"S_{name}"] = (("t_diss", "frequency"), psd.astype("f4"))

    epsilon_results = compute_epsilon(
        freq, spectra, ds["U_diss"].values, ds["nu"].values
    )

    for name, (eps, k_max) in epsilon_results.items():
        probe_num = name[-1]
        ds[f"eps_{probe_num}"] = ("t_diss", eps.astype("f4"))
        ds[f"k_max_{probe_num}"] = ("t_diss", k_max.astype("f4"))

    ds["w"] = -ds[config.pressure].differentiate("t_slow")
    ds["k"] = ds.frequency / ds.U_diss

    return ds

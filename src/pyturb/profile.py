# Processing on an individual profile contained in an xarray Dataset

import numpy as np
import xarray as xr
from profinder import find_segment

from .shear import estimate_epsilon
from .signal import despike, window_mean, window_psd


def despike_profile(ds: xr.Dataset) -> xr.Dataset:
    variables = ["sh1", "sh2", "gradT1", "gradT2"]

    for var in variables:
        if var not in ds.variables:
            continue
        var_cleaned, _, _, _ = despike(ds[var].values)
        ds[var + "_clean"] = ("t_fast", var_cleaned)

    return ds


def prepare_profile(
    ds: xr.Dataset,
    diss_len_sec: float = 1,
    fft_len_sec: float = 1,
    chop_start: bool = True,
) -> xr.Dataset:
    # First use profinder to indentify valid portion of the profile
    profile_idx = find_segment(
        ds.P_slow, apply_speed_threshold=True, min_speed=0.2, velocity=ds.U_EM
    )
    time_0 = ds.t_slow[profile_idx[0]]
    time_1 = ds.t_slow[profile_idx[1]]
    ds_valid = ds.sel(t_slow=slice(time_0, time_1), t_fast=slice(time_0, time_1))

    n_fft = int(fft_len_sec * ds.fs_fast)
    n_diss = int(diss_len_sec * ds.fs_fast)
    fft_overlap = n_fft // 2
    diss_overlap = fft_overlap

    ds_valid["n_fft"] = n_fft
    ds_valid["n_diss"] = n_diss
    ds_valid["fft_overlap"] = fft_overlap
    ds_valid["diss_overlap"] = diss_overlap

    # Figure out how many fft bins and diss bins we can fit into the data
    n_diss_windows = (ds_valid.t_fast.size - diss_overlap) // (n_diss - diss_overlap)
    sampling_ratio = int(ds.fs_fast / ds.fs_slow)
    ds_valid["sampling_ratio"] = sampling_ratio

    print(
        f"FFT windows per dissipation window: {(n_diss - fft_overlap) // (n_fft - fft_overlap)}"
    )
    print(f"Dissipation windows: {n_diss_windows}")

    n_fast = n_diss_windows * (n_diss - diss_overlap) + diss_overlap
    n_slow = (
        n_diss_windows * (n_diss - diss_overlap) // sampling_ratio
        + diss_overlap // sampling_ratio
    )

    print(f"Reducing fast data from {ds_valid.t_fast.size} to {n_fast}")
    print(f"Reducing slow data from {ds_valid.t_slow.size} to {n_slow}")

    if chop_start:
        return ds_valid.isel(t_fast=slice(-n_fast, None), t_slow=slice(-n_slow, None))
    else:
        return ds_valid.isel(t_fast=slice(0, n_fast), t_slow=slice(0, n_slow))


def calculate_spectra(ds: xr.Dataset) -> xr.Dataset:
    n_fft = ds.n_fft.item()
    n_diss = ds.n_diss.item()
    fs = ds.fs_fast.item()

    variables = ["sh1", "sh2", "gradT1", "gradT2"]
    for var in variables:
        varc = var + "_clean"
        if varc not in ds.variables:
            raise ValueError(f"{varc} not in dataset")

        freq, PSD = window_psd(ds[varc].values, fs, n_fft, n_diss)

        ds["S_" + var] = (("t_diss", "frequency"), PSD.astype("f4"))

    ds["frequency"] = ("frequency", freq)

    return ds


def calculate_means(ds: xr.Dataset) -> xr.Dataset:
    variables = ["t_slow", "P_slow", "U_EM"]
    for var in variables:
        if var not in ds.variables:
            raise ValueError(f"{var} not in dataset")

    n_fft_slow = int(ds.n_fft / ds.sampling_ratio)
    n_diss_slow = int(ds.n_diss / ds.sampling_ratio)

    ds["t_diss"] = (
        ("t_diss"),
        window_mean(ds["t_slow"].values, n_fft_slow, n_diss_slow).astype("f4"),
    )
    ds["P_diss"] = (
        ("t_diss"),
        window_mean(ds["P_slow"].values, n_fft_slow, n_diss_slow).astype("f4"),
    )
    ds["U_diss"] = (
        ("t_diss"),
        window_mean(ds["U_EM"].values, n_fft_slow, n_diss_slow).astype("f4"),
    )

    return ds


def calculate_epsilon(ds: xr.Dataset, nu=1e-6) -> xr.Dataset:
    eps_1 = np.full(ds.t_diss.size, np.nan)
    eps_2 = np.full(ds.t_diss.size, np.nan)
    k_max_1 = np.full(ds.t_diss.size, np.nan)
    k_max_2 = np.full(ds.t_diss.size, np.nan)

    for i in range(ds.t_diss.size):
        gp_ = ds.isel(t_diss=i)

        eps_1[i], k_max_1[i] = estimate_epsilon(
            gp_.frequency.values,
            gp_["S_sh1"].values,
            W=gp_.U_diss.values,
            nu=nu,
        )
        eps_2[i], k_max_2[i] = estimate_epsilon(
            gp_.frequency.values,
            gp_["S_sh2"].values,
            W=gp_.U_diss.values,
            nu=nu,
        )

    ds["eps_1"] = (("t_diss"), eps_1.astype("f4"))
    ds["eps_2"] = (("t_diss"), eps_2.astype("f4"))
    ds["k_max_1"] = (("t_diss"), k_max_1.astype("f4"))
    ds["k_max_2"] = (("t_diss"), k_max_2.astype("f4"))

    return ds


def process_profile(
    ds: xr.Dataset,
    diss_len_sec: float = 4,
    fft_len_sec: float = 1,
    chop_start: bool = True,
    nu=1e-6,
) -> xr.Dataset:
    ds = despike_profile(ds)

    ds = prepare_profile(ds, diss_len_sec, fft_len_sec, chop_start)

    ds = calculate_means(ds)

    ds = calculate_spectra(ds)

    ds["w"] = -ds.P_slow.differentiate("t_slow")

    ds["k"] = ds.frequency / ds.U_diss

    ds = calculate_epsilon(ds, nu=nu)

    return ds

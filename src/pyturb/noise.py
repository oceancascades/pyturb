"""
Electronic noise models for RSI instrument channels.

These models predict the noise floor of the signal conditioning electronics
as a function of frequency. They are useful for validating bench test data
and identifying faulty instruments.

``thermistor_noise_phi`` combines ``noise_thermchannel`` with the FP07
Steinhart-Hart scale factor and thermal-response correction to predict the
gradT noise floor in the same physical units and domain as the observed,
response-corrected spectrum (see ``pyturb.profile``'s ``S_gradT1``/
``S_gradT2``) -- used by :func:`pyturb.temperature.estimate_chi` to cap the
chi integration band at the wavenumber where signal meets noise.
"""

import numpy as np

from .temperature import single_pole_correction as _fp07_response_correction


def _channel_calibration_params(ds, probe: str) -> dict:
    """``cal_<key>`` attrs for ``probe`` (e.g. ``"T1"``), plus ``diff_gain``
    from its pre-emphasized channel (e.g. ``"T1_dT1"``).

    Prefers the in-situ ``calibrate-fp07`` coefficients over the factory
    ones if the channel was recalibrated -- ``apply_probe_calibration``
    rebuilds the data with the new coefficients but leaves the original
    ``cal_t_0``/``cal_beta_1``/``cal_beta_2`` attrs untouched (they record
    the factory calibration), so those must not be used here or the
    predicted noise floor would not match what actually produced the data.

    A small, deliberate duplicate of ``fp07_calibration._channel_params``'s
    attr-reading (rather than importing it): ``fp07_calibration`` imports
    ``profile``, and ``profile`` needs this function, so importing
    ``fp07_calibration`` here would create a cycle.

    Returns {} if ``probe`` isn't present or carries no ``cal_*`` attrs.
    """
    if probe not in ds:
        return {}
    attrs = ds[probe].attrs
    params = {k[len("cal_") :]: v for k, v in attrs.items() if k.startswith("cal_")}
    if not params:
        return {}
    if attrs.get(f"{probe}_fp07_recalibrated"):
        params["t_0"] = attrs[f"{probe}_fp07_new_T_0"]
        params["beta_1"] = attrs[f"{probe}_fp07_new_beta_1"]
        new_beta_2 = attrs[f"{probe}_fp07_new_beta_2"]
        if np.isfinite(new_beta_2):
            params["beta_2"] = new_beta_2
        else:
            params.pop("beta_2", None)
    dT_name = f"{probe}_d{probe}"
    if dT_name in ds and "cal_diff_gain" in ds[dT_name].attrs:
        params["diff_gain"] = ds[dT_name].attrs["cal_diff_gain"]
    return params


def _log_R_from_T(
    T_celsius: np.ndarray, t_0: float, beta_1: float, beta_2: float | None = None
) -> np.ndarray:
    """Invert the Steinhart-Hart equation: absolute temperature -> log(R_T/R_0).

    The forward direction is ``fp07_calibration.steinhart_hart_forward``;
    this is only needed here because the noise floor is built from a
    window-mean temperature rather than raw counts.
    """
    T_abs = np.asarray(T_celsius, dtype=float) + 273.15
    c = 1.0 / t_0 - 1.0 / T_abs
    if beta_2 is not None:
        a = 1.0 / beta_2
        b = 1.0 / beta_1
        disc = b * b - 4 * a * c
        return (-b + np.sqrt(disc)) / (2 * a)
    return -c * beta_1


def thermistor_noise_phi(
    f: np.ndarray,
    W: float,
    T_celsius: float,
    params: dict,
    fs_fast: float,
) -> np.ndarray:
    """Predicted FP07 gradT noise floor phi(k) [K2 m-2 cpm-1], k = f / W.

    Combines the electronics noise model (``noise_thermchannel``) with the
    Steinhart-Hart counts-to-K/s scale factor (evaluated at the window-mean
    temperature) and the FP07 thermal single-pole response correction --
    the same corrections applied to the real signal (see
    ``pyturb.profile``'s ``gradT_response_correction`` and
    ``fp07_calibration.make_gradT``) -- so it is directly comparable to the
    observed, response-corrected spectrum.

    f    : frequency (Hz)
    W    : mean fall speed for this window (m/s)
    T_celsius : window-mean temperature (degC) -- sets the thermistor's
        resistance ratio, and hence the scale factor, operating point.
    params : calibration dict from :func:`_channel_calibration_params`
        (a, b, g, e_b, adc_fs, adc_bits, t_0, beta_1, beta_2 [optional],
        diff_gain).
    fs_fast : fast sampling rate (Hz).
    """
    b = float(params["b"])
    adc_fs = float(params["adc_fs"])
    adc_bits = int(float(params["adc_bits"]))
    g = float(params["g"])
    e_b = float(params["e_b"])
    t_0 = float(params["t_0"])
    beta_1 = float(params["beta_1"])
    beta_2 = params.get("beta_2")
    beta_2 = (
        float(beta_2) if beta_2 is not None and np.isfinite(float(beta_2)) else None
    )
    diff_gain = float(params["diff_gain"])

    log_R = _log_R_from_T(T_celsius, t_0, beta_1, beta_2)
    R = np.exp(log_R)
    eta = (b / 2) * (2**adc_bits) * g * e_b / adc_fs
    sf = 1 + 2 * (beta_1 / beta_2) * log_R if beta_2 is not None else 1.0
    T_abs = T_celsius + 273.15
    scale_factor = sf * T_abs**2 * (1 + R) ** 2 / (2 * eta * beta_1 * R)
    R_ohms = 3000.0 * R  # nominal thermistor R_0 (ODAS default)

    with np.errstate(divide="ignore", invalid="ignore"):
        # f=0 (the DC bin, always present in production spectra) makes
        # noise_thermchannel's flicker-noise terms (~1/f) blow up to inf/nan
        # -- harmless (thermistor_noise_phi's own NaN/Inf checks and
        # _noise_crossing_k's isfinite filtering drop that bin downstream),
        # but noisy without suppressing it here.
        noise_counts_psd = noise_thermchannel(
            f,
            FS=adc_fs,
            Bits=adc_bits,
            gamma_RSI=3.0,
            fs=fs_fast,
            R_0=R_ohms,
            gain=g,
            G_D=diff_gain,
        )
        G_HP = (
            (1 / diff_gain) ** 2
            * (2 * np.pi * diff_gain * f) ** 2
            / (1 + (2 * np.pi * diff_gain * f) ** 2)
        )
        noise_Kdot_psd = noise_counts_psd * G_HP * scale_factor**2  # (K/s)^2/Hz
    noise_S_f = noise_Kdot_psd / W**2  # (K/m)^2/Hz
    return noise_S_f * W * _fp07_response_correction(f, W)  # phi(k) domain


def noise_shearchannel(
    f: np.ndarray,
    T_K: float = 295.0,
    K_B: float = 1.382e-23,
    VFS: float = 4.096,
    Bits: int = 16,
    gamma_RSI: float = 2.5,
    fs: float = 512.0,
    R1: float = 1e9,
    C1: float = 1.5e-9,
    R2: float = 499.0,
    C2: float = 0.94e-6,
    R3: float = 1e6,
    C3: float = 470e-12,
    CP: float = 0.0,
    f_AA: float = 110.0,
    E_1: float = 9e-9,
    fc: float = 50.0,
    I_1: float = 0.56e-15,
) -> np.ndarray:
    """
    Compute the electronic noise spectrum of a shear probe channel.

    Models four stages: charge-transfer amplifier, differentiator,
    anti-aliasing filter (2x 4-pole Butterworth), and ADC sampler.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz at which to evaluate the noise spectrum.
    T_K : float
        Temperature in Kelvin. Default 295.
    K_B : float
        Boltzmann constant in J/K.
    VFS : float
        ADC full-scale voltage. Default 4.096 V.
    Bits : int
        ADC resolution in bits. Default 16.
    gamma_RSI : float
        RSI noise factor for sampler. Default 2.5.
    fs : float
        Sampling rate in Hz. Default 512.
    R1 : float
        Charge-transfer feedback resistor in Ohms. Default 1e9.
    C1 : float
        Charge-transfer capacitor in Farads. Default 1.5e-9.
    R2 : float
        Differentiator input resistor in Ohms. Default 499.
    C2 : float
        Differentiator capacitor in Farads. Default 0.94e-6.
    R3 : float
        Differentiator output resistor in Ohms. Default 1e6.
    C3 : float
        Output capacitor in Farads. Default 470e-12.
    CP : float
        Probe capacitance in Farads. Default 0.
    f_AA : float
        Anti-aliasing filter cutoff in Hz. Default 110.
    E_1 : float
        Op-amp voltage noise density in V/sqrt(Hz). Default 9e-9.
    fc : float
        Flicker noise knee frequency in Hz. Default 50.
    I_1 : float
        Op-amp current noise density in A/sqrt(Hz). Default 0.56e-15.

    Returns
    -------
    ndarray
        Noise power spectral density in counts^2/Hz.
    """
    f = np.asarray(f, dtype=np.float64)
    omega = 2 * np.pi * f

    delta_s = VFS / 2**Bits
    fN = fs / 2

    # Stage 1: Charge-transfer amplifier
    V_V1 = E_1**2 * (fc / f) * np.sqrt(1 + (f / fc) ** 2)
    V_I1 = I_1**2 * R1**2 / (1 + (omega * R1 * C1) ** 2)
    V_R1 = 4 * K_B * T_K * R1 / (1 + (omega * R1 * C1) ** 2)
    G_1 = (1 + (omega * R1 * (CP + C1)) ** 2) / (1 + (omega * R1 * C1) ** 2)
    Noise_1 = G_1 * (V_V1 + V_I1) + V_R1

    # Stage 2: Differentiator
    G_2 = (omega * R3 * C2) ** 2 / (
        (1 + (omega * R2 * C2) ** 2) * (1 + (omega * R3 * C3) ** 2)
    )
    Noise_2 = (Noise_1 + V_V1) * G_2

    # Stage 3: Anti-aliasing filter (2x 4-pole Butterworth)
    G_AA = 1 / (1 + (f / f_AA) ** 8) ** 2
    Noise_3 = Noise_2 * G_AA

    # Stage 4: ADC sampler
    Noise_4 = Noise_3 + gamma_RSI * delta_s**2 / (12 * fN)

    # Convert from V^2/Hz to counts^2/Hz
    return Noise_4 / delta_s**2


def noise_thermchannel(
    f: np.ndarray,
    T_K: float = 295.0,
    K_B: float = 1.382e-23,
    FS: float = 4.096,
    Bits: int = 16,
    gamma_RSI: float = 3.0,
    fs: float = 512.0,
    R_0: float = 3000.0,
    gain: float = 6.0,
    G_D: float = 0.94,
    f_AA: float = 110.0,
    E_n: float = 4e-9,
    fc: float = 18.7,
    E_n2: float = 8e-9,
    fc_2: float = 42.0,
) -> np.ndarray:
    """
    Compute the electronic noise spectrum of an FP07 thermistor channel.

    Models four stages: bridge excitation + first amplifier, pre-emphasis
    differentiator, anti-aliasing filter (2x 4-pole Butterworth), and
    ADC sampler.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz at which to evaluate the noise spectrum.
    T_K : float
        Temperature in Kelvin. Default 295.
    K_B : float
        Boltzmann constant in J/K.
    FS : float
        ADC full-scale voltage. Default 4.096 V.
    Bits : int
        ADC resolution in bits. Default 16.
    gamma_RSI : float
        RSI noise factor for sampler. Default 3.
    fs : float
        Sampling rate in Hz. Default 512.
    R_0 : float
        Nominal thermistor resistance in Ohms. Default 3000.
    gain : float
        First-stage circuit gain. Default 6.
    G_D : float
        Differentiator time constant in seconds. Default 0.94.
    f_AA : float
        Anti-aliasing filter cutoff in Hz. Default 110.
    E_n : float
        First-stage op-amp voltage noise in V/sqrt(Hz). Default 4e-9.
    fc : float
        First-stage flicker knee frequency in Hz. Default 18.7.
    E_n2 : float
        Second-stage op-amp voltage noise in V/sqrt(Hz). Default 8e-9.
    fc_2 : float
        Second-stage flicker knee frequency in Hz. Default 42.

    Returns
    -------
    ndarray
        Noise power spectral density in counts^2/Hz.
    """
    f = np.asarray(f, dtype=np.float64)

    delta_s = FS / 2**Bits
    fN = fs / 2

    # Stage 1: Bridge excitation + first amplifier
    V1 = 2 * E_n**2 * np.sqrt(1 + (f / fc) ** 2) / (f / fc)
    phi_R = 4 * K_B * R_0 * T_K
    Noise_1 = gain**2 * (V1 + phi_R)

    # Stage 2: Pre-emphasis differentiator
    G_2 = 1 + (2 * np.pi * G_D * f) ** 2
    V2 = 2 * E_n2**2 * np.sqrt(1 + (f / fc_2) ** 2) / (f / fc_2)
    Noise_2 = G_2 * (Noise_1 + V2)

    # Stage 3: Anti-aliasing filter (2x 4-pole Butterworth)
    G_AA = 1 / (1 + (f / f_AA) ** 8) ** 2
    Noise_3 = Noise_2 * G_AA

    # Stage 4: ADC sampler
    Noise_4 = Noise_3 + gamma_RSI * delta_s**2 / (12 * fN)

    # Convert from V^2/Hz to counts^2/Hz
    return Noise_4 / delta_s**2

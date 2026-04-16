"""
Electronic noise models for RSI instrument channels.

Translated from MATLAB ODAS library v4.5.1:
- noise_shearchannel.m
- noise_thermchannel.m

These models predict the noise floor of the signal conditioning electronics
as a function of frequency. They are useful for validating bench test data
and identifying faulty instruments.
"""

import numpy as np


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

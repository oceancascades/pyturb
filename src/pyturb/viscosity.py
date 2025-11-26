"""Kinematic and dynamic viscosity of seawater."""

import numpy as np
from numpy.typing import ArrayLike


def viscosity(
    S: ArrayLike, T: ArrayLike, rho: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate kinematic and dynamic viscosity of seawater.

    Parameters
    ----------
    S : array_like
        Salinity in practical salinity units (PSU).
    T : array_like
        Temperature in degrees Celsius.
    rho : array_like
        Density in kg/m³.

    Returns
    -------
    nu : ndarray
        Kinematic viscosity in m²/s.
    mu : ndarray
        Dynamic viscosity in kg/(m·s).

    Notes
    -----
    Follows Millero (1974). Valid for 5 ≤ T ≤ 25°C and 0 ≤ S ≤ 40 PSU
    at atmospheric pressure.

    Check value: T=25, S=40, rho=rho(T,S) → mu = 9.6541e-4

    References
    ----------
    Millero, J. F., 1974, The Sea, Vol 5, M. N. Hill, Ed, John Wiley, NY, p. 3.
    Peters and Siedler, in Landolt-Bornstein New Series V/3a (Oceanography), pp 234.
    """
    S = np.asarray(S)
    T = np.asarray(T)
    rho = np.asarray(rho)

    # Coefficients
    a0, a1 = 2.5116e-6, 1.2199e-6
    b0, b1 = 1.4342e-6, 1.8267e-8
    c0 = 1.002e-3
    d0, d1, d2 = 1.1709, 1.827e-3, 89.93

    # Dynamic viscosity at S=0
    mu0 = c0 * 10.0 ** ((d0 * (20 - T) - d1 * (T - 20) ** 2) / (T + d2))

    # Dynamic viscosity at S
    mu = mu0 * (1 + (a0 + a1 * T) * (rho * S) ** 0.5 + (b0 + b1 * T) * rho * S)

    # Kinematic viscosity
    nu = mu / rho

    return nu, mu

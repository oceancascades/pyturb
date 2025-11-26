# Methods for computing temperature variance dissipation

import numpy as np


def kraichnan_spectrum(k, chi, eps, nu=1e-6, kappa_T=1.4e-7, q_K=5.26):
    k_B = (eps / nu / kappa_T**2) ** 0.25  # Batchelor wavenumber
    psi = (
        2
        * np.pi
        * k
        * chi
        * (nu / eps) ** 0.5
        * q_K
        * np.exp(-((6 * q_K) ** 0.5) * 2 * np.pi * k / k_B)
    )
    return psi

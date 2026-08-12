"""Tests for compute_epsilon's NaN-spectrum warning."""

import logging

import numpy as np

from pyturb.profile import compute_epsilon
from pyturb.shear import nasmyth_spectrum

W = 0.6
NU = 1e-6


def _good_psd(freq: np.ndarray) -> np.ndarray:
    k = freq / W
    return nasmyth_spectrum(k, 1e-8, NU) / W


def test_warns_with_window_count_when_spectrum_has_nan(caplog):
    freq = np.linspace(1.0, 100.0, 65)
    psd = np.tile(_good_psd(freq), (3, 1))
    psd[1, 5] = np.nan

    with caplog.at_level(logging.WARNING):
        results = compute_epsilon(freq, {"sh1": psd}, np.full(3, W), np.full(3, NU))

    assert np.isnan(results["sh1"][0][1])
    assert "sh1: 1/3 windows have a NaN spectrum" in caplog.text


def test_no_warning_when_spectrum_is_finite(caplog):
    freq = np.linspace(1.0, 100.0, 65)
    psd = np.tile(_good_psd(freq), (3, 1))

    with caplog.at_level(logging.WARNING):
        compute_epsilon(freq, {"sh1": psd}, np.full(3, W), np.full(3, NU))

    assert "NaN spectrum" not in caplog.text

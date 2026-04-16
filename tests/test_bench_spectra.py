"""
Bench test spectral validation.

Verifies that raw channel spectra from a bench test (instrument on a table,
not in water) match the expected electronic noise models. This catches
reader bugs (byte ordering, channel mapping, sign conventions) that would
distort the spectral shape, as well as confirming that the noise models
themselves are correctly ported from MATLAB.

The test checks that measured spectral points fall within a tolerance band
around the theoretical noise model — similar to the shaded acceptance
region shown in Rockland's bench test GUI.
"""

import numpy as np
import pytest
from scipy.signal import welch

from pyturb.noise import noise_shearchannel, noise_thermchannel

# Tolerance band: measured spectrum must be within this factor of the model.
# A factor of 3 means the acceptance region spans model/3 to model*3.
# This is generous enough to accommodate component tolerances and probe
# capacitance variation, but tight enough to catch real problems.
TOLERANCE_FACTOR = 3.0

# Fraction of spectral points that must fall within the tolerance band.
# We allow some outliers (spectral leakage at band edges, low-frequency
# variance) but require the vast majority to be in-band.
MIN_FRACTION_IN_BAND = 0.90

# Frequency range for comparison. Below ~1 Hz there are too few FFT
# averages for stable estimates; above ~200 Hz the AA filter has already
# killed the signal and quantization noise dominates.
F_MIN = 1.0
F_MAX = 200.0

# FFT length in seconds (matches quick_bench.m default)
FFT_LENGTH_SEC = 2.0


def _compute_psd(signal, fs):
    """Compute PSD using Welch's method with quick_bench.m-equivalent parameters."""
    nfft = round(FFT_LENGTH_SEC * fs)
    f, psd = welch(signal.astype(np.float64), fs=fs, nperseg=nfft, noverlap=nfft // 2)
    return f, psd


def _fraction_in_band(measured, model, factor):
    """Return fraction of points where measured is within [model/factor, model*factor]."""
    in_band = (measured > model / factor) & (measured < model * factor)
    return np.mean(in_band)


class TestShearNoiseSpectrum:
    """Verify shear channel spectra match the electronic noise model."""

    @pytest.fixture
    def shear_spectra(self, raw_data):
        fs = raw_data["fs_fast"]
        f1, psd1 = _compute_psd(raw_data["sh1"], fs)
        f2, psd2 = _compute_psd(raw_data["sh2"], fs)
        mask = (f1 >= F_MIN) & (f1 <= F_MAX)
        model = noise_shearchannel(f1[mask], fs=fs)
        return {
            "f": f1[mask],
            "psd_sh1": psd1[mask],
            "psd_sh2": psd2[mask],
            "model": model,
        }

    def test_sh1_within_tolerance_band(self, shear_spectra):
        frac = _fraction_in_band(
            shear_spectra["psd_sh1"], shear_spectra["model"], TOLERANCE_FACTOR
        )
        assert frac >= MIN_FRACTION_IN_BAND, (
            f"Only {frac:.0%} of sh1 spectral points within {TOLERANCE_FACTOR}x "
            f"of noise model (need {MIN_FRACTION_IN_BAND:.0%})"
        )

    def test_sh2_within_tolerance_band(self, shear_spectra):
        frac = _fraction_in_band(
            shear_spectra["psd_sh2"], shear_spectra["model"], TOLERANCE_FACTOR
        )
        assert frac >= MIN_FRACTION_IN_BAND, (
            f"Only {frac:.0%} of sh2 spectral points within {TOLERANCE_FACTOR}x "
            f"of noise model (need {MIN_FRACTION_IN_BAND:.0%})"
        )

    def test_sh1_median_ratio_near_unity(self, shear_spectra):
        ratio = np.median(shear_spectra["psd_sh1"] / shear_spectra["model"])
        assert 0.3 < ratio < 3.0, f"sh1 median ratio = {ratio:.2f}, expected near 1.0"

    def test_sh2_median_ratio_near_unity(self, shear_spectra):
        ratio = np.median(shear_spectra["psd_sh2"] / shear_spectra["model"])
        assert 0.3 < ratio < 3.0, f"sh2 median ratio = {ratio:.2f}, expected near 1.0"

    def test_sh1_sh2_similar_levels(self, shear_spectra):
        """Two shear probes on the same instrument should have similar noise floors."""
        ratio = np.median(shear_spectra["psd_sh1"] / shear_spectra["psd_sh2"])
        assert 0.3 < ratio < 3.0, f"sh1/sh2 ratio = {ratio:.2f}, probes differ too much"


class TestThermistorNoiseSpectrum:
    """Verify thermistor channel spectra match the electronic noise model."""

    @pytest.fixture
    def therm_spectra(self, raw_data):
        fs = raw_data["fs_fast"]
        f1, psd1 = _compute_psd(raw_data["T1_dT1"], fs)
        f2, psd2 = _compute_psd(raw_data["T2_dT2"], fs)
        mask = (f1 >= F_MIN) & (f1 <= F_MAX)
        model = noise_thermchannel(f1[mask], fs=fs)
        return {
            "f": f1[mask],
            "psd_T1": psd1[mask],
            "psd_T2": psd2[mask],
            "model": model,
        }

    def test_T1_within_tolerance_band(self, therm_spectra):
        frac = _fraction_in_band(
            therm_spectra["psd_T1"], therm_spectra["model"], TOLERANCE_FACTOR
        )
        assert frac >= MIN_FRACTION_IN_BAND, (
            f"Only {frac:.0%} of T1_dT1 spectral points within {TOLERANCE_FACTOR}x "
            f"of noise model (need {MIN_FRACTION_IN_BAND:.0%})"
        )

    def test_T2_within_tolerance_band(self, therm_spectra):
        frac = _fraction_in_band(
            therm_spectra["psd_T2"], therm_spectra["model"], TOLERANCE_FACTOR
        )
        assert frac >= MIN_FRACTION_IN_BAND, (
            f"Only {frac:.0%} of T2_dT2 spectral points within {TOLERANCE_FACTOR}x "
            f"of noise model (need {MIN_FRACTION_IN_BAND:.0%})"
        )

    def test_T1_median_ratio_near_unity(self, therm_spectra):
        ratio = np.median(therm_spectra["psd_T1"] / therm_spectra["model"])
        assert 0.3 < ratio < 3.0, f"T1 median ratio = {ratio:.2f}, expected near 1.0"

    def test_T2_median_ratio_near_unity(self, therm_spectra):
        ratio = np.median(therm_spectra["psd_T2"] / therm_spectra["model"])
        assert 0.3 < ratio < 3.0, f"T2 median ratio = {ratio:.2f}, expected near 1.0"

    def test_T1_T2_similar_levels(self, therm_spectra):
        """Two thermistors on the same instrument should have similar noise floors."""
        ratio = np.median(therm_spectra["psd_T1"] / therm_spectra["psd_T2"])
        assert 0.3 < ratio < 3.0, (
            f"T1/T2 ratio = {ratio:.2f}, thermistors differ too much"
        )

    def test_preemphasis_shape(self, therm_spectra):
        """Thermistor spectrum should rise with frequency due to pre-emphasis.

        The pre-emphasis differentiator adds gain proportional to f^2.
        Check that the spectrum at 50 Hz is significantly higher than at 2 Hz.
        """
        f = therm_spectra["f"]
        psd = therm_spectra["psd_T1"]

        low_mask = (f >= 1.5) & (f <= 3.0)
        high_mask = (f >= 40) & (f <= 60)

        if low_mask.any() and high_mask.any():
            low_level = np.median(psd[low_mask])
            high_level = np.median(psd[high_mask])
            # Pre-emphasis should cause at least 10x increase from 2 to 50 Hz
            assert high_level / low_level > 10, (
                f"Pre-emphasis gain ratio = {high_level / low_level:.1f}, expected > 10"
            )

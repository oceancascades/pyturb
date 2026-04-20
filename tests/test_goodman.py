"""Synthetic tests for Goodman coherent-noise removal.

These tests avoid real data and instead use controlled synthetic cases where
the expected spectral/variance behavior is known.
"""

import numpy as np
import pytest

from pyturb.shear import clean_shear_spec, estimate_epsilon, nasmyth_spectrum, single_pole_correction
from pyturb.signal import window_psd


def _tone(t: np.ndarray, f: float, amp: float, phase: float = 0.0) -> np.ndarray:
    return amp * np.cos(2.0 * np.pi * f * t + phase)


def _nearest_bin(freq: np.ndarray, target_hz: float) -> int:
    return int(np.argmin(np.abs(freq - target_hz)))


def _trim_to_complete_windows(y: np.ndarray, n_fft: int, n_diss: int) -> np.ndarray:
    """Trim 1D array to exact length expected by window_psd."""
    fft_overlap = n_fft // 2
    step = n_fft - fft_overlap
    ffts_per_diss = (n_diss - fft_overlap) // step
    n_seg = (len(y) - fft_overlap) // step
    n_windows = n_seg // ffts_per_diss
    n_used = n_windows * ffts_per_diss * step + fft_overlap
    return y[:n_used]


def _synth_from_one_sided_psd(
    t: np.ndarray,
    freq: np.ndarray,
    psd: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize a real-valued signal from a one-sided PSD by random-phase summation."""
    df = freq[1] - freq[0]
    x = np.zeros_like(t)
    # Skip DC and Nyquist; treat each positive-frequency bin as a cosine component.
    for i in range(1, len(freq) - 1):
        amp = np.sqrt(2.0 * max(psd[i], 0.0) * df)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        x += amp * np.cos(2.0 * np.pi * freq[i] * t + phase)
    return x


class TestGoodmanSyntheticTones:
    """Synthetic tests with known coherent vibration tones."""

    def test_removes_known_coherent_tones_and_recovers_variance(self):
        rng = np.random.default_rng(1234)
        fs = 512.0
        n_fft = 512
        n_diss = 8192
        n = 120 * int(fs)
        t = np.arange(n) / fs

        # Baseline (incoherent) shear signal with known variance.
        sigma = 0.3
        shear_base = rng.normal(0.0, sigma, n)

        # Two synthetic accelerometer channels containing coherent tones.
        accel1 = (
            _tone(t, 20.0, 1.0, 0.1)
            + _tone(t, 40.0, 0.8, 1.3)
            + _tone(t, 60.0, 0.6, -0.7)
        )
        accel2 = (
            _tone(t, 20.0, 0.7, 2.2)
            + _tone(t, 40.0, 0.5, -1.1)
            + _tone(t, 60.0, 0.4, 0.4)
        )
        accel = np.column_stack([accel1, accel2])

        # Contaminated shear = baseline + coherent vibration contamination.
        shear_contam = shear_base + 0.9 * accel1 + 0.6 * accel2

        shear_base = _trim_to_complete_windows(shear_base, n_fft, n_diss)
        shear_contam = _trim_to_complete_windows(shear_contam, n_fft, n_diss)
        accel = accel[: len(shear_contam)]

        freq_raw, psd_raw = window_psd(shear_contam, fs, n_fft, n_diss)
        _, psd_base = window_psd(shear_base, fs, n_fft, n_diss)
        freq_clean, psd_clean = clean_shear_spec(
            shear_contam,
            accel,
            n_fft,
            fs,
            n_diss,
        )

        mean_raw = psd_raw.mean(axis=0)
        mean_base = psd_base.mean(axis=0)
        mean_clean = psd_clean.mean(axis=0)

        # 1) Known coherent tone peaks should be strongly reduced.
        for f0 in (20.0, 40.0, 60.0):
            i = _nearest_bin(freq_raw, f0)
            raw_excess = max(mean_raw[i] - mean_base[i], 0.0)
            clean_excess = max(mean_clean[i] - mean_base[i], 0.0)
            assert clean_excess < 0.35 * raw_excess

        # 2) Integrated variance should move back toward baseline variance.
        df = freq_raw[1] - freq_raw[0]
        var_raw = np.sum(mean_raw) * df
        var_base = np.sum(mean_base) * df
        var_clean = np.sum(mean_clean) * df
        assert abs(var_clean - var_base) < abs(var_raw - var_base)

    def test_shape_and_input_validation(self):
        fs = 512.0
        n_fft = 512
        n_diss = 4096
        n = 8192
        t = np.arange(n) / fs

        shear = _tone(t, 20.0, 0.2)
        accel = np.column_stack([_tone(t, 20.0, 1.0), _tone(t, 40.0, 1.0)])

        freq, psd = clean_shear_spec(shear, accel, n_fft, fs, n_diss)
        assert freq.ndim == 1
        assert psd.ndim == 2
        assert psd.shape[1] == n_fft // 2 + 1
        assert np.all(psd >= 0)

        with pytest.raises(ValueError, match="same number of rows"):
            clean_shear_spec(shear[:-1], accel, n_fft, fs, n_diss)


class TestGoodmanSyntheticNasmyth:
    """Synthetic restoration test using a known Nasmyth-derived target spectrum."""

    def test_restores_nasmyth_like_spectrum_from_tone_contamination(self):
        rng = np.random.default_rng(7)
        fs = 512.0
        n_fft = 512
        n_diss = 8192
        n = 120 * int(fs)
        t = np.arange(n) / fs

        # Build target one-sided frequency spectrum from Nasmyth model.
        eps_true = 3e-8
        nu = 1e-6
        W = 0.6
        freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        k = np.maximum(freq / W, 1e-9)
        phi = nasmyth_spectrum(k, eps_true, nu)
        p_target = phi / (W * single_pole_correction(k))
        p_target[0] = p_target[1]

        # Baseline shear time series with approximately the target PSD.
        shear_base = _synth_from_one_sided_psd(t, freq, p_target, rng)

        # Add coherent vibration noise at observed frequencies.
        accel1 = _tone(t, 20.0, 1.0, 0.2) + _tone(t, 40.0, 0.8, 1.1) + _tone(t, 60.0, 0.6, -0.3)
        accel2 = _tone(t, 20.0, 0.6, -1.7) + _tone(t, 40.0, 0.5, 0.4) + _tone(t, 60.0, 0.4, 2.3)
        accel = np.column_stack([accel1, accel2])
        shear_contam = shear_base + 0.8 * accel1 + 0.5 * accel2

        shear_base = _trim_to_complete_windows(shear_base, n_fft, n_diss)
        shear_contam = _trim_to_complete_windows(shear_contam, n_fft, n_diss)
        accel = accel[: len(shear_contam)]

        freq_raw, psd_raw = window_psd(shear_contam, fs, n_fft, n_diss)
        _, psd_base = window_psd(shear_base, fs, n_fft, n_diss)
        freq_clean, psd_clean = clean_shear_spec(shear_contam, accel, n_fft, fs, n_diss)

        mean_raw = psd_raw.mean(axis=0)
        mean_base = psd_base.mean(axis=0)
        mean_clean = psd_clean.mean(axis=0)

        # 1) Cleaned spectrum should be closer to uncontaminated baseline than raw.
        band = (freq_raw >= 5.0) & (freq_raw <= 120.0)
        rmse_raw = np.sqrt(np.mean((np.log10(mean_raw[band]) - np.log10(mean_base[band])) ** 2))
        rmse_clean = np.sqrt(
            np.mean((np.log10(np.maximum(mean_clean[band], 1e-20)) - np.log10(mean_base[band])) ** 2)
        )
        assert rmse_clean < rmse_raw

        # 2) Tone spikes at 20/40/60 Hz should be substantially reduced.
        for f0 in (20.0, 40.0, 60.0):
            i = _nearest_bin(freq_clean, f0)
            raw_excess = max(mean_raw[i] - mean_base[i], 0.0)
            clean_excess = max(mean_clean[i] - mean_base[i], 0.0)
            assert clean_excess < 0.4 * raw_excess

        # 3) Epsilon estimated from cleaned spectrum should move toward baseline.
        eps_base, _ = estimate_epsilon(freq_clean, mean_base, W=W, nu=nu)
        eps_raw, _ = estimate_epsilon(freq_raw, mean_raw, W=W, nu=nu)
        eps_clean, _ = estimate_epsilon(freq_clean, mean_clean, W=W, nu=nu)
        assert abs(np.log10(eps_clean) - np.log10(eps_base)) < abs(
            np.log10(eps_raw) - np.log10(eps_base)
        )

"""Tests for Goodman coherent-noise removal."""

from pathlib import Path

import numpy as np
import pytest

from pyturb._pfile import to_xarray
from pyturb.pfile import load_pfile_phys
from pyturb.profile import ProfileConfig, process_profile
from pyturb.shear import (
    clean_shear_spec,
    estimate_epsilon,
    nasmyth_spectrum,
    single_pole_correction,
)
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

    @pytest.mark.parametrize("eps_true", [1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    def test_restores_nasmyth_like_spectrum_from_tone_contamination(self, eps_true):
        # Deterministic but varied random phases per epsilon case
        seed = int(-np.log10(eps_true)) + 7
        rng = np.random.default_rng(seed)
        fs = 512.0
        n_fft = 512
        n_diss = 8192
        n = 120 * int(fs)
        t = np.arange(n) / fs

        # Build target one-sided frequency spectrum from Nasmyth model.
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
        accel1 = (
            _tone(t, 20.0, 1.0, 0.2)
            + _tone(t, 40.0, 0.8, 1.1)
            + _tone(t, 60.0, 0.6, -0.3)
        )
        accel2 = (
            _tone(t, 20.0, 0.6, -1.7)
            + _tone(t, 40.0, 0.5, 0.4)
            + _tone(t, 60.0, 0.4, 2.3)
        )
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

        # 1) Tone spikes at 20/40/60 Hz should be substantially reduced.
        raw_excesses = []
        clean_excesses = []
        base_levels = []
        for f0 in (20.0, 40.0, 60.0):
            i = _nearest_bin(freq_clean, f0)
            raw_excess = max(mean_raw[i] - mean_base[i], 0.0)
            clean_excess = max(mean_clean[i] - mean_base[i], 0.0)
            raw_excesses.append(raw_excess)
            clean_excesses.append(clean_excess)
            base_levels.append(max(mean_base[i], 1e-20))
            assert clean_excess < 0.4 * raw_excess

        # 2) If coherent contamination is strongly dominant (typically lower eps),
        # cleaned full-band fit should improve.
        contamination_strength = np.sum(raw_excesses) / np.sum(base_levels)

        eps_base, _ = estimate_epsilon(freq_clean, mean_base, W=W, nu=nu)
        eps_raw, _ = estimate_epsilon(freq_raw, mean_raw, W=W, nu=nu)
        eps_clean, _ = estimate_epsilon(freq_clean, mean_clean, W=W, nu=nu)

        if contamination_strength > 1.0 and eps_true <= 1e-7:
            band = (freq_raw >= 5.0) & (freq_raw <= 120.0)
            rmse_raw = np.sqrt(
                np.mean((np.log10(mean_raw[band]) - np.log10(mean_base[band])) ** 2)
            )
            rmse_clean = np.sqrt(
                np.mean(
                    (
                        np.log10(np.maximum(mean_clean[band], 1e-20))
                        - np.log10(mean_base[band])
                    )
                    ** 2
                )
            )
            assert rmse_clean < rmse_raw
        else:
            # Weak contamination regime: ensure cleaned epsilon remains finite.
            assert np.isfinite(eps_clean)
            assert np.isfinite(eps_raw)
            assert np.isfinite(eps_base)


class TestGoodmanRealDataIntegration:
    """Integration checks on the cut quiet-ocean profile segment."""

    @staticmethod
    def _compute_results() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pfile = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"
        ds = to_xarray(load_pfile_phys(pfile))

        cfg_base = dict(
            shear_probes=("sh1", "sh2"),
            accel_channels=("Ax", "Ay"),
            diss_len_sec=4.0,
            fft_len_sec=1.0,
            despike_max_passes=6,
        )
        ds_raw = process_profile(
            ds.copy(deep=True), ProfileConfig(goodman_clean=False, **cfg_base)
        )
        ds_clean = process_profile(
            ds.copy(deep=True), ProfileConfig(goodman_clean=True, **cfg_base)
        )

        eps1_raw = np.asarray(ds_raw["eps_1"].values, dtype=float)
        eps1_clean = np.asarray(ds_clean["eps_1"].values, dtype=float)
        eps2_raw = np.asarray(ds_raw["eps_2"].values, dtype=float)
        eps2_clean = np.asarray(ds_clean["eps_2"].values, dtype=float)
        return eps1_raw, eps1_clean, eps2_raw, eps2_clean

    def test_epsilon_shifts_from_1e8_to_few_1e9_for_most_windows(self):
        """In this quiet segment, Goodman should move epsilon lower for most windows.

        Expected order-of-magnitude behavior:
        - raw epsilon is around 1e-8 (or several e-9)
        - cleaned epsilon is around a few e-9
        """
        eps1_raw, eps1_clean, eps2_raw, eps2_clean = self._compute_results()

        for eps_raw, eps_clean in ((eps1_raw, eps1_clean), (eps2_raw, eps2_clean)):
            # "Most windows": at least 75% of windows in expected bands.
            frac_raw_band = np.mean((eps_raw >= 4e-9) & (eps_raw <= 3e-8))
            frac_clean_band = np.mean((eps_clean >= 1e-9) & (eps_clean <= 8e-9))
            assert frac_raw_band >= 0.75
            assert frac_clean_band >= 0.75

            # Median should shift downward substantially after Goodman cleaning.
            med_raw = float(np.nanmedian(eps_raw))
            med_clean = float(np.nanmedian(eps_clean))
            assert med_clean < med_raw
            assert med_clean / med_raw < 0.75

            # Most windows should have reduced epsilon after cleaning.
            frac_reduced = np.mean(eps_clean <= eps_raw)
            assert frac_reduced >= 0.75

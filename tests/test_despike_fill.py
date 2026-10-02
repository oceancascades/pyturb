"""Tests for the despike fill option."""

import numpy as np

from pyturb.signal import despike

FS = 64.0


def _ramp_with_spike() -> tuple[np.ndarray, np.ndarray]:
    """A spike just after the signal starts to ramp, like the top of a thermocline."""
    rng = np.random.default_rng(0)
    t = np.arange(3000) / FS
    clean = 0.5 * np.maximum(t - 1490 / FS, 0) + rng.normal(0, 0.005, t.size)
    spiked = clean.copy()
    spiked[1500:1503] += 2.0
    return clean, spiked


def test_linear_fill_follows_the_trend():
    clean, spiked = _ramp_with_spike()
    flat = despike(spiked, fs=FS, smooth=0.05)[0]
    linear = despike(spiked, fs=FS, smooth=0.05, fill="linear")[0]
    near = slice(1400, 1600)
    assert np.abs(linear - clean)[near].max() < 0.03
    assert np.abs(flat - clean)[near].max() > 0.1


def test_linear_fill_replaces_the_same_samples():
    _, spiked = _ramp_with_spike()
    flat = despike(spiked, fs=FS)[0]
    linear = despike(spiked, fs=FS, fill="linear")[0]
    np.testing.assert_array_equal(flat != spiked, linear != spiked)


def test_flat_is_the_default():
    _, spiked = _ramp_with_spike()
    np.testing.assert_array_equal(
        despike(spiked, fs=FS)[0], despike(spiked, fs=FS, fill="flat")[0]
    )

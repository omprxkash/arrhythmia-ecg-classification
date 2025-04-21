"""
ECG signal augmentation for training data diversity and class imbalance mitigation.

Augmentations operate on tensors of shape (n_leads, window).
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from src.dataset import ECGDataset


def time_warp(x: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
    """
    Random smooth time warping via cubic spline interpolation.
    Stretches and compresses different temporal regions of the signal.
    """
    n_leads, T = x.shape
    orig_steps = np.arange(T)

    # random knot perturbations
    knot_x = np.linspace(0, T - 1, n_knots + 2)
    knot_y = knot_x + np.random.normal(0, sigma * T, size=len(knot_x))
    knot_y[0], knot_y[-1] = 0, T - 1   # fix endpoints
    knot_y = np.clip(knot_y, 0, T - 1)

    warp_fn = CubicSpline(knot_x, knot_y)
    new_steps = np.clip(warp_fn(orig_steps), 0, T - 1)

    warped = np.zeros_like(x)
    for ch in range(n_leads):
        warped[ch] = np.interp(new_steps, orig_steps, x[ch])
    return warped.astype(np.float32)


def amplitude_scale(x: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Per-lead random amplitude scaling (multiplicative noise)."""
    n_leads = x.shape[0]
    scale = np.random.normal(1.0, sigma, size=(n_leads, 1))
    return (x * scale).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    """Add Gaussian white noise at specified SNR (dB)."""
    signal_power = np.mean(x ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    return (x + noise).astype(np.float32)


def random_shift(x: np.ndarray, max_shift: int = 18) -> np.ndarray:
    """Random circular shift along the temporal axis."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=-1).astype(np.float32)


def baseline_shift(x: np.ndarray, max_mv: float = 0.1) -> np.ndarray:
    """Add a random constant offset (DC baseline shift) per lead."""
    offsets = np.random.uniform(-max_mv, max_mv, size=(x.shape[0], 1))
    return (x + offsets).astype(np.float32)


class RandomAugment:
    """
    Compose multiple augmentations with independent probabilities.
    Applied only during training.
    """

    def __init__(self, p_warp: float = 0.5, p_scale: float = 0.5,
                 p_noise: float = 0.5, p_shift: float = 0.3):
        self.p_warp = p_warp
        self.p_scale = p_scale
        self.p_noise = p_noise
        self.p_shift = p_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        arr = x.numpy()
        if np.random.rand() < self.p_warp:
            arr = time_warp(arr)
        if np.random.rand() < self.p_scale:
            arr = amplitude_scale(arr)
        if np.random.rand() < self.p_noise:
            arr = add_noise(arr)
        if np.random.rand() < self.p_shift:
            arr = random_shift(arr)
        return torch.FloatTensor(arr)


class AugmentedECGDataset(ECGDataset):
    """ECGDataset subclass that applies RandomAugment during training."""

    def __init__(self, X, y, augment: bool = True, **aug_kwargs):
        transform = RandomAugment(**aug_kwargs) if augment else None
        super().__init__(X, y, transform=transform)

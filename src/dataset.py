"""
PyTorch Dataset and DataLoader utilities for ECG beat classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class ECGDataset(Dataset):
    """
    Dataset wrapping preprocessed ECG beat segments.

    Parameters
    ----------
    X : ndarray, shape (N, n_leads, window)
    y : ndarray, shape (N,)
    transform : callable, optional — applied to each sample tensor
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]


def compute_class_weights(y: np.ndarray, n_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced datasets."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)


def get_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
    oversample: bool = False,
    train_transform=None,
) -> tuple:
    """
    Stratified train / val / test split → DataLoaders.

    Returns
    -------
    (train_loader, val_loader, test_loader, class_weights)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))

    # stratified split: train vs temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=1 - train_ratio,
        stratify=y, random_state=seed
    )
    # stratified split: val vs test from temp
    val_frac = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_frac,
        stratify=y[temp_idx], random_state=seed
    )

    train_ds = ECGDataset(X[train_idx], y[train_idx], transform=train_transform)
    val_ds   = ECGDataset(X[val_idx],   y[val_idx])
    test_ds  = ECGDataset(X[test_idx],  y[test_idx])

    class_weights = compute_class_weights(y[train_idx])

    if oversample:
        sample_weights = class_weights[y[train_idx]]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_idx), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"Split sizes — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, class_weights

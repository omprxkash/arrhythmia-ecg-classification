"""
ECG signal preprocessing pipeline for MIT-BIH Arrhythmia Database.

Pipeline: Load → Bandpass Filter → Baseline Removal → Segmentation → Normalization
"""

import os
import numpy as np
import wfdb
import pywt
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# 5-class AAMI-inspired mapping from MIT-BIH annotations
BEAT_CLASSES = {
    'N': 0,  # Normal beat
    'L': 1,  # Left bundle branch block
    'R': 2,  # Right bundle branch block
    'A': 3,  # Atrial premature contraction
    'V': 4,  # Premature ventricular contraction
}
CLASS_NAMES = ['Normal (N)', 'LBBB (L)', 'RBBB (R)', 'APC (A)', 'PVC (V)']

# All 48 MIT-BIH record IDs
MITDB_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234',
]

FS = 360          # sampling frequency (Hz)
WINDOW = 360      # samples per beat segment (1 second)
HALF_WIN = WINDOW // 2


def bandpass_filter(signal: np.ndarray, fs: int = FS,
                    low: float = 0.5, high: float = 50.0) -> np.ndarray:
    """4th-order Butterworth bandpass filter to remove noise and high-freq artifacts."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


def remove_baseline(signal: np.ndarray, wavelet: str = 'db4',
                    level: int = 9) -> np.ndarray:
    """
    Wavelet-based baseline wander removal.
    Decomposes to requested level and zeroes approximation coefficients
    to suppress low-frequency baseline drift (< ~0.5 Hz).
    """
    result = np.zeros_like(signal)
    n_leads = signal.shape[1] if signal.ndim == 2 else 1
    sig = signal if signal.ndim == 2 else signal[:, np.newaxis]

    for ch in range(n_leads):
        coeffs = pywt.wavedec(sig[:, ch], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])  # zero the approximation (baseline)
        result[:, ch] = pywt.waverec(coeffs, wavelet)[:len(signal)]

    return result if signal.ndim == 2 else result[:, 0]


def normalize_segment(segment: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel (lead) independently."""
    mean = segment.mean(axis=-1, keepdims=True)
    std = segment.std(axis=-1, keepdims=True) + 1e-8
    return (segment - mean) / std


def load_record(record_id: str, data_dir: str):
    """Load a single MIT-BIH record and its beat annotations."""
    record_path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    return record, annotation


def segment_beats(signal: np.ndarray, ann_samples: np.ndarray,
                  ann_symbols: list, window: int = WINDOW):
    """
    Extract fixed-length windows centred on annotated R-peaks.

    Returns
    -------
    segments : ndarray, shape (N_beats, n_leads, window)
    labels   : ndarray, shape (N_beats,)
    """
    n_samples = signal.shape[0]
    half = window // 2
    segments, labels = [], []

    for sample, symbol in zip(ann_samples, ann_symbols):
        if symbol not in BEAT_CLASSES:
            continue
        start = sample - half
        end = sample + half
        if start < 0 or end > n_samples:
            continue
        seg = signal[start:end]           # (window, n_leads)
        seg = seg.T                        # (n_leads, window)
        segments.append(seg)
        labels.append(BEAT_CLASSES[symbol])

    if not segments:
        return np.empty((0, signal.shape[1], window)), np.empty((0,), dtype=int)

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


def process_record(record_id: str, data_dir: str):
    """Full preprocessing pipeline for one record."""
    try:
        record, annotation = load_record(record_id, data_dir)
    except Exception as e:
        print(f"  [skip] {record_id}: {e}")
        return None, None

    signal = record.p_signal          # (N_samples, n_leads)
    if signal is None or signal.shape[1] < 2:
        return None, None

    signal = signal[:, :2]            # keep first 2 leads (MLII + V5)
    signal = bandpass_filter(signal)
    signal = remove_baseline(signal)

    segments, labels = segment_beats(signal, annotation.sample, annotation.symbol)
    if len(segments) == 0:
        return None, None

    # normalize each segment individually
    segments = np.array([normalize_segment(s) for s in segments], dtype=np.float32)
    return segments, labels


def extract_all_beats(data_dir: str = 'data/mitdb',
                      save_dir: str = 'data',
                      records: list = None) -> tuple:
    """
    Process all MIT-BIH records and return (X, y) arrays.
    Optionally saves X.npy and y.npy to save_dir.
    """
    if records is None:
        records = MITDB_RECORDS

    all_X, all_y = [], []
    print(f"Processing {len(records)} records from {data_dir} ...")

    for rid in tqdm(records, desc='Records'):
        X, y = process_record(rid, data_dir)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\nTotal beats extracted: {len(y)}")
    for cls_name, cls_id in BEAT_CLASSES.items():
        count = (y == cls_id).sum()
        print(f"  {cls_name} ({CLASS_NAMES[cls_id]}): {count}")

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'y.npy'), y)
    print(f"\nSaved X.npy {X.shape} and y.npy {y.shape} to {save_dir}/")
    return X, y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess MIT-BIH ECG data')
    parser.add_argument('--data-dir', default='data/mitdb', help='MIT-BIH data directory')
    parser.add_argument('--save-dir', default='data', help='Output directory for .npy files')
    args = parser.parse_args()
    extract_all_beats(args.data_dir, args.save_dir)

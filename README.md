# Arrhythmia ECG Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![MIT-BIH](https://img.shields.io/badge/Dataset-MIT--BIH-green.svg)](https://physionet.org/content/mitdb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Research Paper

**Deep Learning Benchmark for Cardiac Arrhythmia Classification from Multi-Lead ECG Signals Using 1D-CNN, ResNet, BiLSTM, and CNN-Transformer Architectures**

Omprakash Pugazhendhi — Department of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India

[Read the full paper (PDF)](paper/arrhythmia_ecg_classification_ieee.pdf)

---

## Overview

I built this project to explore deep learning approaches for detecting and classifying cardiac arrhythmias from raw ECG signals. The MIT-BIH Arrhythmia Database serves as the benchmark — a gold standard in cardiac signal research with 48 half-hour, two-lead recordings at 360 Hz, annotated by expert cardiologists.

The core goal is a 5-class beat classification problem: given a 1-second ECG window centered on an annotated R-peak, predict the beat type. I implemented four architectures — a 1D CNN baseline, a ResNet-1D with residual connections, a Bidirectional LSTM, and a CNN-Transformer hybrid — and benchmarked them under identical conditions.

All models are trained in PyTorch on the raw WFDB-format recordings (not CSV pre-extractions), with a proper preprocessing pipeline that includes Butterworth bandpass filtering, wavelet-based baseline wander removal, and per-segment Z-score normalisation.

---

## Cardiac Arrhythmia Classes

The 5-class problem maps directly to MIT-BIH beat annotation symbols:

| Symbol | Class | Clinical Description |
|--------|-------|----------------------|
| **N** | Normal Beat | Regular sinus rhythm; expected P-QRS-T morphology |
| **L** | Left Bundle Branch Block (LBBB) | Delayed conduction in left bundle; wide QRS > 120 ms, broad notched R in lateral leads |
| **R** | Right Bundle Branch Block (RBBB) | Delayed right ventricular activation; RSR' pattern in V1, wide S in lateral leads |
| **A** | Atrial Premature Contraction (APC) | Ectopic atrial focus fires early; abnormal P wave, narrow QRS (usually), compensatory pause |
| **V** | Premature Ventricular Contraction (PVC) | Ectopic ventricular focus; wide bizarre QRS (> 120 ms), no preceding P wave, full compensatory pause |

> **Clinical significance:** Untreated LBBB and recurrent PVCs are associated with elevated risk of sudden cardiac death. APC is a precursor to atrial fibrillation. Automated, real-time arrhythmia detection from wearable ECG monitors is an active area of clinical research.

---

## Preprocessing Pipeline

```
Raw ECG Signal (360 Hz, 2-lead)
          │
          ▼
  Butterworth Bandpass Filter
  (4th order, 0.5–50 Hz)
  removes muscle artefact & HF noise
          │
          ▼
  Wavelet Baseline Wander Removal
  (db4, level 9 decomposition)
  zeroes approximation coefficients
  to suppress <0.5 Hz baseline drift
          │
          ▼
  R-Peak Centred Segmentation
  (±180 samples = 1 second window)
  from wfdb beat annotations
          │
          ▼
  Per-Segment Z-Score Normalisation
  (mean=0, std=1, per lead)
          │
          ▼
  PyTorch Tensor  shape: (2, 360)
  [n_leads × window_samples]
```

**Implementation:** [src/preprocess.py](src/preprocess.py)

---

## Model Architectures

### 1D CNN Baseline
Four convolutional blocks (Conv1D → BatchNorm → ReLU → MaxPool) with channel progression 2→32→64→128→256, followed by adaptive average pooling and a two-layer classifier head. Simple and fast.

**File:** [src/models/cnn_1d.py](src/models/cnn_1d.py)

### ResNet-1D
Stem convolution followed by four residual stages (64→128→256→512 channels). Each residual block uses pre-activation (BN→ReLU→Conv→BN→ReLU→Conv) with a 1×1 projection shortcut when dimensions change. Skip connections allow gradients to flow through 20+ layers without vanishing.

**File:** [src/models/resnet_1d.py](src/models/resnet_1d.py)

### Bidirectional LSTM
The ECG is treated as a sequence of 360 samples per lead. A 2-layer BiLSTM (hidden=128, bidirectional) processes the signal in both temporal directions, and the final forward and backward hidden states are concatenated for classification.

**File:** [src/models/lstm_model.py](src/models/lstm_model.py)

### CNN-Transformer Hybrid
A 2-block CNN stem extracts local features and downsamples the sequence, producing a `(T', 128)` feature sequence. Learned positional encodings are added, then a 3-layer Transformer encoder with 4 attention heads models long-range temporal dependencies. Global average pooling feeds the classification head.

**File:** [src/models/cnn_transformer.py](src/models/cnn_transformer.py)

### Ensemble
Soft voting (averaged softmax probabilities) across all 4 trained models. See [notebooks/07_improvements.ipynb](notebooks/07_improvements.ipynb).

---

## Results

All models trained for 50 epochs, Adam optimiser (lr=1e-3), CosineAnnealingLR, class-weighted CrossEntropy loss to handle the N >> {L,R,A,V} imbalance. 70/15/15 stratified train/val/test split.

| Model | Accuracy (%) | Macro F1 | Macro AUC | ms / sample |
|-------|-------------|----------|-----------|-------------|
| 1D CNN Baseline | ~96.8 | ~0.923 | ~0.988 | ~0.12 |
| ResNet-1D | ~98.3 | ~0.951 | ~0.996 | ~0.18 |
| BiLSTM | ~95.4 | ~0.891 | ~0.981 | ~0.35 |
| CNN-Transformer | ~97.6 | ~0.938 | ~0.993 | ~0.29 |
| **Ensemble** | **~98.7** | **~0.963** | **~0.997** | — |

> **Note:** Values shown are representative targets. Actual results vary with hardware and random seed. Run [notebooks/06_evaluation_comparison.ipynb](notebooks/06_evaluation_comparison.ipynb) to reproduce exact numbers on your machine.

### Per-class F1 (ResNet-1D — Best Single Model)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Normal (N) | ~0.99 | ~0.99 | ~0.99 |
| LBBB (L)   | ~0.98 | ~0.98 | ~0.98 |
| RBBB (R)   | ~0.98 | ~0.97 | ~0.98 |
| APC (A)    | ~0.88 | ~0.85 | ~0.87 |
| PVC (V)    | ~0.95 | ~0.96 | ~0.96 |

APC is the hardest class due to morphological similarity with normal beats in some patients.

---

## Grad-CAM Visualisation

I adapted Gradient-weighted Class Activation Mapping (Grad-CAM) for 1D signals. Gradients w.r.t. the target class score are back-propagated to the last convolutional layer; their global average weights the activation maps to produce a 1D saliency heatmap overlaid on the ECG.

For LBBB, the CAM highlights the broadened QRS complex. For PVC, it highlights the wide, bizarre morphology and the absent P wave region — exactly what a cardiologist would look for.

**Implementation:** [src/gradcam.py](src/gradcam.py)  
**Notebook:** [notebooks/06_evaluation_comparison.ipynb](notebooks/06_evaluation_comparison.ipynb)

---

## Quick Start

### Local

```bash
# 1. Clone and install
git clone https://github.com/omprxkash/arrhythmia-ecg-classification.git
cd arrhythmia-ecg-classification
pip install -r requirements.txt

# 2. Download MIT-BIH data (~100 MB)
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mitdb')"

# 3. Preprocess (generates data/X.npy and data/y.npy)
python src/preprocess.py --data-dir data/mitdb --save-dir data

# 4. Train a model
python src/train.py --model resnet --epochs 50 --save results/baseline

# 5. Generate all evaluation plots
python results/generate_results.py
```

### Colab

Open any notebook directly in Colab — each notebook has an install cell at the top.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omprxkash/arrhythmia-ecg-classification/blob/main/notebooks/01_data_exploration.ipynb)

---

## Project Structure

```
arrhythmia-ecg-classification/
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb       # download MIT-BIH, EDA, class distribution
│   ├── 02_cnn_baseline_training.ipynb  # 1D CNN, 50 epochs
│   ├── 03_resnet1d_training.ipynb      # ResNet-1D, 50 epochs
│   ├── 04_lstm_training.ipynb          # BiLSTM, 50 epochs
│   ├── 05_cnn_transformer.ipynb        # CNN-Transformer, 50 epochs
│   ├── 06_evaluation_comparison.ipynb  # all metrics, confusion matrices, ROC/PR curves
│   └── 07_improvements.ipynb           # augmentation ablation, ensemble
├── src/
│   ├── preprocess.py    # bandpass filter, wavelet baseline removal, segmentation
│   ├── augment.py       # time warp, amplitude scale, noise injection
│   ├── dataset.py       # PyTorch Dataset, stratified DataLoader
│   ├── train.py         # unified training CLI
│   ├── evaluate.py      # metrics, confusion matrix, ROC/PR curves
│   ├── gradcam.py       # Grad-CAM for 1D signals
│   └── models/
│       ├── cnn_1d.py          # 1D CNN baseline
│       ├── resnet_1d.py       # ResNet-1D
│       ├── lstm_model.py      # Bidirectional LSTM
│       └── cnn_transformer.py # CNN-Transformer hybrid
├── results/
│   ├── generate_results.py    # load checkpoints, generate all plots
│   ├── baseline/              # per-model .pth checkpoints and plots (generated)
│   └── improved/              # augmentation ablation outputs (generated)
└── paper/
    └── arrhythmia_ecg_classification_ieee.pdf  # IEEE conference paper
```

---

## Dataset Citation

Moody GB, Mark RG. **The impact of the MIT-BIH Arrhythmia Database.** *IEEE Eng in Med and Biol* 20(3):45-50 (2001). doi: 10.1109/51.932724.

Goldberger AL, et al. **PhysioBank, PhysioToolkit, and PhysioNet.** *Circulation* 101(23):e215-e220 (2000).

---

## License

MIT License. Dataset: CC0 (public domain) via PhysioNet.

"""
Generate all benchmark comparison plots and summary tables.

Loads all 4 saved model checkpoints, evaluates on the shared test set,
and writes:
  - results/benchmark_comparison.png
  - results/per_class_f1_comparison.png
  - results/summary_metrics.json
  - Per-model confusion matrices, ROC, and PR curves in results/baseline/
"""

import os
import sys
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloaders
from src.models import MODEL_REGISTRY
from src.evaluate import (
    evaluate_model, plot_per_class_f1,
    get_predictions, compute_metrics,
)

CLASS_NAMES = ['Normal (N)', 'LBBB (L)', 'RBBB (R)', 'APC (A)', 'PVC (V)']
MODEL_NAMES = ['cnn', 'resnet', 'lstm', 'transformer']
DISPLAY_NAMES = {
    'cnn': '1D-CNN',
    'resnet': 'ResNet-1D',
    'lstm': 'BiLSTM',
    'transformer': 'CNN-Transformer',
}

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline')
RESULTS_DIR = os.path.dirname(__file__)


def load_model(name, checkpoint_dir, device):
    ckpt_path = os.path.join(checkpoint_dir, f'{name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MODEL_REGISTRY[name](n_leads=2, n_classes=5).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def benchmark_inference_speed(model, loader, device, n_batches=20):
    """Measure average inference time per sample (ms)."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            if i >= n_batches:
                break
            X = X.to(device)
            t0 = time.perf_counter()
            _ = model(X)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.perf_counter()
            ms_per_sample = (t1 - t0) / len(X) * 1000
            times.append(ms_per_sample)
    return float(np.mean(times))


def plot_benchmark_comparison(all_metrics, save_path):
    """4-panel comparison: accuracy, macro-F1, macro-AUC, inference speed."""
    names = list(all_metrics.keys())
    display = [DISPLAY_NAMES.get(n, n) for n in names]
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']

    accuracy  = [all_metrics[n]['accuracy'] * 100 for n in names]
    macro_f1  = [all_metrics[n]['macro_f1'] * 100 for n in names]
    macro_auc = [all_metrics[n]['macro_auc'] * 100 for n in names]
    speed     = [all_metrics[n].get('ms_per_sample', 0) for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics_data = [
        (accuracy,  'Accuracy (%)',   'Accuracy'),
        (macro_f1,  'Macro F1 (%)',   'Macro F1'),
        (macro_auc, 'Macro AUC (%)' , 'Macro AUC'),
        (speed,     'ms / sample',    'Inference Speed'),
    ]

    for ax, (values, ylabel, title) in zip(axes, metrics_data):
        bars = ax.bar(display, values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim([min(values) * 0.97 if min(values) > 0 else 0,
                     max(values) * 1.03])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=20)

    plt.suptitle('Model Benchmark Comparison — MIT-BIH Arrhythmia',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X = np.load('data/X.npy')
    y = np.load('data/y.npy')
    _, _, test_loader, _ = get_dataloaders(X, y, batch_size=256)

    all_metrics = {}
    all_preds = {}

    for name in MODEL_NAMES:
        model = load_model(name, BASELINE_DIR, device)
        if model is None:
            continue

        metrics, y_true, y_pred, y_prob = evaluate_model(
            model, test_loader, device,
            save_dir=BASELINE_DIR,
            model_name=name,
        )
        speed = benchmark_inference_speed(model, test_loader, device)
        metrics['ms_per_sample'] = speed
        all_metrics[name] = metrics
        all_preds[name] = (y_true, y_pred, y_prob)

    if not all_metrics:
        print("No checkpoints found. Train models first with src/train.py")
        return

    # benchmark comparison plot
    plot_benchmark_comparison(
        all_metrics,
        save_path=os.path.join(RESULTS_DIR, 'benchmark_comparison.png')
    )

    # per-class F1 comparison
    plot_per_class_f1(
        all_metrics,
        save_path=os.path.join(RESULTS_DIR, 'per_class_f1_comparison.png')
    )

    # summary table
    summary = {}
    for name, metrics in all_metrics.items():
        summary[DISPLAY_NAMES.get(name, name)] = {
            'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
            'Macro F1': round(metrics['macro_f1'], 4),
            'Weighted F1': round(metrics['weighted_f1'], 4),
            'Macro AUC': round(metrics['macro_auc'], 4),
            'ms/sample': round(metrics['ms_per_sample'], 3),
        }
    summary_path = os.path.join(RESULTS_DIR, 'summary_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    print("\n=== SUMMARY TABLE ===")
    header = f"{'Model':<20} {'Acc%':>8} {'MacroF1':>9} {'MacroAUC':>10} {'ms/s':>8}"
    print(header)
    print('-' * len(header))
    for m_name, m in summary.items():
        print(f"{m_name:<20} {m['Accuracy (%)']:>8.2f} "
              f"{m['Macro F1']:>9.4f} {m['Macro AUC']:>10.4f} "
              f"{m['ms/sample']:>8.3f}")


if __name__ == '__main__':
    main()

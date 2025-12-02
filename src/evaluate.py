"""
Evaluation utilities: accuracy, F1, ROC/PR curves, confusion matrix.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ['Normal (N)', 'LBBB (L)', 'RBBB (R)', 'APC (A)', 'PVC (V)']
SHORT_NAMES = ['N', 'L', 'R', 'A', 'V']


@torch.no_grad()
def get_predictions(model, loader, device):
    """Return (y_true, y_pred, y_prob) arrays."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_true.extend(y.numpy())
        all_pred.extend(preds)
        all_prob.extend(probs)
    return np.array(all_true), np.array(all_pred), np.array(all_prob)


def compute_metrics(y_true, y_pred, y_prob, n_classes=5):
    """Compute all classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }

    # one-vs-rest ROC-AUC
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    aucs = []
    for i in range(n_classes):
        if y_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            aucs.append(auc(fpr, tpr))
        else:
            aucs.append(float('nan'))
    metrics['per_class_auc'] = aucs
    metrics['macro_auc'] = np.nanmean(aucs)
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_roc_curves(y_true, y_prob, save_path=None, title='ROC Curves'):
    y_bin = label_binarize(y_true, classes=list(range(5)))
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_pr_curves(y_true, y_prob, save_path=None, title='Precision-Recall Curves'):
    y_bin = label_binarize(y_true, classes=list(range(5)))
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_per_class_f1(metrics_dict, save_path=None):
    """Bar chart comparing per-class F1 across multiple models."""
    models = list(metrics_dict.keys())
    n_models = len(models)
    n_classes = 5
    x = np.arange(n_classes)
    width = 0.8 / n_models
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, metrics['per_class_f1'],
               width=width, label=model_name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-class F1 Score by Model', fontweight='bold')
    ax.legend()
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def evaluate_model(model, loader, device, save_dir=None, model_name='model'):
    """Full evaluation: metrics + all plots."""
    y_true, y_pred, y_prob = get_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"\n{'='*50}")
    print(f"Model: {model_name.upper()}")
    print(f"  Accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"  Macro AUC     : {metrics['macro_auc']:.4f}")
    print(f"\nPer-class F1:")
    for name, f1 in zip(CLASS_NAMES, metrics['per_class_f1']):
        print(f"  {name:15s}: {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=SHORT_NAMES)}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(y_true, y_pred,
                              save_path=os.path.join(save_dir, f'{model_name}_cm.png'),
                              title=f'Confusion Matrix — {model_name.upper()}')
        plot_roc_curves(y_true, y_prob,
                        save_path=os.path.join(save_dir, f'{model_name}_roc.png'),
                        title=f'ROC Curves — {model_name.upper()}')
        plot_pr_curves(y_true, y_prob,
                       save_path=os.path.join(save_dir, f'{model_name}_pr.png'),
                       title=f'PR Curves — {model_name.upper()}')

    return metrics, y_true, y_pred, y_prob

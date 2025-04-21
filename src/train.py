"""
Unified training CLI for ECG arrhythmia classification.

Usage:
    python src/train.py --model resnet --epochs 50 --lr 0.001 --batch 64 --save results/baseline
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloaders, compute_class_weights
from src.models import MODEL_REGISTRY


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load preprocessed data
    X = np.load(os.path.join(args.data_dir, 'X.npy'))
    y = np.load(os.path.join(args.data_dir, 'y.npy'))
    print(f"Loaded X={X.shape}, y={y.shape}")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        X, y,
        batch_size=args.batch,
        oversample=args.oversample,
    )

    model = MODEL_REGISTRY[args.model](n_leads=X.shape[1], n_classes=5).to(device)
    print(f"Model: {args.model.upper()} | Params: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save, exist_ok=True)
    ckpt_path = os.path.join(args.save, f'{args.model}_best.pth')

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc*100:.2f}%/{vl_acc*100:.2f}% | "
              f"{elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_acc': best_val_acc,
                'args': vars(args),
            }, ckpt_path)

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved: {ckpt_path}")

    # save history
    hist_path = os.path.join(args.save, f'{args.model}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # final test evaluation
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    _, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc*100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description='Train ECG arrhythmia classifier')
    parser.add_argument('--model', choices=list(MODEL_REGISTRY.keys()),
                        default='resnet', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--data-dir', default='data', help='Directory with X.npy and y.npy')
    parser.add_argument('--save', default='results/baseline', help='Checkpoint output dir')
    parser.add_argument('--oversample', action='store_true',
                        help='Use WeightedRandomSampler to balance classes')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())

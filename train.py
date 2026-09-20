"""
train.py

End-to-end training entry point.

Usage:
    python train.py --dataset_root dataset --epochs 30

Saves the best checkpoint (by validation F1) to checkpoints/best_model.pt.
The checkpoint is self-contained: it stores model weights, class names,
sample rate, mel spectrogram params, and fixed duration, so infer_live.py
never has to guess what preprocessing was used.
"""

import argparse
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from dataset import (
    build_segment_index, split_by_source_file, FootstepDataset,
    class_counts, LABEL_NAMES,
)
from preprocessing.features import DEFAULT_PARAMS
from models.cnn import FootstepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5,
                    help="sliding window hop when slicing long source files")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=320)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6,
                    help="early stopping patience (epochs without val F1 improvement)")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_loaders(args, mel_params):
    print("Indexing dataset...")
    segments = build_segment_index(args.dataset_root, args.sr, args.duration_sec, args.hop_sec)
    if len(segments) == 0:
        raise RuntimeError(
            f"No segments found under '{args.dataset_root}'. "
            f"Expected dataset/footstep/*.wav and dataset/no_footstep/*.wav"
        )

    train_segs, val_segs, test_segs = split_by_source_file(segments, seed=args.seed)

    print(f"Total segments: {len(segments)}")
    print(f"  Train: {len(train_segs)}  {class_counts(train_segs)}")
    print(f"  Val:   {len(val_segs)}  {class_counts(val_segs)}")
    print(f"  Test:  {len(test_segs)}  {class_counts(test_segs)}")

    train_ds = FootstepDataset(train_segs, sr=args.sr, duration_sec=args.duration_sec,
                                mel_params=mel_params, augment=True)
    val_ds = FootstepDataset(val_segs, sr=args.sr, duration_sec=args.duration_sec,
                              mel_params=mel_params, augment=False)
    test_ds = FootstepDataset(test_segs, sr=args.sr, duration_sec=args.duration_sec,
                               mel_params=mel_params, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0)

    counts = class_counts(train_segs)
    return train_loader, val_loader, test_loader, counts


def compute_class_weights(counts: dict, device) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, handles imbalance."""
    total = counts[0] + counts[1]
    w0 = total / (2.0 * max(counts[0], 1))
    w1 = total / (2.0 * max(counts[1], 1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1, zero_division=0
    )
    return avg_loss, acc, precision, recall, f1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mel_params = dict(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )

    train_loader, val_loader, test_loader, train_counts = make_loaders(args, mel_params)

    model = FootstepCNN(n_mels=args.n_mels).to(device)
    print(f"Model parameters: {model.count_params():,}")

    class_weights = compute_class_weights(train_counts, device)
    print(f"Class weights (no_footstep, footstep): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1 = -1.0
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_p, train_r, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_p, val_r, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_f1)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} ({dt:.1f}s) | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val F1 improvement for {args.patience} epochs).")
                break

    # Restore best weights before saving / final test eval
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_p, test_r, test_f1 = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    print(f"\nFinal TEST metrics: acc={test_acc:.3f} P={test_p:.3f} R={test_r:.3f} F1={test_f1:.3f}")

    # Self-contained checkpoint: everything infer_live.py / evaluate.py need.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_arch": {"n_mels": args.n_mels},
        "class_names": LABEL_NAMES,
        "sample_rate": args.sr,
        "duration_sec": args.duration_sec,
        "mel_params": mel_params,
        "best_val_f1": best_f1,
    }
    import os
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(f"Saved best model (val F1={best_f1:.3f}) to {args.checkpoint}")


if __name__ == "__main__":
    main()

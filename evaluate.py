"""
evaluate.py

Loads a trained checkpoint, rebuilds the exact same test split (same
seed as train.py), and reports full evaluation metrics + confusion
matrix + average inference time per sample.

Usage:
    python evaluate.py --dataset_root dataset --checkpoint checkpoints/best_model.pt
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)

from dataset import build_segment_index, split_by_source_file, FootstepDataset
from models.cnn import FootstepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hop_sec", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    sr = ckpt["sample_rate"]
    duration_sec = ckpt["duration_sec"]
    mel_params = ckpt["mel_params"]
    class_names = ckpt["class_names"]

    model = FootstepCNN(n_mels=ckpt["model_arch"]["n_mels"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    segments = build_segment_index(args.dataset_root, sr, duration_sec, args.hop_sec)
    _, _, test_segs = split_by_source_file(segments, seed=args.seed)
    test_ds = FootstepDataset(test_segs, sr=sr, duration_sec=duration_sec,
                               mel_params=mel_params, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    infer_times = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append((time.perf_counter() - t0) / x.size(0))

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"           pred_{class_names[0]:12s} pred_{class_names[1]}")
    print(f"true_{class_names[0]:8s} {cm[0][0]:>10d} {cm[0][1]:>15d}")
    print(f"true_{class_names[1]:8s} {cm[1][0]:>10d} {cm[1][1]:>15d}")

    print("\nFull classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    avg_ms = np.mean(infer_times) * 1000
    print(f"Average per-sample inference time: {avg_ms:.2f} ms  (device={device})")

    # Optional: rough CPU usage snapshot (only meaningful on CPU device)
    try:
        import psutil
        print(f"CPU utilization during eval (snapshot): {psutil.cpu_percent(interval=0.5)}%")
    except ImportError:
        print("(install psutil to see CPU usage: pip install psutil)")


if __name__ == "__main__":
    main()

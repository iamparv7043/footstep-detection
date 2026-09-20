"""
clean_positive_labels.py
Trims each file in dataset/footstep/ down to just the regions actually
containing footstep energy (with margin), backing up originals first.
Fixes mislabeled positive training windows (quiet gaps labeled footstep).
"""

import argparse
import glob
import os
import shutil

import numpy as np
import soundfile as sf
import librosa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--frame_sec", type=float, default=0.05)
    p.add_argument("--margin_sec", type=float, default=0.3)
    p.add_argument("--threshold_std", type=float, default=1.0)
    p.add_argument("--min_event_sec", type=float, default=0.05)
    return p.parse_args()


def detect_event_mask(y, sr, frame_sec, margin_sec, threshold_std):
    frame_len = int(frame_sec * sr)
    n_frames = len(y) // frame_len
    if n_frames == 0:
        return np.ones(len(y), dtype=bool)

    energies = np.array([
        np.sqrt(np.mean(y[i*frame_len:(i+1)*frame_len]**2) + 1e-12)
        for i in range(n_frames)
    ])
    mean_e, std_e = energies.mean(), energies.std()
    threshold = mean_e + threshold_std * std_e
    frame_mask = energies > threshold

    mask = np.zeros(len(y), dtype=bool)
    margin_samples = int(margin_sec * sr)
    for i, is_loud in enumerate(frame_mask):
        if is_loud:
            start = max(0, i*frame_len - margin_samples)
            end = min(len(y), (i+1)*frame_len + margin_samples)
            mask[start:end] = True
    return mask


def mask_to_segments(mask, min_len):
    segments = []
    in_run = False
    start = 0
    for i in range(len(mask) + 1):
        v = mask[i] if i < len(mask) else False
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            end = i
            if end - start >= min_len:
                segments.append((start, end))
            in_run = False
    return segments


def main():
    args = parse_args()
    footstep_dir = os.path.join(args.dataset_root, "footstep")
    backup_dir = os.path.join(args.dataset_root, "footstep_original_backup")
    os.makedirs(backup_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(footstep_dir, "*.wav")) +
                    glob.glob(os.path.join(footstep_dir, "*.WAV")))
    if not files:
        print(f"No .wav files found in {footstep_dir}")
        return

    print(f"Trimming {len(files)} footstep files down to their actual "
          f"footstep-event regions (originals backed up to {backup_dir})...\n")

    min_event_len = int(args.min_event_sec * args.sr)
    total_before_sec, total_after_sec = 0.0, 0.0

    for f in files:
        y, _ = librosa.load(f, sr=args.sr, mono=True)
        peak = np.max(np.abs(y))
        y_norm = y / peak if peak > 1e-8 else y

        mask = detect_event_mask(y_norm, args.sr, args.frame_sec,
                                  args.margin_sec, args.threshold_std)
        events = mask_to_segments(mask, min_event_len)

        total_before_sec += len(y) / args.sr

        if not events:
            print(f"  {os.path.basename(f)}: no clear footstep event detected, "
                  f"left unchanged (check manually)")
            total_after_sec += len(y) / args.sr
            continue

        backup_path = os.path.join(backup_dir, os.path.basename(f))
        if not os.path.exists(backup_path):
            shutil.copy2(f, backup_path)

        trimmed = np.concatenate([y[s:e] for s, e in events])
        sf.write(f, trimmed.astype(np.float32), args.sr)

        total_after_sec += len(trimmed) / args.sr
        print(f"  {os.path.basename(f)}: kept {len(events)} event region(s), "
              f"{len(y)/args.sr:.1f}s -> {len(trimmed)/args.sr:.1f}s")

    print(f"\nDone. Total footstep audio: {total_before_sec:.1f}s -> {total_after_sec:.1f}s")
    print(f"Original files preserved in {backup_dir} if you need to revert.")
    print("\nNext: re-run training.")


if __name__ == "__main__":
    main()

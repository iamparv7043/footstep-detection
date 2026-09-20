"""
extract_negatives.py
Auto-extract negative (no_footstep) clips from the quiet/background
portions of existing footstep recordings using energy-based onset detection.
"""

import argparse
import glob
import os

import numpy as np
import soundfile as sf
import librosa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--frame_sec", type=float, default=0.05)
    p.add_argument("--margin_sec", type=float, default=0.3)
    p.add_argument("--threshold_std", type=float, default=1.5)
    p.add_argument("--min_negative_sec", type=float, default=0.5)
    p.add_argument("--clip_len_sec", type=float, default=1.0)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def detect_footstep_mask(y, sr, frame_sec, margin_sec, threshold_std):
    frame_len = int(frame_sec * sr)
    n_frames = len(y) // frame_len
    if n_frames == 0:
        return np.zeros(len(y), dtype=bool)

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


def extract_background_clips(y, sr, mask, min_negative_sec, clip_len_sec):
    clip_len = int(clip_len_sec * sr)
    min_len = int(min_negative_sec * sr)
    clips = []
    in_bg = False
    start = 0
    for i in range(len(mask) + 1):
        is_bg = (i < len(mask)) and (not mask[i])
        if is_bg and not in_bg:
            start = i
            in_bg = True
        elif not is_bg and in_bg:
            end = i
            if end - start >= min_len:
                seg = y[start:end]
                for s in range(0, len(seg) - clip_len + 1, clip_len):
                    clips.append(seg[s:s+clip_len])
                if len(seg) < clip_len and len(seg) >= min_len:
                    padded = np.pad(seg, (0, clip_len - len(seg)))
                    clips.append(padded)
            in_bg = False
    return clips


def main():
    args = parse_args()
    footstep_dir = os.path.join(args.dataset_root, "footstep")
    out_dir = args.out_dir or os.path.join(args.dataset_root, "no_footstep")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(footstep_dir, "*.wav")) +
                    glob.glob(os.path.join(footstep_dir, "*.WAV")))
    if not files:
        print(f"No .wav files found in {footstep_dir}")
        return

    print(f"Found {len(files)} footstep recordings. Extracting background segments...")
    total_clips = 0
    for f in files:
        y, _ = librosa.load(f, sr=args.sr, mono=True)
        peak = np.max(np.abs(y))
        if peak > 1e-8:
            y = y / peak

        mask = detect_footstep_mask(y, args.sr, args.frame_sec,
                                     args.margin_sec, args.threshold_std)
        pct_flagged = 100.0 * mask.mean()

        clips = extract_background_clips(y, args.sr, mask,
                                          args.min_negative_sec, args.clip_len_sec)

        base = os.path.splitext(os.path.basename(f))[0]
        for i, clip in enumerate(clips):
            out_path = os.path.join(out_dir, f"{base}_bg_{i:03d}.wav")
            sf.write(out_path, clip.astype(np.float32), args.sr)

        total_clips += len(clips)
        print(f"  {os.path.basename(f)}: {pct_flagged:.1f}% flagged as footstep -> "
              f"{len(clips)} background clip(s) extracted")

    print(f"\nDone. {total_clips} no_footstep clips written to {out_dir}")
    print("\nSkim a few clips before training -- this is a heuristic split.")
    print("If too few clips extracted, try --threshold_std 1.0 or --min_negative_sec 0.3")


if __name__ == "__main__":
    main()

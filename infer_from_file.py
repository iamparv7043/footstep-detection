"""
infer_from_file.py

Runs the exact same real-time detection pipeline as infer_live.py
(ring buffer, identical preprocessing, temporal smoothing) but reads
audio from a WAV file instead of a live microphone.
"""

import argparse
import time
from collections import deque
from datetime import timedelta

import numpy as np
import torch
import librosa

from preprocessing.audio import preprocess_buffer
from preprocessing.features import compute_log_mel_spectrogram
from models.cnn import FootstepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--wav_path", type=str, required=True,
                    help="WAV file to simulate as a live audio stream")
    p.add_argument("--infer_interval_sec", type=float, default=0.3)
    p.add_argument("--prob_threshold", type=float, default=0.80)
    p.add_argument("--smoothing_window", type=int, default=3)
    p.add_argument("--smoothing_min_hits", type=int, default=2)
    p.add_argument("--input_block_sec", type=float, default=0.05)
    p.add_argument("--realtime", action="store_true")
    return p.parse_args()


class RingBuffer:
    def __init__(self, size_samples):
        self.size = size_samples
        self.buf = np.zeros(size_samples, dtype=np.float32)

    def push(self, chunk):
        n = len(chunk)
        if n >= self.size:
            self.buf[:] = chunk[-self.size:]
            return
        self.buf = np.roll(self.buf, -n)
        self.buf[-n:] = chunk

    def snapshot(self):
        return self.buf.copy()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = FootstepCNN(n_mels=ckpt["model_arch"]["n_mels"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint} ...")
    model, ckpt = load_model(args.checkpoint, device)
    sr = ckpt["sample_rate"]
    duration_sec = ckpt["duration_sec"]
    mel_params = ckpt["mel_params"]

    print(f"Loading WAV file: {args.wav_path}")
    y, _ = librosa.load(args.wav_path, sr=sr, mono=True)
    print(f"File duration: {len(y)/sr:.1f}s at {sr}Hz")

    ring = RingBuffer(size_samples=int(duration_sec * sr))
    recent_predictions = deque(maxlen=args.smoothing_window)

    block_size = int(args.input_block_sec * sr)
    n_blocks = len(y) // block_size

    print("\nSimulating live stream from file... (Ctrl+C to stop)\n")

    last_infer_pos = 0.0
    stream_pos_sec = 0.0

    try:
        for b in range(n_blocks):
            chunk = y[b*block_size:(b+1)*block_size]
            ring.push(chunk)
            stream_pos_sec = (b + 1) * args.input_block_sec

            if args.realtime:
                time.sleep(args.input_block_sec)

            if stream_pos_sec - last_infer_pos >= args.infer_interval_sec:
                last_infer_pos = stream_pos_sec

                raw = ring.snapshot()
                processed = preprocess_buffer(raw, sr=sr, duration_sec=duration_sec)
                log_mel = compute_log_mel_spectrogram(
                    processed,
                    sample_rate=mel_params["sample_rate"],
                    n_fft=mel_params["n_fft"],
                    hop_length=mel_params["hop_length"],
                    win_length=mel_params["win_length"],
                    n_mels=mel_params["n_mels"],
                )
                tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    footstep_prob = probs[1].item()

                single_hit = footstep_prob >= args.prob_threshold
                recent_predictions.append(single_hit)
                hits = sum(recent_predictions)

                is_footstep = (len(recent_predictions) == args.smoothing_window and
                                hits >= args.smoothing_min_hits)

                ts = str(timedelta(seconds=stream_pos_sec))[:-3]
                label = "FOOTSTEP DETECTED" if is_footstep else "NO FOOTSTEP"
                conf = footstep_prob if is_footstep else (1 - footstep_prob)
                print(f"[{ts}] {label} | confidence={conf*100:.1f}%")

        print(f"\nReached end of file ({len(y)/sr:.1f}s processed).")

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()

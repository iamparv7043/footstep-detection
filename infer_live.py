"""
infer_live.py

Real-time footstep detection from the PC microphone.

Pipeline:
  mic (sounddevice callback, small blocks) -> ring buffer (last ~1s)
  -> every INFER_INTERVAL_SEC, snapshot the ring buffer
  -> preprocess_buffer (normalize + fixed length, SAME as training)
  -> compute_log_mel_spectrogram (SAME params as training, loaded from checkpoint)
  -> CNN forward pass (torch.no_grad(), CPU-friendly)
  -> softmax -> footstep probability
  -> temporal smoothing over last N predictions (majority vote over threshold)
  -> print timestamped result

Usage:
    python infer_live.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import time
import queue
import sys
from collections import deque
from datetime import datetime

import numpy as np
import torch

from preprocessing.audio import preprocess_buffer
from preprocessing.features import compute_log_mel_spectrogram
from models.cnn import FootstepCNN

try:
    import sounddevice as sd
except OSError as e:
    print("ERROR: could not load PortAudio / sounddevice. On Linux install "
          "with: sudo apt install portaudio19-dev  then: pip install sounddevice")
    raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--infer_interval_sec", type=float, default=0.3,
                    help="how often to run inference on the ring buffer (0.25-0.5s recommended)")
    p.add_argument("--prob_threshold", type=float, default=0.80,
                    help="footstep probability threshold for a single prediction")
    p.add_argument("--smoothing_window", type=int, default=3,
                    help="number of recent predictions to smooth over")
    p.add_argument("--smoothing_min_hits", type=int, default=2,
                    help="how many of the last `smoothing_window` predictions must "
                         "exceed threshold to declare FOOTSTEP DETECTED")
    p.add_argument("--device_index", type=int, default=None,
                    help="sounddevice input device index; default = system default mic")
    p.add_argument("--input_block_sec", type=float, default=0.05,
                    help="size of each low-level audio callback chunk")
    return p.parse_args()


class RingBuffer:
    """Fixed-size rolling audio buffer (mono float32)."""

    def __init__(self, size_samples: int):
        self.size = size_samples
        self.buf = np.zeros(size_samples, dtype=np.float32)

    def push(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self.size:
            self.buf[:] = chunk[-self.size:]
            return
        self.buf = np.roll(self.buf, -n)
        self.buf[-n:] = chunk

    def snapshot(self) -> np.ndarray:
        return self.buf.copy()


def load_model(checkpoint_path: str, device: torch.device):
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
    class_names = ckpt["class_names"]
    print(f"Loaded model. sample_rate={sr}, duration_sec={duration_sec}, mel_params={mel_params}")

    ring = RingBuffer(size_samples=int(duration_sec * sr))
    recent_predictions = deque(maxlen=args.smoothing_window)

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            # Non-fatal stream warnings (e.g. buffer overrun); don't crash live loop
            print(f"[mic warning] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    block_size = int(args.input_block_sec * sr)

    try:
        stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            device=args.device_index,
            callback=audio_callback,
        )
    except Exception as e:
        print(f"ERROR: could not open microphone stream: {e}")
        print("Available input devices:")
        print(sd.query_devices())
        return

    print("\nListening for footsteps... (Ctrl+C to stop)\n")

    last_infer_time = time.time()

    try:
        with stream:
            while True:
                # Drain whatever audio has arrived since last loop iteration
                try:
                    while True:
                        chunk = audio_q.get_nowait()
                        ring.push(chunk)
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_infer_time >= args.infer_interval_sec:
                    last_infer_time = now

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
                    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,n_mels,n_frames)

                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1)[0]
                        footstep_prob = probs[1].item()

                    single_hit = footstep_prob >= args.prob_threshold
                    recent_predictions.append(single_hit)
                    hits = sum(recent_predictions)

                    is_footstep = (len(recent_predictions) == args.smoothing_window and
                                    hits >= args.smoothing_min_hits)

                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    label = "FOOTSTEP DETECTED" if is_footstep else "NO FOOTSTEP"
                    conf = footstep_prob if is_footstep else (1 - footstep_prob)
                    print(f"[{ts}] {label} | confidence={conf*100:.1f}%")

                time.sleep(0.01)  # small yield to avoid busy-looping the CPU

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()

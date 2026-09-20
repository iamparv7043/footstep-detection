"""
preprocessing/audio.py

Handles all raw-audio level preprocessing:
- Loading WAV files (any sample rate, mono or stereo)
- Converting to mono
- Resampling to a target sample rate (16 kHz)
- Amplitude normalization
- Converting a variable-length signal into a fixed-duration segment
  (pad with zeros if too short, center-crop / take a window if too long)

This module is used identically by training (dataset.py) and by
real-time inference (infer_live.py), so behaviour MUST stay in sync.
"""

import numpy as np
import librosa


def load_wav_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load a WAV file, convert to mono, resample to target_sr.
    Handles files with different original sample rates and stereo/mono
    automatically (librosa.load does the heavy lifting).

    Returns:
        1-D float32 numpy array in range roughly [-1, 1].
    """
    # sr=target_sr triggers automatic resampling; mono=True averages channels
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def normalize_amplitude(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1]. Guards against silent/zero clips.
    """
    peak = np.max(np.abs(y))
    if peak < eps:
        return y  # silent clip, leave as-is (all zeros)
    return y / peak


def to_fixed_length(y: np.ndarray, sr: int, duration_sec: float = 1.0,
                     random_crop: bool = False) -> np.ndarray:
    """
    Force a 1-D signal to exactly `duration_sec` seconds at sample rate `sr`.

    - If shorter: zero-pad (silence) symmetrically-ish (pad at end, simplest
      and matches how a live 1s ring buffer will look at stream start).
    - If longer: either center-crop (default, used for val/test/live) or
      take a random crop (used for training augmentation when
      random_crop=True) to get temporal diversity from longer recordings.
    """
    target_len = int(duration_sec * sr)
    cur_len = len(y)

    if cur_len == target_len:
        return y

    if cur_len < target_len:
        pad_amount = target_len - cur_len
        return np.pad(y, (0, pad_amount), mode="constant")

    # cur_len > target_len -> crop
    if random_crop:
        max_start = cur_len - target_len
        start = np.random.randint(0, max_start + 1)
    else:
        start = (cur_len - target_len) // 2  # center crop
    return y[start:start + target_len]


def preprocess_file(path: str, sr: int = 16000, duration_sec: float = 1.0,
                     random_crop: bool = False) -> np.ndarray:
    """
    Full pipeline for a single file: load -> mono -> resample -> normalize
    -> fixed length. This is the canonical entry point used by the dataset
    loader for training/val/test.
    """
    y = load_wav_mono(path, target_sr=sr)
    y = normalize_amplitude(y)
    y = to_fixed_length(y, sr, duration_sec, random_crop=random_crop)
    return y


def preprocess_buffer(y: np.ndarray, sr: int = 16000,
                       duration_sec: float = 1.0) -> np.ndarray:
    """
    Same pipeline as preprocess_file, but for an in-memory buffer coming
    from the live microphone ring buffer (already mono + at target sr,
    since sounddevice is opened directly at 16kHz mono). Still normalizes
    and enforces fixed length so behaviour matches training exactly.
    """
    y = normalize_amplitude(y.astype(np.float32))
    y = to_fixed_length(y, sr, duration_sec, random_crop=False)
    return y

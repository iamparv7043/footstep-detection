"""
preprocessing/features.py

Log-Mel Spectrogram feature extraction. Same function is used for
training, evaluation and live inference so the CNN always sees
identically-shaped, identically-scaled input.
"""

import numpy as np
import librosa

# Default mel parameters (also stored in the checkpoint so inference
# always uses whatever the model was actually trained with).
DEFAULT_PARAMS = dict(
    sample_rate=16000,
    n_fft=1024,
    hop_length=320,
    win_length=1024,
    n_mels=64,
)


def compute_log_mel_spectrogram(y: np.ndarray,
                                 sample_rate: int = 16000,
                                 n_fft: int = 1024,
                                 hop_length: int = 320,
                                 win_length: int = 1024,
                                 n_mels: int = 64) -> np.ndarray:
    """
    Compute a Log-Mel spectrogram for a 1-D audio signal.

    Returns:
        2-D float32 array of shape (n_mels, n_frames).
        With 1s @16kHz, hop_length=320 -> n_frames = 16000/320 + 1 = 51.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )
    # Convert power spectrogram to dB (log scale) - standard for CNN input.
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to roughly [0, 1] using a fixed dB floor so scale is
    # consistent across different clips (top_db default clip is -80..0).
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = np.clip(log_mel, 0.0, 1.0)

    return log_mel.astype(np.float32)


def features_from_params(y: np.ndarray, params: dict) -> np.ndarray:
    """Convenience wrapper that unpacks a params dict (as stored in a
    checkpoint) into the spectrogram function."""
    return compute_log_mel_spectrogram(
        y,
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        win_length=params["win_length"],
        n_mels=params["n_mels"],
    )

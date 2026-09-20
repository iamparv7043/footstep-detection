"""
models/cnn.py

Small CNN for binary audio event classification on Log-Mel spectrograms.

Architecture (as specified):
Conv2D -> BatchNorm -> ReLU -> MaxPool   (x3, increasing channels)
Global Average Pooling
Linear -> Dropout -> Linear(2)

Input shape:  (batch, 1, n_mels, n_frames)   e.g. (B, 1, 64, 51)
Output shape: (batch, 2)  raw logits for [no_footstep, footstep]
"""

import torch
import torch.nn as nn


class FootstepCNN(nn.Module):
    def __init__(self, n_mels: int = 64, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Global Average Pooling collapses (freq, time) -> 1x1 regardless
        # of exact spectrogram size, so input duration/hop can change
        # later without reshaping the classifier head.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 logits: [no_footstep, footstep]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, n_frames)
        x = self.conv_block(x)          # (B, 64, n_mels/8, n_frames/8)
        x = self.gap(x)                 # (B, 64, 1, 1)
        x = x.flatten(1)                # (B, 64)
        logits = self.classifier(x)     # (B, 2)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape sanity check: 1s @16kHz, hop=320, n_mels=64 -> 51 frames
    model = FootstepCNN(n_mels=64)
    dummy = torch.randn(4, 1, 64, 51)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 2)
    print("Param count:", model.count_params())

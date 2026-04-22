"""
model.py — Defines SensaCalibNet, a tiny 1D convolutional neural network
that predicts calibrated PM2.5 from a single SEN55 sensor reading.

HOW A CNN WORKS (for EEs):
    A convolutional layer is a bank of FIR filters. Each filter has a small
    kernel (e.g., 3 taps) that slides across the input. The filter learns
    to detect a pattern — here, a pattern across adjacent sensor channels.
    For example, the relationship between PM1 and PM2.5 encodes particle
    size distribution, which correlates with the SEN55's optical bias.

WHY A 1D CNN RATHER THAN A FULLY-CONNECTED NETWORK:
    The 8 sensor channels are not arbitrary — they are physically ordered by
    particle size (PM1 < PM2.5 < PM4 < PM10). A 1D convolution with a kernel
    of size 3 naturally learns pairwise and triplet relationships between
    adjacent size fractions. A fully-connected layer treats all inputs as
    independent, losing that structure.
    Additionally, when you later upgrade to Option B (windowed time-series),
    the exact same CNN architecture applies to a sequence of readings with
    no structural changes.

ARCHITECTURE:
    Input (8 features)
      └─ reshape to (batch, 1 channel, 8 features)
          └─ Conv1d block 1: 1 → 16 channels, kernel=3
              └─ Conv1d block 2: 16 → 32 channels, kernel=3
                  └─ Global average pool → (batch, 32)
                      └─ Linear 32 → 16
                          └─ Linear 16 → 1 (calibrated PM2.5)

PARAMETER COUNT:
    ~2,200 float32 parameters ≈ 8.5 KB float32 ≈ 2.1 KB int8 quantised.
    Well under the 30 KB ESP32-S3 weight budget.
"""

import torch
import torch.nn as nn


class SensaCalibNet(nn.Module):
    """
    Tiny 1D CNN for single-sample PM2.5 sensor calibration.

    Takes one reading from the SEN55 (8 channels) and returns a single
    float: the estimated true PM2.5 value as measured by a reference sensor.

    Designed to operate within ESP32-S3 constraints after int8 quantisation:
      - Weight footprint: ~2 KB
      - Runtime RAM:      ~4 KB activations
      - Inference time:   << 10 ms
    """

    def __init__(self, n_features: int = 8, n_channels: int = 16):
        """
        Args:
            n_features: Number of SEN55 channels used as model input.
                        Default 8: pm1, pm2_5, pm4, pm10, temp, rh, voc, nox.
                        If you later drop some channels (e.g. use PM-only),
                        change this and retrain.
            n_channels: Base width of the conv layers.
                        Keep ≤ 32 to stay within the ESP32 memory budget.
        """
        # nn.Module.__init__ registers all sub-layers so PyTorch can find
        # their parameters during backpropagation and checkpointing.
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────────
        # nn.Sequential is a container that chains layers in order, like a
        # signal processing chain. Data flows left-to-right through the list.
        self.conv_block = nn.Sequential(
            # Layer 1: Conv1d
            #   in_channels=1  : the input has one "channel" (the feature vector)
            #   out_channels=16: learn 16 different filter patterns
            #   kernel_size=3  : each filter looks at 3 adjacent features at a time
            #   padding=1      : pad the edges so output length equals input length
            nn.Conv1d(in_channels=1, out_channels=n_channels, kernel_size=3, padding=1),
            # BatchNorm normalises the activations to zero mean, unit variance.
            # This stabilises training — analogous to removing DC offset before
            # an amplifier stage. It is folded into the conv weights during
            # TFLite export, so it has zero runtime cost on the ESP32.
            nn.BatchNorm1d(n_channels),
            # ReLU: the activation function. Sets negative values to zero.
            # This is what gives the network its non-linear (calibration curve)
            # fitting ability. Think of it as a half-wave rectifier.
            nn.ReLU(),

            # Layer 2: wider conv — learns combinations of the patterns from layer 1
            nn.Conv1d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channels * 2),
            nn.ReLU(),
        )

        # ── Global average pool ───────────────────────────────────────────────
        # Collapses the spatial (feature) dimension by averaging across it.
        # Input: (batch, channels, 8 features) → Output: (batch, channels, 1)
        # This makes the network insensitive to the exact number of input
        # features, which will matter if you later add or remove channels.
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # ── Regression head ───────────────────────────────────────────────────
        # Two fully-connected layers that map the pooled features to the
        # single output value (calibrated PM2.5 in µg/m³).
        self.regressor = nn.Sequential(
            nn.Linear(n_channels * 2, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, 1),  # Final output: one number (calibrated PM2.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass (inference or training step).

        In PyTorch, you define the forward pass explicitly. The backward pass
        (gradient computation for training) is derived automatically.

        Args:
            x: Tensor of shape (batch_size, n_features).
               One row per sensor reading; each row has 8 normalised values.

        Returns:
            Tensor of shape (batch_size, 1): calibrated PM2.5 prediction.
        """
        # Conv1d expects 3D input: (batch, channels, length).
        # Our input is 2D: (batch, features).
        # unsqueeze(1) inserts a size-1 dimension at position 1, giving
        # (batch, 1 channel, features).
        x = x.unsqueeze(1)

        # Apply the two convolutional layers + BatchNorm + ReLU.
        x = self.conv_block(x)   # → (batch, n_channels*2, n_features)

        # Pool across the feature dimension → (batch, n_channels*2, 1)
        x = self.pool(x)

        # Remove the trailing size-1 dimension → (batch, n_channels*2)
        x = x.squeeze(-1)

        # Map to the scalar output → (batch, 1)
        x = self.regressor(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Return the total number of trainable parameters in a model.

    Useful for quickly verifying the model stays within the ESP32 weight budget.
    At float32 (4 bytes/param): budget is ~7,500 params for 30 KB.
    At int8 (1 byte/param): budget is ~30,000 params for 30 KB.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

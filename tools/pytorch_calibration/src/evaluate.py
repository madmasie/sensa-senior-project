"""
evaluate.py — Compute accuracy metrics and generate diagnostic plots.

METRICS EXPLAINED (all in µg/m³ unless noted):

    MAE  — Mean Absolute Error: the average absolute difference between
            prediction and truth. The most intuitive single-number summary.
            "On average, the calibrated sensor is off by X µg/m³."

    RMSE — Root Mean Squared Error: like MAE but larger errors are penalised
            more heavily. If RMSE >> MAE, there are occasional large blunders.
            "Typical error, with outliers weighted more."

    R²   — Coefficient of determination: the fraction of variance in the
            reference PM2.5 that the model explains. Range [0, 1].
            R² = 1.0 means perfect predictions; R² = 0 means no better than
            predicting the mean every time.

    MBE  — Mean Bias Error: mean of (predicted − true). A positive MBE means
            the model systematically reads high; negative means it reads low.
            Like the DC offset of a measurement system. Ideally near zero.

EPA LOW-COST SENSOR ACCURACY GUIDELINES (PM2.5, co-located with FRM/FEM):
    These are voluntary targets from the EPA's 2021 interim guidance for
    air sensor data quality (EPA/600/R-20/231):
      - MAE  < 5 µg/m³ (or RMSE < 7 µg/m³)
      - R²   > 0.80
      - MBE  within ±5 µg/m³
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
) -> Dict[str, float]:
    """
    Run inference on a data split and compute calibration quality metrics.

    Args:
        model:      Trained SensaCalibNet (best checkpoint loaded).
        loader:     DataLoader for the split to evaluate (usually test set).
        device:     Compute device.
        split_name: Label for print output (e.g., "test", "validation").

    Returns:
        Dict with keys: 'mae', 'rmse', 'r2', 'mbe'
    """
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch).cpu().numpy()    # move back to CPU for numpy
            targets = y_batch.numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate all batches and flatten to 1D arrays
    preds   = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    errors = preds - targets

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2   = float(r2_score(targets, preds))
    mbe  = float(np.mean(errors))  # positive = predictions biased high

    print(f"\n{'─'*45}")
    print(f"  Evaluation results — {split_name} set")
    print(f"{'─'*45}")
    print(f"  MAE  (µg/m³) : {mae:>8.3f}   (avg absolute error)")
    print(f"  RMSE (µg/m³) : {rmse:>8.3f}   (RMS error)")
    print(f"  R²           : {r2:>8.4f}   (1.0 = perfect)")
    print(f"  MBE  (µg/m³) : {mbe:>8.3f}   (bias; 0 = unbiased)")
    print()

    # ── Interpret against EPA guidelines ─────────────────────────────────────
    if mae < 5.0 and r2 > 0.80 and abs(mbe) < 5.0:
        print("  ✓ Meets EPA interim low-cost sensor accuracy guidelines.")
    elif mae < 10.0 and r2 > 0.60:
        print("  ~ Acceptable but below EPA recommended thresholds.")
        print("    Suggestions: collect more data, diversify conditions,")
        print("    or increase model capacity slightly (n_channels in config.yaml).")
    else:
        print("  ✗ Poor performance. Likely causes:")
        print("    - Insufficient training data (< ~500 paired samples)")
        print("    - Narrow range of PM2.5 conditions during collection")
        print("    - Sensor hardware issue (check raw SEN55 vs BAM scatter plot)")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mbe': mbe}


def plot_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "models/predictions.png",
) -> None:
    """
    Generate a predicted-vs-true scatter plot and a residual histogram.

    These two plots are standard diagnostics for a regression model:

    Scatter plot (predicted vs. true):
        Points should cluster tightly along the diagonal (y = x line).
        Systematic curvature indicates the model underfits in some PM range.

    Residual histogram (predicted − true):
        Should be a narrow bell curve centred at zero.
        A non-zero centre indicates systematic bias (MBE ≠ 0).
        Heavy tails indicate occasional large errors.

    Args:
        model:     Trained SensaCalibNet.
        loader:    DataLoader for the evaluation split.
        device:    Compute device.
        save_path: Where to write the PNG.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            all_preds.append(model(X_batch).cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    errors  = preds - targets

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SensaCalibNet — Calibration Diagnostics", fontsize=14)

    # ── Scatter: predicted vs. true ───────────────────────────────────────────
    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.4, s=15, color='steelblue', label='Predictions')
    lim = [min(targets.min(), preds.min()) - 2, max(targets.max(), preds.max()) + 2]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect calibration (y=x)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Reference PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title("Predicted vs. Reference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Histogram: residuals ──────────────────────────────────────────────────
    ax = axes[1]
    ax.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0,    color='red',    linestyle='--', linewidth=1.5, label='Zero error')
    ax.axvline(errors.mean(), color='orange', linestyle='-', linewidth=1.5,
               label=f'Mean bias = {errors.mean():.2f} µg/m³')
    ax.set_xlabel("Prediction error (µg/m³)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostic plot saved → {save_path}")

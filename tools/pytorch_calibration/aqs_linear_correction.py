"""
aqs_linear_correction.py — Calibration by post-hoc linear correction of the
AQS-pretrained neural network's predictions.

WHY THIS EXISTS:
    `finetune.py` fine-tunes the AQS pretrained network's weights on the
    local SEN55+PurpleAir data. At our scale (N=17) that has too many free
    parameters and overfits. `linear_calibration.py` fits a 2-parameter
    affine map directly to the SEN55 reading, which works well but ignores
    everything the AQS dataset taught the network.

    This script combines both: it uses the AQS pretrained network as a
    *frozen feature extractor*, runs the local SEN55 inputs through it,
    and then learns a 2-parameter affine correction on top of the network's
    output:

        pm_calibrated = a · NN_AQS(pm_sen55, temp, rh) + b

    The NN supplies the inductive bias from 14k AQS site-hours (captured
    in 2,273 trained weights); the 2-parameter head learns the offset and
    gain needed for *this* SEN55 + this site. With only 17 local samples
    this is the maximum amount of transfer information we can responsibly
    extract — every AQS-learned feature is reused, but only 2 weights are
    fit on the small local set.

WHAT IT DOES:
    1. Load the AQS pretrained model (models_public/best_model_public.pt)
       and the scaler that was fit on AQS inputs (models_public/scaler.pkl).
    2. For each local row, normalize (pm2_5, temp, rh) with the AQS scaler
       and run a forward pass through the AQS model. Call the output
       "aqs_pred" — the AQS model's best guess at the reference PM2.5.
    3. Leave-one-out cross-validate a one-feature linear regression:
            pm_ref = a · aqs_pred + b
       on the 17 (aqs_pred, pm_ref) pairs.
    4. Report MAE / RMSE / MBE / R² and save the predictions JSON in the
       same format paper_figures.py reads for the other methods.

This script does NOT modify the pretrained model. It writes to its own
output directory (models_aqs_corrected/) and reuses cv_predictions /
summarise / plot_results from linear_calibration.py.

USAGE:
    python aqs_linear_correction.py
    python aqs_linear_correction.py --cv 5
    python aqs_linear_correction.py --pretrained models_public
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from linear_calibration import (
    cv_predictions, summarise, plot_results,
)
from src.dataset import load_scaler
from src.model import SensaCalibNet


def _read_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def aqs_forward(
    df: pd.DataFrame,
    features: list[str],
    scaler,
    model: SensaCalibNet,
) -> np.ndarray:
    """
    Run every row of `df` through the AQS pretrained model and return its
    scalar predictions.

    Inputs are normalised with the *AQS-fit* scaler (never refit on the
    local data — that would invalidate the pretrained weights). The model
    is held in eval mode and gradients are disabled.
    """
    X_raw = df[features].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    model.eval()
    with torch.no_grad():
        # shape: (N, n_features) → SensaCalibNet handles the unsqueeze
        preds = model(torch.tensor(X_scaled)).cpu().numpy().flatten()
    return preds.astype(np.float64)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="AQS-network output corrected by a 2-parameter affine head."
    )
    parser.add_argument(
        "--csv", default="data/paired/paired_dataset.csv",
        help="Paired dataset CSV (default: data/paired/paired_dataset.csv).",
    )
    parser.add_argument(
        "--pretrained", default="models_public",
        help="Directory holding best_model_public.pt and scaler.pkl "
             "(default: models_public).",
    )
    parser.add_argument(
        "--target", default="bam_pm2_5",
        help="Reference column in the paired CSV (default: bam_pm2_5).",
    )
    parser.add_argument(
        "--cv", type=int, default=0, metavar="K",
        help="Number of CV folds. 0 (default) = leave-one-out.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for K-fold shuffling (default: 42).",
    )
    parser.add_argument(
        "--out-dir", default="models_aqs_corrected",
        help="Where to write JSON + plot (default: models_aqs_corrected).",
    )
    parser.add_argument(
        "--config", default=str(script_dir.parent / "config.yaml"),
        help="Path to tools/config.yaml (for the AQS model dimensions).",
    )
    args = parser.parse_args()

    # ── Load paired data ────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"ERROR: {csv_path} not found.\n"
            "Run prepare_purpleair_data.py first.")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # ── Load AQS scaler + model ─────────────────────────────────────────────
    pretrained_dir = Path(args.pretrained)
    ckpt_path = pretrained_dir / "best_model_public.pt"
    scaler_path = pretrained_dir / "scaler.pkl"
    if not ckpt_path.exists() or not scaler_path.exists():
        raise SystemExit(
            f"ERROR: AQS pretrained artefacts not found in {pretrained_dir}.\n"
            "Run `python train_public.py` first.")

    config = _read_config(Path(args.config))
    pub_cfg = config["public_data"]
    # AQS schema:  features = [pm2_5_optical, temp, rh], 3 inputs, 16 channels.
    # Local CSV has the same data under different names — map them through.
    aqs_features = pub_cfg["features"]
    local_to_aqs = {"pm2_5_optical": "pm2_5", "temp": "temp", "rh": "rh"}
    local_feature_cols = [local_to_aqs.get(f, f) for f in aqs_features]
    missing = [c for c in local_feature_cols + [args.target] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR: paired CSV missing columns {missing}. Have: {list(df.columns)}")

    df = df.dropna(subset=local_feature_cols + [args.target]).reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise SystemExit(f"ERROR: need at least 3 paired rows, got {n}.")

    scaler = load_scaler(str(scaler_path))
    model = SensaCalibNet(
        n_features=pub_cfg["model"]["n_features"],
        n_channels=pub_cfg["model"]["n_channels"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print(f"\nLoaded AQS model: {ckpt_path}")

    # ── Run the AQS model on every local row ────────────────────────────────
    aqs_pred = aqs_forward(df, local_feature_cols, scaler, model)
    y = df[args.target].to_numpy(dtype=np.float64)
    raw_sensor = df["pm2_5"].to_numpy(dtype=np.float64)

    print(f"\nAQS model output range : "
          f"{aqs_pred.min():.2f} – {aqs_pred.max():.2f} µg/m³")
    print(f"Reference range        : {y.min():.2f} – {y.max():.2f} µg/m³")

    # Baseline: how does the AQS model do on the local data with NO correction?
    summarise("AQS model (no correction)", aqs_pred, y)

    # ── 2-parameter affine head, CV'd ───────────────────────────────────────
    k_folds = args.cv if (args.cv and args.cv > 0) else n
    k_folds = min(k_folds, n)
    mode = "leave-one-out" if k_folds == n else f"{k_folds}-fold"
    print(f"\nCV mode: {mode}")

    X = aqs_pred.reshape(-1, 1)
    cv_preds, fold_assignment, per_fold = cv_predictions(
        X, y, k_folds=k_folds, seed=args.seed,
    )
    print(f"\n  Per-fold MAE: " + " ".join(f"{m['mae']:5.2f}" for m in per_fold))
    metrics_cv = summarise(f"AQS + 2-param affine head ({mode} CV)", cv_preds, y)

    # ── Final fit on all data → deployable coefficients ─────────────────────
    from sklearn.linear_model import LinearRegression
    final = LinearRegression().fit(X, y)
    a = float(final.coef_[0]); b = float(final.intercept_)
    print(f"\n  Final fit (all {n} samples):")
    print(f"    pm_calibrated ≈ {a:.4f} · aqs_pred {b:+.4f}")
    summarise("Training fit (in-sample, optimistic)", final.predict(X), y)

    # ── Persist results ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aqs_linear_correction.json"
    with open(json_path, "w") as f:
        json.dump({
            "method":      "aqs_linear",
            "pretrained":  str(pretrained_dir),
            "features":    local_feature_cols,
            "target":      args.target,
            "n_samples":   n,
            "k_folds":     k_folds,
            "cv_mode":     mode,
            "coef":        [a],
            "intercept":   b,
            "aggregate":   metrics_cv,
            "per_fold":    per_fold,
            "predictions": [
                {"fold": int(fold_assignment[i]),
                 "reference":  float(y[i]),
                 "prediction": float(cv_preds[i]),
                 "aqs_pred":   float(aqs_pred[i])}
                for i in range(n)
            ],
            "deployment_formula": (
                f"pm_calibrated = {a:.6f} * NN_AQS(pm2_5, temp, rh) + {b:+.6f}"
            ),
        }, f, indent=2)
    print(f"\n  Coefficients saved → {json_path}")

    plot_path = out_dir / "aqs_linear_cv_predictions.png"
    plot_results(y, cv_preds, raw_sensor, plot_path, metrics_cv, k_folds)
    print(f"  Diagnostic plot saved → {plot_path}")

    print("\n  Compare with finetune.py --cv and linear_calibration.py to see")
    print("  which transfer-learning strategy works best at this dataset size.")


if __name__ == "__main__":
    main()

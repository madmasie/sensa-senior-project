"""
linear_calibration.py — Baseline 2-coefficient linear calibration.

WHY THIS EXISTS:
    Establishes the simplest possible calibration to compare the fine-tuned
    neural network against. With very small datasets (~17 paired windows),
    a 2-parameter affine fit  y_ref = a*x_sen55 + b  often matches or beats
    a 2000+ parameter network. If the NN doesn't beat this number, the NN
    is overfitting and the linear fit is the better deployment choice.

WHAT IT DOES:
    1. Loads the same paired_dataset.csv that finetune.py consumes.
    2. Runs leave-one-out cross-validation by default (--cv K for K-fold).
       For each fold:
           - Fit an OLS linear regression on the non-test rows.
           - Predict the held-out row(s).
           - Store the prediction.
       Every sample gets predicted by a model that never saw it during fit.
    3. Reports aggregate MAE / RMSE / R² / MBE across the held-out predictions.
       This is directly comparable to finetune.py --cv's aggregate numbers.
    4. Fits one final model on ALL the data and prints the deployable
       coefficients (a, b — or weights + intercept for multi-feature fits).
    5. Saves a side-by-side scatter / residual plot and a small JSON file
       containing the final coefficients.

MODULARITY:
    This script does NOT import from src/ or modify finetune.py / main.py /
    prepare_purpleair_data.py. It reads the same paired CSV and writes its
    output to its own directory (models_linear/ by default). Removing it
    has zero effect on the rest of the pipeline.

USAGE:
    # Single-feature LOO baseline (SEN55 pm2_5 → PurpleAir reference):
    python linear_calibration.py

    # K-fold CV instead of LOO:
    python linear_calibration.py --cv 5

    # Multi-feature linear regression (e.g. PM + humidity + temperature):
    python linear_calibration.py --features pm2_5 rh temp

    # Different paired CSV or output dir:
    python linear_calibration.py --csv data/paired/paired_dataset.csv \\
                                  --out-dir models_linear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run K-fold (or LOO if k_folds >= N) CV with OLS linear regression.

    Returns:
        held_out_preds: array of length N with each sample's prediction
                        from a fold where it was held out.
        per_fold:       list of {'fold', 'n_test', 'mae', 'rmse', 'mbe'} dicts.
    """
    n = len(y)
    if k_folds > n:
        k_folds = n  # leave-one-out

    # Shuffle once with a reproducible seed so K-fold splits are stable.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    X_shuf, y_shuf = X[order], y[order]

    kf = KFold(n_splits=k_folds, shuffle=False)
    preds = np.full(n, np.nan, dtype=np.float64)
    per_fold: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_shuf), start=1):
        model = LinearRegression()
        model.fit(X_shuf[train_idx], y_shuf[train_idx])
        fold_preds = model.predict(X_shuf[test_idx])

        # Store predictions back at the original (un-shuffled) row indices so
        # the final aggregate aligns with the input arrays.
        original_rows = order[test_idx]
        preds[original_rows] = fold_preds

        err = fold_preds - y_shuf[test_idx]
        per_fold.append({
            "fold":   fold_idx,
            "n_test": int(len(test_idx)),
            "mae":    float(np.mean(np.abs(err))),
            "rmse":   float(np.sqrt(np.mean(err ** 2))),
            "mbe":    float(np.mean(err)),
        })

    return preds, per_fold


def summarise(name: str, preds: np.ndarray, targets: np.ndarray) -> dict:
    """Print and return MAE / RMSE / MBE / R² for a set of predictions."""
    err  = preds - targets
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe  = float(np.mean(err))
    r2   = float(r2_score(targets, preds)) if len(targets) > 1 else float("nan")
    print(f"\n  {name}:")
    print(f"    MAE  (µg/m³) : {mae:8.3f}")
    print(f"    RMSE (µg/m³) : {rmse:8.3f}")
    print(f"    MBE  (µg/m³) : {mbe:+8.3f}")
    print(f"    R²           : {r2:8.4f}")
    return {"mae": mae, "rmse": rmse, "mbe": mbe, "r2": r2}


def plot_results(
    targets: np.ndarray,
    cv_preds: np.ndarray,
    raw_sensor: np.ndarray,
    out_path: Path,
    metrics: dict,
    n_folds: int,
) -> None:
    """Save a 2-panel diagnostic plot: scatter + residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Linear calibration — {n_folds}-fold CV  "
        f"({len(targets)} held-out predictions)", fontsize=13,
    )

    # ── Scatter: held-out predictions and raw sensor vs reference ───────────
    ax = axes[0]
    lim_lo = float(min(targets.min(), cv_preds.min(), raw_sensor.min())) - 1.0
    lim_hi = float(max(targets.max(), cv_preds.max(), raw_sensor.max())) + 1.0
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.5,
            label="y = x (perfect)")
    ax.scatter(targets, raw_sensor, s=35, color="lightgray",
               edgecolor="dimgray", alpha=0.85, label="raw SEN55 (no cal)")
    ax.scatter(targets, cv_preds, s=40, color="steelblue",
               edgecolor="white", alpha=0.9, label="linear-cal CV prediction")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Reference PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted / raw PM2.5 (µg/m³)")
    ax.set_title("CV held-out predictions vs reference")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.05,
            f"MAE  = {metrics['mae']:.2f}\nRMSE = {metrics['rmse']:.2f}\n"
            f"MBE  = {metrics['mbe']:+.2f}\nR²   = {metrics['r2']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # ── Residual histogram ──────────────────────────────────────────────────
    err = cv_preds - targets
    ax = axes[1]
    ax.hist(err, bins=max(5, len(err) // 2), color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="zero error")
    ax.axvline(metrics["mbe"], color="orange", linestyle="-", linewidth=1.5,
               label=f"mean bias = {metrics['mbe']:+.2f}")
    ax.set_xlabel("Prediction − reference (µg/m³)")
    ax.set_ylabel("Count")
    ax.set_title("CV residual distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear-calibration baseline with cross-validated metrics."
    )
    parser.add_argument(
        "--csv", default="data/paired/paired_dataset.csv",
        help="Paired dataset CSV (default: data/paired/paired_dataset.csv).",
    )
    parser.add_argument(
        "--features", nargs="+", default=["pm2_5"],
        help="Input feature column(s). Default: pm2_5 (single-feature affine "
             "fit y_ref = a*pm2_5 + b). Pass multiple names for a "
             "multivariate linear fit, e.g. --features pm2_5 rh temp.",
    )
    parser.add_argument(
        "--target", default="bam_pm2_5",
        help="Reference column name in the CSV (default: bam_pm2_5 — the "
             "averaged PurpleAir reading from prepare_purpleair_data.py).",
    )
    parser.add_argument(
        "--cv", type=int, default=0, metavar="K",
        help="Number of CV folds. 0 (default) = leave-one-out. Pass --cv 5 "
             "for 5-fold, etc. Any K >= N is treated as LOO.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for K-fold shuffling (default: 42).",
    )
    parser.add_argument(
        "--out-dir", default="models_linear",
        help="Where to write the plot and coefficient JSON "
             "(default: models_linear).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"ERROR: {csv_path} not found.\n"
            "Run prepare_purpleair_data.py to produce the paired CSV first."
        )

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    missing = [c for c in args.features + [args.target] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR: CSV missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    # Drop rows where any required column is null — keeps OLS happy.
    df = df.dropna(subset=args.features + [args.target]).reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise SystemExit(f"ERROR: need at least 3 paired rows, got {n}.")

    X = df[args.features].to_numpy(dtype=np.float64)
    y = df[args.target].to_numpy(dtype=np.float64)
    raw_sensor = df["pm2_5"].to_numpy(dtype=np.float64) if "pm2_5" in df.columns else X[:, 0]

    k_folds = args.cv if (args.cv and args.cv > 0) else n
    k_folds = min(k_folds, n)
    mode = "leave-one-out" if k_folds == n else f"{k_folds}-fold"
    print(f"\nDataset: {csv_path}   N = {n}   features = {args.features}   "
          f"target = {args.target}")
    print(f"CV mode: {mode}")

    # ── Cross-validated metrics ─────────────────────────────────────────────
    cv_preds, per_fold = cv_predictions(X, y, k_folds=k_folds, seed=args.seed)
    print(f"\n  Per-fold MAE: " +
          " ".join(f"{m['mae']:5.2f}" for m in per_fold))
    metrics_cv = summarise(f"{mode} cross-validation", cv_preds, y)

    # ── Raw-sensor baseline for context ─────────────────────────────────────
    summarise("Raw SEN55 (no calibration)", raw_sensor, y)

    # ── Final fit on ALL data (this is what you would deploy) ───────────────
    final = LinearRegression().fit(X, y)
    print(f"\n  Final fit on all {n} samples:")
    if len(args.features) == 1:
        a = float(final.coef_[0])
        b = float(final.intercept_)
        print(f"    y_ref ≈ {a:.4f} * {args.features[0]} + {b:+.4f}")
    else:
        for fname, coef in zip(args.features, final.coef_):
            print(f"    coef[{fname:<6}] = {coef:+.4f}")
        print(f"    intercept   = {float(final.intercept_):+.4f}")
    summarise("Training fit (in-sample, optimistic)", final.predict(X), y)

    # ── Persist coefficients + metrics ──────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "linear_calibration.json"
    with open(json_path, "w") as f:
        json.dump({
            "features":   args.features,
            "target":     args.target,
            "n_samples":  n,
            "cv_mode":    mode,
            "coef":       [float(c) for c in final.coef_],
            "intercept":  float(final.intercept_),
            "cv_metrics": metrics_cv,
            "per_fold":   per_fold,
            "deployment_formula": (
                f"y_ref = {float(final.coef_[0]):.6f}*{args.features[0]} + "
                f"{float(final.intercept_):+.6f}"
                if len(args.features) == 1
                else "y_ref = sum(coef_i * feature_i) + intercept"
            ),
        }, f, indent=2)
    print(f"\n  Coefficients saved → {json_path}")

    plot_path = out_dir / "linear_cv_predictions.png"
    plot_results(y, cv_preds, raw_sensor, plot_path, metrics_cv, k_folds)
    print(f"  Diagnostic plot saved → {plot_path}")

    print("\n  This number is directly comparable to finetune.py --cv. If the")
    print("  linear MAE is similar or lower than the fine-tuned NN's MAE, the")
    print("  NN is overfitting and the linear fit is the better deployment.")


if __name__ == "__main__":
    main()

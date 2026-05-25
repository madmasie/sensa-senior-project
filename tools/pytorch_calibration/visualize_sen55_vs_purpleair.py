"""
visualize_sen55_vs_purpleair.py — Plot the raw SEN55 PM2.5 readings against the
PurpleAir reference, to see how much the SEN55 is off BEFORE any calibration.

This is a model-independent diagnostic. It reads the paired CSV produced by
prepare_purpleair_data.py and answers:
    - How well does the raw SEN55 already track the PurpleAir reference?
    - Is the error proportional (slope ≠ 1) or an offset (intercept ≠ 0)?
    - Does the error depend on humidity, temperature, or concentration?

The four panels (left → right, top → bottom):

    1. Scatter of SEN55 PM2.5 vs PurpleAir PM2.5
       Each dot is one ~10-minute co-location window. A red dashed line shows
       perfect agreement (y = x); a blue line is the ordinary least-squares
       best fit. Slope ≠ 1 indicates a gain error; intercept ≠ 0 indicates an
       offset error.

    2. Residual histogram (SEN55 − PurpleAir)
       Centre = mean bias error (MBE). Spread = noise / variability.

    3. Residual vs PurpleAir PM2.5
       Reveals concentration-dependent error. A trend here means the SEN55's
       error scales with PM2.5 — typical for optical sensors that under-read
       at low concentrations and over- or under-read at high ones.

    4. Residual vs relative humidity
       PM optical sensors notoriously over-read in humid air (hygroscopic
       growth of aerosols). A positive trend confirms this in the local data.

PRINTED METRICS:
    MAE, RMSE, R², MBE — same definitions as in src/evaluate.py.
    slope, intercept   — linear-fit parameters of SEN55 on PurpleAir.
                         An ideal sensor would be slope=1, intercept=0.

USAGE:
    python visualize_sen55_vs_purpleair.py
    python visualize_sen55_vs_purpleair.py --csv data/paired/paired_dataset.csv
    python visualize_sen55_vs_purpleair.py --out diagnostics/sen55_vs_pa.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least-squares fit y ≈ slope * x + intercept."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize raw SEN55 PM2.5 vs PurpleAir reference."
    )
    parser.add_argument(
        "--csv", default="data/paired/paired_dataset.csv",
        help="Paired dataset CSV (default: data/paired/paired_dataset.csv).",
    )
    parser.add_argument(
        "--out", default="diagnostics/sen55_vs_purpleair.png",
        help="Output PNG path (default: diagnostics/sen55_vs_purpleair.png).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"ERROR: {csv_path} not found.\n"
            "Run prepare_purpleair_data.py first to produce the paired CSV."
        )

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} paired rows from {csv_path}")

    sen55 = df['pm2_5'].to_numpy()
    ref   = df['bam_pm2_5'].to_numpy()   # PurpleAir-averaged reference
    rh    = df['rh'].to_numpy()
    temp  = df['temp'].to_numpy()
    err   = sen55 - ref                  # residual: SEN55 minus reference

    # ── Metrics ──────────────────────────────────────────────────────────────
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe  = float(np.mean(err))           # +ve = SEN55 reads higher than PurpleAir
    r2   = float(r2_score(ref, sen55)) if len(ref) > 1 else float('nan')
    slope, intercept = linear_fit(ref, sen55) if len(ref) > 1 else (float('nan'), float('nan'))

    print("\n── Raw SEN55 vs PurpleAir ──────────────────────────────────────")
    print(f"  N samples   : {len(df)}")
    print(f"  MAE (µg/m³) : {mae:7.3f}")
    print(f"  RMSE        : {rmse:7.3f}")
    print(f"  MBE (bias)  : {mbe:+7.3f}  ({'SEN55 reads HIGH' if mbe > 0 else 'SEN55 reads LOW'})")
    print(f"  R²          : {r2:7.4f}")
    print(f"  Linear fit  : SEN55 ≈ {slope:.3f} × PurpleAir {intercept:+.3f}")
    print(f"                (ideal sensor would be 1.000 × ref + 0.000)")

    # ── Figure ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Raw SEN55 vs PurpleAir reference  ({len(df)} co-location windows)",
        fontsize=14,
    )

    # 1. Scatter — SEN55 vs PurpleAir, with 1:1 line and OLS fit ───────────────
    ax = axes[0, 0]
    ax.scatter(ref, sen55, s=40, color='steelblue', alpha=0.8, edgecolor='white')
    lo = float(min(ref.min(), sen55.min())) - 1.0
    hi = float(max(ref.max(), sen55.max())) + 1.0
    line = np.array([lo, hi])
    ax.plot(line, line, 'r--', linewidth=1.5, label='y = x (perfect)')
    ax.plot(line, slope * line + intercept, color='navy', linewidth=1.5,
            label=f'fit: y = {slope:.2f}x + {intercept:+.2f}')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("PurpleAir reference PM2.5 (µg/m³)")
    ax.set_ylabel("SEN55 PM2.5 (µg/m³)")
    ax.set_title("SEN55 vs PurpleAir")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.05,
            f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nMBE = {mbe:+.2f}\nR² = {r2:.3f}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Residual histogram ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(err, bins=max(5, len(err) // 2), color='steelblue',
            edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='zero error')
    ax.axvline(mbe, color='orange', linestyle='-', linewidth=1.5,
               label=f'mean bias = {mbe:+.2f}')
    ax.set_xlabel("SEN55 − PurpleAir (µg/m³)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Residual vs reference concentration ───────────────────────────────────
    ax = axes[1, 0]
    sc = ax.scatter(ref, err, c=rh, cmap='viridis', s=40, alpha=0.85,
                    edgecolor='white')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
    if len(ref) > 1:
        s2, i2 = linear_fit(ref, err)
        ax.plot(line, s2 * line + i2, color='navy', linewidth=1.2,
                label=f'trend: {s2:+.3f} per µg/m³')
        ax.legend(loc='upper right')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("RH (%)")
    ax.set_xlabel("PurpleAir reference PM2.5 (µg/m³)")
    ax.set_ylabel("SEN55 − PurpleAir (µg/m³)")
    ax.set_title("Residual vs concentration  (color = RH)")
    ax.grid(True, alpha=0.3)

    # 4. Residual vs relative humidity ─────────────────────────────────────────
    ax = axes[1, 1]
    sc = ax.scatter(rh, err, c=temp, cmap='plasma', s=40, alpha=0.85,
                    edgecolor='white')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
    if len(rh) > 1:
        s3, i3 = linear_fit(rh, err)
        rh_line = np.array([rh.min() - 1, rh.max() + 1])
        ax.plot(rh_line, s3 * rh_line + i3, color='navy', linewidth=1.2,
                label=f'trend: {s3:+.3f} per %RH')
        ax.legend(loc='upper right')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Temp (°C)")
    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel("SEN55 − PurpleAir (µg/m³)")
    ax.set_title("Residual vs humidity  (color = temp)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved diagnostic plot → {out_path}")


if __name__ == "__main__":
    main()

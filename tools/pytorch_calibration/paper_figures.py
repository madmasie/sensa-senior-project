"""
paper_figures.py — Publication-quality figures for the Sensa report,
generated from the *actual* PM2.5 calibration results.

This script produces four figures:

  1. METHOD COMPARISON — PREDICTED vs REFERENCE
     Three side-by-side scatter panels: raw SEN55 (no calibration), the
     transfer-learned neural network, and the linear baseline. Each panel
     plots predicted vs reference PM2.5 with the y=x line, the OLS fit
     line, and MAE / RMSE / MBE / R² in an annotation box.

  2. PER-FOLD MAE DISTRIBUTION
     Violin plot of per-fold MAE for each method across the
     cross-validation folds. Mean and median markers overlaid; each fold's
     value plotted as a jittered dot. Shows both the central tendency and
     the spread — a method can have a good mean MAE but a terrible
     worst-fold MAE, and that is paper-worthy information.

  3. PER-FOLD AGGREGATE TABLE
     A clean tabular summary (MAE / RMSE / MBE / R²) of the three methods
     at the aggregate level. Renders the table as a matplotlib figure so
     it drops straight into the paper alongside the other figures.

  4. CNN ARCHITECTURE DIAGRAM
     A block diagram of SensaCalibNet (see src/model.py) drawn with
     matplotlib patches. Independent of the data — included so the paper
     can present what the network *would* learn given more data.

──────────────────────────────────────────────────────────────────────────────
INPUT FILES
──────────────────────────────────────────────────────────────────────────────
  data/paired/paired_dataset.csv         — produced by prepare_purpleair_data.py.
                                           Provides the raw SEN55 vs reference
                                           pairs (the "no calibration" baseline).
  models_linear/linear_calibration.json  — produced by linear_calibration.py.
                                           Holds per-fold predictions for the
                                           linear baseline.
  models_finetuned/cv_results.json       — produced by `finetune.py --cv K`.
                                           Holds per-fold predictions for the
                                           fine-tuned neural network.

If a method's JSON is missing, that method is silently skipped (the figures
still render with whatever is available). To regenerate everything from scratch:

    python prepare_purpleair_data.py
    python linear_calibration.py
    python finetune.py --cv 17 --freeze-conv
    python paper_figures.py

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────
    python paper_figures.py
    python paper_figures.py --outdir paper_figures
    python paper_figures.py --paired data/paired/paired_dataset.csv \\
                            --nn  models_finetuned/cv_results.json \\
                            --linear models_linear/linear_calibration.json

Output: 300-dpi PNG plus vector PDF for each figure, in --outdir.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import r2_score


# Okabe-Ito-derived palette, one colour per method.
METHOD_COLORS = {
    "raw":             "#999999",
    "nn":              "#0072B2",
    "nn_head":         "#56B4E9",
    "linear":          "#009E73",
    "linear_humidity": "#CC79A7",
    "aqs_linear":      "#E69F00",
}
METHOD_LABELS = {
    "raw":             "Raw SEN55 (no calibration)",
    "nn":              "Fine-tuned NN (transfer learning)",
    "nn_head":         "Fine-tuned NN (head-only, 17 params)",
    "linear":          "Linear baseline",
    "linear_humidity": "Linear + humidity correction",
    "aqs_linear":      "AQS frozen + 2-param head",
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class MethodResult:
    """One calibration method's predictions, per-fold breakdown, and label."""
    key:         str                # 'raw' / 'nn' / 'linear'
    label:       str
    predictions: np.ndarray         # shape (n_samples,)
    references:  np.ndarray         # shape (n_samples,)
    fold:        Optional[np.ndarray]  # shape (n_samples,) or None for raw
    color:       str

    @property
    def errors(self) -> np.ndarray:
        return self.predictions - self.references

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.errors)))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.errors ** 2)))

    @property
    def mbe(self) -> float:
        return float(np.mean(self.errors))

    @property
    def r2(self) -> float:
        if len(self.references) < 2:
            return float("nan")
        return float(r2_score(self.references, self.predictions))

    def per_fold_mae(self) -> np.ndarray:
        """Return MAE per fold, or a single-element array if no fold info."""
        if self.fold is None:
            return np.array([self.mae])
        out = []
        for f in np.unique(self.fold):
            mask = self.fold == f
            out.append(float(np.mean(np.abs(self.errors[mask]))))
        return np.array(out)


# ═════════════════════════════════════════════════════════════════════════════
# LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_raw_sen55(paired_csv: Path) -> Optional[MethodResult]:
    """Build the 'no calibration' baseline from the paired CSV."""
    if not paired_csv.exists():
        print(f"  [skip] paired CSV not found: {paired_csv}")
        return None
    df = pd.read_csv(paired_csv)
    if "pm2_5" not in df.columns or "bam_pm2_5" not in df.columns:
        print(f"  [skip] {paired_csv} missing pm2_5 / bam_pm2_5 columns")
        return None
    return MethodResult(
        key="raw",
        label=METHOD_LABELS["raw"],
        predictions=df["pm2_5"].to_numpy(dtype=float),
        references=df["bam_pm2_5"].to_numpy(dtype=float),
        fold=None,
        color=METHOD_COLORS["raw"],
    )


def load_method_json(path: Path, key: str) -> Optional[MethodResult]:
    """Read a *_results.json or linear_calibration.json file."""
    if not path.exists():
        print(f"  [skip] {key} results file not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    if "predictions" not in data:
        print(f"  [skip] {path} has no 'predictions' list — was it written by "
              "an older version of the script?")
        return None
    preds = np.array([p["prediction"] for p in data["predictions"]], dtype=float)
    refs  = np.array([p["reference"]  for p in data["predictions"]], dtype=float)
    fold  = np.array([p["fold"]       for p in data["predictions"]], dtype=int)
    return MethodResult(
        key=key,
        label=METHOD_LABELS[key],
        predictions=preds,
        references=refs,
        fold=fold,
        color=METHOD_COLORS[key],
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — METHOD COMPARISON SCATTER
# ═════════════════════════════════════════════════════════════════════════════
def plot_method_comparison(methods: list[MethodResult], save_path: Path) -> None:
    """One scatter per method, predicted vs reference, with y=x and an OLS fit."""
    n = len(methods)
    # Lay out as a grid: 1 row for ≤3 methods, 2 rows for ≤6, 3 rows otherwise.
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    all_refs   = np.concatenate([m.references  for m in methods])
    all_preds  = np.concatenate([m.predictions for m in methods])
    lo = float(min(all_refs.min(), all_preds.min())) - 1.0
    hi = float(max(all_refs.max(), all_preds.max())) + 1.0
    line = np.array([lo, hi])

    for ax, m in zip(axes, methods):
        ax.plot(line, line, "r--", linewidth=1.4, label="y = x")
        if len(m.references) >= 2:
            slope, intercept = np.polyfit(m.references, m.predictions, 1)
            ax.plot(line, slope * line + intercept, color="navy",
                    linewidth=1.4,
                    label=f"fit: y = {slope:.2f}x {intercept:+.2f}")
        ax.scatter(m.references, m.predictions, s=42, color=m.color,
                   edgecolor="white", alpha=0.92, linewidth=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Reference PM2.5 (µg/m³)", fontsize=11)
        ax.set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=11)
        ax.set_title(m.label, fontsize=12, pad=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.text(0.98, 0.05,
                f"MAE  = {m.mae:.2f}\nRMSE = {m.rmse:.2f}\n"
                f"MBE  = {m.mbe:+.2f}\nR²   = {m.r2:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="#aaaaaa", alpha=0.95))

    # Hide any unused grid cells.
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("PM2.5 calibration — predicted vs PurpleAir reference  "
                 f"(N = {len(methods[0].references)} held-out predictions)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PER-FOLD MAE VIOLIN PLOT
# ═════════════════════════════════════════════════════════════════════════════
def plot_per_fold_mae(methods: list[MethodResult], save_path: Path) -> None:
    """Violin of per-fold MAE, one violin per method."""
    # Drop methods that have no fold information (e.g. 'raw' baseline).
    cv_methods = [m for m in methods if m.fold is not None]
    if not cv_methods:
        print("  [warn] no methods have per-fold info — skipping violin plot")
        return

    fig, ax = plt.subplots(figsize=(max(8.5, 2.2 * len(cv_methods)), 6.0))
    positions = np.arange(1, len(cv_methods) + 1)
    data = [m.per_fold_mae() for m in cv_methods]

    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=False, showextrema=False)
    for body, m in zip(parts["bodies"], cv_methods):
        body.set_facecolor(m.color); body.set_edgecolor("black")
        body.set_alpha(0.55); body.set_linewidth(1.0)

    rng = np.random.default_rng(0)
    annotations: list[tuple[float, str, str]] = []
    for pos, m, vals in zip(positions, cv_methods, data):
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(pos + jitter, vals, s=22, color=m.color,
                   edgecolor="black", linewidth=0.5, zorder=3, alpha=0.92)
        mean, median = float(np.mean(vals)), float(np.median(vals))
        ax.scatter(pos, mean, marker="D", s=55, color="white",
                   edgecolor="black", linewidth=1.2, zorder=4)
        ax.hlines(median, pos - 0.22, pos + 0.22, color="black",
                  linewidth=1.5, zorder=4)
        # Collect annotation strings to draw via fig.text below the axis,
        # so they sit below the rotated x-tick labels instead of overlapping.
        annotations.append((
            pos,
            m.color,
            f"agg MAE = {m.mae:.2f}\nfold mean ± std\n"
            f"{mean:.2f} ± {np.std(vals):.2f}",
        ))

    # Compact labels: drop the parenthetical so x-axis stays readable.
    def _short(label: str) -> str:
        return (label.split(" (")[0]
                if " (" in label and "transfer" not in label.lower()
                else label.replace(" (transfer learning)", "")
                          .replace(" (no calibration)", ""))
    ax.set_xticks(positions)
    ax.set_xticklabels([_short(m.label) for m in cv_methods],
                       fontsize=9.5, rotation=15, ha="right")
    ax.set_ylabel("Per-fold MAE (µg/m³)", fontsize=12)
    n_folds = len(cv_methods[0].per_fold_mae())
    ax.set_title(f"Cross-validation MAE per fold  ({n_folds}-fold CV, "
                 f"N = {len(cv_methods[0].references)} samples)",
                 fontsize=13, pad=12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6); ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markeredgecolor="black", markersize=7, label="Individual fold"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9,
              framealpha=0.95)

    # Reserve space below the axis: rotated tick labels on top, annotation
    # boxes below them. Convert each violin's x-position to figure
    # coordinates so the boxes line up with their violin.
    fig.subplots_adjust(bottom=0.32)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for pos, color, text in annotations:
        x_data = ax.transData.transform((pos, 0))[0]
        x_fig = inv.transform((x_data, 0))[0]
        fig.text(x_fig, 0.04, text, ha="center", va="bottom",
                 fontsize=8.5, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=color, linewidth=1.0, alpha=0.95))
    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — AGGREGATE METRICS TABLE
# ═════════════════════════════════════════════════════════════════════════════
def plot_metrics_table(methods: list[MethodResult], save_path: Path) -> None:
    """Render a clean aggregate-metrics table for direct inclusion in the paper."""
    rows = []
    for m in methods:
        rows.append([
            m.label,
            f"{m.mae:.2f}",
            f"{m.rmse:.2f}",
            f"{m.mbe:+.2f}",
            f"{m.r2:.3f}",
        ])

    cols = ["Method", "MAE (µg/m³)", "RMSE (µg/m³)", "MBE (µg/m³)", "R²"]

    fig, ax = plt.subplots(figsize=(11.5, 0.7 + 0.55 * len(rows)))
    ax.axis("off")

    # Wider first column for the method label so it doesn't clip.
    col_widths = [0.42, 0.145, 0.145, 0.145, 0.145]
    table = ax.table(
        cellText=rows, colLabels=cols, loc="center", cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)
    # Left-align the method-name column for readability.
    for r in range(len(rows) + 1):
        table[r, 0].set_text_props(ha="left")
        table[r, 0].PAD = 0.04

    # Header style + per-method tint on the Method column.
    n_cols = len(cols)
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    for r, m in enumerate(methods, start=1):
        # Tint just the leading "Method" column.
        method_cell = table[r, 0]
        method_cell.set_facecolor(m.color)
        method_cell.set_alpha(0.35)
        method_cell.set_text_props(fontweight="bold")
        # Highlight the lowest-MAE row by drawing a green border.
        if m.mae == min(x.mae for x in methods):
            for c in range(n_cols):
                table[r, c].set_edgecolor("#009E73")
                table[r, c].set_linewidth(2.0)

    ax.set_title("Aggregate cross-validation accuracy by method",
                 fontsize=13, pad=12)
    fig.tight_layout()
    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — CNN ARCHITECTURE DIAGRAM
# ═════════════════════════════════════════════════════════════════════════════
def plot_cnn_architecture(save_path: Path, n_features: int = 3,
                          n_channels: int = 16) -> None:
    """
    Block diagram of SensaCalibNet (see src/model.py).

    Default geometry shows the 3-feature transfer-learning configuration used
    by finetune.py; override --n-features / --n-channels for the 8-feature
    from-scratch (main.py) configuration.
    """
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    c = n_channels
    stages = [
        (8,  "Input",            f"SEN55 reading\n{n_features} features",
         f"({n_features},)",                       "#E8E8E8"),
        (26, "Conv1d Block 1",   f"Conv1d 1→{c}, k=3, pad=1\n"
                                 f"BatchNorm → ReLU",
         f"({c}, {n_features})",                   "#9ECAE1"),
        (44, "Conv1d Block 2",   f"Conv1d {c}→{2*c}, k=3, pad=1\n"
                                 f"BatchNorm → ReLU",
         f"({2*c}, {n_features})",                 "#6BAED6"),
        (61, "Global Avg Pool",  "AdaptiveAvgPool1d\nover feature axis",
         f"({2*c},)",                              "#FDD0A2"),
        (77, "Regression Head",  f"Linear {2*c}→{c} → ReLU\n"
                                 f"Linear {c}→1",
         "(1,)",                                   "#A1D99B"),
        (93, "AQI Category",     "Threshold on\nEPA AQI breakpoints",
         "5 classes",                              "#FCBBA1"),
    ]

    box_w, box_h, y0 = 13.0, 9.0, 13.0
    for i, (xc, title, sub, shape, color) in enumerate(stages):
        box = FancyBboxPatch(
            (xc - box_w / 2, y0), box_w, box_h,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=color, edgecolor="black", linewidth=1.4)
        ax.add_patch(box)
        ax.text(xc, y0 + box_h - 1.9, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(xc, y0 + box_h / 2 - 1.4, sub, ha="center", va="center",
                fontsize=8.3, color="#222222")
        ax.text(xc, y0 - 2.3, shape, ha="center", va="center",
                fontsize=8.5, style="italic", color="#444444",
                family="monospace")
        if i < len(stages) - 1:
            x_next = stages[i + 1][0]
            arrow = FancyArrowPatch(
                (xc + box_w / 2, y0 + box_h / 2),
                (x_next - box_w / 2, y0 + box_h / 2),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=1.6, color="#333333")
            ax.add_patch(arrow)

    def _bracket(x_left, x_right, label):
        y = y0 - 5.2
        ax.plot([x_left, x_left, x_right, x_right],
                [y + 0.8, y, y, y + 0.8], color="#666666", linewidth=1.2)
        ax.text((x_left + x_right) / 2, y - 1.4, label, ha="center",
                va="center", fontsize=9, style="italic", color="#666666")

    _bracket(26 - box_w / 2, 44 + box_w / 2, "Convolutional feature extractor")
    _bracket(61 - box_w / 2, 77 + box_w / 2, "Calibration regressor")

    ax.set_title(
        "SensaCalibNet — 1D CNN Architecture for SEN55 PM2.5 Calibration",
        fontsize=13.5, fontweight="bold", pad=14)
    ax.text(50, 1.2,
            f"~2.2k trainable parameters  ·  int8-quantised for "
            f"ESP32-S3 deployment  ·  scalar PM2.5 output binned into "
            f"5 EPA AQI categories",
            ha="center", va="center", fontsize=8.5, color="#666666")
    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _save(fig, save_path: Path) -> None:
    """Write the figure as both PNG (300 dpi raster) and PDF (vector)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {save_path}")
    print(f"  saved  {pdf_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate paper figures from the actual PM2.5 "
                    "calibration results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--paired",
        default=str(script_dir / "data" / "paired" / "paired_dataset.csv"),
        help="Paired SEN55 + reference dataset (raw baseline).")
    parser.add_argument(
        "--nn",
        default=str(script_dir / "models_finetuned" / "cv_results.json"),
        help="Fine-tuned NN CV results JSON (from finetune.py --cv K).")
    parser.add_argument(
        "--linear",
        default=str(script_dir / "models_linear" / "linear_calibration.json"),
        help="Linear baseline CV results JSON (from linear_calibration.py).")
    parser.add_argument(
        "--linear-humidity",
        default=str(script_dir / "models_linear" / "linear_calibration_humidity.json"),
        help="Linear + humidity correction CV results JSON "
             "(from linear_calibration.py --humidity-correction).")
    parser.add_argument(
        "--aqs-linear",
        default=str(script_dir / "models_aqs_corrected" / "aqs_linear_correction.json"),
        help="AQS-frozen + 2-param head CV results JSON "
             "(from aqs_linear_correction.py).")
    parser.add_argument(
        "--nn-head",
        default=str(script_dir / "models_finetuned" / "cv_results_head.json"),
        help="Head-only NN CV results JSON "
             "(rename cv_results.json to cv_results_head.json after running "
             "`finetune.py --cv 17 --freeze-all-but-head`).")
    parser.add_argument(
        "--outdir", default=str(script_dir / "paper_figures"),
        help="Directory where the figure files are written.")
    parser.add_argument(
        "--n-features", type=int, default=3,
        help="CNN input feature count for the architecture figure "
             "(3 for the fine-tuned model, 8 for from-scratch).")
    parser.add_argument(
        "--n-channels", type=int, default=16,
        help="CNN base conv width for the architecture figure.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    print(f"\nWriting figures to: {outdir}\n")

    # ── Load every method that has results on disk ──────────────────────────
    print("Loading method results")
    methods: list[MethodResult] = []
    raw = load_raw_sen55(Path(args.paired))
    if raw is not None:
        methods.append(raw)
    for path, key in [
        (args.nn,              "nn"),
        (args.nn_head,         "nn_head"),
        (args.linear,          "linear"),
        (args.linear_humidity, "linear_humidity"),
        (args.aqs_linear,      "aqs_linear"),
    ]:
        m = load_method_json(Path(path), key)
        if m is not None:
            methods.append(m)

    if not methods:
        raise SystemExit(
            "\nERROR: no method results could be loaded. Run "
            "prepare_purpleair_data.py, linear_calibration.py, and "
            "`finetune.py --cv 17 --freeze-conv` first.")

    print()
    for m in methods:
        n_folds = len(np.unique(m.fold)) if m.fold is not None else "N/A"
        print(f"  {m.key:7s}  N={len(m.references):3d}  "
              f"folds={n_folds:>3}  "
              f"MAE={m.mae:6.3f}  RMSE={m.rmse:6.3f}  "
              f"MBE={m.mbe:+6.3f}  R²={m.r2:7.3f}")

    # ── Figure 1: predicted vs reference scatter ────────────────────────────
    print("\nFigure 1 — method comparison scatter")
    plot_method_comparison(methods, outdir / "fig_method_comparison.png")

    # ── Figure 2: per-fold MAE violin ───────────────────────────────────────
    print("\nFigure 2 — per-fold MAE distribution")
    plot_per_fold_mae(methods, outdir / "fig_per_fold_mae.png")

    # ── Figure 3: aggregate metrics table ───────────────────────────────────
    print("\nFigure 3 — aggregate metrics table")
    plot_metrics_table(methods, outdir / "fig_metrics_table.png")

    # ── Figure 4: CNN architecture (no result data needed) ──────────────────
    print("\nFigure 4 — CNN architecture diagram")
    plot_cnn_architecture(outdir / "fig_cnn_architecture.png",
                          n_features=args.n_features,
                          n_channels=args.n_channels)

    print("\nDone.")


if __name__ == "__main__":
    main()

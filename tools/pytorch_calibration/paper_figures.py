"""
paper_figures.py — Generate publication-quality matplotlib figures for the
Sensa report/paper from the AI calibration model's evaluation results.

This script produces three figures:

  1. K-FOLD CROSS-VALIDATION VIOLIN PLOT
     Distribution of six classification metrics — Accuracy, Precision,
     F1-score, Recall, Specificity, Sensitivity — across the k folds.
     A violin plot shows the full spread (not just mean ± std), so the
     reader can see whether performance is stable or fold-dependent.

  2. CONFUSION MATRIX
     A classic 5×5 confusion matrix over the EPA AQI categories
     (GOOD, MODERATE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS). Each cell is
     annotated with the count and the row-normalised percentage.

  3. CNN ARCHITECTURE DIAGRAM
     A block diagram of SensaCalibNet (see src/model.py) drawn with
     matplotlib patches — input features, the two Conv1d blocks, global
     average pooling, the regression head, and the downstream AQI
     categorisation step that turns the scalar PM2.5 output into one of
     the five classes scored by figures 1 and 2.

──────────────────────────────────────────────────────────────────────────────
INPUT FILE FORMATS
──────────────────────────────────────────────────────────────────────────────
The script reads two result files. Neither is required to run — if a file is
missing, synthetic demo data is substituted so the figure still renders (with
a clear "[DEMO DATA]" banner on the figure). Replace with real results before
using the figures in the paper.

  --kfold  <path>   CSV, one row per fold. Required column 'fold' plus the six
                    metric columns. Header names are case-insensitive and may
                    use any of these aliases:
                        accuracy
                        precision
                        f1 | f1_score | f1-score
                        recall
                        specificity
                        sensitivity
                    Metric values may be fractions (0-1) or percentages
                    (0-100); the script auto-detects and normalises to 0-1.
                    For the 5-class AQI task, precision/recall/F1/specificity/
                    sensitivity are expected to be macro-averaged across the
                    five classes (one-vs-rest), so each fold contributes one
                    value per metric.

                    Example (analysis/kfold_results.example.csv):
                        fold,accuracy,precision,f1,recall,specificity,sensitivity
                        1,0.94,0.91,0.90,0.89,0.97,0.89
                        2,0.93,0.90,0.89,0.88,0.97,0.88
                        ...

  --confusion <path>  CSV holding a 5×5 integer matrix of sample counts.
                    Rows = TRUE AQI category, columns = PREDICTED AQI category,
                    both in the fixed order GOOD, MODERATE, UNHEALTHY,
                    VERY_UNHEALTHY, HAZARDOUS. A header row/index column is
                    optional — if present it is ignored and the fixed order is
                    assumed. Typically this is the matrix summed over all
                    held-out fold predictions.

                    Example (analysis/confusion_matrix.example.csv):
                        ,GOOD,MODERATE,UNHEALTHY,VERY_UNHEALTHY,HAZARDOUS
                        GOOD,820,14,0,0,0
                        MODERATE,22,460,9,0,0
                        ...

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────
    # Real results:
    python paper_figures.py --kfold analysis/kfold_results.csv \\
                            --confusion analysis/confusion_matrix.csv \\
                            --outdir paper_figures

    # Demo (synthetic data, no result files needed) — useful for checking
    # layout/styling before the real numbers are in:
    python paper_figures.py --demo

Output: three PNG files (300 dpi) plus matching PDF (vector) copies in
--outdir, suitable for direct inclusion in a LaTeX/Word document.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed, just writes files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── EPA AQI categories ───────────────────────────────────────────────────────
# These mirror the Classification enum used by the firmware classifier
# (lib/classify/src/classify.cpp) and the breakpoints in
# include/aqi_thresholds.h. Order is fixed and is the row/column order assumed
# for the confusion matrix.
AQI_CLASSES = ["GOOD", "MODERATE", "UNHEALTHY", "VERY_UNHEALTHY", "HAZARDOUS"]
AQI_LABELS = ["Good", "Moderate", "Unhealthy", "Very\nUnhealthy", "Hazardous"]

# ── The six metrics, in display order ────────────────────────────────────────
# Each entry: canonical key, display label, accepted header aliases.
METRICS = [
    ("accuracy",    "Accuracy",    ("accuracy", "acc")),
    ("precision",   "Precision",   ("precision", "prec")),
    ("f1",          "F1-score",    ("f1", "f1_score", "f1-score", "f1score")),
    ("recall",      "Recall",      ("recall",)),
    ("specificity", "Specificity", ("specificity", "spec")),
    ("sensitivity", "Sensitivity", ("sensitivity", "sens")),
]

# Colour-blind-safe palette (Okabe-Ito), one colour per metric.
METRIC_COLORS = ["#0072B2", "#E69F00", "#009E73",
                 "#D55E00", "#56B4E9", "#CC79A7"]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def _normalise_metric_scale(values: np.ndarray) -> np.ndarray:
    """
    Return metric values on the 0-1 scale.

    Accepts either fractions (0-1) or percentages (0-100). If any value exceeds
    1.0 the column is assumed to be in percent and divided by 100.
    """
    values = np.asarray(values, dtype=float)
    if np.nanmax(values) > 1.0:
        values = values / 100.0
    return values


def load_fold_metrics(path: Path) -> pd.DataFrame:
    """
    Load per-fold classification metrics from a CSV file.

    Returns a DataFrame with exactly the six canonical metric columns
    (accuracy, precision, f1, recall, specificity, sensitivity), one row per
    fold, all values on the 0-1 scale.

    Raises:
        FileNotFoundError: if `path` does not exist (caller handles the
                           demo-data fallback).
        ValueError:        if a required metric column cannot be found.
    """
    df_raw = pd.read_csv(path)
    # Normalise header names so alias matching is case/space insensitive.
    lookup = {c.strip().lower().replace(" ", "_"): c for c in df_raw.columns}

    out = {}
    for key, label, aliases in METRICS:
        col = next((lookup[a] for a in aliases if a in lookup), None)
        if col is None:
            raise ValueError(
                f"Column for metric '{label}' not found in {path}. "
                f"Looked for any of: {', '.join(aliases)}. "
                f"Found columns: {', '.join(df_raw.columns)}"
            )
        out[key] = _normalise_metric_scale(df_raw[col].to_numpy())

    return pd.DataFrame(out)


def load_confusion_matrix(path: Path) -> np.ndarray:
    """
    Load a 5×5 AQI confusion matrix of sample counts from a CSV file.

    A header row and/or index column are tolerated — the loader keeps only the
    numeric 5×5 block and assumes the fixed AQI_CLASSES order. Rows are the
    TRUE category, columns the PREDICTED category.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError:        if the numeric block is not 5×5.
    """
    # header=None first so we can detect whether row 0 / col 0 are labels.
    df = pd.read_csv(path, header=None)
    # Drop any fully non-numeric leading row or column (the header/index).
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cm = df.to_numpy(dtype=float)

    if cm.shape != (len(AQI_CLASSES), len(AQI_CLASSES)):
        raise ValueError(
            f"Confusion matrix in {path} is {cm.shape}, expected "
            f"{(len(AQI_CLASSES), len(AQI_CLASSES))} (5 AQI classes)."
        )
    return cm.astype(int)


def demo_fold_metrics(n_folds: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic per-fold metrics for layout/preview purposes only.

    NOT REAL RESULTS — values are drawn from plausible-looking distributions so
    the violin plot can be styled before the genuine k-fold numbers exist.
    """
    rng = np.random.default_rng(seed)
    # Plausible mean / spread for each metric (macro-averaged, 5-class AQI).
    centres = {"accuracy": 0.93, "precision": 0.89, "f1": 0.88,
               "recall": 0.87, "specificity": 0.97, "sensitivity": 0.87}
    data = {}
    for key, _, _ in METRICS:
        vals = rng.normal(centres[key], 0.025, n_folds)
        data[key] = np.clip(vals, 0.0, 1.0)
    return pd.DataFrame(data)


def demo_confusion_matrix(seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic 5×5 AQI confusion matrix for preview purposes only.

    NOT REAL RESULTS — most mass sits on the diagonal with small off-diagonal
    leakage into neighbouring categories, as a real classifier would show.
    """
    rng = np.random.default_rng(seed)
    # Rough class prevalence in low-cost-sensor data: mostly Good/Moderate.
    support = np.array([900, 520, 180, 60, 25])
    cm = np.zeros((5, 5), dtype=int)
    for i, n in enumerate(support):
        # ~92% correct; the rest leaks mostly to adjacent categories.
        probs = np.zeros(5)
        probs[i] = 0.92
        if i > 0:
            probs[i - 1] = 0.05
        if i < 4:
            probs[i + 1] += 0.03
        probs /= probs.sum()
        cm[i] = rng.multinomial(n, probs)
    return cm


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — K-FOLD CROSS-VALIDATION VIOLIN PLOT
# ═════════════════════════════════════════════════════════════════════════════
def plot_kfold_violin(metrics: pd.DataFrame, save_path: Path,
                      is_demo: bool = False) -> None:
    """
    Draw a violin plot of the six classification metrics across the k folds.

    For each metric the violin shows the kernel-density estimate of the
    per-fold values; the individual fold values are overlaid as jittered dots,
    and the mean is marked with a white diamond and printed beneath the axis.

    Args:
        metrics:   DataFrame with the six canonical metric columns, one row
                   per fold (output of load_fold_metrics / demo_fold_metrics).
        save_path: Destination path; a .pdf sibling is also written.
        is_demo:   If True, stamp a "[DEMO DATA]" banner on the figure.
    """
    n_folds = len(metrics)
    keys = [k for k, _, _ in METRICS]
    labels = [lbl for _, lbl, _ in METRICS]
    data = [metrics[k].to_numpy() for k in keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, len(keys) + 1)

    # ── Violins ──────────────────────────────────────────────────────────────
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], METRIC_COLORS):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
        body.set_linewidth(1.0)

    # ── Per-fold points (jittered) + mean / median markers ───────────────────
    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, METRIC_COLORS):
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(pos + jitter, vals, s=22, color=color, edgecolor="black",
                   linewidth=0.5, zorder=3, alpha=0.9)
        # Mean — white diamond; median — short black bar.
        mean, median = float(np.mean(vals)), float(np.median(vals))
        ax.scatter(pos, mean, marker="D", s=55, color="white",
                   edgecolor="black", linewidth=1.2, zorder=4)
        ax.hlines(median, pos - 0.22, pos + 0.22, color="black",
                  linewidth=1.5, zorder=4)
        # Annotate mean ± std in the empty space below each violin.
        ax.text(pos, 0.10, f"{mean:.3f}\n± {np.std(vals):.3f}",
                ha="center", va="center", fontsize=9, color="#333333",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, linewidth=1.0, alpha=0.95))

    # ── Cosmetics ────────────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.4, len(keys) + 0.6)
    ax.set_title(
        f"Distribution of {n_folds}-Fold Cross-Validation Performance\n"
        f"(5-class EPA AQI classification, macro-averaged metrics)",
        fontsize=13, pad=12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    # Legend explaining the overlay markers.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markeredgecolor="black", markersize=7, label="Individual fold"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
    ]
    # Anchored just above the mean±std boxes, in the violins' empty lower band.
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.012, 0.17), fontsize=9, framealpha=0.95)

    _stamp_demo(fig, is_demo)
    fig.tight_layout()
    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm: np.ndarray, save_path: Path,
                          is_demo: bool = False) -> None:
    """
    Draw a classic confusion matrix over the five EPA AQI categories.

    Cells are coloured by row-normalised rate (so class imbalance does not wash
    out the colour scale) and annotated with the raw count and that rate.
    Overall accuracy (trace / total) is printed in the title.

    Args:
        cm:        5×5 integer array; rows = true class, cols = predicted.
        save_path: Destination path; a .pdf sibling is also written.
        is_demo:   If True, stamp a "[DEMO DATA]" banner on the figure.
    """
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero for any class with no true samples.
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm),
                        where=row_sums != 0)
    total = cm.sum()
    accuracy = np.trace(cm) / total if total else 0.0

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised rate", fontsize=10)

    n = len(AQI_CLASSES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(AQI_LABELS, fontsize=10)
    ax.set_yticklabels(AQI_LABELS, fontsize=10)
    ax.set_xlabel("Predicted AQI category", fontsize=12)
    ax.set_ylabel("True AQI category", fontsize=12)
    ax.set_title(f"Confusion Matrix — EPA AQI Classification\n"
                 f"Overall accuracy = {accuracy:.3f}  (N = {int(total):,})",
                 fontsize=13, pad=12)

    # Annotate every cell: count on top, row-rate below.
    for i in range(n):
        for j in range(n):
            # White text on dark cells, black on light cells.
            txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i - 0.13, f"{int(cm[i, j]):,}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=txt_color)
            ax.text(j, i + 0.20, f"{cm_norm[i, j] * 100:.1f}%",
                    ha="center", va="center", fontsize=8.5, color=txt_color)

    # Light gridlines between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    _stamp_demo(fig, is_demo)
    fig.tight_layout()
    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CNN ARCHITECTURE DIAGRAM
# ═════════════════════════════════════════════════════════════════════════════
def plot_cnn_architecture(save_path: Path, n_features: int = 8,
                          n_channels: int = 16) -> None:
    """
    Draw a block diagram of SensaCalibNet (see src/model.py).

    The diagram is a left-to-right data-flow chain: input feature vector → two
    Conv1d + BatchNorm + ReLU blocks → global average pool → two-layer
    regression head → scalar calibrated PM2.5 → AQI categorisation into the
    five classes scored by the other two figures.

    Args:
        save_path:  Destination path; a .pdf sibling is also written.
        n_features: Number of input SEN55 channels (config.yaml model.n_features).
        n_channels: Base conv width (config.yaml model.n_channels). The second
                    conv block is 2× this.
    """
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    # Each stage: (x-centre, label, sublabel, output-shape, facecolor).
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

    box_w, box_h, y0 = 13.0, 9.0, 13.0  # width, height, bottom-y of each box

    for i, (xc, title, sub, shape, color) in enumerate(stages):
        # ── Box ──────────────────────────────────────────────────────────────
        box = FancyBboxPatch(
            (xc - box_w / 2, y0), box_w, box_h,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=color, edgecolor="black", linewidth=1.4)
        ax.add_patch(box)
        ax.text(xc, y0 + box_h - 1.9, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(xc, y0 + box_h / 2 - 1.4, sub, ha="center", va="center",
                fontsize=8.3, color="#222222")
        # Output tensor shape printed beneath each box.
        ax.text(xc, y0 - 2.3, shape, ha="center", va="center",
                fontsize=8.5, style="italic", color="#444444",
                family="monospace")

        # ── Arrow to the next stage ──────────────────────────────────────────
        if i < len(stages) - 1:
            x_next = stages[i + 1][0]
            arrow = FancyArrowPatch(
                (xc + box_w / 2, y0 + box_h / 2),
                (x_next - box_w / 2, y0 + box_h / 2),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=1.6, color="#333333")
            ax.add_patch(arrow)

    # ── Brackets grouping the conceptual sections ────────────────────────────
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

    # Footnote with the parameter budget context from src/model.py.
    ax.text(50, 1.2,
            f"~2.2k trainable parameters  ·  int8-quantised for "
            f"ESP32-S3 deployment  ·  scalar PM2.5 output binned into "
            f"5 EPA AQI categories",
            ha="center", va="center", fontsize=8.5, color="#666666")

    _save(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _stamp_demo(fig, is_demo: bool) -> None:
    """Overlay a translucent '[DEMO DATA]' banner when synthetic data is used."""
    if not is_demo:
        return
    fig.text(0.5, 0.5, "[DEMO DATA]", fontsize=46, color="red",
             alpha=0.16, ha="center", va="center", rotation=25,
             fontweight="bold", zorder=10)


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
        description="Generate paper figures from AI calibration results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--kfold", default=str(script_dir / "analysis" / "kfold_results.csv"),
        help="CSV of per-fold classification metrics.")
    parser.add_argument(
        "--confusion",
        default=str(script_dir / "analysis" / "confusion_matrix.csv"),
        help="CSV of the 5x5 AQI confusion matrix (counts).")
    parser.add_argument(
        "--outdir", default=str(script_dir / "paper_figures"),
        help="Directory where the figure files are written.")
    parser.add_argument(
        "--demo", action="store_true",
        help="Ignore the input files and use synthetic demo data.")
    parser.add_argument(
        "--n-features", type=int, default=8,
        help="CNN input feature count (for the architecture figure).")
    parser.add_argument(
        "--n-channels", type=int, default=16,
        help="CNN base conv width (for the architecture figure).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    print(f"\nWriting figures to: {outdir}\n")

    # ── Figure 1: k-fold violin plot ─────────────────────────────────────────
    print("Figure 1 — k-fold cross-validation violin plot")
    kfold_path = Path(args.kfold)
    if args.demo or not kfold_path.exists():
        if not args.demo:
            print(f"  [!] {kfold_path} not found — using demo data.")
        metrics = demo_fold_metrics()
        kfold_is_demo = True
    else:
        metrics = load_fold_metrics(kfold_path)
        print(f"  loaded {len(metrics)} folds from {kfold_path}")
        kfold_is_demo = False
    plot_kfold_violin(metrics, outdir / "fig_kfold_violin.png",
                      is_demo=kfold_is_demo)

    # ── Figure 2: confusion matrix ───────────────────────────────────────────
    print("\nFigure 2 — confusion matrix")
    conf_path = Path(args.confusion)
    if args.demo or not conf_path.exists():
        if not args.demo:
            print(f"  [!] {conf_path} not found — using demo data.")
        cm = demo_confusion_matrix()
        conf_is_demo = True
    else:
        cm = load_confusion_matrix(conf_path)
        print(f"  loaded confusion matrix from {conf_path}")
        conf_is_demo = False
    plot_confusion_matrix(cm, outdir / "fig_confusion_matrix.png",
                          is_demo=conf_is_demo)

    # ── Figure 3: CNN architecture (no result data needed) ───────────────────
    print("\nFigure 3 — CNN architecture diagram")
    plot_cnn_architecture(outdir / "fig_cnn_architecture.png",
                          n_features=args.n_features,
                          n_channels=args.n_channels)

    print("\nDone.")


if __name__ == "__main__":
    main()

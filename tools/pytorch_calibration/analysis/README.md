# analysis/ — Result files for the paper figures

`../paper_figures.py` reads two result files from this directory and turns them
into publication figures. Drop your real k-fold results here using the same
column layout as the `.example.csv` templates, then run:

```bash
cd tools/pytorch_calibration
python paper_figures.py            # uses analysis/kfold_results.csv + confusion_matrix.csv
python paper_figures.py --demo     # synthetic data, no result files needed
```

Figures are written to `../paper_figures/` as 300 dpi PNG **and** vector PDF.

## `kfold_results.csv`

One row per cross-validation fold. Columns (case-insensitive, aliases allowed):

| Column        | Meaning                                                       |
|---------------|---------------------------------------------------------------|
| `fold`        | Fold index (1…k). Not plotted directly; just for your records.|
| `accuracy`    | Overall classification accuracy on the held-out fold.         |
| `precision`   | Macro-averaged precision across the 5 AQI classes.            |
| `f1`          | Macro-averaged F1-score (aliases: `f1_score`, `f1-score`).    |
| `recall`      | Macro-averaged recall across the 5 AQI classes.               |
| `specificity` | Macro-averaged one-vs-rest specificity.                       |
| `sensitivity` | Macro-averaged sensitivity (equals recall for this task).     |

Values may be fractions (0–1) or percentages (0–100) — the script auto-detects
and normalises. See `kfold_results.example.csv`.

## `confusion_matrix.csv`

A 5×5 integer matrix of sample counts. **Rows = true category, columns =
predicted category**, both in the fixed order:

`GOOD, MODERATE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS`

(matching the firmware `Classification` enum in `lib/classify/src/classify.cpp`).
A header row and index column are optional and ignored. Typically this is the
matrix summed over every fold's held-out predictions. See
`confusion_matrix.example.csv`.

> The `.example.csv` files contain **illustrative placeholder numbers only** —
> replace them with real results before using any figure in the paper.

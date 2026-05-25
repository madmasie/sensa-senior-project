"""
prepare_purpleair_data.py — Pair SEN55 .pkl recordings with PurpleAir reference
PM2.5 readings collected during the same co-location windows.

DATA LAYOUT EXPECTED:
    <repo>/data/
        purpleairdata.csv            ← one row per co-location window
        sen55_YYYY-MM-DD_HH-MM-SS.pkl ← raw SEN55 captures (~10 min each)

    purpleairdata.csv columns:
        filename   — name of the matching sen55_*.pkl file
        pm2.5_1    — PurpleAir channel A average over that window  (µg/m³)
        pm2.5_2    — PurpleAir channel B average over that window  (µg/m³)

WHAT THIS DOES:
    1. Reads purpleairdata.csv.
    2. For each row, opens the corresponding sen55_*.pkl.
    3. Averages all 8 SEN55 channels over the file's full window.
    4. Pairs that average with mean(pm2.5_1, pm2.5_2) as bam_pm2_5.
    5. Writes paired_dataset.csv in the schema main.py / finetune.py expect:
           timestamp, pm1, pm2_5, pm4, pm10, temp, rh, voc, nox, bam_pm2_5

PurpleAir's two channels are independent laser modules in the same device;
averaging them is the standard way to get a single reference value per window
(EPA's PurpleAir correction equations do the same thing).

NOTE ON USAGE:
    With a small number of co-location windows (this script will tell you how
    many it found), the right training entry point is fine-tuning, not
    from-scratch training:
        python finetune.py            # transfer-learn from the AQS model
        python finetune.py --freeze-conv   # safer for very small datasets
    main.py trains from scratch and needs hundreds of paired hours to be
    useful — but the paired CSV produced here is compatible with either.

USAGE:
    # Default paths (repo root data/ folder, config.yaml output location):
    python prepare_purpleair_data.py

    # Custom paths:
    python prepare_purpleair_data.py \
        --source-dir ../../data \
        --csv-name purpleairdata.csv \
        --out-dir data/paired
"""

import argparse
from pathlib import Path

import pandas as pd


SEN55_COLUMNS = ['pm1', 'pm2_5', 'pm4', 'pm10', 'temp', 'rh', 'voc', 'nox']

# Skip pkl files shorter than this — a 10-minute window should yield hundreds
# of samples at the SEN55's ~0.6 Hz rate; anything tiny is a glitched capture
# (e.g. a session that was started and stopped immediately).
MIN_SAMPLES_PER_PKL = 30


def load_purpleair_csv(csv_path: Path) -> pd.DataFrame:
    """Read the PurpleAir reference CSV and add the averaged reference column."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PurpleAir reference CSV not found: {csv_path}\n"
            "Expected columns: filename, pm2.5_1, pm2.5_2"
        )

    df = pd.read_csv(csv_path)
    required = {'filename', 'pm2.5_1', 'pm2.5_2'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}\n"
            f"Found: {list(df.columns)}"
        )

    # Drop rows where either channel is missing — we need both to average.
    before = len(df)
    df = df.dropna(subset=['pm2.5_1', 'pm2.5_2']).copy()
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows missing a PurpleAir reading.")

    df['bam_pm2_5'] = (df['pm2.5_1'] + df['pm2.5_2']) / 2.0
    print(f"  Loaded {len(df)} PurpleAir reference rows from {csv_path.name}")
    print(f"  PurpleAir-avg PM2.5 range: "
          f"{df['bam_pm2_5'].min():.2f} – {df['bam_pm2_5'].max():.2f} µg/m³")
    return df


def average_sen55_pkl(pkl_path: Path) -> dict | None:
    """
    Average a SEN55 .pkl recording into a single row.

    Returns a dict with the 8 SEN55 channel means plus the recording's mean
    timestamp, or None if the file is too short to be a real 10-minute window.
    """
    if not pkl_path.exists():
        print(f"  [skip] {pkl_path.name}: file not found")
        return None

    df = pd.read_pickle(pkl_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"  [skip] {pkl_path.name}: index is not a DatetimeIndex")
        return None

    if len(df) < MIN_SAMPLES_PER_PKL:
        print(f"  [skip] {pkl_path.name}: only {len(df)} samples "
              f"(< {MIN_SAMPLES_PER_PKL}) — likely a glitched capture")
        return None

    missing_cols = [c for c in SEN55_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  [skip] {pkl_path.name}: missing columns {missing_cols}")
        return None

    means = df[SEN55_COLUMNS].mean()
    # Use the midpoint of the recording window as the paired timestamp. This
    # is the natural pairing point with a window-averaged reference value.
    midpoint = df.index.min() + (df.index.max() - df.index.min()) / 2
    return {'timestamp': midpoint, **means.to_dict()}


def build_paired_dataset(pa_df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """For every PurpleAir row, average its matching SEN55 .pkl and join them."""
    rows = []
    for _, pa_row in pa_df.iterrows():
        pkl_path = source_dir / pa_row['filename']
        sen55_means = average_sen55_pkl(pkl_path)
        if sen55_means is None:
            continue

        rows.append({
            **sen55_means,
            'bam_pm2_5': float(pa_row['bam_pm2_5']),
            'pa_pm2_5_a': float(pa_row['pm2.5_1']),
            'pa_pm2_5_b': float(pa_row['pm2.5_2']),
            'source_file': pa_row['filename'],
        })

    if not rows:
        raise RuntimeError(
            "No usable (SEN55, PurpleAir) pairs were produced. "
            "Check that the .pkl files referenced in the CSV exist alongside it."
        )

    paired = pd.DataFrame(rows)

    # Column order matches what main.py / finetune.py expect; trailing columns
    # (pa_pm2_5_a, pa_pm2_5_b, source_file) are extra context for inspection,
    # ignored by the training pipeline since it only reads the named features
    # plus bam_pm2_5.
    col_order = ['timestamp', *SEN55_COLUMNS, 'bam_pm2_5',
                 'pa_pm2_5_a', 'pa_pm2_5_b', 'source_file']
    paired = paired[col_order].sort_values('timestamp').reset_index(drop=True)

    print(f"\n  Paired rows: {len(paired)}")
    print(f"  SEN55 PM2.5 range : "
          f"{paired['pm2_5'].min():.2f} – {paired['pm2_5'].max():.2f} µg/m³")
    print(f"  PurpleAir PM2.5   : "
          f"{paired['bam_pm2_5'].min():.2f} – {paired['bam_pm2_5'].max():.2f} µg/m³")
    pm_range = paired['bam_pm2_5'].max() - paired['bam_pm2_5'].min()
    if pm_range < 20.0:
        print(f"\n  WARNING: PM2.5 range is only {pm_range:.1f} µg/m³.")
        print("  The model may not generalise well outside this range. Collect")
        print("  data across more diverse air-quality conditions, or rely on")
        print("  fine-tuning so the AQS pretraining covers the missing regimes.")
    return paired


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Pair SEN55 .pkl recordings with PurpleAir reference data."
    )
    parser.add_argument(
        "--source-dir", default=str(repo_root / "data"),
        help="Directory containing purpleairdata.csv and sen55_*.pkl files "
             "(default: <repo>/data)",
    )
    parser.add_argument(
        "--csv-name", default="purpleairdata.csv",
        help="Name of the PurpleAir reference CSV inside --source-dir "
             "(default: purpleairdata.csv)",
    )
    parser.add_argument(
        "--out-dir", default="data/paired",
        help="Where to write paired_dataset.csv "
             "(default: data/paired, matches tools/config.yaml)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Loading PurpleAir reference CSV ────────────────────────────")
    pa_df = load_purpleair_csv(source_dir / args.csv_name)

    print("\n── Averaging SEN55 .pkl files per window ──────────────────────")
    paired = build_paired_dataset(pa_df, source_dir)

    out_path = out_dir / "paired_dataset.csv"
    paired.to_csv(out_path, index=False)
    print(f"\n  Paired dataset saved → {out_path}")
    print()
    print("  Next steps:")
    print("    python finetune.py            # transfer-learn from the AQS model")
    print("    python finetune.py --freeze-conv  # safer for very small datasets")
    print("    python main.py                # from-scratch local training "
          "(needs many more pairs)")


if __name__ == "__main__":
    main()

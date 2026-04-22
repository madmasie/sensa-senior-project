"""
prepare_data.py — Pair SEN55 readings with BAM (or AQS) reference PM2.5 values.

RUN THIS ONCE before training whenever you collect new data.

WHAT THIS DOES:
    1. Loads all SEN55 .pkl files from data/raw/  (output of uart_logger.py)
    2. Loads the reference PM2.5 CSV from data/raw/  (from BAM or AQS station)
    3. Resamples the SEN55 data to hourly averages to match the BAM's reporting rate
    4. Merges the two datasets by timestamp
    5. Writes the paired dataset to data/paired/paired_dataset.csv

WHY HOURLY AVERAGING:
    The SEN55 samples at 1–2 Hz (thousands of readings per hour).
    BAM machines report one hourly-average PM2.5 value.
    You cannot directly pair a 1-Hz SEN55 reading with an hourly BAM average
    because they represent different integration windows.
    The standard approach in the sensor calibration literature is to average
    the SEN55 over the same hourly window as the BAM, then pair those.
    This reduces your dataset to N_hours pairs, but each pair is directly
    comparable. N_hours ~ 1,000 is typically sufficient to train this model.

REFERENCE CSV FORMAT:
    The file data/raw/bam_reference.csv must have these two columns:
        timestamp   — ISO 8601 datetime, hourly resolution  (e.g. 2024-03-15 14:00:00)
        bam_pm2_5   — PM2.5 concentration in µg/m³

    BAM machines typically export Excel or CSV. Rename or reformat the
    timestamp and PM2.5 columns to match the names above.

    If you are using an EPA AQS download instead of a physical BAM:
        1. Download the hourly PM2.5 file for your state/year from
           https://aqs.epa.gov/aqsweb/airdata/download_files.html
        2. Filter to the monitoring station nearest to your SEN55 deployment.
        3. Rename "Date GMT" + "Time GMT" → timestamp, "Sample Measurement" → bam_pm2_5.
        IMPORTANT: the EPA station must be within ~1–2 km of your SEN55 for the
        PM2.5 values to be spatially correlated. Do not use distant stations.

Usage:
    python prepare_data.py
    python prepare_data.py --raw-dir path/to/raw --out-dir path/to/paired
"""

import argparse
from pathlib import Path

import pandas as pd


def load_sen55_pkls(raw_dir: Path) -> pd.DataFrame:
    """
    Load all SEN55 .pkl files in raw_dir and concatenate them.

    Each .pkl was written by uart_logger.py and contains a DataFrame with
    a DatetimeIndex and columns: pm1, pm2_5, pm4, pm10, temp, rh, voc, nox.

    Args:
        raw_dir: Directory containing one or more sen55_*.pkl files.

    Returns:
        A single DataFrame with all sessions concatenated, sorted by timestamp.
    """
    pkl_files = sorted(raw_dir.glob("sen55_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No sen55_*.pkl files found in {raw_dir}.\n"
            "Run uart_logger.py to collect SEN55 data first."
        )

    frames = []
    for path in pkl_files:
        df = pd.read_pickle(path)
        # Ensure the index is a DatetimeIndex (uart_logger.py sets this up,
        # but we check in case files were modified manually).
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{path.name}: expected a DatetimeIndex but got {type(df.index)}. "
                "Check that this file was produced by uart_logger.py."
            )
        frames.append(df)
        print(f"  Loaded {len(df)} rows from {path.name}")

    combined = pd.concat(frames).sort_index()
    combined.index.name = 'timestamp'
    print(f"  Total SEN55 rows: {len(combined)}")
    return combined


def load_reference_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the BAM or AQS reference PM2.5 CSV.

    Expected columns: timestamp, bam_pm2_5
    The timestamp should be at hourly resolution.

    Args:
        csv_path: Path to the reference CSV file.

    Returns:
        DataFrame with a DatetimeIndex and column 'bam_pm2_5'.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Reference file not found: {csv_path}\n"
            "Create data/raw/bam_reference.csv with columns: timestamp, bam_pm2_5\n"
            "See the module docstring for format details."
        )

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()

    if 'bam_pm2_5' not in df.columns:
        raise ValueError(
            f"{csv_path.name} must have a column named 'bam_pm2_5'. "
            f"Found: {list(df.columns)}"
        )

    print(f"  Loaded {len(df)} reference rows from {csv_path.name}")
    print(f"  Reference PM2.5 range: {df['bam_pm2_5'].min():.1f} – {df['bam_pm2_5'].max():.1f} µg/m³")
    return df[['bam_pm2_5']]


def pair_datasets(sen55_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average the SEN55 data into hourly bins and merge with the reference.

    BAM machines report one value per hour. To create comparable pairs:
      1. Truncate each SEN55 timestamp to the start of its hour (floor to hour).
      2. Group all SEN55 readings in the same hour and take the mean.
      3. Merge this hourly SEN55 DataFrame with the BAM DataFrame on timestamp.

    Args:
        sen55_df:     1–2 Hz SEN55 DataFrame with DatetimeIndex.
        reference_df: Hourly BAM/AQS DataFrame with DatetimeIndex and 'bam_pm2_5'.

    Returns:
        Paired DataFrame with columns: pm1, pm2_5, ..., bam_pm2_5
        One row per hour for which both sensors have data.
    """
    # Floor each timestamp to the nearest hour.
    # E.g., 14:37:22 → 14:00:00
    sen55_hourly = (
        sen55_df
        .copy()
        .assign(hour=sen55_df.index.floor('h'))
        .groupby('hour')
        .mean()
    )
    sen55_hourly.index.name = 'timestamp'

    # Inner join: keep only hours where both the SEN55 and the BAM have data.
    paired = sen55_hourly.join(reference_df, how='inner')

    if len(paired) == 0:
        raise ValueError(
            "No overlapping hours found between SEN55 and reference data.\n"
            "Check that the timestamps are in the same timezone and that both\n"
            "sensors were running during the same time window."
        )

    print(f"  Paired rows: {len(paired)} hours with both SEN55 and reference data")
    print(f"  Date range: {paired.index.min()} → {paired.index.max()}")
    print(f"  BAM PM2.5 range in paired set: "
          f"{paired['bam_pm2_5'].min():.1f} – {paired['bam_pm2_5'].max():.1f} µg/m³")

    # Warn if the PM2.5 range is too narrow — the model needs to see
    # a variety of conditions to learn a good calibration curve.
    pm_range = paired['bam_pm2_5'].max() - paired['bam_pm2_5'].min()
    if pm_range < 20.0:
        print(f"\n  WARNING: PM2.5 range is only {pm_range:.1f} µg/m³.")
        print("  The model may not generalise well outside this range.")
        print("  Try to collect data across more diverse air quality conditions")
        print("  (clean days, cooking events, traffic, wildfire smoke, etc.).")

    return paired.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair SEN55 .pkl files with BAM reference CSV → paired_dataset.csv"
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Directory containing sen55_*.pkl files and bam_reference.csv (default: data/raw)"
    )
    parser.add_argument(
        "--out-dir", default="data/paired",
        help="Directory to write paired_dataset.csv (default: data/paired)"
    )
    parser.add_argument(
        "--ref-file", default="bam_reference.csv",
        help="Name of the reference CSV file inside --raw-dir (default: bam_reference.csv)"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Loading SEN55 data ─────────────────────────────────────────")
    sen55_df = load_sen55_pkls(raw_dir)

    print("\n── Loading reference data ─────────────────────────────────────")
    reference_df = load_reference_csv(raw_dir / args.ref_file)

    print("\n── Pairing datasets ───────────────────────────────────────────")
    paired_df = pair_datasets(sen55_df, reference_df)

    out_path = out_dir / "paired_dataset.csv"
    paired_df.to_csv(out_path, index=False)
    print(f"\n  Paired dataset saved → {out_path}")
    print(f"  Ready for training. Run: python main.py")


if __name__ == "__main__":
    main()

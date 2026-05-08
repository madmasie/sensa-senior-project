"""
fetch_public_data.py — Download EPA AQS data and build a paired calibration dataset.

This script is the public-data counterpart to prepare_data.py. It fetches
freely-available EPA AQS bulk files, finds sites where a continuous optical
PM2.5 monitor (param 88502) was co-located with a reference FRM/FEM monitor
(param 88101), applies optional physics-based pre-corrections, and writes a
paired CSV ready for train_public.py.

DATA SOURCES (no API key required)
-----------------------------------
EPA AQS bulk hourly files:
    https://aqs.epa.gov/aqsweb/airdata/download_files.html

Parameters downloaded:
    88101 — PM2.5 FRM/FEM       : reference "true" PM2.5 (target variable)
    88502 — PM2.5 continuous     : optical monitor, SEN55 analog (input feature)
    62101 — Outdoor temperature  : input feature + temperature correction
    62201 — Relative humidity    : input feature + humidity correction
    42401 — SO2  [optional]      : needed only with --include-gas
    42602 — NO2  [optional]      : needed only with --include-gas

OUTPUT
------
    data/public/paired_public.csv — columns:
        site_id          AQS site identifier (state-county-site)
        timestamp        UTC datetime, hourly resolution
        pm2_5_optical    Pre-corrected continuous PM2.5 reading (µg/m³)
        temp             Outdoor temperature (°C)
        rh               Relative humidity (%)
        so2              SO2 concentration in ppb  [if --include-gas]
        no2              NO2 concentration in ppb  [if --include-gas]
        pm2_5_reference  FRM/FEM reference PM2.5 (µg/m³) — training target

USAGE
-----
    python fetch_public_data.py
    python fetch_public_data.py --state CA --years 2020 2021 2022
    python fetch_public_data.py --include-gas
    python fetch_public_data.py --no-corrections
    python fetch_public_data.py --help

NOTE ON CALIFORNIA DATA
-----------------------
California has some of the densest AQS monitoring networks in the US.
Many sites in the South Coast (LA), Bay Area, and San Joaquin Valley run
co-located FRM + continuous monitors, giving several hundred to a few
thousand paired hourly rows per site per year.

If param 88502 (non-FRM continuous) yields few co-located pairs, you can
widen the feature set to also include FEM-continuous monitors (coded 88101
with method types "FEM"). This is discussed in the code comments below.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ingestion.collocate import merge_parameters
from ingestion.corrections import CorrectionPipeline
from ingestion.epa_aqs import PARAM, STATE_FIPS, download_aqs_hourly, load_aqs_state

# Human-readable column names for the output CSV
_RENAME = {
    f"param_{PARAM['PM25_FRM']}":  "pm2_5_reference",
    f"param_{PARAM['PM25_CONT']}": "pm2_5_optical",
    f"param_{PARAM['TEMP']}":      "temp",
    f"param_{PARAM['RH']}":        "rh",
    f"param_{PARAM['SO2']}":       "so2",
    f"param_{PARAM['NO2']}":       "no2",
}

# Physical validity bounds for quality filtering
_BOUNDS = {
    "pm2_5_reference": (0.0,  1000.0),
    "pm2_5_optical":   (0.0,  1000.0),
    "temp":            (-40.0,  60.0),
    "rh":              (0.0,   100.0),
    "so2":             (0.0,   999.0),
    "no2":             (0.0,   999.0),
}


def _quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with physically implausible values."""
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in _BOUNDS.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi)
    before = len(df)
    df = df[mask].copy()
    removed = before - len(df)
    if removed:
        print(f"  Quality filter removed {removed:,} implausible rows.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download EPA AQS data and build a paired PM2.5 calibration dataset "
            "for use with train_public.py."
        )
    )
    parser.add_argument(
        "--state", default="CA",
        choices=list(STATE_FIPS.keys()),
        help="US state to download data for (default: CA).",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2021, 2022],
        metavar="YEAR",
        help="Calendar years to download (default: 2021 2022).",
    )
    parser.add_argument(
        "--include-gas", action="store_true",
        help=(
            "Also download SO2 (42401) and NO2 (42602). "
            "Required to enable SO2Correction / NO2Correction. "
            "Adds ~200-400 MB of downloads."
        ),
    )
    parser.add_argument(
        "--no-corrections", action="store_true",
        help="Skip physics-based pre-corrections and save raw optical PM2.5.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--cache-dir", default="data/public/aqs_cache",
        help="Directory for cached raw AQS CSV downloads (default: data/public/aqs_cache).",
    )
    parser.add_argument(
        "--out-dir", default="data/public",
        help="Directory to write paired_public.csv (default: data/public).",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download AQS files even if cached copies exist.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    state_fips = STATE_FIPS[args.state]
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Decide which parameters to fetch ───────────────────────────────────
    core_params = [
        PARAM["PM25_FRM"],   # reference target
        PARAM["PM25_CONT"],  # optical sensor analog
        PARAM["TEMP"],
        PARAM["RH"],
    ]
    gas_params = [PARAM["SO2"], PARAM["NO2"]] if args.include_gas else []
    all_params = core_params + gas_params
    feature_params = [p for p in all_params if p != PARAM["PM25_FRM"]]

    # ── Download and load ───────────────────────────────────────────────────
    print(f"\n── Downloading EPA AQS data ─────────────────────────────────────")
    print(f"   State: {args.state}  |  Years: {args.years}")
    print(f"   Parameters: {all_params}\n")

    yearly_frames: dict[int, list[pd.DataFrame]] = {p: [] for p in all_params}

    for year in args.years:
        print(f"\n  Year {year}:")
        for param in all_params:
            csv_path = download_aqs_hourly(
                param, year, cache_dir, force_download=args.force_download
            )
            df = load_aqs_state(csv_path, state_fips, param)
            if not df.empty:
                yearly_frames[param].append(df)

    # Concatenate across years for each parameter
    combined: dict[int, pd.DataFrame] = {}
    for param, frames in yearly_frames.items():
        if frames:
            combined[param] = pd.concat(frames, ignore_index=True)
        else:
            print(f"\n  WARNING: No data found for param {param} in {args.state}. "
                  f"Removing from feature set.")

    # Remove params with no data (may happen for sparse gas monitors)
    feature_params = [p for p in feature_params if p in combined]
    if PARAM["PM25_FRM"] not in combined:
        raise RuntimeError(
            "No PM2.5 FRM data (param 88101) found for this state/year combination. "
            "Cannot build a paired dataset without the reference variable."
        )

    # ── Merge co-located monitors ───────────────────────────────────────────
    print(f"\n── Merging co-located monitors ──────────────────────────────────")
    paired = merge_parameters(
        dfs=combined,
        primary_param=PARAM["PM25_FRM"],
        feature_params=feature_params,
    )

    # ── Apply physics-based corrections ────────────────────────────────────
    optical_col = f"param_{PARAM['PM25_CONT']}"
    if not args.no_corrections and optical_col in paired.columns:
        print(f"\n── Applying correction pipeline ─────────────────────────────────")
        corr_cfg = cfg.get("public_data", {}).get("corrections")
        if corr_cfg:
            pipeline = CorrectionPipeline.from_config(corr_cfg, target_col=optical_col)
        else:
            pipeline = CorrectionPipeline.default(target_col=optical_col)
        paired = pipeline.apply(paired)
        applied = [f.__class__.__name__ for f in pipeline.factors]
        print(f"  Applied: {applied}")
    elif args.no_corrections:
        print("\n── Skipping corrections (--no-corrections) ──────────────────────")

    # ── Rename to human-readable column names ──────────────────────────────
    paired = paired.rename(columns={k: v for k, v in _RENAME.items() if k in paired.columns})

    # Drop internal columns not needed for training
    keep_cols = ["site_id", "timestamp"] + [
        c for c in ["pm2_5_optical", "pm2_5_reference", "temp", "rh", "so2", "no2"]
        if c in paired.columns
    ]
    paired = paired[keep_cols]

    # ── Quality filter ──────────────────────────────────────────────────────
    paired = _quality_filter(paired)

    # ── Summary statistics ──────────────────────────────────────────────────
    print(f"\n── Dataset summary ──────────────────────────────────────────────")
    print(f"  Rows:   {len(paired):,}")
    print(f"  Sites:  {paired['site_id'].nunique()}")
    print(f"  Columns: {list(paired.columns)}")
    ref_col = "pm2_5_reference"
    if ref_col in paired.columns:
        print(
            f"  PM2.5 reference:  "
            f"{paired[ref_col].min():.1f} – {paired[ref_col].max():.1f} µg/m³  "
            f"(mean {paired[ref_col].mean():.1f})"
        )
    if "pm2_5_optical" in paired.columns:
        print(
            f"  PM2.5 optical:    "
            f"{paired['pm2_5_optical'].min():.1f} – {paired['pm2_5_optical'].max():.1f} µg/m³  "
            f"(mean {paired['pm2_5_optical'].mean():.1f})"
        )

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = out_dir / "paired_public.csv"
    paired.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Next step: python train_public.py")


if __name__ == "__main__":
    main()

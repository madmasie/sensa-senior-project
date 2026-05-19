"""
train_public.py — Train a PM2.5 calibration model on public EPA AQS data.

This is the public-data counterpart to main.py. It reuses the same model
architecture (SensaCalibNet), training loop, evaluation, and export code
from src/, but operates on paired_public.csv files produced by
fetch_public_data.py.  The original main.py / local pipeline is untouched.

─────────────────────────────────────────────────────────────────────────────
OVERVIEW: TWO PATHS TO A CALIBRATED MODEL
─────────────────────────────────────────────────────────────────────────────

  PATH A  — Public EPA AQS data (this script)
  ──────────────────────────────────────────
  Use when: you don't yet have access to a BAM machine or co-location time.

  Data comes from EPA AQS monitoring sites where a co-located continuous
  optical PM2.5 monitor (param 88502, analogous to what the SEN55 reads) sits
  beside a reference FRM/BAM instrument (param 88101).  A model trained on
  these pairs learns the optical → reference correction using humidity and
  temperature as auxiliary features.  The same correction transfers to the
  SEN55 because both are optical particle counters.

  At SEN55 inference time, map:
      SEN55 pm2_5  →  pm2_5_optical   (the 3-feature model input)
      SEN55 temp   →  temp
      SEN55 rh     →  rh
  The extra SEN55 channels (pm1, pm4, pm10, voc, nox) are unused by this model.

  PATH B  — Local SEN55 + BAM co-location (main.py)
  ──────────────────────────────────────────────────
  Use when: you have direct access to a BAM machine and can run the SEN55
  beside it for at least several days (ideally across varied PM conditions).

  uart_logger.py captures SEN55 readings.  prepare_data.py pairs them with
  the BAM's hourly CSV.  main.py trains an 8-feature model (pm1, pm2_5, pm4,
  pm10, temp, rh, voc, nox → bam_pm2_5) that uses ALL SEN55 channels and
  is specific to your unit and deployment site.  This typically achieves
  better accuracy than Path A once enough paired hours are collected (~500+).

  COMBINING BOTH (recommended sequence):
    1. Train and deploy the public model immediately (Path A).
    2. While waiting, begin co-location data collection alongside a BAM.
    3. Once ~500+ paired hours are collected, re-train with main.py (Path B).
    4. The Path B model replaces the Path A model in firmware.

─────────────────────────────────────────────────────────────────────────────
SELECTING DATASETS
─────────────────────────────────────────────────────────────────────────────

By default, this script reads the file set in config.yaml under
public_data.paired_file (data/public/paired_public.csv).

Use --data to point at a different file, a list of files, or directories:

    # Default (uses config.yaml)
    python train_public.py

    # One specific file
    python train_public.py --data data/public/CA_2022/paired_public.csv

    # Multiple files — concatenated before training
    python train_public.py --data data/public/CA_2021/paired_public.csv \\
                                   data/public/CA_2022/paired_public.csv

    # Directories — all *.csv files inside each are loaded and concatenated
    python train_public.py --data data/public/CA_2021 data/public/CA_2022

    # Mix of files and directories
    python train_public.py --data data/public/CA_2021 \\
                                   data/public/CA_2022/paired_public.csv

All CSVs must share the same feature and target columns defined in
config.yaml (public_data.features and public_data.target).

To produce per-region or per-year CSVs, pass --out-dir when downloading:
    python fetch_public_data.py --state CA --years 2021 --out-dir data/public/CA_2021
    python fetch_public_data.py --state CA --years 2022 --out-dir data/public/CA_2022
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    PM25CalibrationDataset,
    load_paired_csv,
    save_scaler,
    load_scaler,
    split_dataset,
)
from src.evaluate import evaluate, plot_predictions
from src.export import convert_onnx_to_tflite, export_scaler_as_c_header, export_to_onnx
from src.model import SensaCalibNet, count_parameters
from src.train import train


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU detected — training on CPU.")
    return device


def resolve_data_paths(paths: List[str]) -> List[Path]:
    """
    Expand a list of file and/or directory paths into a flat list of CSV files.

    Directories are searched for *.csv files (non-recursive). Files are used
    as-is. Raises FileNotFoundError if a path does not exist or a directory
    contains no CSVs.
    """
    csv_files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            csv_files.append(p)
        elif p.is_dir():
            found = sorted(p.glob("*.csv"))
            if not found:
                raise FileNotFoundError(
                    f"No .csv files found in directory: {p}\n"
                    "Run fetch_public_data.py with --out-dir pointing here first."
                )
            csv_files.extend(found)
        else:
            raise FileNotFoundError(
                f"Path not found: {p}\n"
                "Check that fetch_public_data.py has run and produced output here."
            )
    return csv_files


def load_datasets(
    data_paths: Optional[List[str]],
    default_csv: Path,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Load and concatenate one or more paired CSV files.

    If data_paths is None or empty, falls back to default_csv (the path from
    config.yaml). Returns a single concatenated DataFrame.
    """
    if data_paths:
        csv_files = resolve_data_paths(data_paths)
        frames = []
        for csv_path in csv_files:
            print(f"  Loading {csv_path}")
            frames.append(
                load_paired_csv(str(csv_path), feature_cols=feature_cols, target_col=target_col)
            )
        df = pd.concat(frames, ignore_index=True)
        print(f"  Combined total: {len(df):,} rows from {len(csv_files)} file(s)")
        return df
    else:
        if not default_csv.exists():
            print(f"\nERROR: {default_csv} not found.")
            print(
                "Either run fetch_public_data.py to generate the default dataset,\n"
                "or pass --data <path> to point at an existing paired CSV."
            )
            sys.exit(1)
        return load_paired_csv(str(default_csv), feature_cols=feature_cols, target_col=target_col)


def run_public_pipeline(
    config: dict,
    device: torch.device,
    export: bool = True,
    data_paths: Optional[List[str]] = None,
) -> None:
    pub_cfg   = config["public_data"]
    train_cfg = config["training"]
    data_cfg  = config["data"]

    feature_cols = pub_cfg["features"]
    target_col   = pub_cfg["target"]
    model_cfg    = pub_cfg["model"]
    export_cfg   = pub_cfg["export"]

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n── Loading data ─────────────────────────────────────────────────")
    df = load_datasets(
        data_paths=data_paths,
        default_csv=Path(pub_cfg["paired_file"]),
        feature_cols=feature_cols,
        target_col=target_col,
    )

    train_df, val_df, test_df = split_dataset(
        df,
        val_frac=data_cfg["val_frac"],
        test_frac=data_cfg["test_frac"],
        seed=data_cfg["seed"],
    )

    # ── Normalise ───────────────────────────────────────────────────────────
    print("\n── Normalising features ─────────────────────────────────────────")
    train_ds = PM25CalibrationDataset(
        train_df, fit_scaler=True,
        input_features=feature_cols, target_column=target_col,
    )
    val_ds = PM25CalibrationDataset(
        val_df, scaler=train_ds.scaler, fit_scaler=False,
        input_features=feature_cols, target_column=target_col,
    )
    test_ds = PM25CalibrationDataset(
        test_df, scaler=train_ds.scaler, fit_scaler=False,
        input_features=feature_cols, target_column=target_col,
    )

    model_dir = Path(export_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(train_ds.scaler, str(model_dir / "scaler.pkl"))

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg["batch_size"], shuffle=False)

    # ── Build model ─────────────────────────────────────────────────────────
    print("\n── Building model ───────────────────────────────────────────────")
    print(f"  Input features ({len(feature_cols)}): {feature_cols}")
    print(f"  Target: {target_col}")

    model = SensaCalibNet(
        n_features=model_cfg["n_features"],
        n_channels=model_cfg["n_channels"],
    )
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}  ({n_params * 4 / 1024:.1f} KB float32 / "
          f"{n_params / 1024:.1f} KB int8 quantised)")

    # train() reads config['export'] for the checkpoint path, so we pass a
    # modified config that routes output to models_public/ instead of models/.
    train_config = {**config, "model": model_cfg, "export": export_cfg}

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n── Training ─────────────────────────────────────────────────────")
    model, _ = train(model, train_loader, val_loader, train_config, device)

    # ── Evaluate ────────────────────────────────────────────────────────────
    print("\n── Evaluating on test set ───────────────────────────────────────")
    evaluate(model, test_loader, device, split_name="test (public data)")
    plot_predictions(
        model, test_loader, device,
        save_path=str(model_dir / "predictions.png"),
    )

    if not export:
        print("\nSkipping export (--no-export flag set).")
        return

    # ── Export ──────────────────────────────────────────────────────────────
    print("\n── Exporting model ──────────────────────────────────────────────")

    checkpoint = model_dir / export_cfg["pytorch_checkpoint"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to("cpu")

    onnx_path = str(model_dir / export_cfg["onnx_file"])
    export_to_onnx(model, onnx_path, n_features=model_cfg["n_features"])

    rep_data    = train_ds.X.numpy()
    tflite_path = str(model_dir / export_cfg["tflite_file"])
    convert_onnx_to_tflite(
        onnx_path=onnx_path,
        tflite_path=tflite_path,
        quantize=export_cfg["quantize"],
        representative_data=rep_data,
    )

    scaler = load_scaler(str(model_dir / "scaler.pkl"))
    export_scaler_as_c_header(scaler, export_cfg["c_header_path"])

    print("\n── Done ─────────────────────────────────────────────────────────")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  ONNX       : {onnx_path}")
    print(f"  TFLite     : {tflite_path}")
    print(f"  C header   : {export_cfg['c_header_path']}")
    print()
    print("  SEN55 firmware inference mapping:")
    print("    SEN55 pm2_5  →  model input 0  (pm2_5_optical)")
    print("    SEN55 temp   →  model input 1  (temp)")
    print("    SEN55 rh     →  model input 2  (rh)")
    print("    pm1 / pm4 / pm10 / voc / nox  →  not used by this model")
    print()
    print("  To upgrade to the full 8-feature model once BAM data is available:")
    print("    python prepare_data.py && python main.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PM2.5 calibration model on public EPA AQS data.\n\n"
            "By default reads the paired CSV set in config.yaml. Use --data to\n"
            "supply one or more specific files or directories instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", nargs="+", default=None,
        metavar="PATH",
        help=(
            "One or more paths to paired CSV files or directories containing them. "
            "Directories are searched for *.csv files. Multiple inputs are "
            "concatenated before training. If omitted, uses public_data.paired_file "
            "from config.yaml."
        ),
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Train the model but skip ONNX/TFLite export.",
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--config", default=str(script_dir.parent / "config.yaml"),
        help="Path to the shared tools/config.yaml (default: tools/config.yaml).",
    )
    args = parser.parse_args()

    # Resolve user-supplied paths against the caller's cwd, then move into the
    # script's own directory so the relative paths inside config.yaml
    # (data/, models/, ../../include/) resolve correctly no matter where
    # this script was launched from.
    config_path = Path(args.config).resolve()
    data_paths = [str(Path(p).resolve()) for p in args.data] if args.data else None
    os.chdir(script_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    run_public_pipeline(
        config,
        device,
        export=not args.no_export,
        data_paths=data_paths,
    )


if __name__ == "__main__":
    main()

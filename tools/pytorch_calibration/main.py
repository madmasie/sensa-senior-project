"""
main.py — Entry point for the Sensa PM2.5 calibration model pipeline.

This script ties together all the modules in src/ and runs the full pipeline:
    1. Load and split the paired dataset
    2. Build and train the model
    3. Evaluate accuracy on the held-out test set
    4. Export to ONNX and TFLite for ESP32 deployment

BEFORE RUNNING THIS SCRIPT:
    1. Collect paired data with uart_logger.py + your BAM/AQS reference.
    2. Run prepare_data.py to generate data/paired/paired_dataset.csv.
    3. Install dependencies: pip install -r requirements.txt

USAGE:
    # Full pipeline (train + export):
    python main.py

    # Training only (no export):
    python main.py --no-export

    # Export only (load an existing checkpoint):
    python main.py --export-only

    # Run a quick smoke test with synthetic data (no real data needed):
    python main.py --demo

GPU NOTE:
    On Windows with an AMD GPU, PyTorch does not support AMD (ROCm).
    Training will automatically run on CPU — this is fine for this model.
    On Linux with an AMD GPU, install the ROCm version of PyTorch from
    https://pytorch.org/get-started/locally/ and ROCm will be detected
    automatically as 'cuda'.
    NVIDIA GPU users: install the CUDA-enabled PyTorch wheel and CUDA
    will be detected automatically.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    PM25CalibrationDataset,
    load_paired_csv,
    load_scaler,
    save_scaler,
    split_dataset,
)
from src.evaluate import evaluate, plot_predictions
from src.export import convert_onnx_to_tflite, export_scaler_as_c_header, export_to_onnx
from src.model import SensaCalibNet, count_parameters
from src.train import train


def load_config(path: str = "config.yaml") -> dict:
    """Load the YAML config file into a Python dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """
    Choose the best available compute device.

    Priority: CUDA (NVIDIA) > CPU
    Note: ROCm (AMD) on Linux is exposed as 'cuda' by PyTorch, so it is
    automatically detected here. On Windows, AMD GPUs are not supported.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("No GPU detected — training on CPU.")
        print("(This is fine for this model; expect < 1 minute per epoch.)")
    return device


def make_demo_dataset(n_samples: int = 2000, seed: int = 42) -> None:
    """
    Generate a synthetic paired dataset so the pipeline can be tested
    without real sensor data.

    The synthetic SEN55 readings are drawn from realistic ranges.
    The "calibration" relationship applied here is a simple linear bias
    (the SEN55 under-reads by 20% and has a +3 µg/m³ offset) plus noise.
    A well-trained model should learn to invert this relationship.

    NOTE: This is for pipeline testing only. A model trained on synthetic
    data will NOT calibrate a real SEN55 sensor correctly.

    Args:
        n_samples: Number of synthetic (SEN55, BAM) pairs to generate.
        seed:      Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Simulate realistic SEN55 readings
    pm2_5_true = rng.uniform(2.0, 80.0, n_samples)   # "true" PM2.5
    pm1        = pm2_5_true * rng.uniform(0.4, 0.6, n_samples)
    pm4        = pm2_5_true * rng.uniform(1.1, 1.3, n_samples)
    pm10       = pm2_5_true * rng.uniform(1.2, 1.5, n_samples)

    # Simulate SEN55 optical bias: 20% under-read + 3 µg/m³ offset + noise
    pm2_5_sen55 = pm2_5_true * 0.80 + 3.0 + rng.normal(0, 1.5, n_samples)

    temp = rng.uniform(10.0, 35.0, n_samples)
    rh   = rng.uniform(20.0, 90.0, n_samples)
    voc  = rng.uniform(50.0, 250.0, n_samples)
    nox  = rng.uniform(1.0, 50.0, n_samples)

    # BAM reference = true PM2.5 with small measurement noise
    bam_pm2_5 = pm2_5_true + rng.normal(0, 0.5, n_samples)

    import pandas as pd
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
        'pm1':      pm1.clip(0),
        'pm2_5':    pm2_5_sen55.clip(0),
        'pm4':      pm4.clip(0),
        'pm10':     pm10.clip(0),
        'temp':     temp,
        'rh':       rh,
        'voc':      voc,
        'nox':      nox,
        'bam_pm2_5': bam_pm2_5.clip(0),
    })

    out_dir = Path("data/paired")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paired_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"  Demo dataset written → {out_path}  ({n_samples} synthetic samples)")


def run_pipeline(config: dict, device: torch.device, export: bool = True) -> None:
    """
    Load data → train model → evaluate → export.

    Args:
        config: Loaded config.yaml dict.
        device: Compute device ('cpu' or 'cuda').
        export: If False, skip the ONNX/TFLite export step.
    """
    data_cfg  = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    export_cfg = config['export']

    paired_csv = Path(data_cfg['paired_dir']) / data_cfg['paired_file']
    if not paired_csv.exists():
        print(f"\nERROR: {paired_csv} not found.")
        print("Run prepare_data.py first to generate the paired dataset.")
        sys.exit(1)

    # ── Load and split data ───────────────────────────────────────────────────
    print("\n── Loading data ───────────────────────────────────────────────")
    df = load_paired_csv(str(paired_csv))
    train_df, val_df, test_df = split_dataset(
        df,
        val_frac  = data_cfg['val_frac'],
        test_frac = data_cfg['test_frac'],
        seed      = data_cfg['seed'],
    )

    # Build Dataset objects.
    # The scaler is fit on the training set ONLY, then reused for val/test.
    print("\n── Normalising features ──────────────────────────────────────")
    train_ds = PM25CalibrationDataset(train_df, fit_scaler=True)
    val_ds   = PM25CalibrationDataset(val_df,   scaler=train_ds.scaler, fit_scaler=False)
    test_ds  = PM25CalibrationDataset(test_df,  scaler=train_ds.scaler, fit_scaler=False)

    # Save the scaler so export.py can generate the C header.
    model_dir = Path(export_cfg['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(train_ds.scaler, str(model_dir / "scaler.pkl"))

    # DataLoaders: these wrap the Dataset and handle batching + shuffling.
    # shuffle=True for training so the model sees samples in random order
    # each epoch (prevents it from learning the temporal order of the data).
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg['batch_size'], shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n── Building model ─────────────────────────────────────────────")
    model = SensaCalibNet(
        n_features = model_cfg['n_features'],
        n_channels = model_cfg['n_channels'],
    )
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}  ({n_params * 4 / 1024:.1f} KB float32 / "
          f"{n_params / 1024:.1f} KB int8 quantised)")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n── Training ───────────────────────────────────────────────────")
    model, history = train(model, train_loader, val_loader, config, device)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n── Evaluating on test set ─────────────────────────────────────")
    metrics = evaluate(model, test_loader, device, split_name="test")
    plot_predictions(model, test_loader, device,
                     save_path=str(model_dir / "predictions.png"))

    # ── Export ────────────────────────────────────────────────────────────────
    if not export:
        print("\nSkipping export (--no-export flag set).")
        return

    print("\n── Exporting model ────────────────────────────────────────────")

    # Reload best checkpoint (train() already does this, but be explicit)
    checkpoint = model_dir / export_cfg['pytorch_checkpoint']
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to('cpu')  # Export must be done on CPU

    # Step 1: PyTorch → ONNX
    onnx_path = str(model_dir / export_cfg['onnx_file'])
    export_to_onnx(model, onnx_path, n_features=model_cfg['n_features'])

    # Step 2: ONNX → TFLite
    # Draw representative data from the training set for int8 quantisation.
    rep_data = train_ds.X.numpy()  # All training inputs as numpy array
    tflite_path = str(model_dir / export_cfg['tflite_file'])
    convert_onnx_to_tflite(
        onnx_path    = onnx_path,
        tflite_path  = tflite_path,
        quantize     = export_cfg['quantize'],
        representative_data = rep_data,
    )

    # Step 3: Generate C header with normalisation constants for firmware
    scaler = load_scaler(str(model_dir / "scaler.pkl"))
    export_scaler_as_c_header(scaler, export_cfg['c_header_path'])

    print("\n── Done ───────────────────────────────────────────────────────")
    print(f"  Trained model     : {checkpoint}")
    print(f"  ONNX model        : {onnx_path}")
    print(f"  TFLite model      : {tflite_path}")
    print(f"  C scaler header   : {export_cfg['c_header_path']}")
    print()
    print("  Next steps:")
    print("  1. Copy the .tflite file into the firmware project.")
    print("  2. Copy include/calib_scaler.h into include/.")
    print("  3. Use the TFLite Micro runtime to run inference on the ESP32-S3.")


def run_export_only(config: dict) -> None:
    """
    Load an existing checkpoint and run only the export step.
    Useful if you have already trained and just need a new TFLite file.
    """
    model_cfg  = config['model']
    export_cfg = config['export']
    model_dir  = Path(export_cfg['model_dir'])

    checkpoint = model_dir / export_cfg['pytorch_checkpoint']
    if not checkpoint.exists():
        print(f"ERROR: No checkpoint found at {checkpoint}.")
        print("Run training first (python main.py) before using --export-only.")
        sys.exit(1)

    scaler_path = model_dir / "scaler.pkl"
    if not scaler_path.exists():
        print(f"ERROR: Scaler not found at {scaler_path}.")
        print("The scaler is saved during training. Run full training first.")
        sys.exit(1)

    print("\n── Loading model for export ───────────────────────────────────")
    model = SensaCalibNet(n_features=model_cfg['n_features'], n_channels=model_cfg['n_channels'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    onnx_path   = str(model_dir / export_cfg['onnx_file'])
    tflite_path = str(model_dir / export_cfg['tflite_file'])

    export_to_onnx(model, onnx_path, n_features=model_cfg['n_features'])

    # No representative data available without the training set.
    # The model will be exported in float32. Re-run full pipeline for int8.
    convert_onnx_to_tflite(onnx_path, tflite_path, quantize=False)

    scaler = load_scaler(str(scaler_path))
    export_scaler_as_c_header(scaler, export_cfg['c_header_path'])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export the Sensa PM2.5 calibration model."
    )
    parser.add_argument(
        '--no-export', action='store_true',
        help="Train the model but skip ONNX/TFLite export."
    )
    parser.add_argument(
        '--export-only', action='store_true',
        help="Skip training; export an existing checkpoint to TFLite."
    )
    parser.add_argument(
        '--demo', action='store_true',
        help="Generate synthetic data and run the full pipeline as a smoke test. "
             "No real sensor data needed."
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help="Path to the config YAML file (default: config.yaml)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    if args.demo:
        print("\n── Demo mode: generating synthetic data ───────────────────")
        make_demo_dataset()
        run_pipeline(config, device, export=False)  # skip export in demo
        return

    if args.export_only:
        run_export_only(config)
        return

    run_pipeline(config, device, export=not args.no_export)


if __name__ == "__main__":
    main()

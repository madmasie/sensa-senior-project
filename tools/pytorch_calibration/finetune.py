"""
finetune.py — Transfer-learn the calibration model: pretrain on EPA AQS, then
fine-tune on local SEN55 + BAM co-location data.

─────────────────────────────────────────────────────────────────────────────
WHY THIS SCRIPT EXISTS
─────────────────────────────────────────────────────────────────────────────
There are two sources of paired calibration data, and each has a weakness:

  - EPA AQS public data (train_public.py): HUGE and diverse (many sites,
    seasons, PM regimes) — but the optical monitor in that data is a
    research-grade instrument, NOT a SEN55. A model trained only on AQS is
    miscalibrated for our cheap sensor.

  - Local SEN55 + BAM co-location (main.py / prepare_data.py): exactly our
    sensor — but SCARCE (a co-location campaign yields maybe a few hundred
    paired hours).

Transfer learning combines them the correct way round:

      STEP 1  Pretrain on AQS        →  learns the general shape of the
              (train_public.py)         optical-PM → reference-PM correction
                                        from a large, diverse dataset.

      STEP 2  Fine-tune on local     →  nudges those pretrained weights to
              SEN55+BAM (this file)     close the SEN55-specific gap, using
                                        a LOW learning rate so the local
                                        data adapts the model without
                                        erasing what AQS taught it.

ORDER MATTERS. Pretraining must use the large set and fine-tuning the small
set — never the reverse. If you fine-tuned on the larger dataset, its many
gradient updates would simply overwrite the small dataset's contribution
("catastrophic forgetting"). The dataset trained on LAST has the final say,
and we want the final say to belong to our actual sensor.

─────────────────────────────────────────────────────────────────────────────
TWO THINGS THIS SCRIPT IS CAREFUL ABOUT
─────────────────────────────────────────────────────────────────────────────
1. SCALER REUSE. The pretrained model expects its inputs normalised by the
   MinMaxScaler that was fit on the AQS data. We MUST reuse that exact scaler
   for the local data — fitting a fresh scaler would shift the input range
   and make the pretrained weights meaningless. So this script loads
   models_public/scaler.pkl and applies it (transform only, never fit).

2. LOW LEARNING RATE. Fine-tuning uses ~10x lower LR than from-scratch
   training (see finetune.learning_rate in config.yaml). Big steps would
   overwrite the pretrained knowledge; small steps refine it.

─────────────────────────────────────────────────────────────────────────────
FEATURE SCHEMA
─────────────────────────────────────────────────────────────────────────────
The AQS pretrained model has 3 inputs: [pm2_5_optical, temp, rh]. The local
paired CSV has all 8 SEN55 channels, so this script keeps only the 3 the
model uses and renames them to the AQS schema:
      local  pm2_5      →  pm2_5_optical
      local  temp       →  temp
      local  rh         →  rh
      local  bam_pm2_5  →  pm2_5_reference   (the target)

USAGE
    # 1. First produce the pretrained model:
    python train_public.py
    # 2. Collect local co-location data and pair it:
    python prepare_data.py
    # 3. Fine-tune:
    python finetune.py
    python finetune.py --freeze-conv      # adapt only the regression head
    python finetune.py --no-export        # skip the ONNX/TFLite export
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
from src.export import (
    convert_onnx_to_tflite,
    export_scaler_as_c_header,
    export_tflite_as_c_header,
    export_to_onnx,
)
from src.model import SensaCalibNet, count_parameters
from src.train import validate  # the validation loop is identical to training


def get_device() -> torch.device:
    """Pick CUDA if available, otherwise CPU (fine for this tiny model)."""
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected — fine-tuning on CPU (expect a few seconds/epoch).")
    return torch.device("cpu")


def freeze_conv_block(model: SensaCalibNet) -> None:
    """
    Freeze the convolutional feature extractor so only the regression head
    is fine-tuned.

    WHEN TO USE THIS:
        With very little local data (say < ~200 paired hours), letting the
        whole network move can overfit. Freezing the conv layers keeps the
        AQS-learned feature extractor fixed and only re-fits the final
        mapping to PM2.5 — far fewer free parameters, much less overfitting.

    HOW IT WORKS:
        Setting requires_grad = False on a parameter tells PyTorch not to
        compute or apply gradients for it — the weight is held constant.
    """
    for param in model.conv_block.parameters():
        param.requires_grad = False


def finetune_loop(
    model: SensaCalibNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ft_cfg: dict,
    device: torch.device,
    freeze_conv: bool,
) -> SensaCalibNet:
    """
    Fine-tuning training loop. Mirrors src/train.py's train() but with the
    transfer-learning specifics: a low learning rate, optional frozen conv
    layers, and frozen-BatchNorm handling.

    Args:
        model:        The pretrained SensaCalibNet (AQS weights already loaded).
        train_loader: Local training data.
        val_loader:   Local validation data.
        ft_cfg:       The 'finetune' section of config.yaml.
        device:       'cpu' or 'cuda'.
        freeze_conv:  If True, only the regression head is updated.

    Returns:
        The model with the best (lowest validation loss) fine-tuned weights.
    """
    save_dir = Path(ft_cfg["export"]["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / ft_cfg["export"]["pytorch_checkpoint"]

    lr           = ft_cfg["learning_rate"]
    weight_decay = ft_cfg.get("weight_decay", 1e-4)
    max_epochs   = ft_cfg["epochs"]
    patience     = ft_cfg.get("early_stopping_patience", 15)

    # The optimizer only ever updates parameters with requires_grad = True.
    # When freeze_conv is set, the conv layers were already frozen, so this
    # list contains just the regression head.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

    # Halve the LR if validation loss stalls for 5 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    model.to(device)
    print(f"\nFine-tuning on: {device}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}  {'Time':>7}")
    print("─" * 58)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- one training epoch ---
        model.train()
        # If the conv block is frozen, also put its BatchNorm layers back into
        # eval mode. model.train() above re-enabled them; left in train mode
        # they would keep updating their running mean/variance toward the
        # small local set, which quietly undoes the point of freezing.
        if freeze_conv:
            model.conv_block.eval()

        total_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # --- validation ---
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
            f"{current_lr:>10.2e}  {time.time() - t0:>5.1f}s"
        )

        # Keep the checkpoint with the lowest validation loss seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"         ✓ Checkpoint saved (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    # Restore the best weights before returning.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\nFine-tuning complete. Best validation loss: {best_val_loss:.6f}")
    return model


def run_finetune(config: dict, device: torch.device,
                 export: bool = True, freeze_conv: bool = False) -> None:
    """
    Load the pretrained AQS model, fine-tune it on local SEN55+BAM data,
    evaluate the before/after improvement, and export for the firmware.
    """
    ft_cfg    = config["finetune"]
    pub_cfg   = config["public_data"]
    data_cfg  = config["data"]
    train_cfg = config["training"]

    # The fine-tuned model keeps the AQS model's 3-feature architecture.
    features  = pub_cfg["features"]      # ['pm2_5_optical', 'temp', 'rh']
    target    = pub_cfg["target"]        # 'pm2_5_reference'
    model_cfg = pub_cfg["model"]         # n_features=3, n_channels=16
    ft_export = ft_cfg["export"]

    # ── Load the pretrained AQS model + the scaler it was trained with ──────
    print("\n── Loading pretrained AQS model ─────────────────────────────────")
    pretrained_dir = Path(ft_cfg["pretrained_dir"])
    checkpoint  = pretrained_dir / ft_cfg["pretrained_checkpoint"]
    scaler_path = pretrained_dir / ft_cfg["pretrained_scaler"]

    if not checkpoint.exists():
        print(f"\nERROR: pretrained checkpoint not found: {checkpoint}")
        print("Run `python train_public.py` first to produce the AQS model.")
        sys.exit(1)
    if not scaler_path.exists():
        print(f"\nERROR: pretrained scaler not found: {scaler_path}")
        print("Run `python train_public.py` first — it saves the scaler.")
        sys.exit(1)

    # Reuse the AQS scaler — see the module docstring, point 1. Fitting a new
    # scaler here would invalidate the pretrained weights.
    aqs_scaler = load_scaler(str(scaler_path))

    model = SensaCalibNet(
        n_features=model_cfg["n_features"],
        n_channels=model_cfg["n_channels"],
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    print(f"  Loaded {checkpoint}")
    n_total = count_parameters(model)
    print(f"  Model parameters: {n_total:,}")

    # ── Load the local SEN55 + BAM data, mapped to the 3-feature schema ─────
    print("\n── Loading local SEN55 + BAM data ───────────────────────────────")
    local_csv = Path(ft_cfg["local_paired_file"])
    if not local_csv.exists():
        print(f"\nERROR: local paired dataset not found: {local_csv}")
        print("Run `python prepare_data.py` after collecting co-location data.")
        sys.exit(1)

    # load_paired_csv() cleans the data using the LOCAL column names; we then
    # rename the 3 kept columns to the AQS schema the model expects.
    df = load_paired_csv(
        str(local_csv),
        feature_cols=["pm2_5", "temp", "rh"],
        target_col="bam_pm2_5",
    )
    df = df.rename(columns={"pm2_5": "pm2_5_optical", "bam_pm2_5": "pm2_5_reference"})

    if len(df) < 30:
        print(f"\n  WARNING: only {len(df)} local paired rows. Fine-tuning on so")
        print("  few samples is unreliable — try to collect more co-location data,")
        print("  or use --freeze-conv to limit the number of trainable weights.")

    train_df, val_df, test_df = split_dataset(
        df,
        val_frac=data_cfg["val_frac"],
        test_frac=data_cfg["test_frac"],
        seed=data_cfg["seed"],
    )
    if len(train_df) < 2 or len(val_df) < 1 or len(test_df) < 1:
        print("\nERROR: not enough local data to form train/val/test splits.")
        print("Collect more co-location data before fine-tuning.")
        sys.exit(1)

    # Build datasets with the REUSED AQS scaler (fit_scaler=False).
    print("\n── Normalising with the AQS scaler (reused, not refit) ──────────")
    train_ds = PM25CalibrationDataset(
        train_df, scaler=aqs_scaler, fit_scaler=False,
        input_features=features, target_column=target,
    )
    val_ds = PM25CalibrationDataset(
        val_df, scaler=aqs_scaler, fit_scaler=False,
        input_features=features, target_column=target,
    )
    test_ds = PM25CalibrationDataset(
        test_df, scaler=aqs_scaler, fit_scaler=False,
        input_features=features, target_column=target,
    )

    # Batch size: cap at the training-set size for tiny datasets. drop_last
    # avoids a final batch of exactly 1 sample, which BatchNorm cannot handle
    # while in training mode.
    batch = min(train_cfg["batch_size"], len(train_ds))
    drop_last = (len(train_ds) % batch == 1)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=drop_last)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    # ── Baseline: how does the un-fine-tuned AQS model do on OUR sensor? ────
    # This is the "before" number. Comparing it with the "after" number below
    # tells you exactly how much fine-tuning helped.
    print("\n── Baseline: pretrained AQS model on the local test set ─────────")
    evaluate(model, test_loader, device, split_name="local test — BEFORE fine-tuning")

    # ── Fine-tune ───────────────────────────────────────────────────────────
    if freeze_conv:
        freeze_conv_block(model)
        n_trainable = count_parameters(model)
        print(f"\n  Conv feature extractor FROZEN — "
              f"{n_trainable:,} of {n_total:,} parameters will be fine-tuned.")
    else:
        print(f"\n  Full fine-tuning — all {n_total:,} parameters will be updated.")

    print("\n── Fine-tuning on local data ────────────────────────────────────")
    model = finetune_loop(model, train_loader, val_loader, ft_cfg, device, freeze_conv)

    # ── After: re-evaluate on the same local test set ───────────────────────
    print("\n── Result: fine-tuned model on the local test set ───────────────")
    evaluate(model, test_loader, device, split_name="local test — AFTER fine-tuning")

    model_dir = Path(ft_export["model_dir"])
    plot_predictions(model, test_loader, device,
                     save_path=str(model_dir / "predictions.png"))
    # Save the (reused) scaler next to the fine-tuned model so the export step
    # and any later --no-train run can find it.
    save_scaler(aqs_scaler, str(model_dir / "scaler.pkl"))

    if not export:
        print("\nSkipping export (--no-export flag set).")
        return

    # ── Export the fine-tuned model for the ESP32 firmware ──────────────────
    print("\n── Exporting fine-tuned model ───────────────────────────────────")
    model.to("cpu")

    onnx_path   = str(model_dir / ft_export["onnx_file"])
    tflite_path = str(model_dir / ft_export["tflite_file"])

    # Step 1: PyTorch → ONNX
    export_to_onnx(model, onnx_path, n_features=model_cfg["n_features"])

    # Step 2: ONNX → int8 TFLite. The representative dataset is drawn from the
    # local training inputs so the int8 quantisation ranges match what the
    # firmware will actually see.
    convert_onnx_to_tflite(
        onnx_path=onnx_path,
        tflite_path=tflite_path,
        quantize=ft_export["quantize"],
        representative_data=train_ds.X.numpy(),
    )

    # Step 3: scaler constants header. The model has 3 inputs, so we pass the
    # 3 feature names explicitly (the default is the 8-feature local order).
    export_scaler_as_c_header(
        aqs_scaler, ft_export["c_header_path"],
        feature_names=["pm2_5", "temp", "rh"],
    )

    # Step 4: model bytes header.
    export_tflite_as_c_header(tflite_path, ft_export["c_model_path"])

    print("\n── Done ─────────────────────────────────────────────────────────")
    print(f"  Fine-tuned checkpoint : {model_dir / ft_export['pytorch_checkpoint']}")
    print(f"  TFLite model          : {tflite_path}")
    print(f"  C model header        : {ft_export['c_model_path']}")
    print(f"  C scaler header       : {ft_export['c_header_path']}")
    print()
    print("  The two C headers are written into the firmware include/ folder.")
    print("  src/ml/calibrate.cpp picks them up automatically on the next")
    print("  `pio run` (it auto-detects the 3-feature model).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the AQS-pretrained calibration model on local "
                    "SEN55 + BAM co-location data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Fine-tune but skip the ONNX/TFLite export step.",
    )
    parser.add_argument(
        "--freeze-conv", action="store_true",
        help="Freeze the conv feature extractor; fine-tune only the "
             "regression head. Use when local data is very scarce. "
             "Overrides finetune.freeze_conv in config.yaml.",
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--config", default=str(script_dir.parent / "config.yaml"),
        help="Path to the shared tools/config.yaml (default: tools/config.yaml).",
    )
    args = parser.parse_args()

    # Resolve config against the caller's cwd, then move into the script's own
    # directory so the relative paths inside config.yaml resolve correctly.
    config_path = Path(args.config).resolve()
    os.chdir(script_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "finetune" not in config:
        print("ERROR: config.yaml has no 'finetune:' section.")
        sys.exit(1)

    # Command-line --freeze-conv overrides the config default.
    freeze_conv = args.freeze_conv or config["finetune"].get("freeze_conv", False)

    device = get_device()
    run_finetune(config, device, export=not args.no_export, freeze_conv=freeze_conv)


if __name__ == "__main__":
    main()

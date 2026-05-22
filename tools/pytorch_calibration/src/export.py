"""
export.py — Convert the trained PyTorch model to TFLite for ESP32 deployment.

CONVERSION PIPELINE:
    1. PyTorch (.pt checkpoint)
       ↓  torch.onnx.export()
    2. ONNX (.onnx)                 ← portable intermediate format
       ↓  onnx2tf
    3. TensorFlow SavedModel        ← temporary, discarded after step 4
       ↓  tf.lite.TFLiteConverter
    4. TFLite (.tflite)             ← deployed to ESP32-S3
       ↓  (manual step)
    5. C byte array in firmware     ← embedded in flash as a .h file

WHY THREE STEPS INSTEAD OF ONE:
    PyTorch and TFLite have no direct converter. ONNX (Open Neural Network
    Exchange) is an industry-standard intermediate format both ecosystems
    understand. Think of it as the EDF or STEP format of neural networks.

INT8 QUANTISATION:
    The trained model uses float32 weights (32 bits per parameter).
    The ESP32-S3's LX7 core emulates float operations in software — they
    are roughly 10× slower than integer operations.

    Quantisation maps each weight from float32 → int8 (1 byte).
    Result: model shrinks ~4×, inference speeds up ~5–10×, with < 1%
    accuracy loss on regression tasks like this one.

    To quantise, the converter needs a "representative dataset" — a small
    sample of real inputs so it can measure the range of activations inside
    the network and compute the int8 scaling factors. We draw this sample
    from the training set automatically in main.py.

C HEADER GENERATION:
    The ESP32 cannot run Python's scikit-learn scaler. Instead, export.py
    reads the fitted MinMaxScaler and writes a C header file containing
    the min/max value for each input channel as float constants.
    The firmware applies: scaled = (raw - MIN[i]) / (MAX[i] - MIN[i])
    before passing the values to the TFLite inference engine.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ── Step 1: PyTorch → ONNX ───────────────────────────────────────────────────

def export_to_onnx(
    model: nn.Module,
    save_path: str,
    n_features: int = 8,
) -> None:
    """
    Trace the model with a dummy input and serialise it to ONNX format.

    ONNX export works by "tracing" — PyTorch runs one forward pass with
    the dummy input while recording every operation into a graph. That
    graph is what gets written to the .onnx file.

    Args:
        model:      Trained SensaCalibNet with best weights loaded.
        save_path:  Destination path for the .onnx file.
        n_features: Number of input features (must match training config).
    """
    model.eval()

    # A dummy input with the same shape as one real sample.
    # Batch size of 1 matches ESP32 inference (one reading at a time).
    dummy_input = torch.zeros(1, n_features)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        # Force the legacy TorchScript exporter. Recent PyTorch defaults to the
        # dynamo-based exporter, which requires the extra `onnxscript` package
        # and emits a different graph style. opset_version=11 + dynamic_axes
        # below are legacy-exporter idioms, and onnx2tf expects that graph.
        dynamo=False,
        # opset_version=11 is the oldest version supported by onnx2tf.
        # Higher versions support newer ops but may not be supported by
        # downstream tools. 11 is the safe, widely compatible choice.
        opset_version=11,
        input_names=['sensor_input'],
        output_names=['calibrated_pm2_5'],
        # dynamic_axes lets the exported model accept any batch size,
        # even though we always send batch=1 on the ESP32.
        dynamic_axes={
            'sensor_input':      {0: 'batch_size'},
            'calibrated_pm2_5':  {0: 'batch_size'},
        },
    )
    print(f"  ONNX model saved → {save_path}")


# ── Step 2: ONNX → TFLite ────────────────────────────────────────────────────

def convert_onnx_to_tflite(
    onnx_path: str,
    tflite_path: str,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> None:
    """
    Convert an ONNX model to TFLite, with optional int8 post-training quantisation.

    DEPENDENCIES (must be installed):
        pip install onnx2tf tensorflow-cpu

    Args:
        onnx_path:            Path to the .onnx file from export_to_onnx().
        tflite_path:          Destination path for the .tflite file.
        quantize:             If True, apply int8 post-training quantisation.
        representative_data:  NumPy array of shape (N, n_features) drawn from
                              the training set. Required for int8 quantisation.
                              500 samples is sufficient.
    """
    # ── Import onnx2tf (optional dependency) ─────────────────────────────────
    try:
        import onnx2tf
    except ImportError:
        print("ERROR: onnx2tf is not installed.")
        print("       Run: pip install onnx2tf")
        return

    # ── Import TensorFlow (needed only for the TFLite conversion step) ───────
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: tensorflow is not installed.")
        print("       Run: pip install tensorflow-cpu")
        return

    import tempfile

    # onnx2tf outputs a TF SavedModel directory, not a single file.
    # We use a temporary directory so it is cleaned up automatically.
    with tempfile.TemporaryDirectory() as tmpdir:
        savedmodel_dir = Path(tmpdir) / "savedmodel"

        print("  Converting ONNX → TensorFlow SavedModel ...")
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=str(savedmodel_dir),
            non_verbose=True,
        )

        # ── TFLite conversion ─────────────────────────────────────────────────
        print("  Converting TF SavedModel → TFLite ...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))

        if quantize and representative_data is not None:
            print("  Applying int8 post-training quantisation ...")

            # Tell the converter to use integer arithmetic throughout.
            converter.optimizations           = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Set both input and output tensors to int8 so the ESP32 does not
            # need to perform float↔int conversion at the boundary.
            converter.inference_input_type    = tf.int8
            converter.inference_output_type   = tf.int8

            # The representative dataset is a generator that yields one sample
            # at a time. The converter calls it ~500 times to measure the range
            # of activations in each layer and determine the int8 scale factors.
            def representative_dataset_gen():
                n = min(500, len(representative_data))
                for i in range(n):
                    sample = representative_data[i : i + 1].astype(np.float32)
                    yield [sample]

            converter.representative_dataset = representative_dataset_gen

        elif quantize and representative_data is None:
            # Without representative data we cannot determine scale factors,
            # so int8 quantisation is skipped. The model will still work but
            # will be ~4× larger and slower on the ESP32.
            print("  WARNING: quantize=True but no representative_data provided.")
            print("           Falling back to float32 TFLite. Pass training data to main.py.")

        tflite_model = converter.convert()

        # Write the .tflite binary to disk
        Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model saved → {tflite_path}  ({size_kb:.1f} KB)")

    if size_kb > 30:
        print(f"  WARNING: {size_kb:.1f} KB exceeds the 30 KB ESP32 budget.")
        print("           Reduce n_channels in config.yaml and retrain.")
    else:
        print(f"  Model is within ESP32 budget ({size_kb:.1f} / 30 KB).")


# ── Step 3: Generate C header for firmware normalisation ─────────────────────

def export_scaler_as_c_header(
    scaler: MinMaxScaler,
    output_path: str,
    feature_names: Optional[List[str]] = None,
) -> None:
    """
    Write the MinMaxScaler's parameters as a C header file for the ESP32 firmware.

    WHAT THIS IS FOR:
        The ESP32 cannot run Python. Before calling the TFLite inference engine,
        the firmware must normalise each raw SEN55 reading to [0, 1] using the
        same formula that was applied during training:

            scaled = (raw_value - MIN[i]) / (MAX[i] - MIN[i])

        This function hard-codes the per-channel min and max values into a C
        header so the firmware can do that calculation without Python.

    Args:
        scaler:        Fitted MinMaxScaler from a PM25CalibrationDataset.
        output_path:   Destination path (e.g., "../../include/calib_scaler.h").
        feature_names: Input feature names, in model-input order. Defaults to
                       the 8-feature local SEN55 order. The 3-feature
                       combined/public model passes ['pm2_5', 'temp', 'rh'].
                       This MUST match the model's actual input count.
    """
    # Feature order must exactly match the model's input order (INPUT_FEATURES
    # in dataset.py for the local model, public_data.features for the others).
    # Any mismatch produces silently wrong predictions on the ESP32.
    if feature_names is None:
        feature_names = ['pm1', 'pm2_5', 'pm4', 'pm10', 'temp', 'rh', 'voc', 'nox']

    mins  = scaler.data_min_
    maxes = scaler.data_max_

    # Guard against the silent-bug case where the scaler was fit on a
    # different number of features than feature_names describes.
    if len(feature_names) != len(mins):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but the scaler "
            f"was fit on {len(mins)} features. They must match — pass the "
            f"correct feature_names for this model."
        )

    lines = [
        "// calib_scaler.h — Auto-generated by tools/pytorch_calibration/src/export.py",
        "// DO NOT edit manually. Re-run: python main.py --export-only",
        "//",
        "// Normalisation parameters for SensaCalibNet PM2.5 calibration model.",
        "// Apply these to raw SEN55 readings before TFLite inference.",
        "//",
        "// Formula: scaled_val = (raw_val - CALIB_MIN[i]) / (CALIB_MAX[i] - CALIB_MIN[i])",
        "// where i is the feature index defined by CALIB_FEATURE_ORDER below.",
        "",
        "#pragma once",
        "",
        "// Number of input features fed to the model.",
        f"#define CALIB_N_FEATURES {len(feature_names)}",
        "",
        "// Feature order (index → channel name):",
        "//   " + "  ".join(f"[{i}] {name}" for i, name in enumerate(feature_names)),
        "",
        f"static const float CALIB_MIN[{len(feature_names)}] = {{",
        "    " + ", ".join(f"{v:.6f}f" for v in mins),
        "};",
        "",
        f"static const float CALIB_MAX[{len(feature_names)}] = {{",
        "    " + ", ".join(f"{v:.6f}f" for v in maxes),
        "};",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  C scaler header saved → {output_path}")
    print("  Copy this file into include/ and #include it in your inference code.")


# ── Step 4: Embed the .tflite model as a C byte array ────────────────────────

def export_tflite_as_c_header(
    tflite_path: str,
    output_path: str,
    var_name: str = "g_calib_model_tflite",
) -> None:
    """
    Convert a .tflite binary into a C header containing it as a byte array.

    WHAT THIS IS FOR:
        A microcontroller has no filesystem to load a .tflite file from at
        runtime. Instead the model bytes are compiled directly into the
        firmware image and stored in flash. This function turns the binary
        .tflite file into a C array the firmware can #include.

        This is the same job the Unix `xxd -i` tool does, but done here so the
        whole export is one reproducible Python step with no manual commands.

    The generated header defines:
        <var_name>[]      — the model bytes (const, 8-byte aligned for TFLM)
        <var_name>_len    — the number of bytes

    Args:
        tflite_path: Path to the .tflite file from convert_onnx_to_tflite().
        output_path: Destination .h path (e.g., "../../include/calib_model.h").
        var_name:    C identifier for the generated array.
    """
    data = Path(tflite_path).read_bytes()

    # Format the bytes 12-per-line so the header stays human-readable.
    byte_lines = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        byte_lines.append("    " + " ".join(f"0x{b:02x}," for b in chunk))

    lines = [
        f"// {Path(output_path).name} — Auto-generated by "
        "tools/pytorch_calibration/src/export.py",
        "// DO NOT edit manually. Re-run: python main.py --export-only",
        "//",
        "// The Sensa PM2.5 calibration model as a C byte array, embedded in",
        "// firmware flash and loaded by src/ml/calibrate.cpp at start-up.",
        "",
        "#pragma once",
        "",
        "// alignas(8): TensorFlow Lite Micro requires the model data to be at",
        "// least 8-byte aligned in memory.",
        f"alignas(8) const unsigned char {var_name}[] = {{",
        *byte_lines,
        "};",
        "",
        f"const unsigned int {var_name}_len = {len(data)};",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  C model header saved → {output_path}  "
          f"({len(data) / 1024:.1f} KB array)")

    # Best-effort: list the operators the model uses, so the team can keep the
    # MicroMutableOpResolver in calibrate.cpp in sync. Non-fatal if it fails.
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        ops = sorted({d['op_name'] for d in interp._get_ops_details()})
        print(f"  Model operators ({len(ops)}): {', '.join(ops)}")
        print("  Ensure calibrate.cpp's MicroMutableOpResolver registers each.")
    except Exception:
        # Listing ops is a convenience only — skip silently if TF cannot.
        pass

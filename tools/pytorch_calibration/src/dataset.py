"""
dataset.py — Data loading, cleaning, and normalisation for the calibration model.

WHAT THIS MODULE DOES:
    1. Loads a paired CSV (one row = one SEN55 reading + BAM reference value).
    2. Cleans obvious bad values (negative PM, out-of-range humidity, etc.).
    3. Normalises inputs to [0, 1] so all features have the same scale.
    4. Wraps the data in a PyTorch Dataset that the training loop can iterate.

ABOUT NORMALISATION:
    Neural networks are sensitive to input scale. If PM10 ranges from 0–200
    while temperature ranges from 15–35, the PM10 gradients dominate and the
    network ignores temperature. Min-max scaling maps every feature to [0, 1]
    so each channel has equal influence.

    The scaler's parameters (min and max per feature) must be saved alongside
    the model. The ESP32 firmware must apply the same scaling before calling
    the TFLite inference engine. export.py generates a C header file with
    these constants embedded.

EXPECTED CSV COLUMNS:
    timestamp   — ISO 8601 datetime string (e.g. "2024-03-15 14:30:00")
    pm1         — SEN55 PM1.0 concentration (µg/m³)
    pm2_5       — SEN55 PM2.5 concentration (µg/m³)
    pm4         — SEN55 PM4.0 concentration (µg/m³)
    pm10        — SEN55 PM10 concentration (µg/m³)
    temp        — SEN55 temperature (°C)
    rh          — SEN55 relative humidity (%)
    voc         — SEN55 VOC index (1–500, dimensionless)
    nox         — SEN55 NOx index (1–500, dimensionless)
    bam_pm2_5   — Reference PM2.5 from BAM or AQS station (µg/m³)

    Run prepare_data.py first to generate this CSV from raw .pkl and .csv files.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


# ── Feature column names ──────────────────────────────────────────────────────
# Must match the column names output by uart_logger.py and prepare_data.py.
# The order here is the order in which features are fed to the model —
# this same order must be used in firmware preprocessing.
INPUT_FEATURES = ['pm1', 'pm2_5', 'pm4', 'pm10', 'temp', 'rh', 'voc', 'nox']

# The reference measurement we are trying to predict.
TARGET_COLUMN = 'bam_pm2_5'


class PM25CalibrationDataset(Dataset):
    """
    PyTorch Dataset for paired (SEN55 reading, BAM PM2.5) samples.

    A PyTorch Dataset is a class that knows:
      1. How many samples exist     (__len__)
      2. How to return one sample   (__getitem__)

    The DataLoader (in train.py) calls these methods to batch and shuffle data.
    Think of the Dataset as a shelf of labelled lab samples; the DataLoader
    is the technician who picks them up in random order and hands them to you
    in groups of batch_size.

    Usage:
        df = load_paired_csv("data/paired/paired_dataset.csv")
        train_ds = PM25CalibrationDataset(df_train, fit_scaler=True)
        val_ds   = PM25CalibrationDataset(df_val, scaler=train_ds.scaler, fit_scaler=False)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        scaler: Optional[MinMaxScaler] = None,
        fit_scaler: bool = True,
    ):
        """
        Args:
            dataframe:  A cleaned DataFrame with INPUT_FEATURES and TARGET_COLUMN.
            scaler:     A pre-fit MinMaxScaler. For training data, pass None and
                        set fit_scaler=True. For val/test data, pass the scaler
                        that was fit on the training set so the same scaling
                        is applied consistently.
            fit_scaler: If True, fit a new scaler on this data.
                        Always True for the training split; always False for
                        validation and test splits.
        """
        self.df = dataframe.reset_index(drop=True)

        # Pull out raw numpy arrays for fast indexing
        X_raw = self.df[INPUT_FEATURES].values.astype(np.float32)
        y_raw = self.df[TARGET_COLUMN].values.astype(np.float32)

        # ── Normalise inputs ──────────────────────────────────────────────────
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        else:
            self.scaler = scaler

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            # Use the parameters already computed on the training set.
            # NEVER fit the scaler on validation or test data — that would
            # leak information about those splits into the scaling transform.
            X_scaled = self.scaler.transform(X_raw)

        # ── Convert to PyTorch tensors ────────────────────────────────────────
        # A tensor is the fundamental data container in PyTorch. It behaves
        # like a numpy array but can live on a GPU and tracks gradients.
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        # unsqueeze(1) turns shape (N,) into (N, 1) — the model outputs shape
        # (batch, 1), so labels must match.
        self.y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        """Returns the total number of (sensor reading, reference) pairs."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return one sample by index.

        Args:
            idx: Integer index into the dataset.

        Returns:
            x: Float tensor of shape (8,) — one normalised SEN55 reading.
            y: Float tensor of shape (1,) — BAM PM2.5 reference value (µg/m³).
        """
        return self.X[idx], self.y[idx]


# ── File I/O helpers ──────────────────────────────────────────────────────────

def load_paired_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the paired SEN55 + BAM dataset from CSV and apply basic cleaning.

    Args:
        csv_path: Path to the CSV file produced by prepare_data.py.

    Returns:
        A clean pandas DataFrame ready for PM25CalibrationDataset.
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    initial_rows = len(df)

    # Drop rows missing the reference label — they cannot be used for training.
    df = df.dropna(subset=[TARGET_COLUMN])

    # Drop rows missing any input feature.
    df = df.dropna(subset=INPUT_FEATURES)

    # PM concentration cannot be negative — clamp any noise-induced negatives.
    for col in ['pm1', 'pm2_5', 'pm4', 'pm10']:
        df[col] = df[col].clip(lower=0.0)

    # Relative humidity is bounded [0, 100]% by physical definition.
    df['rh'] = df['rh'].clip(lower=0.0, upper=100.0)

    # VOC and NOx indices are defined on [1, 500].
    for col in ['voc', 'nox']:
        df[col] = df[col].clip(lower=1.0, upper=500.0)

    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing or invalid values.")
    print(f"  Loaded {len(df)} valid paired samples from {csv_path}")

    return df


def split_dataset(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into non-overlapping train, validation, and test sets.

    WHY WE SHUFFLE BEFORE SPLITTING:
        Sensor data is collected in time order. Without shuffling, the training
        set would always be the earliest readings and the test set would be the
        latest. If air quality changed over the collection period (e.g., a
        wildfire), the model would be evaluated on a different distribution
        than it was trained on. Shuffling breaks this temporal dependency and
        gives a more honest accuracy estimate.

        EXCEPTION: if you collected data on distinct days in widely different
        conditions (clean day vs. smoke event), consider splitting by date
        instead, passing each day's data as its own session.

    Args:
        df:        Full cleaned dataset.
        val_frac:  Fraction of rows reserved for validation.
        test_frac: Fraction of rows reserved for final testing.
        seed:      Random seed so the same split is reproduced every run.

    Returns:
        (train_df, val_df, test_df) — three non-overlapping DataFrames.
    """
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)

    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_df  = df_shuffled.iloc[:n_test]
    val_df   = df_shuffled.iloc[n_test : n_test + n_val]
    train_df = df_shuffled.iloc[n_test + n_val :]

    print(f"  Dataset split → {len(train_df)} train | {len(val_df)} val | {len(test_df)} test")
    return train_df, val_df, test_df


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """
    Pickle the fitted MinMaxScaler to disk so it can be reloaded later.

    The scaler must be saved alongside the model because the same min/max
    values used during training must be applied on the ESP32 before inference.
    export.py reads this file and generates a C header with those constants.

    Args:
        scaler: Fitted MinMaxScaler from a PM25CalibrationDataset.
        path:   File path (e.g., "models/scaler.pkl").
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {path}")


def load_scaler(path: str) -> MinMaxScaler:
    """Load a previously saved MinMaxScaler from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)

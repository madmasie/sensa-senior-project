"""
epa_aqs.py — Download and parse EPA AQS bulk hourly data files.

Data source: https://aqs.epa.gov/aqsweb/airdata/download_files.html
No API key required. Files are freely available as annual ZIP archives.

URL pattern:
    https://aqs.epa.gov/aqsweb/airdata/hourly_{param_code}_{year}.zip

Each ZIP contains one CSV covering the entire US for that parameter and year.
After download, we filter to the target state and cache the result so the
large national file is only parsed once per year.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# AQS parameter codes
# ---------------------------------------------------------------------------
PARAM = {
    "PM25_FRM":  88101,  # PM2.5 FRM/FEM — gold-standard reference (BAM, filter)
    "PM25_CONT": 88502,  # PM2.5 continuous non-FRM — optical/TEOM, sensor analog
    "TEMP":      62101,  # Outdoor air temperature (°C)
    "RH":        62201,  # Relative humidity (%)
    "SO2":       42401,  # Sulfur dioxide (ppb) — optional interference correction
    "NO2":       42602,  # Nitrogen dioxide (ppb) — optional interference correction
}

# Reverse map for human-readable column names
PARAM_NAME = {v: k for k, v in PARAM.items()}

# State FIPS codes (zero-padded two-digit strings as stored in AQS)
STATE_FIPS = {
    "CA": "06",
    "TX": "48",
    "NY": "36",
    "FL": "12",
    "CO": "08",
    "WA": "53",
    "OR": "41",
    "AZ": "04",
    "NV": "32",
    "UT": "49",
}

AQS_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"

# Columns to load from the raw AQS CSV (subset keeps memory use reasonable)
_USECOLS = [
    "State Code", "County Code", "Site Num", "Parameter Code",
    "POC", "Latitude", "Longitude",
    "Date GMT", "Time GMT", "Sample Measurement",
    "Units of Measure", "MDL", "Method Type",
]


def download_aqs_hourly(
    param_code: int,
    year: int,
    cache_dir: Path,
    force_download: bool = False,
) -> Path:
    """
    Download an EPA AQS hourly bulk ZIP and extract the CSV inside it.

    Files are cached; subsequent calls return immediately from cache.
    Set force_download=True to re-fetch even if the cache file exists.

    Args:
        param_code:     AQS parameter code (e.g. 88101 for PM2.5 FRM).
        year:           Calendar year (e.g. 2022).
        cache_dir:      Directory to store extracted CSVs.
        force_download: Re-download even if cached CSV already exists.

    Returns:
        Path to the extracted (unzipped) CSV file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"hourly_{param_code}_{year}.csv"

    if csv_path.exists() and not force_download:
        print(f"  Cache hit: {csv_path.name}")
        return csv_path

    url = f"{AQS_BASE_URL}/hourly_{param_code}_{year}.zip"
    print(f"  Downloading {url}")

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total_bytes = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total_bytes, unit="B", unit_scale=True,
              desc=f"  param={param_code} year={year}") as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB chunks
            buf.write(chunk)
            pbar.update(len(chunk))

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(
                f"No CSV found in ZIP from {url}. "
                "Check that the parameter code and year are valid."
            )
        zf.extract(csv_names[0], path=cache_dir)
        extracted = cache_dir / csv_names[0]
        if extracted != csv_path:
            extracted.rename(csv_path)

    print(f"  Saved → {csv_path.name}  ({csv_path.stat().st_size / 1e6:.1f} MB)")
    return csv_path


def load_aqs_state(
    csv_path: Path,
    state_fips: str,
    param_code: int,
) -> pd.DataFrame:
    """
    Load an AQS hourly CSV, filter to one state, and return a tidy DataFrame.

    AQS CSVs use state_code as a zero-padded string (e.g. "06" for California).
    The function handles the leading-zero issue by reading the column as str.

    Output columns:
        site_id   — "{state}-{county}-{site}" identifier string
        timestamp — UTC-aware datetime at hourly resolution
        latitude  — site latitude (float)
        longitude — site longitude (float)
        poc       — Parameter Occurrence Code (int); 1 = primary instrument
        param_{code} — measurement value in the parameter's native units

    Args:
        csv_path:   Path to the downloaded and extracted AQS hourly CSV.
        state_fips: Two-digit FIPS code string (e.g. "06").
        param_code: AQS parameter code; used to name the value column.

    Returns:
        Filtered and cleaned DataFrame. May be empty if the state has no data
        for this parameter/year (e.g. SO2 is not monitored everywhere).
    """
    df = pd.read_csv(
        csv_path,
        usecols=_USECOLS,
        dtype={
            "State Code":   str,
            "County Code":  str,
            "Site Num":     str,
            "Parameter Code": int,
            "POC":          int,
        },
        low_memory=False,
    )

    # Normalize column names: lowercase, replace spaces with underscores
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Filter to target state
    state_code = state_fips.zfill(2)
    df = df[df["state_code"] == state_code].copy()

    if df.empty:
        print(f"  WARNING: No rows for state '{state_fips}' in {csv_path.name}")
        return pd.DataFrame()

    # Build a stable site identifier
    df["site_id"] = (
        df["state_code"]
        + "-" + df["county_code"].str.zfill(3)
        + "-" + df["site_num"].str.zfill(4)
    )

    # Parse UTC timestamp (AQS Date GMT + Time GMT columns are in HH:MM format)
    df["timestamp"] = pd.to_datetime(
        df["date_gmt"] + " " + df["time_gmt"],
        format="%Y-%m-%d %H:%M",
        utc=True,
    )

    # Drop rows without a measurement
    df = df.dropna(subset=["sample_measurement"])

    # Replace measurements below the MDL with half the MDL (standard convention)
    mdl = df["mdl"].fillna(0.0)
    below_mdl = df["sample_measurement"] < mdl
    df.loc[below_mdl, "sample_measurement"] = mdl[below_mdl] / 2.0

    value_col = f"param_{param_code}"
    result = df[["site_id", "timestamp", "latitude", "longitude",
                 "poc", "sample_measurement"]].copy()
    result = result.rename(columns={"sample_measurement": value_col})

    print(
        f"  Loaded {len(result):,} rows | "
        f"{result['site_id'].nunique()} sites | "
        f"param={param_code} ({PARAM_NAME.get(param_code, '?')})"
    )
    return result.reset_index(drop=True)

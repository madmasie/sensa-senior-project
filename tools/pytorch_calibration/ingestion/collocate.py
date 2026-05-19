"""
collocate.py — Merge co-located AQS monitor measurements into paired rows.

"Co-located" means the same AQS site (state-county-site FIPS code) reported
valid readings for multiple parameter codes within the same UTC hour. This is
how calibration-grade paired datasets are constructed: a reference instrument
(PM2.5 FRM/FEM) and a continuous optical monitor at the same physical site
give us (raw_optical, true_pm2_5) pairs for training.

Typical California AQS sites with co-located instruments:
  - Major urban sites (LA, Bay Area, San Joaquin Valley) often run a
    continuous FEM monitor alongside a filter-based FRM sampler.
  - Some research sites add non-FRM continuous monitors (88502) for QA.
  - Meteorological stations (temp, RH) are frequently co-located.

When multiple instruments of the same type are at one site (POC > 1), we
keep the primary instrument (lowest POC) to avoid double-counting.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def merge_parameters(
    dfs: dict[int, pd.DataFrame],
    primary_param: int,
    feature_params: list[int],
    max_sites: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inner-join parameter DataFrames from co-located AQS monitors.

    Rows are retained only when ALL required parameters (primary + features)
    have valid readings for the same site and UTC hour. This inner-join
    strategy is conservative but produces clean, fully-populated training rows.

    Args:
        dfs:           Mapping of param_code → DataFrame (from load_aqs_state).
                       Each DataFrame must have columns: site_id, timestamp,
                       param_{code}, latitude, longitude, poc.
        primary_param: Parameter code for the calibration target (e.g. 88101).
        feature_params: Parameter codes to use as input features.
        max_sites:     If set, limit to this many sites. Useful for a quick
                       sanity check before running the full download.

    Returns:
        Merged DataFrame with one row per (site, UTC hour) where all
        parameters have data. Columns: site_id, timestamp, latitude,
        longitude, param_{primary}, param_{feat1}, param_{feat2}, ...
    """
    required = [primary_param] + feature_params
    missing_keys = [p for p in required if p not in dfs]
    if missing_keys:
        raise ValueError(
            f"No DataFrame provided for parameter(s): {missing_keys}. "
            "Did fetch_public_data.py download all required parameter files?"
        )

    def _keep_primary_poc(df: pd.DataFrame, code: int) -> pd.DataFrame:
        """
        When multiple instruments of the same type share a site, keep the
        one with the lowest POC (POC=1 is the designated primary instrument).
        """
        value_col = f"param_{code}"
        df = df.sort_values("poc")
        return (
            df.groupby(["site_id", "timestamp"], as_index=False)
            .first()
        )

    # Start with the reference (primary) parameter
    base = _keep_primary_poc(dfs[primary_param], primary_param)[
        ["site_id", "timestamp", "latitude", "longitude", f"param_{primary_param}"]
    ]
    print(
        f"  Base param {primary_param}: {len(base):,} rows | "
        f"{base['site_id'].nunique()} sites"
    )

    # Inner-join each feature parameter, reporting how each step shrinks the
    # set so an empty result can be traced to the parameter that caused it.
    for param in feature_params:
        feat_df = _keep_primary_poc(dfs[param], param)[
            ["site_id", "timestamp", f"param_{param}"]
        ]
        # How much of the join key actually overlaps, before merging.
        base_keys = set(zip(base["site_id"], base["timestamp"]))
        feat_keys = set(zip(feat_df["site_id"], feat_df["timestamp"]))
        shared_sites = base["site_id"].isin(feat_df["site_id"]).sum()
        base = base.merge(feat_df, on=["site_id", "timestamp"], how="inner")
        print(
            f"  + join param {param}: {len(base):,} rows | "
            f"{base['site_id'].nunique()} sites "
            f"(feature had {feat_df['site_id'].nunique()} sites; "
            f"{len(base_keys & feat_keys):,} site-hour keys overlapped)"
        )
        if base.empty:
            raise ValueError(
                f"Join with parameter {param} produced zero rows.\n"
                f"The reference set had {shared_sites:,} rows at sites that "
                f"also appear in param {param}, but none matched on the same\n"
                f"UTC hour. Param {param} is the parameter that collapses the "
                f"dataset.\n"
                "Consider:\n"
                "  - Adding more years (--years 2019 2020 2021 2022)\n"
                f"  - Dropping param {param} from the feature set if it is "
                f"rarely co-located\n"
                "  - Removing optional parameters (drop --include-gas)"
            )

    if base.empty:
        raise ValueError(
            "No co-located rows found after merging all parameters.\n"
            "This usually means the selected state has no sites with all\n"
            "required co-located instruments in the requested year range.\n"
            "Consider:\n"
            "  - Adding more years (--years 2019 2020 2021 2022)\n"
            "  - Removing optional parameters (drop --include-gas)\n"
            "  - Checking whether param 88502 (non-FRM) exists for this state"
        )

    # Optionally limit to a subset of sites for quick testing
    if max_sites is not None:
        sites = base["site_id"].unique()[:max_sites]
        base = base[base["site_id"].isin(sites)].copy()

    # Remove rows with any NaN in the value columns
    value_cols = [f"param_{p}" for p in required]
    base = base.dropna(subset=value_cols)

    print(f"  Co-located rows: {len(base):,}")
    print(f"  Sites with all parameters: {base['site_id'].nunique()}")
    print(f"  Date range: {base['timestamp'].min()} – {base['timestamp'].max()}")

    return base.reset_index(drop=True)

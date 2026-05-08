"""
corrections.py — Extensible physics-based correction pipeline for optical PM2.5.

BACKGROUND
----------
Low-cost optical particle counters (including the SEN55 and co-located AQS
continuous monitors) report PM concentrations derived from light-scattering.
Several physical processes cause their readings to diverge from a reference
mass-based instrument (BAM/FRM):

  1. Hygroscopic growth (humidity): Water-soluble particles absorb moisture
     at high RH, grow larger, scatter more light, and inflate the optical
     reading. This is the dominant correction in most environments.

  2. Temperature: Volatile species (ammonium nitrate, organic aerosols)
     evaporate at high temperatures, causing under-reading. Sensor detector
     efficiency also drifts with temperature.

  3. Gas-phase interference (SO2, NO2): Some OPC wavelengths overlap with
     SO2/NO2 absorption bands, producing false particle counts.

  4. Particle composition: The assumed refractive index (usually that of
     ammonium sulfate) may not match the local aerosol mix.

DESIGN
------
Each correction factor is an independent, composable object:
  - Subclass CorrectionFactor and implement apply() + required_columns.
  - CorrectionPipeline chains any number of factors in sequence.
  - Build the pipeline from config.yaml via CorrectionPipeline.from_config().

Corrections modify the OPTICAL PM2.5 column (target_col) before ML training.
The corrected value becomes a "pre-cleaned" input feature. The ML model then
learns any residual correction on top of the physics-based pre-processing.
The reference column (param_88101) is never modified.

ADDING A NEW CORRECTION
-----------------------
1. Subclass CorrectionFactor (see HumidityCorrection as a template).
2. Implement required_columns (list of DataFrame column names needed).
3. Implement apply(df, target_col) → modified DataFrame.
4. Register the class name in CorrectionPipeline._FACTOR_MAP.
5. Add the correction parameters to config.yaml under public_data.corrections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CorrectionFactor(ABC):
    """Abstract base for a single physics-based PM2.5 correction step."""

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """DataFrame columns that must exist before apply() is called."""
        ...

    @abstractmethod
    def apply(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Apply the correction to target_col (optical PM2.5) and return a new
        DataFrame. Must not modify any reference/ground-truth columns.

        Args:
            df:         Input DataFrame containing required_columns + target_col.
            target_col: Name of the optical PM2.5 column to correct in place.

        Returns:
            New DataFrame (copy) with target_col updated.
        """
        ...

    def _assert_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires columns {missing} "
                f"which are not present in the DataFrame."
            )


class HumidityCorrection(CorrectionFactor):
    """
    Correct optical PM2.5 for hygroscopic particle growth at high RH.

    At elevated relative humidity, hygroscopic particles absorb water,
    swell, and scatter proportionally more light — causing optical PM to
    read artificially high. This correction inverts the growth factor.

    Formula (Hänel 1976 / single-kappa model):
        f(RH) = 1 + kappa * RH / (100 - RH)
        PM2.5_corrected = PM2.5_optical / f(RH)

    kappa — hygroscopic growth coefficient:
        0.3–0.4  low-hygroscopicity aerosol (mineral dust, biomass smoke)
        0.4–0.6  mixed urban aerosol (typical default: 0.5)
        0.6–0.8  high-hygroscopicity (ammonium sulfate, marine aerosol)

    Reference:
        Kruse, M. et al. (2020). "Towards improved PM2.5 sensor corrections
        at high relative humidity." Atmos. Meas. Tech.
    """

    def __init__(
        self,
        kappa: float = 0.5,
        rh_col: str = "param_62201",
    ):
        self.kappa = kappa
        self.rh_col = rh_col

    @property
    def required_columns(self) -> list[str]:
        return [self.rh_col]

    def apply(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        self._assert_columns(df)
        df = df.copy()
        rh = df[self.rh_col].clip(0.0, 99.0)  # avoid division by zero at RH=100
        growth_factor = 1.0 + self.kappa * rh / (100.0 - rh)
        df[target_col] = (df[target_col] / growth_factor).clip(lower=0.0)
        return df


class TemperatureCorrection(CorrectionFactor):
    """
    Correct optical PM2.5 for temperature-dependent measurement bias.

    At temperatures above T_ref, semi-volatile species (ammonium nitrate,
    secondary organics) evaporate and are lost from the particle phase, so
    optical sensors under-read relative to filter-based reference methods.
    At low temperatures the opposite occurs.

    Formula (first-order linear approximation):
        PM2.5_corrected = PM2.5_optical * (1 + alpha * (T - T_ref))

    alpha — temperature sensitivity coefficient in 1/°C.
    Default 0.0 (correction disabled) because the coefficient is
    highly aerosol-composition-dependent and not yet characterised for
    the SEN55. Override once you have co-location data spanning a wide
    temperature range.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        t_ref: float = 20.0,
        temp_col: str = "param_62101",
    ):
        self.alpha = alpha
        self.t_ref = t_ref
        self.temp_col = temp_col

    @property
    def required_columns(self) -> list[str]:
        return [self.temp_col]

    def apply(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.alpha == 0.0:
            return df  # no-op: coefficient not yet characterised
        self._assert_columns(df)
        df = df.copy()
        delta_t = df[self.temp_col] - self.t_ref
        df[target_col] = (df[target_col] * (1.0 + self.alpha * delta_t)).clip(lower=0.0)
        return df


class SO2Correction(CorrectionFactor):
    """
    Correct for sulfur dioxide interference in optical particle counters.

    Some OPC scattering channels are sensitive to SO2 gas, which produces
    a false particle-count signal proportional to [SO2].

    Formula:
        PM2.5_corrected = PM2.5_optical - beta * SO2_ppb

    beta — interference coefficient in (µg/m³)/ppb.
    Default 0.0 (disabled). Requires --include-gas when downloading data.
    """

    def __init__(self, beta: float = 0.0, so2_col: str = "param_42401"):
        self.beta = beta
        self.so2_col = so2_col

    @property
    def required_columns(self) -> list[str]:
        return [self.so2_col]

    def apply(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.beta == 0.0:
            return df
        self._assert_columns(df)
        df = df.copy()
        df[target_col] = (df[target_col] - self.beta * df[self.so2_col]).clip(lower=0.0)
        return df


class NO2Correction(CorrectionFactor):
    """
    Correct for nitrogen dioxide interference in optical particle counters.

    NO2 absorbs light at wavelengths used by some OPC laser sources, creating
    an apparent scattering signal that inflates PM readings.

    Formula:
        PM2.5_corrected = PM2.5_optical - gamma * NO2_ppb

    gamma — interference coefficient in (µg/m³)/ppb.
    Default 0.0 (disabled). Requires --include-gas when downloading data.
    """

    def __init__(self, gamma: float = 0.0, no2_col: str = "param_42602"):
        self.gamma = gamma
        self.no2_col = no2_col

    @property
    def required_columns(self) -> list[str]:
        return [self.no2_col]

    def apply(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.gamma == 0.0:
            return df
        self._assert_columns(df)
        df = df.copy()
        df[target_col] = (df[target_col] - self.gamma * df[self.no2_col]).clip(lower=0.0)
        return df


class CorrectionPipeline:
    """
    Apply a sequence of CorrectionFactor steps to the optical PM2.5 column.

    Steps are applied in order. Each step receives the output of the previous
    step, so corrections compose: the humidity-corrected value is further
    corrected for temperature, and so on.

    If a correction's required columns are absent (e.g. SO2 was not downloaded),
    that step is silently skipped with a warning rather than raising an error.

    Example:
        pipeline = CorrectionPipeline(
            factors=[HumidityCorrection(kappa=0.5), TemperatureCorrection()],
            target_col="param_88502",
        )
        df_corrected = pipeline.apply(df_raw)
    """

    # Registry: config.yaml key → correction class
    _FACTOR_MAP: dict[str, type[CorrectionFactor]] = {
        "humidity":    HumidityCorrection,
        "temperature": TemperatureCorrection,
        "so2":         SO2Correction,
        "no2":         NO2Correction,
    }

    def __init__(
        self,
        factors: list[CorrectionFactor],
        target_col: str = "param_88502",
    ):
        self.factors = factors
        self.target_col = target_col

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all factors in sequence to self.target_col."""
        for factor in self.factors:
            missing = [c for c in factor.required_columns if c not in df.columns]
            if missing:
                print(
                    f"  Skipping {factor.__class__.__name__}: "
                    f"required columns not available: {missing}"
                )
                continue
            df = factor.apply(df, self.target_col)
        return df

    @classmethod
    def from_config(
        cls,
        corrections_cfg: dict,
        target_col: str = "param_88502",
    ) -> "CorrectionPipeline":
        """
        Build a CorrectionPipeline from the 'corrections' sub-dict in config.yaml.

        Expected config structure:
            corrections:
              humidity:
                kappa: 0.5
              temperature:
                alpha: 0.0
                t_ref: 20.0
              so2:
                beta: 0.0

        Each key matches a name in _FACTOR_MAP; the value dict is passed as
        keyword arguments to the correction's __init__.
        """
        factors = []
        for name, params in (corrections_cfg or {}).items():
            if name not in cls._FACTOR_MAP:
                raise ValueError(
                    f"Unknown correction '{name}'. "
                    f"Available: {list(cls._FACTOR_MAP.keys())}"
                )
            factors.append(cls._FACTOR_MAP[name](**(params or {})))
        return cls(factors=factors, target_col=target_col)

    @classmethod
    def default(cls, target_col: str = "param_88502") -> "CorrectionPipeline":
        """Return a default pipeline with humidity correction only (kappa=0.5)."""
        return cls(factors=[HumidityCorrection(kappa=0.5)], target_col=target_col)

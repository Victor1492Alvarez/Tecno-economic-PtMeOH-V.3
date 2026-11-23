from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd

from domain.units import (
    DEFAULT_ANNUAL_HOURS,
    HOURS_PER_DAY,
    interval_energy_kwh_to_power_mw,
    interval_energy_mwh_to_power_mw,
    kw_to_mw,
)


SUPPORTED_PROFILE_UNITS = (
    "MW",
    "kW",
    "MWh/day",
    "kWh/day",
    "MWh/interval",
    "kWh/interval",
)


def _positive_scalar(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a finite number greater than zero."
        )

    return float(value)


def _regular_time_step_h(
    timestamps: pd.Series,
    single_row_interval_h: float = 1.0,
) -> float:
    """
    Infer a uniform interval from interval-start timestamps.

    One record requires an explicit or default interval assumption.
    Missing intervals are not filled automatically.
    """

    if len(timestamps) == 0:
        raise ValueError(
            "A time step cannot be inferred from an empty profile."
        )

    if timestamps.isna().any():
        raise ValueError(
            "The profile contains invalid timestamps."
        )

    if timestamps.duplicated().any():
        raise ValueError(
            "The profile contains duplicate timestamps."
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Profile timestamps must be sorted chronologically."
        )

    if len(timestamps) == 1:
        return _positive_scalar(
            single_row_interval_h,
            "Single-record interval [h]",
        )

    differences_h = (
        timestamps.diff().dropna().dt.total_seconds() / 3600.0
    )

    inferred_h = float(differences_h.median())
    _positive_scalar(inferred_h, "Profile interval [h]")

    tolerance_h = max(1e-8, inferred_h * 1e-6)

    if (
        (differences_h - inferred_h).abs() > tolerance_h
    ).any():
        raise ValueError(
            "The renewable profile has irregular intervals or missing "
            "records. Provide a regular profile before simulation; "
            "missing periods are not filled automatically."
        )

    return inferred_h


def infer_profile_meta_from_normalized(
    df: pd.DataFrame,
    source_name: str,
    source_kind: str,
    raw_rows: int | None = None,
    unit_mode: str | None = None,
    expanded_to_hourly: bool | None = None,
    time_step_h: float | None = None,
) -> dict:
    """
    Describe a normalized profile without modifying it.

    Each timestamp marks the start of one complete interval.
    median_step_h always describes the effective profile interval.
    """

    expanded = bool(expanded_to_hourly)

    if df is None or df.empty:
        return {
            "source_name": source_name,
            "source_kind": source_kind,
            "raw_rows": 0 if raw_rows is None else int(raw_rows),
            "normalized_rows": 0,
            "median_step_h": None,
            "time_step_h": None,
            "original_step_h": None,
            "looks_daily": False,
            "effective_is_daily": False,
            "expanded_to_hourly": expanded,
            "unit_mode": unit_mode,
            "duration_h": 0.0,
            "total_energy_mwh": 0.0,
            "start": None,
            "end": None,
            "interval_end": None,
            "warnings": ["The renewable profile is empty."],
        }

    required = {"timestamp", "renewable_power_mw"}

    if not required.issubset(df.columns):
        raise ValueError(
            "A normalized profile requires timestamp "
            "and renewable_power_mw columns."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "The profile contains duplicate column names."
        )

    timestamps = df["timestamp"]

    if not pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
        raise ValueError(
            "Normalized timestamps must have a datetime dtype."
        )

    fallback_h = 1.0 if time_step_h is None else time_step_h
    effective_h = _regular_time_step_h(
        timestamps,
        single_row_interval_h=fallback_h,
    )

    if time_step_h is not None:
        declared_h = _positive_scalar(
            time_step_h,
            "Declared profile interval [h]",
        )

        if not np.isclose(
            effective_h,
            declared_h,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "The declared time step does not match "
                "the profile timestamps."
            )

    power = df["renewable_power_mw"]

    if (
        not pd.api.types.is_numeric_dtype(power.dtype)
        or pd.api.types.is_bool_dtype(power.dtype)
        or pd.api.types.is_complex_dtype(power.dtype)
        or power.isna().any()
    ):
        raise ValueError(
            "Normalized renewable power must contain finite real "
            "numbers without missing values."
        )

    power_values = power.to_numpy(dtype=float)

    if not np.isfinite(power_values).all():
        raise ValueError(
            "Normalized renewable power contains infinite values."
        )

    if (power_values < 0).any():
        raise ValueError(
            "Normalized renewable power cannot be negative."
        )

    if expanded and not np.isclose(
        effective_h,
        1.0,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "A daily-to-hourly expanded profile must have "
            "a one-hour interval."
        )

    effective_is_daily = bool(
        np.isclose(effective_h, HOURS_PER_DAY)
    )
    duration_h = float(len(df) * effective_h)
    energy_mwh = float(power_values.sum() * effective_h)

    if not isfinite(energy_mwh):
        raise ValueError(
            "The profile energy total is not finite."
        )

    warnings: list[str] = []

    if len(df) == 1:
        warnings.append(
            f"The profile has one record. An interval of "
            f"{effective_h:g} h is assumed."
        )

    if expanded:
        warnings.append(
            "Daily values were expanded into flat hourly values. "
            "Daily energy is preserved, but intraday variability "
            "is not reconstructed."
        )
    elif effective_h > 1.0 + 1e-8:
        warnings.append(
            f"The simulation interval is {effective_h:g} h. "
            "Variations and storage operations within each interval "
            "are not resolved."
        )

    if duration_h < DEFAULT_ANNUAL_HOURS - 1e-6:
        warnings.append(
            f"The profile covers {duration_h:,.2f} h, less than "
            "8,760 h. Annualization must be explicitly selected "
            "if an annual equivalent is required."
        )

    return {
        "source_name": source_name,
        "source_kind": source_kind,
        "raw_rows": (
            int(len(df)) if raw_rows is None else int(raw_rows)
        ),
        "normalized_rows": int(len(df)),
        "median_step_h": effective_h,
        "time_step_h": effective_h,
        "original_step_h": (
            HOURS_PER_DAY if expanded else effective_h
        ),
        "looks_daily": expanded or effective_is_daily,
        "effective_is_daily": effective_is_daily,
        "expanded_to_hourly": expanded,
        "unit_mode": unit_mode,
        "duration_h": duration_h,
        "total_energy_mwh": energy_mwh,
        "start": str(timestamps.iloc[0]),
        "end": str(timestamps.iloc[-1]),
        "interval_end": str(
            timestamps.iloc[-1]
            + pd.Timedelta(hours=effective_h)
        ),
        "warnings": warnings,
    }


def normalize_renewable_profile(
    raw_profile_df: pd.DataFrame,
    timestamp_col: str,
    renewable_col: str,
    unit_mode: str,
    expand_daily_to_hourly: bool = True,
    single_row_interval_h: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize renewable data to timestamps and average power in MW.

    Daily expansion repeats each daily average across 24 hours.
    It preserves energy without introducing a synthetic daily shape.

    Without expansion, the original regular interval is retained.
    """

    if (
        not isinstance(raw_profile_df, pd.DataFrame)
        or raw_profile_df.empty
    ):
        raise ValueError(
            "The uploaded renewable profile is empty or invalid."
        )

    if raw_profile_df.columns.duplicated().any():
        raise ValueError(
            "The uploaded profile contains duplicate column names."
        )

    if timestamp_col == renewable_col:
        raise ValueError(
            "Select different timestamp and renewable-value columns."
        )

    for column in (timestamp_col, renewable_col):
        if column not in raw_profile_df.columns:
            raise ValueError(
                f"The selected column {column!r} does not exist."
            )

    if unit_mode not in SUPPORTED_PROFILE_UNITS:
        raise ValueError(
            f"Unsupported renewable profile unit: {unit_mode!r}."
        )

    raw_rows = len(raw_profile_df)
    df = raw_profile_df[
        [timestamp_col, renewable_col]
    ].copy()
    df.columns = ["timestamp", "raw_value"]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )
    df["raw_value"] = pd.to_numeric(
        df["raw_value"],
        errors="coerce",
    )

    invalid_rows = df[
        ["timestamp", "raw_value"]
    ].isna().any(axis=1)

    if invalid_rows.any():
        raise ValueError(
            f"The selected columns contain "
            f"{int(invalid_rows.sum())} invalid or missing records. "
            "Correct these records before loading the profile."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"].dtype
    ):
        raise ValueError(
            "Timestamps must use a consistent datetime representation. "
            "Normalize mixed time zones before uploading."
        )

    raw_values = df["raw_value"].to_numpy(dtype=float)

    if not np.isfinite(raw_values).all():
        raise ValueError(
            "The renewable profile contains infinite values."
        )

    if (raw_values < 0).any():
        raise ValueError(
            "The renewable profile contains negative values. "
            "They are not automatically clipped to zero."
        )

    was_sorted = df["timestamp"].is_monotonic_increasing
    df = df.sort_values("timestamp").reset_index(drop=True)

    daily_energy_units = unit_mode in (
        "MWh/day",
        "kWh/day",
    )

    fallback_h = (
        HOURS_PER_DAY if daily_energy_units else 1.0
    )

    if single_row_interval_h is not None:
        fallback_h = _positive_scalar(
            single_row_interval_h,
            "Single-record interval [h]",
        )

    original_h = _regular_time_step_h(
        df["timestamp"],
        single_row_interval_h=fallback_h,
    )
    looks_daily = bool(
        np.isclose(original_h, HOURS_PER_DAY)
    )

    if daily_energy_units and not looks_daily:
        raise ValueError(
            "MWh/day and kWh/day inputs require one record per "
            "24-hour interval. For other intervals, select "
            "MWh/interval or kWh/interval."
        )

    if unit_mode == "MW":
        power_mw = raw_values.copy()
    elif unit_mode == "kW":
        power_mw = kw_to_mw(raw_values)
    elif unit_mode == "MWh/day":
        power_mw = interval_energy_mwh_to_power_mw(
            raw_values,
            HOURS_PER_DAY,
        )
    elif unit_mode == "kWh/day":
        power_mw = interval_energy_kwh_to_power_mw(
            raw_values,
            HOURS_PER_DAY,
        )
    elif unit_mode == "MWh/interval":
        power_mw = interval_energy_mwh_to_power_mw(
            raw_values,
            original_h,
        )
    else:
        power_mw = interval_energy_kwh_to_power_mw(
            raw_values,
            original_h,
        )

    df["renewable_power_mw"] = power_mw
    original_energy_mwh = float(
        df["renewable_power_mw"].sum() * original_h
    )

    expanded = looks_daily and bool(expand_daily_to_hourly)

    if expanded:
        hours_per_record = int(HOURS_PER_DAY)

        expanded_timestamps = (
            pd.DatetimeIndex(
                df["timestamp"].repeat(hours_per_record)
            )
            + pd.to_timedelta(
                np.tile(
                    np.arange(hours_per_record),
                    len(df),
                ),
                unit="h",
            )
        )

        out = pd.DataFrame(
            {
                "timestamp": expanded_timestamps,
                "renewable_power_mw": np.repeat(
                    df["renewable_power_mw"].to_numpy(dtype=float),
                    hours_per_record,
                ),
            }
        )
        effective_h = 1.0
    else:
        out = df[
            ["timestamp", "renewable_power_mw"]
        ].copy()
        effective_h = original_h

    out = out.reset_index(drop=True)

    meta = infer_profile_meta_from_normalized(
        df=out,
        source_name="uploaded_profile",
        source_kind="uploaded",
        raw_rows=raw_rows,
        unit_mode=unit_mode,
        expanded_to_hourly=expanded,
        time_step_h=effective_h,
    )

    meta["original_step_h"] = original_h
    meta["looks_daily"] = looks_daily

    if not was_sorted:
        meta["warnings"].append(
            "The uploaded records were sorted chronologically."
        )

    if not np.isclose(
        original_energy_mwh,
        meta["total_energy_mwh"],
        rtol=1e-10,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Profile normalization did not preserve total energy."
        )

    return out, meta


def build_default_hourly_profile(
    peak_power_mw: float,
    year_start: str = "2024-01-01 00:00:00",
    hours: int = 8760,
) -> pd.DataFrame:
    """
    Build the original deterministic synthetic hourly profile.

    The original formula and default start date are retained.
    The synthetic profile is not a measured resource assessment.
    """

    peak = _positive_scalar(
        peak_power_mw,
        "Renewable peak power [MW]",
    )

    if (
        isinstance(hours, bool)
        or not isinstance(hours, Integral)
        or hours <= 0
    ):
        raise ValueError(
            "Profile hours must be a positive integer."
        )

    hours = int(hours)

    index = pd.date_range(
        start=year_start,
        periods=hours,
        freq="h",
    )

    hour_of_day = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()

    daily_shape = np.maximum(
        0.0,
        np.sin(
            (hour_of_day - 6.0)
            / HOURS_PER_DAY
            * 2.0
            * np.pi
        ),
    )

    seasonal_shape = (
        0.55
        + 0.35
        * np.sin(
            (day_of_year - 80.0)
            / 365.0
            * 2.0
            * np.pi
        )
    )

    noise = (
        0.08
        * (np.sin(np.arange(hours) * 0.17) + 1.0)
        / 2.0
    )

    renewable_power_mw = peak * np.clip(
        daily_shape * seasonal_shape + noise,
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "timestamp": index,
            "renewable_power_mw": renewable_power_mw,
        }
    )

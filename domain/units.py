from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Shared unit constants.
# ---------------------------------------------------------------------


KG_PER_T = 1000.0

W_PER_KW = 1000.0
KW_PER_MW = 1000.0
W_PER_MW = 1_000_000.0

KWH_PER_MWH = 1000.0
MWH_PER_GWH = 1000.0

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0
DEFAULT_ANNUAL_HOURS = 8760.0

# Rounded molar masses, expressed in kg/kmol.
MOLAR_MASS_H2_KG_PER_KMOL = 2.016
MOLAR_MASS_CO2_KG_PER_KMOL = 44.01
MOLAR_MASS_H2O_KG_PER_KMOL = 18.015
MOLAR_MASS_O2_KG_PER_KMOL = 31.998
MOLAR_MASS_MEOH_KG_PER_KMOL = 32.042


NumericInput: TypeAlias = (
    float
    | int
    | list[float]
    | tuple[float, ...]
    | np.ndarray
    | pd.Series
)

NumericOutput: TypeAlias = float | np.ndarray | pd.Series


# ---------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------


def _finite_array(
    values: NumericInput,
    name: str = "values",
) -> np.ndarray:
    """
    Convert numeric input to a finite floating-point NumPy array.

    Missing values, booleans, complex numbers, and nonnumeric inputs
    are rejected. The original input is not modified.
    """

    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from exc

    if raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must contain real numeric values "
            "without missing entries."
        )

    array = raw.astype(float, copy=True)

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


def _restore_type(
    original: NumericInput,
    converted: np.ndarray,
) -> NumericOutput:
    """Preserve scalar or Series structure where applicable."""

    if isinstance(original, pd.Series):
        return pd.Series(
            converted,
            index=original.index,
            name=original.name,
            dtype=float,
        )

    if converted.ndim == 0:
        return float(converted.item())

    return converted


def _scale(
    values: NumericInput,
    factor: float,
) -> NumericOutput:
    """Apply a conversion factor without changing the original input."""

    array = _finite_array(values)

    with np.errstate(over="ignore", invalid="ignore"):
        converted = array * factor

    if not np.isfinite(converted).all():
        raise ValueError(
            "The unit conversion produced a non-finite result."
        )

    return _restore_type(values, converted)


def _positive_scalar(
    value: float,
    name: str,
) -> float:
    """Validate a finite, strictly positive scalar."""

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


# ---------------------------------------------------------------------
# Power conversions.
# ---------------------------------------------------------------------


def w_to_kw(values: NumericInput) -> NumericOutput:
    return _scale(values, 1.0 / W_PER_KW)


def kw_to_w(values: NumericInput) -> NumericOutput:
    return _scale(values, W_PER_KW)


def w_to_mw(values: NumericInput) -> NumericOutput:
    return _scale(values, 1.0 / W_PER_MW)


def mw_to_w(values: NumericInput) -> NumericOutput:
    return _scale(values, W_PER_MW)


def kw_to_mw(values: NumericInput) -> NumericOutput:
    return _scale(values, 1.0 / KW_PER_MW)


def mw_to_kw(values: NumericInput) -> NumericOutput:
    return _scale(values, KW_PER_MW)


# ---------------------------------------------------------------------
# Mass and mass-flow conversions.
#
# Scaling kg to t also converts kg/h to t/h when the time basis
# remains unchanged.
# ---------------------------------------------------------------------


def kg_to_t(values: NumericInput) -> NumericOutput:
    return _scale(values, 1.0 / KG_PER_T)


def t_to_kg(values: NumericInput) -> NumericOutput:
    return _scale(values, KG_PER_T)


def kg_per_s_to_kg_per_h(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, SECONDS_PER_HOUR)


def kg_per_h_to_kg_per_s(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, 1.0 / SECONDS_PER_HOUR)


def kg_per_s_to_t_per_h(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, SECONDS_PER_HOUR / KG_PER_T)


def kmol_per_h_to_kg_per_h(
    values: NumericInput,
    molar_mass_kg_per_kmol: float,
) -> NumericOutput:
    molar_mass = _positive_scalar(
        molar_mass_kg_per_kmol,
        "Molar mass [kg/kmol]",
    )
    return _scale(values, molar_mass)


def kg_per_h_to_kmol_per_h(
    values: NumericInput,
    molar_mass_kg_per_kmol: float,
) -> NumericOutput:
    molar_mass = _positive_scalar(
        molar_mass_kg_per_kmol,
        "Molar mass [kg/kmol]",
    )
    return _scale(values, 1.0 / molar_mass)


def mass_flow_to_kg_per_h(
    values: NumericInput,
    unit: str,
    molar_mass_kg_per_kmol: float | None = None,
) -> NumericOutput:
    """
    Convert an explicitly declared mass-flow unit to kg/h.

    Molar mass is required for kmol/h.
    Units are never inferred from the magnitude of the values.
    """

    if not isinstance(unit, str):
        raise ValueError("The mass-flow unit must be a string.")

    normalized_unit = unit.strip().lower()

    if normalized_unit == "kg/h":
        return _scale(values, 1.0)

    if normalized_unit == "t/h":
        return t_to_kg(values)

    if normalized_unit == "kg/s":
        return kg_per_s_to_kg_per_h(values)

    if normalized_unit == "kmol/h":
        if molar_mass_kg_per_kmol is None:
            raise ValueError(
                "Molar mass is required to convert kmol/h to kg/h."
            )

        return kmol_per_h_to_kg_per_h(
            values,
            molar_mass_kg_per_kmol,
        )

    raise ValueError(
        f"Unsupported mass-flow unit: {unit!r}. "
        "Supported units: kg/h, t/h, kg/s, kmol/h."
    )


# ---------------------------------------------------------------------
# Energy conversions.
# ---------------------------------------------------------------------


def kwh_to_mwh(values: NumericInput) -> NumericOutput:
    return _scale(values, 1.0 / KWH_PER_MWH)


def mwh_to_kwh(values: NumericInput) -> NumericOutput:
    return _scale(values, KWH_PER_MWH)


def mwh_to_gwh(values: NumericInput) -> NumericOutput:
    return _scale(values, 1.0 / MWH_PER_GWH)


def gwh_to_mwh(values: NumericInput) -> NumericOutput:
    return _scale(values, MWH_PER_GWH)


# ---------------------------------------------------------------------
# Time integration.
#
# Each value represents the average rate over one complete interval.
# Integration uses a rectangular sum, consistent with stepwise dispatch.
# No trapezoidal endpoint correction or annualization is applied.
# ---------------------------------------------------------------------


def integrate_rate(
    rate_values: NumericInput,
    time_step_h: float,
) -> float:
    """
    Integrate a rate expressed per hour over uniform time intervals.

    Examples:
        MW integrated over hours gives MWh.
        kg/h integrated over hours gives kg.
        t/h integrated over hours gives t.

    A scalar represents one interval.
    """

    dt = _positive_scalar(time_step_h, "Time step [h]")
    array = _finite_array(rate_values, "Rate values")

    if array.ndim > 1:
        raise ValueError(
            "Rate values must be a scalar or a one-dimensional series."
        )

    if array.size == 0:
        raise ValueError(
            "An empty series cannot be integrated."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(array, dtype=float) * dt)

    if not isfinite(total):
        raise ValueError(
            "Rate integration produced a non-finite total."
        )

    return total


def power_mw_to_energy_mwh(
    power_mw: NumericInput,
    time_step_h: float,
) -> float:
    return integrate_rate(power_mw, time_step_h)


def power_w_to_energy_kwh(
    power_w: NumericInput,
    time_step_h: float,
) -> float:
    power_kw = w_to_kw(power_w)
    return integrate_rate(power_kw, time_step_h)


def mass_flow_kg_per_h_to_mass_kg(
    mass_flow_kg_per_h: NumericInput,
    time_step_h: float,
) -> float:
    return integrate_rate(mass_flow_kg_per_h, time_step_h)


def mass_flow_kg_per_h_to_mass_t(
    mass_flow_kg_per_h: NumericInput,
    time_step_h: float,
) -> float:
    return integrate_rate(
        mass_flow_kg_per_h,
        time_step_h,
    ) / KG_PER_T


# ---------------------------------------------------------------------
# Interval energy to average power.
# ---------------------------------------------------------------------


def interval_energy_mwh_to_power_mw(
    energy_mwh: NumericInput,
    interval_h: float,
) -> NumericOutput:
    interval = _positive_scalar(
        interval_h,
        "Energy interval [h]",
    )
    return _scale(energy_mwh, 1.0 / interval)


def interval_energy_kwh_to_power_mw(
    energy_kwh: NumericInput,
    interval_h: float,
) -> NumericOutput:
    interval = _positive_scalar(
        interval_h,
        "Energy interval [h]",
    )
    return _scale(
        energy_kwh,
        1.0 / KWH_PER_MWH / interval,
    )


# ---------------------------------------------------------------------
# Price-unit conversions.
#
# These functions change the quantity basis, not the currency.
# For example, EUR/kWh becomes EUR/MWh.
# They never convert USD to EUR.
# ---------------------------------------------------------------------


def price_per_kwh_to_per_mwh(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, KWH_PER_MWH)


def price_per_mwh_to_per_kwh(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, 1.0 / KWH_PER_MWH)


def price_per_t_to_per_kg(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, 1.0 / KG_PER_T)


def price_per_kg_to_per_t(
    values: NumericInput,
) -> NumericOutput:
    return _scale(values, KG_PER_T)

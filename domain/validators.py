from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

import pandas as pd

from domain.data_models import CaseInputs, TEAInputs


class InputValidationError(ValueError):
    """Input errors that prevent an interpretable calculation."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        message = "The calculation cannot proceed:\n" + "\n".join(
            f"- {error}" for error in self.errors
        )
        super().__init__(message)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise InputValidationError(self.errors)

    def extend(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def _number(
    report: ValidationReport,
    name: str,
    value: Any,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
    integer: bool = False,
) -> bool:
    """Validate a scalar without converting or replacing its value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
    ):
        report.errors.append(
            f"{name}: must be a finite number; received {value!r}."
        )
        return False

    numeric = float(value)
    valid = True

    if integer and not numeric.is_integer():
        report.errors.append(f"{name}: must be an integer.")
        valid = False

    if strictly_positive and numeric <= 0:
        report.errors.append(f"{name}: must be greater than zero.")
        valid = False
    elif minimum is not None and numeric < minimum:
        report.errors.append(f"{name}: must be at least {minimum}.")
        valid = False

    if maximum is not None and numeric > maximum:
        report.errors.append(f"{name}: must not exceed {maximum}.")
        valid = False

    return valid


def _nonnegative(
    report: ValidationReport,
    name: str,
    value: Any,
) -> bool:
    return _number(report, name, value, minimum=0.0)


def _fraction(
    report: ValidationReport,
    name: str,
    value: Any,
) -> bool:
    return _number(report, name, value, minimum=0.0, maximum=1.0)


def _choice(
    report: ValidationReport,
    name: str,
    value: Any,
    allowed: tuple[str, ...],
) -> None:
    if value not in allowed:
        report.errors.append(
            f"{name}: unrecognized value {value!r}. "
            f"Available options: {', '.join(allowed)}."
        )


def _validate_profile(
    case: CaseInputs,
    report: ValidationReport,
) -> None:
    profile = case.renewable_profile

    if not isinstance(profile, pd.DataFrame):
        report.errors.append(
            "The renewable profile must be a pandas.DataFrame."
        )
        return

    if profile.empty:
        report.errors.append("The renewable profile is empty.")
        return

    required = {"timestamp", "renewable_power_mw"}
    missing = required.difference(profile.columns)

    if missing:
        report.errors.append(
            "The renewable profile is missing required columns: "
            + ", ".join(sorted(missing))
            + "."
        )
        return

    if profile.columns.duplicated().any():
        report.errors.append(
            "The renewable profile contains duplicate column names."
        )
        return

    power = profile["renewable_power_mw"]

    if (
        not pd.api.types.is_numeric_dtype(power.dtype)
        or pd.api.types.is_bool_dtype(power.dtype)
    ):
        report.errors.append(
            "renewable_power_mw must contain numeric data."
        )
    else:
        finite_power = power.map(
            lambda value: (
                pd.notna(value) and isfinite(float(value))
            )
        )

        if not finite_power.all():
            report.errors.append(
                "The renewable profile contains missing, NaN, "
                "or infinite power values."
            )
        elif (power < 0).any():
            report.errors.append(
                "The renewable profile contains negative power values. "
                "Check the unit conversion or source data."
            )
        elif power.eq(0).all():
            report.warnings.append(
                "The entire renewable profile has zero power. "
                "Execution is allowed, but production may be zero."
            )

    timestamps = profile["timestamp"]

    if not pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
        report.errors.append(
            "timestamp must be converted to datetime "
            "before running the simulation."
        )
        return

    if timestamps.isna().any():
        report.errors.append(
            "The renewable profile contains invalid timestamps."
        )
        return

    if timestamps.duplicated().any():
        report.errors.append(
            "The renewable profile contains duplicate timestamps."
        )
        return

    if not timestamps.is_monotonic_increasing:
        report.errors.append(
            "The renewable profile must be sorted chronologically."
        )
        return

    dt = case.time_step_h

    if (
        isinstance(dt, Real)
        and not isinstance(dt, bool)
        and isfinite(float(dt))
        and dt > 0
        and len(timestamps) > 1
    ):
        intervals = (
            timestamps.diff().dropna().dt.total_seconds() / 3600.0
        )
        tolerance = max(1e-8, float(dt) * 1e-6)

        if ((intervals - float(dt)).abs() > tolerance).any():
            report.errors.append(
                "The profile intervals do not match "
                f"time_step_h={dt}. Normalize the profile or adjust "
                "the time step before execution."
            )

    if (
        isinstance(dt, Real)
        and not isinstance(dt, bool)
        and isfinite(float(dt))
        and dt > 0
    ):
        duration_h = len(profile) * float(dt)

        if duration_h < 8760.0 - 1e-6:
            report.warnings.append(
                f"The profile represents {duration_h:,.2f} hours. "
                "It must not automatically be interpreted as a full "
                "year. Review the TEA annualization assumption."
            )


def inspect_tea_inputs(tea: TEAInputs) -> ValidationReport:
    """Check the new economic configuration expressed in EUR."""

    report = ValidationReport()

    _number(
        report,
        "Project start year",
        tea.project_start_year,
        minimum=1,
        integer=True,
    )
    _number(
        report,
        "Project lifetime",
        tea.project_years,
        strictly_positive=True,
        integer=True,
    )
    _number(
        report,
        "Price base year",
        tea.price_base_year,
        minimum=1,
        integer=True,
    )
    _nonnegative(report, "Discount rate", tea.discount_rate)

    if tea.price_base_year != tea.project_start_year:
        report.warnings.append(
            "The price base year differs from the investment year. "
            "No implicit monetary adjustment will be applied."
        )

    _number(
        report,
        "Reference annual hours",
        tea.annual_hours,
        strictly_positive=True,
    )

    prices = tea.prices

    for label, value in [
        ("Electricity [EUR/kWh]", prices.electricity_eur_per_kwh),
        ("Cooling water [EUR/t]", prices.cooling_water_eur_per_t),
        ("Captured CO2 [EUR/t]", prices.captured_co2_eur_per_t),
        ("Electrolysis water [EUR/t]", prices.electrolysis_water_eur_per_t),
        ("Methanol [EUR/t]", prices.methanol_eur_per_t),
    ]:
        _nonnegative(report, label, value)

    capex = tea.capex

    for label, value in [
        ("Reference plant CAPEX [EUR]", capex.reference_plant_cost_eur),
        (
            "Reference MeOH capacity [t/h]",
            capex.reference_meoh_capacity_t_per_h,
        ),
        (
            "Nominal design MeOH capacity [t/h]",
            capex.design_meoh_capacity_t_per_h,
        ),
        ("Scaling exponent", capex.scaling_exponent),
    ]:
        _number(report, label, value, strictly_positive=True)

    _nonnegative(
        report,
        "Electrolyzer cost [EUR/kW]",
        capex.electrolyzer_eur_per_kw,
    )
    _nonnegative(
        report,
        "H2 storage cost [EUR/kg]",
        capex.h2_storage_eur_per_kg,
    )
    _nonnegative(
        report,
        "CO2 storage capacity [kg]",
        capex.co2_storage_capacity_kg,
    )

    if capex.co2_storage_enabled:
        _nonnegative(
            report,
            "CO2 storage cost [EUR/kg]",
            capex.co2_storage_eur_per_kg,
        )
        if capex.co2_storage_capacity_kg == 0:
            report.warnings.append(
                "CO2 storage is enabled with zero capacity."
            )
    elif capex.co2_storage_capacity_kg != 0:
        report.warnings.append(
            "CO2 storage capacity will not be included in CAPEX "
            "because CO2 storage is disabled."
        )

    opex = tea.opex

    for label, value in [
        ("Operating labor [EUR/year]", opex.operating_labor_eur_per_year),
        (
            "Operating supervision [EUR/year]",
            opex.operating_supervision_eur_per_year,
        ),
    ]:
        _nonnegative(report, label, value)

    for name in [
        "maintenance_fraction_of_capex",
        "supplies_fraction_of_maintenance",
        "royalties_fraction_of_variable_cost",
        "annual_taxes_fraction_of_capex",
        "insurance_fraction_of_capex",
        "overhead_fraction",
        "administration_fraction_of_labor",
        "working_capital_interest_fraction_of_capex",
    ]:
        _fraction(report, name, getattr(opex, name))

    utilities = tea.utilities

    _number(
        report,
        "Water specific heat capacity [J/(kg K)]",
        utilities.cooling_water_cp_j_per_kg_k,
        strictly_positive=True,
    )
    _number(
        report,
        "Cooling-water temperature rise [K]",
        utilities.cooling_water_delta_t_k,
        strictly_positive=True,
    )

    _choice(
        report,
        "Cooling-water cost basis",
        utilities.cooling_water_cost_basis,
        ("circulating", "makeup"),
    )

    if utilities.cooling_water_cost_basis == "makeup":
        _fraction(
            report,
            "Cooling-water makeup fraction",
            utilities.cooling_water_makeup_fraction,
        )

    _choice(
        report,
        "Cooling-duty sign convention",
        utilities.cooling_duty_sign,
        ("negative_is_cooling", "positive_is_cooling"),
    )
    _choice(
        report,
        "Heater-duty sign convention",
        utilities.heater_duty_sign,
        ("positive_is_heating", "negative_is_heating"),
    )

    _number(
        report,
        "Electric heater efficiency",
        utilities.electric_heater_efficiency,
        strictly_positive=True,
        maximum=1.0,
    )

    if (
        not isinstance(utilities.cooling_model_names, list)
        or not utilities.cooling_model_names
        or any(
            not isinstance(name, str) or not name.strip()
            for name in utilities.cooling_model_names
        )
    ):
        report.errors.append(
            "Cooling model names must be specified."
        )
    elif len(set(utilities.cooling_model_names)) != len(
        utilities.cooling_model_names
    ):
        report.errors.append(
            "Cooling model names are duplicated; "
            "their consumption would be counted more than once."
        )

    if (
        not isinstance(utilities.electric_heater_model_name, str)
        or not utilities.electric_heater_model_name.strip()
    ):
        report.errors.append(
            "The electric heater model name must be specified."
        )

    _number(
        report,
        "Electrolysis water [kg/kg H2]",
        utilities.electrolysis_water_kg_per_kg_h2,
        strictly_positive=True,
    )
    _nonnegative(
        report,
        "Oxygen production [kg/kg H2]",
        utilities.oxygen_kg_per_kg_h2,
    )

    co2 = tea.co2_feed

    _choice(
        report,
        "CO2 feed source",
        co2.source,
        ("stoichiometric", "surrogate"),
    )
    _choice(
        report,
        "CO2 model output unit",
        co2.output_unit,
        ("kg/h", "t/h", "kmol/h"),
    )

    if co2.source == "surrogate":
        if (
            not isinstance(co2.model_name, str)
            or not co2.model_name.strip()
        ):
            report.errors.append(
                "Surrogate-based CO2 feed was selected, "
                "but the model name is missing."
            )

    if (
        co2.source == "stoichiometric"
        or co2.allow_stoichiometric_fallback
    ):
        _number(
            report,
            "CO2/MeOH ratio [kg/kg]",
            co2.kg_co2_per_kg_meoh,
            strictly_positive=True,
        )

    if co2.source == "stoichiometric":
        report.warnings.append(
            "CO2 consumption will be estimated using a stoichiometric "
            "balance, not the new trained surrogate."
        )

    coproducts = tea.coproducts

    _fraction(
        report,
        "Saleable oxygen fraction",
        coproducts.oxygen_saleable_fraction,
    )
    _nonnegative(
        report,
        "Oxygen conditioning cost [EUR/t]",
        coproducts.oxygen_conditioning_eur_per_t,
    )

    if coproducts.oxygen_sales_enabled:
        _nonnegative(
            report,
            "Oxygen price [EUR/t]",
            prices.oxygen_eur_per_t,
        )
        if coproducts.oxygen_saleable_fraction == 0:
            report.warnings.append(
                "Oxygen sales are enabled, "
                "but the saleable fraction is zero."
            )

    return report


def inspect_case_inputs(case: CaseInputs) -> ValidationReport:
    """Return errors and warnings without running the simulation."""

    report = ValidationReport()

    _number(
        report,
        "Time step [h]",
        case.time_step_h,
        strictly_positive=True,
    )

    electrolyzer = case.electrolyzer

    _number(
        report,
        "Electrolyzer nominal power [MW]",
        electrolyzer.nominal_power_mw,
        strictly_positive=True,
    )
    _number(
        report,
        "Module power [MW]",
        electrolyzer.module_size_mw,
        strictly_positive=True,
    )
    _fraction(
        report,
        "Electrolyzer minimum load fraction",
        electrolyzer.min_load_fraction,
    )
    _number(
        report,
        "Specific energy consumption [kWh/kg H2]",
        electrolyzer.specific_energy_kwh_per_kg_h2,
        strictly_positive=True,
    )

    storage = case.storage

    _nonnegative(
        report,
        "H2 storage capacity [kg]",
        storage.usable_capacity_kg_h2,
    )
    _fraction(
        report,
        "Initial H2 inventory fraction",
        storage.initial_soc_fraction,
    )

    for label, value in [
        ("Maximum H2 charge rate [kg/h]", storage.max_charge_kg_per_h),
        ("Maximum H2 discharge rate [kg/h]", storage.max_discharge_kg_per_h),
    ]:
        if value is not None:
            _nonnegative(report, label, value)

    if storage.enabled and storage.usable_capacity_kg_h2 == 0:
        report.warnings.append(
            "H2 storage is enabled with zero capacity."
        )

    ptmeoh = case.ptmeoh

    target_ok = _nonnegative(
        report,
        "Target H2 feed [kg/h]",
        ptmeoh.target_h2_feed_kg_per_h,
    )
    maximum_ok = _number(
        report,
        "Maximum process H2 feed [kg/h]",
        ptmeoh.max_h2_feed_kg_per_h,
        strictly_positive=True,
    )

    if (
        target_ok
        and maximum_ok
        and ptmeoh.target_h2_feed_kg_per_h
        > ptmeoh.max_h2_feed_kg_per_h
    ):
        report.errors.append(
            "The H2 target exceeds the maximum process capacity. "
            "Adjust the target or the physical capacity."
        )

    _choice(
        report,
        "Operating mode",
        ptmeoh.operating_mode,
        ("quasi_base_load", "flexible"),
    )

    if (
        not isinstance(ptmeoh.surrogate_library, str)
        or not ptmeoh.surrogate_library.strip()
    ):
        report.errors.append(
            "A surrogate model library must be selected."
        )

    _number(
        report,
        "Fallback MeOH yield [t/t H2]",
        ptmeoh.methanol_yield_t_meoh_per_t_h2,
        strictly_positive=True,
    )

    for name in [
        "unmet_h2_warning_threshold",
        "curtailment_warning_threshold",
        "utilization_warning_threshold",
    ]:
        _fraction(report, name, getattr(ptmeoh, name))

    _validate_profile(case, report)

    if case.tea is not None:
        report.extend(inspect_tea_inputs(case.tea))
    else:
        economic = case.economic

        _number(
            report,
            "Project lifetime",
            economic.project_years,
            strictly_positive=True,
            integer=True,
        )
        _nonnegative(
            report,
            "Discount rate",
            economic.discount_rate,
        )

        for name in [
            "electricity_price_usd_per_mwh",
            "co2_price_usd_per_t",
            "methanol_price_usd_per_t",
            "capex_multiplier",
            "opex_multiplier",
        ]:
            _nonnegative(report, name, getattr(economic, name))

        _nonnegative(
            report,
            "Legacy electrolyzer CAPEX [USD/kW]",
            electrolyzer.capex_usd_per_kw,
        )
        _fraction(
            report,
            "Legacy electrolyzer fixed OPEX fraction",
            electrolyzer.fixed_opex_fraction,
        )
        _nonnegative(
            report,
            "Legacy H2 storage CAPEX [USD/kg]",
            storage.capex_usd_per_kg_h2,
        )

        report.warnings.append(
            "The case has no TEA configuration in EUR. "
            "It uses the previous economic methodology in USD."
        )

    return report


def validate_case_inputs(case: CaseInputs) -> list[str]:
    """Engine API: raise on blocking errors and return warnings."""

    report = inspect_case_inputs(case)
    report.raise_if_invalid()
    return report.warnings


def validate_optimization_inputs(case: CaseInputs) -> list[str]:
    """Check the search grid only when optimization is requested."""

    report = ValidationReport()
    optimization = case.optimization

    grids = [
        (
            "Power grid [MW]",
            optimization.electrolyzer_power_grid_mw,
            True,
            False,
        ),
        (
            "H2 storage grid [kg]",
            optimization.storage_grid_kg_h2,
            False,
            False,
        ),
        (
            "H2 target grid [kg/h]",
            optimization.target_h2_grid_kg_per_h,
            False,
            False,
        ),
        (
            "Module count grid",
            optimization.module_count_grid,
            True,
            True,
        ),
    ]

    for label, values, positive, integer in grids:
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            report.errors.append(f"{label}: must contain values.")
            continue

        for index, value in enumerate(values):
            _number(
                report,
                f"{label}, item {index + 1}",
                value,
                minimum=0.0,
                strictly_positive=positive,
                integer=integer,
            )

    for name in [
        "min_ptmeoh_utilization",
        "max_unmet_h2_fraction",
        "max_curtailment_fraction",
        "max_surrogate_out_of_domain_fraction",
    ]:
        _fraction(report, name, getattr(optimization, name))

    _nonnegative(
        report,
        "Allowed electricity deficit [MWh]",
        optimization.max_power_deficit_mwh,
    )
    _choice(
        report,
        "LCOMeOH basis",
        optimization.lcomeoh_basis,
        ("gross", "net"),
    )

    report.raise_if_invalid()
    return report.warnings


def model_value_warnings(
    model_name: str,
    variable_name: str,
    value: Any,
    lower: float | None = None,
    upper: float | None = None,
    unit: str = "",
    value_kind: str = "input",
) -> list[str]:
    """
    Check a model value without blocking or changing it.

    NaN and infinite values are reported but are not made valid.
    The model executor must handle these failures separately.

    Use value_kind="input" or value_kind="output".
    """

    warnings: list[str] = []
    label = f"{model_name} | {value_kind} | {variable_name}"
    suffix = f" {unit}" if unit else ""

    def finite_scalar(candidate: Any) -> bool:
        return (
            isinstance(candidate, Real)
            and not isinstance(candidate, bool)
            and isfinite(float(candidate))
        )

    if not finite_scalar(value):
        return [
            f"{label}: unusable value {value!r}. "
            "It must not be silently replaced with zero."
        ]

    if lower is not None and not finite_scalar(lower):
        warnings.append(
            f"{label}: invalid lower bound; "
            "this bound will not be checked."
        )
        lower = None

    if upper is not None and not finite_scalar(upper):
        warnings.append(
            f"{label}: invalid upper bound; "
            "this bound will not be checked."
        )
        upper = None

    if lower is not None and upper is not None and lower > upper:
        warnings.append(
            f"{label}: bounds are reversed. "
            "The range cannot be verified; calculation continues."
        )
        return warnings

    if lower is None and upper is None:
        warnings.append(
            f"{label}: domain or range is undefined. "
            "Calculation continues without range validation."
        )
        return warnings

    if lower is None or upper is None:
        warnings.append(
            f"{label}: the range is incomplete. "
            "Only the available bound will be checked."
        )

    outside = (
        (lower is not None and value < lower)
        or (upper is not None and value > upper)
    )

    if outside:
        lower_text = (
            f"{lower:g}" if lower is not None else "undefined"
        )
        upper_text = (
            f"{upper:g}" if upper is not None else "undefined"
        )
        detail = (
            "Prediction by extrapolation is allowed."
            if value_kind == "input"
            else "The prediction outside the declared range is retained."
        )

        warnings.append(
            f"{label}: {value:g}{suffix}, outside the range "
            f"[{lower_text}, {upper_text}]{suffix}. "
            f"{detail} Review the technical validity of the result."
        )

    return warnings

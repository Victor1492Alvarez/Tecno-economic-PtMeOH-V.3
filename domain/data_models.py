from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd


DATA_SCHEMA_VERSION = "2.0.0"


# ---------------------------------------------------------------------
# Original data structures.
#
# USD fields remain available for backward compatibility.
# They do not represent EUR and must not be interpreted as EUR.
# The new economic analysis will use CaseInputs.tea.
# ---------------------------------------------------------------------


@dataclass
class ScenarioEconomicInputs:
    electricity_price_usd_per_mwh: float
    co2_price_usd_per_t: float
    methanol_price_usd_per_t: float
    discount_rate: float
    project_years: int
    capex_multiplier: float = 1.0
    opex_multiplier: float = 1.0


@dataclass
class ElectrolyzerInputs:
    nominal_power_mw: float
    module_size_mw: float
    min_load_fraction: float
    specific_energy_kwh_per_kg_h2: float
    capex_usd_per_kw: float
    fixed_opex_fraction: float


@dataclass
class StorageInputs:
    enabled: bool
    usable_capacity_kg_h2: float
    initial_soc_fraction: float
    max_charge_kg_per_h: Optional[float] = None
    max_discharge_kg_per_h: Optional[float] = None
    capex_usd_per_kg_h2: float = 20.0


@dataclass
class PtMeOHInputs:
    operating_mode: str
    surrogate_library: str
    target_h2_feed_kg_per_h: float
    max_h2_feed_kg_per_h: float
    methanol_yield_t_meoh_per_t_h2: float = 7.95
    unmet_h2_warning_threshold: float = 0.10
    curtailment_warning_threshold: float = 0.35
    utilization_warning_threshold: float = 0.70


@dataclass
class OptimizationInputs:
    electrolyzer_power_grid_mw: List[float]
    storage_grid_kg_h2: List[float]
    target_h2_grid_kg_per_h: List[float]
    module_count_grid: List[int]
    objective: str = "min_lcomeoh"
    min_ptmeoh_utilization: float = 0.70
    max_unmet_h2_fraction: float = 0.10
    max_curtailment_fraction: float = 0.35

    # Additional limits for the updated optimizer.
    max_surrogate_out_of_domain_fraction: float = 0.0
    max_power_deficit_mwh: float = 1e-6

    # False keeps demonstration runs available.
    # Results must be labeled as demonstrative when mock predictors
    # have been used.
    require_runtime_models: bool = False

    # The new TEA can optimize gross cost or cost net of coproduct revenue.
    lcomeoh_basis: Literal["gross", "net"] = "gross"


# ---------------------------------------------------------------------
# New techno-economic analysis settings in EUR.
#
# Monetary prices are expressed in base-year EUR.
# Rates and fractions are expressed between 0 and 1.
# No implicit currency conversions are performed.
# ---------------------------------------------------------------------


@dataclass
class TEAPriceInputs:
    electricity_eur_per_kwh: Optional[float] = None
    cooling_water_eur_per_t: Optional[float] = None
    captured_co2_eur_per_t: Optional[float] = None
    electrolysis_water_eur_per_t: Optional[float] = None

    methanol_eur_per_t: Optional[float] = None

    # Required only when oxygen sales are enabled.
    oxygen_eur_per_t: Optional[float] = None


@dataclass
class TEACapexInputs:
    # The reference cost includes the entire plant.
    # Equipment costs must not be added again to the scaled total.
    reference_plant_cost_eur: Optional[float] = None
    reference_meoh_capacity_t_per_h: Optional[float] = None

    # Nominal capacity, not actual annual production.
    # Update this capacity when the process design changes.
    design_meoh_capacity_t_per_h: Optional[float] = None

    scaling_exponent: float = 0.70

    electrolyzer_eur_per_kw: Optional[float] = None
    h2_storage_eur_per_kg: Optional[float] = None

    # Economic representation only; this does not imply a dynamic tank.
    co2_storage_enabled: bool = False
    co2_storage_capacity_kg: float = 0.0
    co2_storage_eur_per_kg: Optional[float] = None


@dataclass
class TEAOpexInputs:
    operating_labor_eur_per_year: float = 819_958.02
    operating_supervision_eur_per_year: float = 122_993.70

    maintenance_fraction_of_capex: float = 0.10
    supplies_fraction_of_maintenance: float = 0.15
    royalties_fraction_of_variable_cost: float = 0.06

    # Annual charge on CAPEX under the requested methodology.
    # This is not a corporate income tax.
    annual_taxes_fraction_of_capex: float = 0.04
    insurance_fraction_of_capex: float = 0.01

    # Basis: operating labor + supervision + maintenance.
    overhead_fraction: float = 0.70

    administration_fraction_of_labor: float = 0.25

    # Assumed annual charge; does not create another year-0 investment.
    working_capital_interest_fraction_of_capex: float = 0.15


@dataclass
class UtilityInputs:
    # Explicit assumptions for converting cooling duty into water demand.
    cooling_water_cp_j_per_kg_k: float = 4180.0

    # Must be defined before cooling-water cost can be calculated.
    cooling_water_delta_t_k: Optional[float] = None

    # circulating: apply the price to total circulating water.
    # makeup: apply the price to the makeup-water fraction.
    cooling_water_cost_basis: Literal["circulating", "makeup"] = (
        "circulating"
    )
    cooling_water_makeup_fraction: Optional[float] = None

    # Verify this against the sign convention of the actual models.
    cooling_duty_sign: Literal[
        "negative_is_cooling",
        "positive_is_cooling",
    ] = "negative_is_cooling"

    cooling_model_names: List[str] = field(
        default_factory=lambda: [
            "Model_Duty_HX1_Watt",
            "Model_Duty_HX4_Watt",
            "Model_Duty_RTR_Watt",
        ]
    )

    electric_heater_model_name: str = "Model_Duty_HX3_Watt"
    electric_heater_efficiency: float = 1.0
    heater_duty_sign: Literal[
        "positive_is_heating",
        "negative_is_heating",
    ] = "positive_is_heating"

    # Simplified electrolysis balances, editable and documented.
    electrolysis_water_kg_per_kg_h2: float = 9.0
    oxygen_kg_per_kg_h2: float = 8.0


@dataclass
class CO2FeedInputs:
    # "surrogate" requires the actual model name and output units.
    # "stoichiometric" provides an explicit fallback estimate.
    source: Literal["stoichiometric", "surrogate"] = "stoichiometric"

    # Pending delivery of the CO2 feed model ZIP.
    model_name: Optional[str] = None

    # Model output unit, not the H2 input unit.
    output_unit: Literal["kg/h", "t/h", "kmol/h"] = "kg/h"

    # Used only for stoichiometric estimation.
    kg_co2_per_kg_meoh: float = 1.375

    # False prevents silent replacement of a requested surrogate.
    allow_stoichiometric_fallback: bool = False


@dataclass
class CoproductInputs:
    oxygen_sales_enabled: bool = False

    # Fraction of produced oxygen assumed to be saleable.
    oxygen_saleable_fraction: float = 0.0

    # Additional cost applied to the saleable quantity.
    oxygen_conditioning_eur_per_t: float = 0.0


@dataclass
class TEAInputs:
    # Year 0 is the investment year.
    # The first operating year is project_start_year + 1.
    project_start_year: int = 2027
    project_years: int = 25
    discount_rate: float = 0.08

    # Constant base-year EUR; no implicit future price escalation.
    price_base_year: int = 2027

    prices: TEAPriceInputs = field(default_factory=TEAPriceInputs)
    capex: TEACapexInputs = field(default_factory=TEACapexInputs)
    opex: TEAOpexInputs = field(default_factory=TEAOpexInputs)
    utilities: UtilityInputs = field(default_factory=UtilityInputs)
    co2_feed: CO2FeedInputs = field(default_factory=CO2FeedInputs)
    coproducts: CoproductInputs = field(default_factory=CoproductInputs)

    # Short profiles are not extrapolated to a full year by default.
    # When enabled, annual_hours defines the equivalent annual horizon.
    annualize_profile: bool = False
    annual_hours: float = 8760.0


# ---------------------------------------------------------------------
# Simulation case.
# ---------------------------------------------------------------------


@dataclass
class CaseInputs:
    scenario_name: str
    economic: ScenarioEconomicInputs
    electrolyzer: ElectrolyzerInputs
    storage: StorageInputs
    ptmeoh: PtMeOHInputs
    optimization: OptimizationInputs
    renewable_profile: pd.DataFrame
    time_step_h: float = 1.0
    case_name: str = "base_case"

    # None preserves cases built using the previous API.
    # The updated engine must identify the economic methodology used.
    tea: Optional[TEAInputs] = None

    # Identifiers and assumptions used for traceability.
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Detailed economic results.
# ---------------------------------------------------------------------


@dataclass
class TEAArtifacts:
    currency: str = "EUR"
    price_base_year: int = 2027
    methodology: str = "discounted_cash_flow_constant_eur"

    # Numeric metrics and status information are stored separately.
    kpis: Dict[str, float] = field(default_factory=dict)
    status: Dict[str, Any] = field(default_factory=dict)

    capex_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    utility_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    opex_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    revenue_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    margins: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)

    # None means the investment is not recovered within the horizon.
    simple_payback_years: Optional[float] = None
    discounted_payback_years: Optional[float] = None

    assumptions: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Simulation and optimization results.
# ---------------------------------------------------------------------


@dataclass
class SimulationArtifacts:
    time_series: pd.DataFrame
    kpis: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    surrogate_info: Dict[str, Any] = field(default_factory=dict)
    model_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    tea: Optional[TEAArtifacts] = None

    # Serializable snapshot of inputs, assumptions, and model versions.
    input_snapshot: Dict[str, Any] = field(default_factory=dict)
    run_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridSearchArtifacts:
    results: pd.DataFrame
    feasible_results: pd.DataFrame
    best_row: pd.Series

    # "not_evaluated" avoids assigning a status to legacy code
    # that has not yet been updated.
    status: Literal[
        "not_evaluated",
        "feasible",
        "no_feasible",
        "empty",
        "failed",
    ] = "not_evaluated"

    # Informational alternative; never equivalent to a feasible optimum.
    best_infeasible_row: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="object")
    )

    warnings: List[str] = field(default_factory=list)

    # Full simulation and economic details for the selected design only.
    best_simulation: Optional[SimulationArtifacts] = None

    run_metadata: Dict[str, Any] = field(default_factory=dict)

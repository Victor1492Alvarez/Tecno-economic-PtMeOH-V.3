from __future__ import annotations

from copy import deepcopy
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from domain.data_models import (
    CaseInputs,
    ElectrolyzerInputs,
    GridSearchArtifacts,
    OptimizationInputs,
    PtMeOHInputs,
    ScenarioEconomicInputs,
    SimulationArtifacts,
    StorageInputs,
    TEAInputs,
)
from domain.optimizer_grid import GridOptimizer
from domain.renewable_profile import (
    build_default_hourly_profile,
    infer_profile_meta_from_normalized,
    normalize_renewable_profile,
)
from domain.sensitivity_analysis import SensitivityAnalyzer
from domain.simulation_engine import SimulationEngine
from domain.validators import validate_case_inputs


ProgressCallback = Callable[[str, int, int], None] | None


class CaseRunner:
    """Build cases and coordinate the application workflows."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

        self.engine = SimulationEngine(self.project_root)
        self.optimizer = GridOptimizer(self.engine)
        self.sensitivity = SensitivityAnalyzer(self.engine)
        self.logger = self.engine.logger

        config_path = (
            self.project_root / "config" / "scenarios.json"
        )

        try:
            payload = json.loads(
                config_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(
                f"The scenario configuration could not be loaded "
                f"from {config_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict) or not payload:
            raise ValueError(
                "config/scenarios.json must contain "
                "a nonempty JSON object."
            )

        self.scenario_config = payload

    @staticmethod
    def _number(
        value: float,
        name: str,
        *,
        strictly_positive: bool = False,
        integer: bool = False,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

        numeric = float(value)

        if numeric < 0 or (
            strictly_positive and numeric == 0
        ):
            requirement = (
                "greater than zero"
                if strictly_positive
                else "nonnegative"
            )
            raise ValueError(
                f"{name} must be {requirement}."
            )

        if integer and not numeric.is_integer():
            raise ValueError(
                f"{name} must be an integer."
            )

        return numeric

    @staticmethod
    def _default_optimization(
        power_mw: float,
        module_count: int,
        storage_enabled: bool,
        storage_kg: float,
        target_h2: float,
        maximum_h2: float,
    ) -> OptimizationInputs:
        """
        Preserve the original search ranges.

        Avoid rounding small positive power values to zero.
        Module size is calculated later from power / module count.
        """

        if not storage_enabled:
            storage_grid = [0.0]
        elif storage_kg == 0:
            storage_grid = [0.0, 500.0]
        else:
            storage_grid = sorted(
                {
                    0.5 * storage_kg,
                    storage_kg,
                    1.5 * storage_kg,
                }
            )

        return OptimizationInputs(
            electrolyzer_power_grid_mw=sorted(
                {
                    0.8 * power_mw,
                    power_mw,
                    1.2 * power_mw,
                }
            ),
            storage_grid_kg_h2=storage_grid,
            target_h2_grid_kg_per_h=sorted(
                {
                    0.85 * target_h2,
                    target_h2,
                    min(1.15 * target_h2, maximum_h2),
                }
            ),
            module_count_grid=sorted(
                {
                    max(module_count - 1, 1),
                    module_count,
                    module_count + 1,
                }
            ),
        )

    def build_case(
        self,
        scenario_name: str,
        electrolyzer_power_mw: float,
        module_count: int,
        storage_enabled: bool,
        storage_kg_h2: float,
        operating_mode: str,
        surrogate_library: str,
        target_h2_kg_per_h: float,
        max_h2_feed_kg_per_h: float,
        renewable_peak_power_mw: float,
        renewable_profile_df: pd.DataFrame | None = None,
        electricity_price_usd_per_kwh: float | None = None,
        *,
        tea_inputs: TEAInputs | None = None,
        optimization_inputs: OptimizationInputs | None = None,
        profile_meta: dict[str, Any] | None = None,
        time_step_h: float | None = None,
        expand_daily_to_hourly: bool = False,
        initial_soc_fraction: float = 0.30,
        max_charge_kg_per_h: float | None = None,
        max_discharge_kg_per_h: float | None = None,
        min_load_fraction: float = 0.15,
        specific_energy_kwh_per_kg_h2: float = 52.0,
        case_name: str = "base_case",
    ) -> CaseInputs:
        """
        Build an independent case configuration.

        renewable_profile_df must already express power in MW.
        Raw energy-unit uploads should first use
        normalize_renewable_profile() in the application layer.

        tea_inputs=None selects the previous USD methodology.
        Passing TEAInputs selects the new EUR methodology.
        """

        if scenario_name not in self.scenario_config:
            raise ValueError(
                f"Unknown economic scenario {scenario_name!r}. "
                "Available scenarios: "
                + ", ".join(sorted(self.scenario_config))
                + "."
            )

        power = self._number(
            electrolyzer_power_mw,
            "Electrolyzer nominal power [MW]",
            strictly_positive=True,
        )
        modules = int(
            self._number(
                module_count,
                "Electrolyzer module count",
                strictly_positive=True,
                integer=True,
            )
        )
        storage_capacity = self._number(
            storage_kg_h2,
            "Usable H2 storage capacity [kg]",
        )
        target = self._number(
            target_h2_kg_per_h,
            "Target H2 feed [kg/h]",
        )
        maximum = self._number(
            max_h2_feed_kg_per_h,
            "Maximum H2 feed [kg/h]",
            strictly_positive=True,
        )

        if target > maximum:
            raise ValueError(
                "Target H2 feed exceeds maximum process intake."
            )

        if not isinstance(storage_enabled, bool):
            raise ValueError(
                "storage_enabled must be True or False."
            )

        if not isinstance(expand_daily_to_hourly, bool):
            raise ValueError(
                "expand_daily_to_hourly must be True or False."
            )

        if tea_inputs is not None and not isinstance(
            tea_inputs,
            TEAInputs,
        ):
            raise ValueError(
                "tea_inputs must be a TEAInputs object or None."
            )

        if optimization_inputs is not None and not isinstance(
            optimization_inputs,
            OptimizationInputs,
        ):
            raise ValueError(
                "optimization_inputs must be an "
                "OptimizationInputs object or None."
            )

        if profile_meta is not None and not isinstance(
            profile_meta,
            dict,
        ):
            raise ValueError(
                "profile_meta must be a dictionary or None."
            )

        if not isinstance(case_name, str) or not case_name.strip():
            raise ValueError(
                "case_name must be a nonempty string."
            )

        if (
            tea_inputs is not None
            and electricity_price_usd_per_kwh is not None
        ):
            raise ValueError(
                "Do not provide a USD electricity-price override "
                "with the EUR TEA. Set "
                "tea_inputs.prices.electricity_eur_per_kwh instead."
            )

        economic_payload = deepcopy(
            self.scenario_config[scenario_name]
        )

        if not isinstance(economic_payload, dict):
            raise ValueError(
                "The selected scenario must contain a JSON object."
            )

        if electricity_price_usd_per_kwh is not None:
            usd_price = self._number(
                electricity_price_usd_per_kwh,
                "Electricity price [USD/kWh]",
            )
            economic_payload[
                "electricity_price_usd_per_mwh"
            ] = usd_price * 1000.0

        try:
            economic = ScenarioEconomicInputs(
                **economic_payload
            )
        except TypeError as exc:
            raise ValueError(
                f"Invalid configuration for scenario "
                f"{scenario_name!r}: {exc}"
            ) from exc

        declared_step = time_step_h

        if declared_step is not None:
            declared_step = self._number(
                declared_step,
                "Declared time step [h]",
                strictly_positive=True,
            )

        upstream_meta = deepcopy(profile_meta or {})

        if renewable_profile_df is None:
            if declared_step is not None and abs(
                declared_step - 1.0
            ) > 1e-8:
                raise ValueError(
                    "The synthetic default profile is hourly. "
                    "Its time step must be one hour."
                )

            peak = self._number(
                renewable_peak_power_mw,
                "Renewable peak power [MW]",
                strictly_positive=True,
            )
            profile = build_default_hourly_profile(
                peak_power_mw=peak,
            )
            effective_meta = infer_profile_meta_from_normalized(
                df=profile,
                source_name="default_synthetic_profile",
                source_kind="synthetic",
                raw_rows=len(profile),
                unit_mode="MW",
                expanded_to_hourly=False,
                time_step_h=1.0,
            )

        else:
            # Copy inputs and preserve all records. The normalizer
            # reports invalid rows instead of silently dropping them.
            fallback_step = declared_step

            if fallback_step is None:
                metadata_step = upstream_meta.get("time_step_h")

                if metadata_step is None:
                    metadata_step = upstream_meta.get(
                        "median_step_h"
                    )

                if metadata_step is not None:
                    fallback_step = self._number(
                        metadata_step,
                        "Profile metadata time step [h]",
                        strictly_positive=True,
                    )

            profile, effective_meta = normalize_renewable_profile(
                raw_profile_df=renewable_profile_df.copy(deep=True),
                timestamp_col="timestamp",
                renewable_col="renewable_power_mw",
                unit_mode="MW",
                expand_daily_to_hourly=expand_daily_to_hourly,
                single_row_interval_h=fallback_step,
            )

            if declared_step is not None:
                original_step = effective_meta["original_step_h"]

                if abs(original_step - declared_step) > max(
                    1e-8,
                    declared_step * 1e-6,
                ):
                    raise ValueError(
                        "The declared time step does not match "
                        "the supplied profile before expansion."
                    )

            effective_meta["source_name"] = upstream_meta.get(
                "source_name",
                "provided_renewable_profile",
            )
            effective_meta["source_kind"] = upstream_meta.get(
                "source_kind",
                "provided",
            )

            # Preserve descriptive upstream expansion history without
            # overriding the interval inferred from the actual records.
            if (
                upstream_meta.get("expanded_to_hourly")
                and abs(effective_meta["time_step_h"] - 1.0) <= 1e-8
            ):
                effective_meta["expanded_to_hourly"] = True
                effective_meta["looks_daily"] = True
                effective_meta["original_step_h"] = 24.0
                effective_meta["warnings"].append(
                    "The supplied profile was already expanded "
                    "from daily to flat hourly values upstream."
                )

        profile_warnings = list(
            dict.fromkeys(effective_meta.get("warnings", []))
        )
        effective_meta["warnings"] = profile_warnings

        for message in profile_warnings:
            self.logger.warning(message)

        electrolyzer = ElectrolyzerInputs(
            nominal_power_mw=power,
            module_size_mw=power / modules,
            min_load_fraction=min_load_fraction,
            specific_energy_kwh_per_kg_h2=(
                specific_energy_kwh_per_kg_h2
            ),
            # Used only by the legacy USD economic calculation.
            capex_usd_per_kw=850.0,
            fixed_opex_fraction=0.03,
        )

        storage = StorageInputs(
            enabled=storage_enabled,
            usable_capacity_kg_h2=storage_capacity,
            initial_soc_fraction=initial_soc_fraction,
            max_charge_kg_per_h=max_charge_kg_per_h,
            max_discharge_kg_per_h=max_discharge_kg_per_h,
        )

        ptmeoh = PtMeOHInputs(
            operating_mode=operating_mode,
            surrogate_library=surrogate_library,
            target_h2_feed_kg_per_h=target,
            max_h2_feed_kg_per_h=maximum,
        )

        optimization = (
            deepcopy(optimization_inputs)
            if optimization_inputs is not None
            else self._default_optimization(
                power_mw=power,
                module_count=modules,
                storage_enabled=storage_enabled,
                storage_kg=storage_capacity,
                target_h2=target,
                maximum_h2=maximum,
            )
        )

        case = CaseInputs(
            case_name=case_name.strip(),
            scenario_name=scenario_name,
            economic=economic,
            electrolyzer=electrolyzer,
            storage=storage,
            ptmeoh=ptmeoh,
            optimization=optimization,
            renewable_profile=profile,
            time_step_h=float(effective_meta["time_step_h"]),
            tea=deepcopy(tea_inputs),
            provenance={
                "profile_meta": effective_meta,
                "upstream_profile_meta": upstream_meta,
                "economic_mode": (
                    "EUR_TEA"
                    if tea_inputs is not None
                    else "legacy_USD"
                ),
                "scenario_applies_to_eur_tea": False,
                "module_count": modules,
            },
        )

        validate_case_inputs(case)

        return case

    def run_simulation(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ) -> SimulationArtifacts:
        self.logger.info("Running base simulation")

        return self.engine.run(
            case,
            progress_callback=progress_callback,
        )

    def run_optimization(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ) -> GridSearchArtifacts:
        self.logger.info("Running optimization")

        return self.optimizer.run(
            case,
            progress_callback=progress_callback,
        )

    def run_sensitivity(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
        baseline_simulation: SimulationArtifacts | None = None,
    ) -> pd.DataFrame:
        self.logger.info("Running sensitivity analysis")

        return self.sensitivity.run(
            case,
            progress_callback=progress_callback,
            baseline_simulation=baseline_simulation,
        )

    def run_sensitivities(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
        baseline_simulation: SimulationArtifacts | None = None,
    ) -> pd.DataFrame:
        """Plural alias for application compatibility."""

        return self.run_sensitivity(
            case,
            progress_callback=progress_callback,
            baseline_simulation=baseline_simulation,
        )

    def run_all(
        self,
        case: CaseInputs,
        run_optimization: bool = True,
        run_sensitivity: bool = True,
        progress_callback: ProgressCallback = None,
    ) -> dict[str, Any]:
        """
        Run the baseline and optional analyses.

        Sensitivity remains centered on the supplied case.
        It does not automatically switch to the optimization winner.
        """

        simulation = self.run_simulation(
            case,
            progress_callback=progress_callback,
        )

        outputs: dict[str, Any] = {
            "simulation": simulation,
            "optimization": None,
            "sensitivity": None,
            "sensitivities": None,
        }

        if run_optimization:
            outputs["optimization"] = self.run_optimization(
                case,
                progress_callback=progress_callback,
            )
        else:
            self.logger.info("Optimization skipped by user")

        if run_sensitivity:
            sensitivity_result = self.run_sensitivity(
                case,
                progress_callback=progress_callback,
                baseline_simulation=simulation,
            )
            outputs["sensitivity"] = sensitivity_result
            outputs["sensitivities"] = sensitivity_result
        else:
            self.logger.info("Sensitivity analysis skipped by user")

        return outputs

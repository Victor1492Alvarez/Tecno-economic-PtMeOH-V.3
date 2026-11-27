from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
from math import isfinite
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from domain.data_models import (
    DATA_SCHEMA_VERSION,
    CaseInputs,
    SimulationArtifacts,
)
from domain.electrolyzer_model import ElectrolyzerModel
from domain.h2_storage_model import H2StorageModel
from domain.ptmeoh_surrogate import MultiSurrogateManager
from domain.technoeconomics import TechnoEconomics
from domain.units import (
    KG_PER_T,
    MOLAR_MASS_CO2_KG_PER_KMOL,
    MOLAR_MASS_H2_KG_PER_KMOL,
    MOLAR_MASS_MEOH_KG_PER_KMOL,
    mass_flow_to_kg_per_h,
)
from domain.validators import validate_case_inputs
from infrastructure.logging_utils import configure_logger
from infrastructure.model_registry import ModelRegistry


ProgressCallback = Callable[[str, int, int], None] | None


class SimulationEngine:
    """
    Sequential deterministic simulation with coupled electrical demand.

    Raw surrogate predictions are retained separately from operational
    quantities. Operational process outputs are zero when H2 feed is zero.

    Electrical deficits and unconverged iterations are reported.
    They are not interpreted as implicit electricity imports.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.logger = configure_logger("ptmeoh_tool.simulation")

        self._surrogate_cache: dict[
            tuple[str, bool],
            tuple[str, MultiSurrogateManager],
        ] = {}

    def clear_surrogate_cache(self) -> None:
        self._surrogate_cache.clear()

    def _get_surrogates(
        self,
        library_name: str,
        require_runtime: bool = False,
    ) -> tuple[MultiSurrogateManager, str]:
        registry = ModelRegistry(self.project_root)
        fingerprint = registry.get_library_fingerprint(library_name)
        key = (library_name, require_runtime)
        cached = self._surrogate_cache.get(key)

        if cached is None or cached[0] != fingerprint:
            self.logger.info(
                "Loading surrogate library '%s'",
                library_name,
            )

            manager = MultiSurrogateManager(
                project_root=self.project_root,
                library_name=library_name,
                allow_mock=not require_runtime,
            )

            # Detect changes during loading rather than mixing versions.
            refreshed = ModelRegistry(self.project_root)
            after_load = refreshed.get_library_fingerprint(library_name)

            if after_load != fingerprint:
                raise RuntimeError(
                    "Model package contents changed during loading. "
                    "Finish installing the packages and retry."
                )

            self._surrogate_cache[key] = (
                fingerprint,
                manager,
            )

        return self._surrogate_cache[key][1], fingerprint

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        if callback is not None:
            callback(stage, current, total)

    @staticmethod
    def _dispatch_h2(
        case: CaseInputs,
        storage: H2StorageModel,
        starting_soc_kg_h2: float,
        produced_h2_kg_per_h: float,
    ) -> dict[str, float]:
        """
        Dispatch all produced H2 without losing excess above process intake.

        quasi_base_load:
            Serve the target, store excess, then discard remaining excess.

        flexible:
            Store production above the target first. The process can absorb
            remaining surplus up to its maximum intake. Storage discharges
            only to support the target.
        """

        produced = float(produced_h2_kg_per_h)

        if not isfinite(produced) or produced < 0:
            raise ValueError(
                "Produced H2 must be finite and nonnegative."
            )

        target = float(case.ptmeoh.target_h2_feed_kg_per_h)
        maximum = float(case.ptmeoh.max_h2_feed_kg_per_h)
        dt = float(case.time_step_h)

        direct = min(produced, target)
        requested_charge = max(produced - direct, 0.0)
        requested_discharge = max(target - direct, 0.0)

        if case.storage.enabled:
            transition = storage.preview(
                requested_charge_kg_per_h=requested_charge,
                requested_discharge_kg_per_h=requested_discharge,
                time_step_h=dt,
                starting_soc_kg_h2=starting_soc_kg_h2,
            )

            tank_in = transition.actual_charge_kg_per_h
            tank_out = transition.actual_discharge_kg_per_h
            new_soc = transition.ending_soc_kg_h2
        else:
            tank_in = 0.0
            tank_out = 0.0
            new_soc = starting_soc_kg_h2

        process_feed = direct + tank_out
        surplus = max(produced - direct - tank_in, 0.0)

        if case.ptmeoh.operating_mode == "flexible":
            additional_feed = min(
                surplus,
                max(maximum - process_feed, 0.0),
            )
            process_feed += additional_feed
            surplus -= additional_feed

        unmet = max(target - process_feed, 0.0)

        residual = (
            produced + tank_out
            - process_feed - tank_in - surplus
        )

        tolerance = max(1e-8, produced * 1e-10)

        if abs(residual) > tolerance:
            raise RuntimeError(
                "The H2 dispatch balance did not close: "
                f"{residual:.12g} kg/h."
            )

        return {
            "target_h2_kg_per_h": target,
            "actual_h2_kg_per_h": process_feed,
            "unmet_h2_kg_per_h": unmet,
            "tank_in_kg_per_h": tank_in,
            "tank_out_kg_per_h": tank_out,
            "tank_full_event": int(
                case.storage.enabled
                and surplus > 1e-9
                and storage.capacity_kg_h2 - new_soc
                <= storage.inventory_tolerance_kg
            ),
            "tank_empty_event": int(
                case.storage.enabled
                and unmet > 1e-9
                and new_soc <= storage.inventory_tolerance_kg
            ),
            "curtailed_h2_kg_per_h": max(surplus, 0.0),
            "new_soc_kg_h2": new_soc,
            "h2_balance_residual_kg_per_h": residual,
        }

    def _operational_outputs(
        self,
        case: CaseInputs,
        manager: MultiSurrogateManager,
        predictions: dict[str, Any],
        h2_feed_kg_per_h: float,
    ) -> tuple[dict[str, float], set[str]]:
        """
        Convert process predictions to explicit simulation units.

        EUR runtime models require declared, unambiguous output units.
        Historical units may be assumed for legacy or mock predictions,
        but every such assumption is reported.
        """

        summary = predictions["model_summary_df"]
        by_name = summary.set_index("model_name")
        messages: set[str] = set()
        process_on = h2_feed_kg_per_h > 0.0

        def raw(
            name: str,
            expected_unit: str,
            allow_missing: bool = False,
        ) -> tuple[float, str]:
            if name not in by_name.index:
                if allow_missing:
                    messages.add(
                        f"{name}: no model is available; "
                        "the explicitly configured fallback is used."
                    )
                    return 0.0, expected_unit

                raise ValueError(
                    f"Required process model {name!r} is unavailable."
                )

            row = by_name.loc[name]
            value = float(row["prediction"])

            if not isfinite(value):
                raise ValueError(
                    f"{name}: the prediction is not finite."
                )

            unit = str(row["output_unit"]).strip()
            ambiguous = not unit or "_or_" in unit

            if ambiguous:
                if (
                    case.tea is not None
                    and row["runtime_mode"] == "runtime"
                ):
                    raise ValueError(
                        f"{name}: declare the actual output unit "
                        "in its TXT parameters or metadata.json "
                        "before using the EUR TEA."
                    )

                unit = expected_unit
                messages.add(
                    f"{name}: output units are unresolved; "
                    f"the historical {expected_unit} convention "
                    "was assumed. Verify before technical use."
                )
            if unit.lower().replace(" ", "") in {"mol/h", "mol/hr"}:
                value = value / 1000.0
                unit = "kmol/h"
            
            return value, unit

        def power_w(name: str, signed: bool = False) -> float:
            value, unit = raw(name, "W")
            factors = {
                "w": 1.0,
                "kw": 1000.0,
                "mw": 1_000_000.0,
            }
            normalized = unit.lower().replace(" ", "")

            if normalized not in factors:
                raise ValueError(
                    f"{name}: unsupported power unit {unit!r}."
                )

            converted = value * factors[normalized]

            if not signed and converted < 0 and process_on:
                raise ValueError(
                    f"{name}: negative electrical consumption "
                    "cannot be used in the operating balance."
                )

            return converted if process_on else 0.0

        def mass_kg_per_h(
            name: str,
            expected_unit: str,
            molar_mass: float,
        ) -> float:
            value, unit = raw(name, expected_unit)
            converted = float(
                mass_flow_to_kg_per_h(
                    value,
                    unit,
                    molar_mass_kg_per_kmol=molar_mass,
                )
            )

            if converted < 0 and process_on:
                raise ValueError(
                    f"{name}: negative material flow cannot "
                    "be used as valid process production or feed."
                )

            return converted if process_on else 0.0

        result: dict[str, float] = {}

        for number in (1, 2, 3):
            value_w = power_w(f"Model_Power_C{number}_Watt")
            result[f"power_c{number}_w"] = value_w
            result[f"power_c{number}_mw"] = value_w / 1_000_000.0

        if case.tea is not None:
            utility_settings = case.tea.utilities
            heater_name = utility_settings.electric_heater_model_name
            heater_duty_w = power_w(heater_name, signed=True)

            if (
                utility_settings.heater_duty_sign
                == "negative_is_heating"
            ):
                heating_w = max(-heater_duty_w, 0.0)
                opposite_sign = heater_duty_w > 0
            else:
                heating_w = max(heater_duty_w, 0.0)
                opposite_sign = heater_duty_w < 0

            if opposite_sign:
                messages.add(
                    f"{heater_name}: the duty has the opposite sign "
                    "to the configured heating convention. "
                    "It does not create electric-heating demand."
                )

            hx3_w = (
                heating_w
                / utility_settings.electric_heater_efficiency
            )

            for name in utility_settings.cooling_model_names:
                result[f"{name}__w"] = power_w(name, signed=True)

        else:
            # The previous USD methodology counted compressors only.
            hx3_w = 0.0

        result["power_hx3_w"] = hx3_w
        result["power_hx3_mw"] = hx3_w / 1_000_000.0

        auxiliary_w = (
            result["power_c1_w"]
            + result["power_c2_w"]
            + result["power_c3_w"]
            + hx3_w
        )

        result["downstream_aux_power_mw"] = (
            auxiliary_w / 1_000_000.0
        )
        result["aux_power_from_models_w"] = auxiliary_w

        if "Model_Prod_MeOH" in by_name.index:
            result["methanol_t_per_h"] = (
                mass_kg_per_h(
                    "Model_Prod_MeOH",
                    "t/h",
                    MOLAR_MASS_MEOH_KG_PER_KMOL,
                )
                / KG_PER_T
            )
        else:
            result["methanol_t_per_h"] = (
                h2_feed_kg_per_h
                / KG_PER_T
                * case.ptmeoh.methanol_yield_t_meoh_per_t_h2
            )
            messages.add(
                "The methanol production model is unavailable. "
                "The configured constant-yield fallback was used."
            )

        if case.tea is not None:
            co2_settings = case.tea.co2_feed

            if co2_settings.source == "surrogate":
                name = co2_settings.model_name
                bundle = manager.bundles.get(name)

                if bundle is None or bundle.runtime_mode != "runtime":
                    if not co2_settings.allow_stoichiometric_fallback:
                        raise ValueError(
                            "Surrogate-based CO2 feed requires its "
                            "selected runtime model. A mock predictor "
                            "is not a valid CO2-feed substitute."
                        )

                    result["co2_feed_kg_per_h"] = (
                        result["methanol_t_per_h"]
                        * KG_PER_T
                        * co2_settings.kg_co2_per_kg_meoh
                    )
                    messages.add(
                        "The selected runtime CO2 model is unavailable. "
                        "The explicitly permitted stoichiometric "
                        "fallback was used."
                    )
                    result["co2_stoichiometric_fallback"] = 1.0
                else:
                    # The declared model unit takes precedence. The TEA
                    # setting is only the fallback for an undeclared unit.
                    result["co2_feed_kg_per_h"] = mass_kg_per_h(
                        name,
                        co2_settings.output_unit,
                        MOLAR_MASS_CO2_KG_PER_KMOL,
                    )
                    result["co2_stoichiometric_fallback"] = 0.0

            else:
                result["co2_feed_kg_per_h"] = (
                    result["methanol_t_per_h"]
                    * KG_PER_T
                    * co2_settings.kg_co2_per_kg_meoh
                )
                result["co2_stoichiometric_fallback"] = 0.0

        # Optional diagnostic streams, converted only with known units.
        for name, column in (
            ("Model_Purge_H2", "h2_purge_kg_per_h"),
            ("Model_RECLP_H2", "h2_recycle_kg_per_h"),
        ):
            if name not in by_name.index:
                continue

            unit = str(by_name.loc[name, "output_unit"]).strip()

            if not unit or "_or_" in unit:
                messages.add(
                    f"{name}: normalized H2 diagnostic flow is "
                    "unavailable because its output unit is unresolved."
                )
                continue

            result[column] = mass_kg_per_h(
                name,
                unit,
                MOLAR_MASS_H2_KG_PER_KMOL,
            )

        if not all(isfinite(value) for value in result.values()):
            raise ValueError(
                "Unit conversion produced a non-finite "
                "operational quantity."
            )

        return result, messages

    @staticmethod
    def _profile_fingerprint(profile: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update(
            repr(
                [
                    (str(column), str(dtype))
                    for column, dtype in zip(
                        profile.columns,
                        profile.dtypes,
                    )
                ]
            ).encode("utf-8")
        )
        digest.update(
            pd.util.hash_pandas_object(
                profile,
                index=False,
            ).to_numpy().tobytes()
        )
        return digest.hexdigest()

    @staticmethod
    def _input_snapshot(
        case: CaseInputs,
        profile_fingerprint: str,
        model_fingerprint: str,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}

        for item in fields(case):
            if item.name == "renewable_profile":
                continue

            value = getattr(case, item.name)

            if hasattr(value, "__dataclass_fields__"):
                value = asdict(value)

            snapshot[item.name] = value

        snapshot["renewable_profile"] = {
            "rows": len(case.renewable_profile),
            "fingerprint": profile_fingerprint,
            "start": str(
                case.renewable_profile["timestamp"].iloc[0]
            ),
            "end": str(
                case.renewable_profile["timestamp"].iloc[-1]
            ),
        }
        snapshot["model_library_fingerprint"] = model_fingerprint

        # Detach nested user-provided dictionaries from the live case.
        from copy import deepcopy

        return deepcopy(snapshot)

    def run(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
        progress_every_steps: int = 250,
        max_power_iterations: int = 40,
        power_tolerance_mw: float = 1e-6,
    ) -> SimulationArtifacts:
        warnings = validate_case_inputs(case)

        for name, value in (
            ("progress_every_steps", progress_every_steps),
            ("max_power_iterations", max_power_iterations),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if (
            not isfinite(power_tolerance_mw)
            or power_tolerance_mw <= 0
        ):
            raise ValueError(
                "power_tolerance_mw must be finite and positive."
            )

        self.logger.info(
            "Simulation started for case '%s'",
            case.case_name,
        )

        electrolyzer = ElectrolyzerModel(
            nominal_power_mw=case.electrolyzer.nominal_power_mw,
            module_size_mw=case.electrolyzer.module_size_mw,
            min_load_fraction=case.electrolyzer.min_load_fraction,
            specific_energy_kwh_per_kg_h2=(
                case.electrolyzer.specific_energy_kwh_per_kg_h2
            ),
        )

        storage = H2StorageModel(
            capacity_kg_h2=case.storage.usable_capacity_kg_h2,
            initial_soc_fraction=case.storage.initial_soc_fraction,
            max_charge_kg_per_h=case.storage.max_charge_kg_per_h,
            max_discharge_kg_per_h=case.storage.max_discharge_kg_per_h,
        )

        if not case.storage.enabled:
            storage.set_inventory(0.0)

        initial_inventory = storage.soc_kg_h2

        manager, model_fingerprint = self._get_surrogates(
            case.ptmeoh.surrogate_library,
            require_runtime=case.optimization.require_runtime_models,
        )

        profile_fingerprint = self._profile_fingerprint(
            case.renewable_profile
        )
        snapshot = self._input_snapshot(
            case,
            profile_fingerprint,
            model_fingerprint,
        )

        warning_set = set(warnings)
        warning_set.update(manager.warnings)

        rows: list[dict[str, Any]] = []
        diagnostic_counts = {
            name: {
                "evaluated_steps": 0,
                "input_outside_steps": 0,
                "input_unknown_steps": 0,
                "output_outside_steps": 0,
                "output_unknown_steps": 0,
                "warning_steps": 0,
            }
            for name in manager.model_names
        }

        dt = float(case.time_step_h)
        total_steps = len(case.renewable_profile)
        model_summary = pd.DataFrame()

        self._emit_progress(
            progress_callback,
            "simulation",
            0,
            total_steps,
        )

        for step_index, row in enumerate(
            case.renewable_profile.itertuples(index=False),
            start=1,
        ):
            renewable_mw = float(row.renewable_power_mw)
            starting_inventory = storage.soc_kg_h2
            auxiliary_guess_mw = 0.0
            converged = False

            for iteration in range(1, max_power_iterations + 1):
                el_state = electrolyzer.step(
                    max(renewable_mw - auxiliary_guess_mw, 0.0)
                )

                dispatch = self._dispatch_h2(
                    case=case,
                    storage=storage,
                    starting_soc_kg_h2=starting_inventory,
                    produced_h2_kg_per_h=(
                        el_state.h2_produced_kg_per_h
                    ),
                )

                predictions = manager.predict_all(
                    dispatch["actual_h2_kg_per_h"]
                )

                operational, conversion_warnings = (
                    self._operational_outputs(
                        case,
                        manager,
                        predictions,
                        dispatch["actual_h2_kg_per_h"],
                    )
                )

                required_auxiliary_mw = operational[
                    "downstream_aux_power_mw"
                ]
                iteration_residual = (
                    required_auxiliary_mw - auxiliary_guess_mw
                )

                if abs(iteration_residual) <= power_tolerance_mw:
                    converged = True
                    break

                # Damping reduces oscillations around module thresholds.
                auxiliary_guess_mw = (
                    0.5 * auxiliary_guess_mw
                    + 0.5 * required_auxiliary_mw
                )

            # Commit inventory only after the final evaluated iteration.
            storage.set_inventory(dispatch["new_soc_kg_h2"])

            model_summary = predictions["model_summary_df"].copy()
            warning_set.update(conversion_warnings)

            total_internal_mw = (
                el_state.power_to_electrolyzer_mw
                + operational["downstream_aux_power_mw"]
            )
            renewable_used_mw = min(
                renewable_mw,
                total_internal_mw,
            )
            deficit_mw = max(
                total_internal_mw - renewable_mw,
                0.0,
            )
            curtailed_mw = max(
                renewable_mw - total_internal_mw,
                0.0,
            )

            record: dict[str, Any] = {
                "timestamp": row.timestamp,
                "renewable_power_mw": renewable_mw,
                "renewable_used_mw": renewable_used_mw,
                "power_to_electrolyzer_mw": (
                    el_state.power_to_electrolyzer_mw
                ),
                "total_internal_power_mw": total_internal_mw,
                "power_deficit_mw": deficit_mw,
                "curtailed_power_mw": curtailed_mw,
                "module_count_online": el_state.module_count_online,
                "online_module_load_fraction": (
                    el_state.online_module_load_fraction
                ),
                "h2_produced_kg_per_h": (
                    el_state.h2_produced_kg_per_h
                ),
                "tank_start_soc_kg_h2": starting_inventory,
                "tank_soc_kg_h2": storage.soc_kg_h2,
                "tank_in_kg_per_h": dispatch["tank_in_kg_per_h"],
                "tank_out_kg_per_h": dispatch["tank_out_kg_per_h"],
                "tank_full_event": dispatch["tank_full_event"],
                "tank_empty_event": dispatch["tank_empty_event"],
                "h2_target_to_ptmeoh_kg_per_h": (
                    dispatch["target_h2_kg_per_h"]
                ),
                "h2_to_ptmeoh_kg_per_h": (
                    dispatch["actual_h2_kg_per_h"]
                ),
                "unmet_h2_kg_per_h": (
                    dispatch["unmet_h2_kg_per_h"]
                ),
                "curtailed_h2_kg_per_h": (
                    dispatch["curtailed_h2_kg_per_h"]
                ),
                "h2_balance_residual_kg_per_h": (
                    dispatch["h2_balance_residual_kg_per_h"]
                ),
                "power_iteration_converged": int(converged),
                "power_iteration_count": iteration,
                "power_iteration_residual_mw": iteration_residual,
                "surrogate_all_models_in_domain": int(
                    predictions["all_models_in_domain"]
                ),
                "surrogate_all_domains_verified": int(
                    predictions["all_input_domains_verified"]
                ),
                "surrogate_runtime_models_count": (
                    predictions["runtime_models_count"]
                ),
                "surrogate_total_models_count": (
                    predictions["total_models_count"]
                ),
                "surrogate_input_outside_count": (
                    predictions["input_out_of_domain_models_count"]
                ),
                "surrogate_input_unknown_count": (
                    predictions["input_domain_unknown_models_count"]
                ),
                "surrogate_output_outside_count": (
                    predictions["output_out_of_range_models_count"]
                ),
                "surrogate_output_unknown_count": (
                    predictions["output_range_unknown_models_count"]
                ),
                "surrogate_warning_models_count": (
                    predictions["models_with_warnings_count"]
                ),
                **operational,
            }

            for name, bundle in manager.bundles.items():
                output_name = bundle.output_column

                # Keep the existing raw output alias when it is safe.
                # Always retain an unambiguous per-model raw column.
                raw_column = f"{name}__raw_prediction"
                std_column = f"{name}__raw_std"

                if raw_column in record or std_column in record:
                    raise ValueError(
                        f"Reserved simulation-column collision: {name}."
                    )

                record[raw_column] = predictions[output_name]
                record[std_column] = predictions[
                    f"{output_name}__std"
                ]

                if output_name not in record:
                    record[output_name] = predictions[output_name]

                alias_std = f"{output_name}__std"

                if alias_std not in record:
                    record[alias_std] = predictions[alias_std]

            for diagnostic in model_summary.to_dict("records"):
                name = diagnostic["model_name"]
                counts = diagnostic_counts[name]
                counts["evaluated_steps"] += 1

                input_status = diagnostic["in_domain"]
                output_status = diagnostic["output_in_range"]

                input_unknown = pd.isna(input_status)
                output_unknown = pd.isna(output_status)

                input_outside = (
                    not input_unknown and not bool(input_status)
                )
                output_outside = (
                    not output_unknown and not bool(output_status)
                )

                counts["input_unknown_steps"] += int(input_unknown)
                counts["input_outside_steps"] += int(input_outside)
                counts["output_unknown_steps"] += int(output_unknown)
                counts["output_outside_steps"] += int(output_outside)
                counts["warning_steps"] += int(
                    bool(diagnostic["warnings"])
                )

                record[f"{name}__input_outside"] = int(input_outside)
                record[f"{name}__input_unknown"] = int(input_unknown)
                record[f"{name}__output_outside"] = int(output_outside)
                record[f"{name}__output_unknown"] = int(output_unknown)

            rows.append(record)

            if (
                step_index == 1
                or step_index % progress_every_steps == 0
                or step_index == total_steps
            ):
                self.logger.info(
                    "Simulation progress %s/%s",
                    step_index,
                    total_steps,
                )
                self._emit_progress(
                    progress_callback,
                    "simulation",
                    step_index,
                    total_steps,
                )

        df = pd.DataFrame(rows)

        te = TechnoEconomics(case)
        economic = te.compute(df)
        tea_artifacts = economic.get("tea_artifacts")

        annual_factor = (
            float(tea_artifacts.assumptions["annualization_factor"])
            if tea_artifacts is not None
            else 1.0
        )

        def period_total(column: str) -> float:
            return float(df[column].sum() * dt)

        def annual_total(column: str) -> float:
            return period_total(column) * annual_factor

        requested_h2 = period_total(
            "h2_target_to_ptmeoh_kg_per_h"
        )
        unmet_h2 = period_total("unmet_h2_kg_per_h")
        renewable_energy = period_total("renewable_power_mw")

        unmet_fraction = (
            unmet_h2 / requested_h2
            if requested_h2 > 0
            else 0.0
        )

        curtailment_fraction = (
            period_total("curtailed_power_mw") / renewable_energy
            if renewable_energy > 0
            else 0.0
        )

        runtime_count = len(
            [
                bundle
                for bundle in manager.bundles.values()
                if bundle.runtime_mode == "runtime"
            ]
        )
        model_count = len(manager.bundles)

        kpis = {
            key: float(value)
            for key, value in economic.items()
            if isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
        }

        kpis.update(
            {
                "annual_methanol_t": float(
                    economic["annual_meoh_t"]
                ),
                "electrolyzer_full_load_hours_h": (
                    annual_total("power_to_electrolyzer_mw")
                    / case.electrolyzer.nominal_power_mw
                ),
                "ptmeoh_utilization_factor": float(
                    df["h2_to_ptmeoh_kg_per_h"].mean()
                    / case.ptmeoh.max_h2_feed_kg_per_h
                ),
                "renewable_utilization_fraction": (
                    period_total("renewable_used_mw")
                    / renewable_energy
                    if renewable_energy > 0
                    else 0.0
                ),
                "h2_not_supplied_t": (
                    unmet_h2 * annual_factor / KG_PER_T
                ),
                "unmet_h2_fraction": unmet_fraction,
                "curtailment_fraction": curtailment_fraction,
                "annual_total_electricity_mwh": float(
                    economic["annual_power_mwh"]
                ),
                "annual_downstream_electricity_mwh": float(
                    economic["annual_downstream_power_mwh"]
                ),
                "power_deficit_mwh": period_total(
                    "power_deficit_mw"
                ),
                "annual_power_deficit_mwh": annual_total(
                    "power_deficit_mw"
                ),
                "surrogate_out_of_domain_fraction": float(
                    (df["surrogate_input_outside_count"] > 0).mean()
                ),
                "surrogate_unknown_domain_fraction": float(
                    (df["surrogate_input_unknown_count"] > 0).mean()
                ),
                "surrogate_output_out_of_range_fraction": float(
                    (df["surrogate_output_outside_count"] > 0).mean()
                ),
                "surrogate_unknown_output_range_fraction": float(
                    (df["surrogate_output_unknown_count"] > 0).mean()
                ),
                "runtime_models_fraction": (
                    runtime_count / model_count
                ),
                "tank_empty_hours": annual_total(
                    "tank_empty_event"
                ),
                "tank_full_hours": annual_total(
                    "tank_full_event"
                ),
                "curtailed_hours": float(
                    (df["curtailed_power_mw"] > 0).sum()
                    * dt * annual_factor
                ),
                "power_nonconvergence_fraction": float(
                    (df["power_iteration_converged"] == 0).mean()
                ),
                "initial_h2_inventory_kg": initial_inventory,
                "final_h2_inventory_kg": storage.soc_kg_h2,
                "net_h2_inventory_change_kg": (
                    storage.soc_kg_h2 - initial_inventory
                ),
                "max_h2_balance_residual_kg_per_h": float(
                    df["h2_balance_residual_kg_per_h"].abs().max()
                ),
            }
        )

        thresholds = [
            (
                unmet_fraction,
                case.ptmeoh.unmet_h2_warning_threshold,
                "Unmet H2 exceeds the configured warning threshold.",
            ),
            (
                curtailment_fraction,
                case.ptmeoh.curtailment_warning_threshold,
                "Renewable curtailment exceeds the configured threshold.",
            ),
        ]

        for actual, limit, message in thresholds:
            if actual > limit:
                warning_set.add(message)

        if (
            kpis["ptmeoh_utilization_factor"]
            < case.ptmeoh.utilization_warning_threshold
        ):
            warning_set.add(
                "PtMeOH utilization is below the configured threshold."
            )

        if kpis["power_nonconvergence_fraction"] > 0:
            warning_set.add(
                "The electrical coupling did not converge in some "
                "time steps. Their final evaluated states are retained "
                "and flagged; they are not validated operating points."
            )

        if kpis["power_deficit_mwh"] > 1e-6:
            warning_set.add(
                "Internal electricity demand exceeded renewable "
                "supply in some time steps. No grid supply was "
                "implicitly added."
            )

        if runtime_count < model_count:
            warning_set.add(
                "DEMONSTRATION RESULTS: at least one selected model "
                "uses a mock predictor."
            )

        if (
            initial_inventory - storage.soc_kg_h2
            > storage.inventory_tolerance_kg
        ):
            warning_set.add(
                "The simulated period consumed net initial H2 inventory. "
                "A cyclic inventory condition is not enforced. "
                "Repeating this period in the TEA assumes that inventory "
                "is replenished; review this boundary condition."
            )

        diagnostic_rows = []

        for name, counts in diagnostic_counts.items():
            diagnostic_rows.append(
                {"model_name": name, **counts}
            )

            if counts["input_outside_steps"]:
                warning_set.add(
                    f"{name}: input extrapolation occurred in "
                    f"{counts['input_outside_steps']} accepted steps. "
                    "Predictions were retained."
                )

            if counts["output_outside_steps"]:
                warning_set.add(
                    f"{name}: outputs exceeded declared ranges in "
                    f"{counts['output_outside_steps']} accepted steps. "
                    "Predictions were retained."
                )

            if counts["input_unknown_steps"]:
                warning_set.add(
                    f"{name}: the input domain was not fully defined "
                    f"in {counts['input_unknown_steps']} accepted steps."
                )

            if counts["output_unknown_steps"]:
                warning_set.add(
                    f"{name}: the output range was not fully defined "
                    f"in {counts['output_unknown_steps']} accepted steps."
                )

        diagnostics = pd.DataFrame(diagnostic_rows)

        if not model_summary.empty:
            model_summary = model_summary.merge(
                diagnostics,
                on="model_name",
                how="left",
                validate="one_to_one",
            )

        if tea_artifacts is not None:
            if (
                "co2_stoichiometric_fallback" in df.columns
                and df["co2_stoichiometric_fallback"].any()
            ):
                tea_artifacts.status["co2_source"] = (
                    "stoichiometric_fallback"
                )

            warning_set.update(tea_artifacts.warnings)

            valid_operating_points = (
                kpis["power_deficit_mwh"] <= 1e-6
                and kpis["power_nonconvergence_fraction"] == 0
            )
            tea_artifacts.status.update(
                {
                    "operating_points_electrically_valid": (
                        valid_operating_points
                    ),
                    "uses_mock_models": runtime_count < model_count,
                    "uses_input_extrapolation": (
                        kpis["surrogate_out_of_domain_fraction"] > 0
                    ),
                }
            )

        completed_at = datetime.now(timezone.utc).isoformat()

        self.logger.info(
            "Simulation finished for case '%s'",
            case.case_name,
        )

        return SimulationArtifacts(
            time_series=df,
            kpis=kpis,
            warnings=sorted(warning_set),
            surrogate_info={
                "library_name": case.ptmeoh.surrogate_library,
                "models_detected": model_count,
                "runtime_models_detected": runtime_count,
                "model_library_fingerprint": model_fingerprint,
                "diagnostic_summary": diagnostics,
                "model_summary_scope": "last accepted time step",
                "range_warning_policy": "calculate_and_warn",
            },
            model_summary=model_summary,
            tea=tea_artifacts,
            input_snapshot=snapshot,
            run_metadata={
                "schema_version": DATA_SCHEMA_VERSION,
                "completed_at_utc": completed_at,
                "currency": (
                    "EUR" if tea_artifacts is not None else "USD"
                ),
                "profile_fingerprint": profile_fingerprint,
                "model_library_fingerprint": model_fingerprint,
                "time_step_h": dt,
                "simulated_duration_h": len(df) * dt,
                "annualization_factor": annual_factor,
                "max_power_iterations": max_power_iterations,
                "power_tolerance_mw": power_tolerance_mw,
                "storage_losses_modeled": False,
                "cyclic_inventory_enforced": False,
                "process_shutdown_policy": (
                    "zero operational process outputs at zero H2 feed; "
                    "raw model predictions retained"
                ),
                "flexible_dispatch_policy": (
                    "serve target, store surplus, then use remaining "
                    "surplus up to maximum process intake"
                ),
            },
        )

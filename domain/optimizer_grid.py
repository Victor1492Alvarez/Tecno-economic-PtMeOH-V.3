from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import itertools
from math import isfinite, prod
from numbers import Real
from typing import Any, Callable

import pandas as pd

from domain.data_models import (
    CaseInputs,
    GridSearchArtifacts,
    SimulationArtifacts,
)
from domain.simulation_engine import SimulationEngine
from domain.validators import (
    validate_case_inputs,
    validate_optimization_inputs,
)


ProgressCallback = Callable[[str, int, int], None] | None


class OptimizationConsistencyError(RuntimeError):
    """Inputs or model packages changed during a grid search."""


class GridOptimizer:
    """
    Exhaustive search over the configured design grid.

    Model extrapolation is reported but does not reject a candidate.

    Operational feasibility, model validity, and demonstration status
    remain separate concepts.
    """

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.logger = getattr(engine, "logger", None)

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
    def _unique(values: list) -> list:
        """Remove duplicate grid entries while preserving their order."""

        return list(dict.fromkeys(values))

    @staticmethod
    def _metric(
        kpis: dict[str, Any],
        name: str,
        allow_infinite: bool = False,
    ) -> float:
        value = kpis.get(name)

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise ValueError(
                f"Required optimization metric {name!r} "
                "is missing or is not numeric."
            )

        numeric = float(value)

        if pd.isna(numeric):
            raise ValueError(
                f"Optimization metric {name!r} is NaN."
            )

        if not allow_infinite and not isfinite(numeric):
            raise ValueError(
                f"Optimization metric {name!r} is not finite."
            )

        return numeric

    @staticmethod
    def _build_candidate(
        base_case: CaseInputs,
        power_mw: float,
        storage_kg: float,
        target_h2: float,
        module_count: int,
        case_index: int,
    ) -> CaseInputs:
        """
        Preserve all existing configuration fields.

        The renewable profile is shared for efficiency and must not
        be modified by the simulation engine.
        """

        electrolyzer = replace(
            base_case.electrolyzer,
            nominal_power_mw=float(power_mw),
            module_size_mw=float(power_mw) / int(module_count),
        )

        storage = replace(
            base_case.storage,
            usable_capacity_kg_h2=float(storage_kg),
        )

        ptmeoh = replace(
            base_case.ptmeoh,
            target_h2_feed_kg_per_h=float(target_h2),
        )

        provenance = deepcopy(base_case.provenance)
        provenance["optimization_candidate"] = {
            "index": case_index,
            "base_case_name": base_case.case_name,
            "electrolyzer_power_mw": float(power_mw),
            "storage_kg_h2": float(storage_kg),
            "target_h2_kg_per_h": float(target_h2),
            "module_count": int(module_count),
        }

        return replace(
            base_case,
            case_name=(
                f"grid_{case_index:05d}"
                f"_P{power_mw:g}"
                f"_S{storage_kg:g}"
                f"_T{target_h2:g}"
                f"_M{module_count}"
            ),
            economic=deepcopy(base_case.economic),
            electrolyzer=electrolyzer,
            storage=storage,
            ptmeoh=ptmeoh,
            optimization=deepcopy(base_case.optimization),
            tea=deepcopy(base_case.tea),
            provenance=provenance,
        )

    def _evaluate(
        self,
        simulation: SimulationArtifacts,
        base_case: CaseInputs,
        currency: str,
        objective: str,
    ) -> dict[str, Any]:
        kpis = simulation.kpis
        settings = base_case.optimization

        utilization = self._metric(
            kpis,
            "ptmeoh_utilization_factor",
        )
        unmet = self._metric(kpis, "unmet_h2_fraction")
        curtailment = self._metric(
            kpis,
            "curtailment_fraction",
        )
        deficit = self._metric(kpis, "power_deficit_mwh")
        nonconvergence = self._metric(
            kpis,
            "power_nonconvergence_fraction",
        )
        methanol = self._metric(kpis, "annual_methanol_t")
        runtime_fraction = self._metric(
            kpis,
            "runtime_models_fraction",
        )

        input_extrapolation = self._metric(
            kpis,
            "surrogate_out_of_domain_fraction",
        )
        output_extrapolation = self._metric(
            kpis,
            "surrogate_output_out_of_range_fraction",
        )
        unknown_input = self._metric(
            kpis,
            "surrogate_unknown_domain_fraction",
        )
        unknown_output = self._metric(
            kpis,
            "surrogate_unknown_output_range_fraction",
        )

        suffix = currency.lower()

        capex_key = f"total_capex_{suffix}"
        npv_key = f"npv_{suffix}"

        if currency == "EUR":
            lcomeoh_key = (
                "lcomeoh_net_eur_per_t_meoh"
                if settings.lcomeoh_basis == "net"
                else "lcomeoh_gross_eur_per_t_meoh"
            )
        else:
            lcomeoh_key = "lcomeoh_usd_per_t_meoh"

        capex = self._metric(kpis, capex_key)
        npv = self._metric(kpis, npv_key)
        lcomeoh = self._metric(
            kpis,
            lcomeoh_key,
            allow_infinite=True,
        )

        if capex < 0:
            raise ValueError("Total CAPEX cannot be negative.")

        tolerance = 1e-9

        checks = {
            "positive_methanol_production": methanol > 0,
            "minimum_ptmeoh_utilization": (
                utilization + tolerance
                >= settings.min_ptmeoh_utilization
            ),
            "maximum_unmet_h2_fraction": (
                unmet <= settings.max_unmet_h2_fraction + tolerance
            ),
            "maximum_curtailment_fraction": (
                curtailment
                <= settings.max_curtailment_fraction + tolerance
            ),
            "maximum_power_deficit": (
                deficit
                <= settings.max_power_deficit_mwh + 1e-9
            ),
            "electrical_coupling_converged": (
                nonconvergence == 0.0
            ),
            "runtime_models_required": (
                not settings.require_runtime_models
                or runtime_fraction >= 1.0 - tolerance
            ),
        }

        if objective == "min_lcomeoh":
            objective_value = lcomeoh
            score = lcomeoh
            objective_metric = lcomeoh_key
        else:
            objective_value = npv
            score = -npv
            objective_metric = npv_key

        checks["finite_objective"] = isfinite(score)

        violations = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        uses_mock = runtime_fraction < 1.0 - tolerance
        uses_extrapolation = (
            input_extrapolation > 0
            or output_extrapolation > 0
        )
        has_unknown_ranges = (
            unknown_input > 0
            or unknown_output > 0
        )

        if uses_mock:
            prediction_status = "demonstration"
        elif uses_extrapolation:
            prediction_status = "extrapolation_warning"
        elif has_unknown_ranges:
            prediction_status = "unverified_ranges"
        else:
            # Range checks alone do not validate a physical model.
            prediction_status = "within_declared_ranges"

        result: dict[str, Any] = {
            "feasible": int(not violations),
            "constraint_violation_count": len(violations),
            "constraint_violations": "; ".join(violations),
            "objective_metric": objective_metric,
            "objective_value": objective_value,
            "objective_score": score,
            "ranking_capex": capex,
            "prediction_status": prediction_status,
            "uses_mock_models": uses_mock,
            "uses_extrapolation": uses_extrapolation,
            "has_unknown_model_ranges": has_unknown_ranges,
        }

        for name, passed in checks.items():
            result[f"passes_{name}"] = bool(passed)

        return result

    def run(
        self,
        base_case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ) -> GridSearchArtifacts:
        preflight_warnings = validate_case_inputs(base_case)
        preflight_warnings.extend(
            validate_optimization_inputs(base_case)
        )

        settings = base_case.optimization
        objective = settings.objective

        if objective not in {"min_lcomeoh", "max_npv"}:
            raise ValueError(
                f"Unsupported optimization objective {objective!r}. "
                "Supported objectives: min_lcomeoh, max_npv."
            )

        currency = "EUR" if base_case.tea is not None else "USD"

        if currency == "USD" and settings.lcomeoh_basis == "net":
            raise ValueError(
                "Net-of-coproduct LCOMeOH requires the EUR TEA. "
                "Use lcomeoh_basis='gross' for the legacy USD mode."
            )

        warnings = list(dict.fromkeys(preflight_warnings))

        if settings.max_surrogate_out_of_domain_fraction != 0:
            warnings.append(
                "max_surrogate_out_of_domain_fraction is retained "
                "for compatibility but is not an exclusion criterion. "
                "Extrapolation follows the calculate-and-warn policy."
            )

        power_grid = self._unique(
            settings.electrolyzer_power_grid_mw
        )
        storage_grid = (
            self._unique(settings.storage_grid_kg_h2)
            if base_case.storage.enabled
            else [0.0]
        )
        target_grid = self._unique(
            settings.target_h2_grid_kg_per_h
        )
        module_grid = self._unique(settings.module_count_grid)

        if not base_case.storage.enabled:
            warnings.append(
                "H2 storage is disabled. Storage grid values were "
                "collapsed to zero to avoid equivalent evaluations."
            )

        total_cases = prod(
            len(grid)
            for grid in (
                power_grid,
                storage_grid,
                target_grid,
                module_grid,
            )
        )

        combinations = itertools.product(
            power_grid,
            storage_grid,
            target_grid,
            module_grid,
        )

        self._emit_progress(
            progress_callback,
            "optimization",
            0,
            total_cases,
        )

        rows: list[dict[str, Any]] = []
        best_simulation: SimulationArtifacts | None = None
        best_rank: tuple[float, int, float, int] | None = None

        expected_profile_fingerprint: str | None = None
        expected_model_fingerprint: str | None = None

        for index, combination in enumerate(
            combinations,
            start=1,
        ):
            power, storage_kg, target, module_count = combination

            candidate_row: dict[str, Any] = {
                "candidate_index": index,
                "case_name": f"grid_{index:05d}",
                "currency": currency,
                "electrolyzer_power_mw": float(power),
                "module_count": int(module_count),
                "module_size_mw": float(power) / int(module_count),
                "storage_kg_h2": float(storage_kg),
                "target_h2_kg_per_h": float(target),
                "evaluation_status": "failed",
                "feasible": 0,
                "warning_count": 0,
                "error_type": "",
                "error_message": "",
            }

            if self.logger is not None:
                self.logger.info(
                    "Optimization case %s/%s | "
                    "P=%s MW | S=%s kg | T=%s kg/h | M=%s",
                    index,
                    total_cases,
                    power,
                    storage_kg,
                    target,
                    module_count,
                )

            try:
                candidate = self._build_candidate(
                    base_case=base_case,
                    power_mw=power,
                    storage_kg=storage_kg,
                    target_h2=target,
                    module_count=module_count,
                    case_index=index,
                )

                candidate_row["case_name"] = candidate.case_name

                simulation = self.engine.run(candidate)

                profile_fingerprint = simulation.run_metadata.get(
                    "profile_fingerprint"
                )
                model_fingerprint = simulation.run_metadata.get(
                    "model_library_fingerprint"
                )

                if not profile_fingerprint or not model_fingerprint:
                    raise ValueError(
                        "The engine did not return profile and "
                        "model fingerprints. Install the updated "
                        "simulation_engine.py before optimization."
                    )

                if expected_profile_fingerprint is None:
                    expected_profile_fingerprint = profile_fingerprint
                    expected_model_fingerprint = model_fingerprint

                elif (
                    profile_fingerprint
                    != expected_profile_fingerprint
                    or model_fingerprint
                    != expected_model_fingerprint
                ):
                    raise OptimizationConsistencyError(
                        "The profile or model library changed during "
                        "the search. Results from different input "
                        "versions cannot be ranked together."
                    )

                if simulation.run_metadata.get("currency") != currency:
                    raise OptimizationConsistencyError(
                        "The simulation returned an unexpected currency."
                    )

                evaluation = self._evaluate(
                    simulation=simulation,
                    base_case=base_case,
                    currency=currency,
                    objective=objective,
                )

                candidate_row.update(simulation.kpis)
                candidate_row.update(evaluation)
                candidate_row.update(
                    {
                        "evaluation_status": "completed",
                        "warning_count": len(simulation.warnings),
                        "warning_messages": "\n".join(
                            simulation.warnings
                        ),
                        "profile_fingerprint": profile_fingerprint,
                        "model_library_fingerprint": (
                            model_fingerprint
                        ),
                    }
                )

                if evaluation["feasible"]:
                    rank = (
                        float(evaluation["objective_score"]),
                        len(simulation.warnings),
                        float(evaluation["ranking_capex"]),
                        index,
                    )

                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_simulation = simulation

            except OptimizationConsistencyError:
                # Never combine different model/profile versions.
                raise

            except Exception as exc:
                candidate_row.update(
                    {
                        "evaluation_status": "failed",
                        "feasible": 0,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "constraint_violations": (
                            "candidate_evaluation_failed"
                        ),
                    }
                )

                if self.logger is not None:
                    self.logger.warning(
                        "Optimization candidate %s failed: %s: %s",
                        index,
                        type(exc).__name__,
                        exc,
                    )

            rows.append(candidate_row)

            self._emit_progress(
                progress_callback,
                "optimization",
                index,
                total_cases,
            )

        results = pd.DataFrame(rows)

        empty_best = pd.Series(
            dtype="object",
            name="best_row",
        )
        empty_alternative = pd.Series(
            dtype="object",
            name="best_infeasible_row",
        )

        if results.empty:
            return GridSearchArtifacts(
                results=results,
                feasible_results=results.copy(),
                best_row=empty_best,
                status="empty",
                warnings=warnings + [
                    "No optimization candidates were generated."
                ],
                run_metadata={
                    "currency": currency,
                    "objective": objective,
                    "evaluated_cases": 0,
                },
            )

        completed = results.loc[
            results["evaluation_status"] == "completed"
        ].copy()

        rank_columns = [
            "objective_score",
            "warning_count",
            "ranking_capex",
            "candidate_index",
        ]

        if completed.empty:
            feasible = completed.copy()
            best = empty_best
            alternative = empty_alternative
            status = "failed"

            warnings.append(
                "Every candidate evaluation failed. "
                "Inspect error_type and error_message in the results."
            )

        else:
            feasible = completed.loc[
                completed["feasible"] == 1
            ].copy()

            if not feasible.empty:
                feasible = feasible.sort_values(
                    rank_columns,
                    ascending=True,
                    kind="stable",
                )
                best = feasible.iloc[0].copy()
                status = "feasible"
            else:
                best = empty_best
                status = "no_feasible"

                warnings.append(
                    "No operationally feasible candidate was found. "
                    "No recommended configuration is returned."
                )

            # This is the economically best completed non-feasible
            # alternative, not the closest design to feasibility.
            alternatives = completed.loc[
                (completed["feasible"] == 0)
                & completed["objective_score"].map(
                    lambda value: isfinite(float(value))
                )
            ].copy()

            if alternatives.empty:
                alternative = empty_alternative
            else:
                alternative = (
                    alternatives.sort_values(
                        rank_columns,
                        ascending=True,
                        kind="stable",
                    )
                    .iloc[0]
                    .copy()
                )

        failed_count = int(
            (results["evaluation_status"] == "failed").sum()
        )

        if failed_count:
            warnings.append(
                f"{failed_count} of {len(results)} candidates failed. "
                "The recommendation, if present, applies only to "
                "successfully evaluated candidates."
            )

        if not best.empty:
            if bool(best["uses_mock_models"]):
                warnings.append(
                    "The selected candidate is operationally feasible "
                    "under the implemented checks, but uses mock "
                    "models. It is a demonstration recommendation."
                )

            if bool(best["uses_extrapolation"]):
                warnings.append(
                    "The selected candidate uses surrogate "
                    "extrapolation. Its predictions were retained; "
                    "technical validity requires review."
                )

            if bool(best["has_unknown_model_ranges"]):
                warnings.append(
                    "The selected candidate has models with "
                    "incompletely defined validity ranges."
                )

        if best_simulation is not None:
            warnings.extend(best_simulation.warnings)

        return GridSearchArtifacts(
            results=results,
            feasible_results=feasible,
            best_row=best,
            status=status,
            best_infeasible_row=alternative,
            best_simulation=best_simulation,
            warnings=list(dict.fromkeys(warnings)),
            run_metadata={
                "currency": currency,
                "objective": objective,
                "lcomeoh_basis": settings.lcomeoh_basis,
                "evaluated_cases": int(len(results)),
                "completed_cases": int(len(completed)),
                "failed_cases": failed_count,
                "feasible_cases": int(len(feasible)),
                "profile_fingerprint": (
                    expected_profile_fingerprint
                ),
                "model_library_fingerprint": (
                    expected_model_fingerprint
                ),
                "extrapolation_policy": "calculate_and_warn",
                "ranking_tiebreakers": [
                    "warning_count",
                    "total_capex",
                    "candidate_index",
                ],
                "process_maximum_intake_varied": False,
                "nominal_meoh_capacity_varied": False,
                "best_infeasible_selection": (
                    "best objective among completed, non-feasible "
                    "candidates with finite objective"
                ),
            },
        )

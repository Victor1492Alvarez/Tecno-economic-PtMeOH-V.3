from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from math import isfinite
from numbers import Real
from typing import Any, Callable

import pandas as pd

from domain.data_models import CaseInputs, SimulationArtifacts
from domain.simulation_engine import SimulationEngine
from domain.technoeconomics import TechnoEconomics
from domain.validators import validate_case_inputs


ProgressCallback = Callable[[str, int, int], None] | None


EXPERIMENTS = [
    ("electricity_price", 0.85, 1.15),
    ("electrolyzer_power", 0.90, 1.10),
    ("storage_capacity", 0.80, 1.20),
    ("target_h2_feed", 0.90, 1.10),
]


class SensitivityConsistencyError(RuntimeError):
    """Results cannot be compared because their input versions differ."""


class SensitivityAnalyzer:
    """One-at-a-time sensitivity around a fixed reference case."""

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
    def _validate_multiplier(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(
                "Sensitivity multipliers must be finite and positive."
            )

        return float(value)

    def _clone_case(
        self,
        base_case: CaseInputs,
        *,
        electricity_price_multiplier: float = 1.0,
        electrolyzer_power_multiplier: float = 1.0,
        storage_multiplier: float = 1.0,
        target_h2_multiplier: float = 1.0,
    ) -> CaseInputs:
        """
        Change only the requested inputs.

        Scaling module size together with total power preserves the
        original module count. No minimum-size floor is introduced.
        """

        electricity_multiplier = self._validate_multiplier(
            electricity_price_multiplier
        )
        power_multiplier = self._validate_multiplier(
            electrolyzer_power_multiplier
        )
        storage_multiplier = self._validate_multiplier(
            storage_multiplier
        )
        target_multiplier = self._validate_multiplier(
            target_h2_multiplier
        )

        economic = deepcopy(base_case.economic)
        tea = deepcopy(base_case.tea)

        if tea is None:
            economic.electricity_price_usd_per_mwh *= (
                electricity_multiplier
            )
        else:
            price = tea.prices.electricity_eur_per_kwh

            if price is None:
                raise ValueError(
                    "The EUR electricity price must be defined."
                )

            tea.prices.electricity_eur_per_kwh = (
                price * electricity_multiplier
            )

        target = min(
            base_case.ptmeoh.target_h2_feed_kg_per_h
            * target_multiplier,
            base_case.ptmeoh.max_h2_feed_kg_per_h,
        )

        return replace(
            base_case,
            economic=economic,
            electrolyzer=replace(
                base_case.electrolyzer,
                nominal_power_mw=(
                    base_case.electrolyzer.nominal_power_mw
                    * power_multiplier
                ),
                module_size_mw=(
                    base_case.electrolyzer.module_size_mw
                    * power_multiplier
                ),
            ),
            storage=replace(
                base_case.storage,
                usable_capacity_kg_h2=(
                    base_case.storage.usable_capacity_kg_h2
                    * storage_multiplier
                ),
            ),
            ptmeoh=replace(
                base_case.ptmeoh,
                target_h2_feed_kg_per_h=target,
            ),
            optimization=deepcopy(base_case.optimization),
            tea=tea,
            provenance=deepcopy(base_case.provenance),
        )

    @staticmethod
    def _parameter_value(
        case: CaseInputs,
        parameter: str,
    ) -> tuple[float, str]:
        if parameter == "electricity_price":
            if case.tea is not None:
                return (
                    float(case.tea.prices.electricity_eur_per_kwh),
                    "EUR/kWh",
                )

            return (
                float(case.economic.electricity_price_usd_per_mwh),
                "USD/MWh",
            )

        if parameter == "electrolyzer_power":
            return case.electrolyzer.nominal_power_mw, "MW"

        if parameter == "storage_capacity":
            return case.storage.usable_capacity_kg_h2, "kg H2"

        if parameter == "target_h2_feed":
            return case.ptmeoh.target_h2_feed_kg_per_h, "kg/h"

        raise ValueError(
            f"Unsupported sensitivity parameter {parameter!r}."
        )

    @staticmethod
    def _lcomeoh_key(case: CaseInputs) -> str:
        if case.tea is None:
            return "lcomeoh_usd_per_t_meoh"

        return (
            "lcomeoh_net_eur_per_t_meoh"
            if case.optimization.lcomeoh_basis == "net"
            else "lcomeoh_gross_eur_per_t_meoh"
        )

    @staticmethod
    def _finite_difference(
        value: Any,
        reference: Any,
    ) -> float:
        """Undefined differences remain NaN rather than becoming zero."""

        try:
            value = float(value)
            reference = float(reference)
        except (TypeError, ValueError):
            return float("nan")

        if not isfinite(value) or not isfinite(reference):
            return float("nan")

        return value - reference

    def _verify_baseline(
        self,
        case: CaseInputs,
        baseline: SimulationArtifacts,
    ) -> None:
        """
        Verify that a supplied baseline belongs to the current case.

        This uses the snapshot and fingerprint helpers of the updated
        simulation engine.
        """

        if not isinstance(baseline, SimulationArtifacts):
            raise ValueError(
                "baseline_simulation must be a SimulationArtifacts object."
            )

        _, model_fingerprint = self.engine._get_surrogates(
            case.ptmeoh.surrogate_library,
            require_runtime=case.optimization.require_runtime_models,
        )

        profile_fingerprint = self.engine._profile_fingerprint(
            case.renewable_profile
        )
        expected_snapshot = self.engine._input_snapshot(
            case,
            profile_fingerprint,
            model_fingerprint,
        )

        expected = json.dumps(
            expected_snapshot,
            sort_keys=True,
            default=str,
        )
        actual = json.dumps(
            baseline.input_snapshot,
            sort_keys=True,
            default=str,
        )

        if expected != actual:
            raise SensitivityConsistencyError(
                "The supplied baseline does not match the current "
                "case, renewable profile, or model versions. "
                "Run a new baseline simulation."
            )

    @staticmethod
    def _reprice(
        case: CaseInputs,
        baseline: SimulationArtifacts,
    ) -> tuple[dict[str, float], list[str]]:
        """
        Recalculate economics without rerunning unchanged dispatch.

        Electricity price does not affect the current dispatch policy.
        """

        economics = TechnoEconomics(case).compute(
            baseline.time_series
        )
        kpis = dict(baseline.kpis)

        for name, value in economics.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                kpis[name] = float(value)

        kpis["annual_methanol_t"] = float(
            economics["annual_meoh_t"]
        )
        kpis["annual_total_electricity_mwh"] = float(
            economics["annual_power_mwh"]
        )
        kpis["annual_downstream_electricity_mwh"] = float(
            economics["annual_downstream_power_mwh"]
        )

        # Remove old economic messages before adding repriced messages.
        old_tea_warnings = (
            set(baseline.tea.warnings)
            if baseline.tea is not None
            else set()
        )
        warnings = [
            message
            for message in baseline.warnings
            if message not in old_tea_warnings
        ]

        artifacts = economics.get("tea_artifacts")

        if artifacts is not None:
            warnings.extend(artifacts.warnings)

        return kpis, list(dict.fromkeys(warnings))

    def run(
        self,
        base_case: CaseInputs,
        progress_callback: ProgressCallback = None,
        baseline_simulation: SimulationArtifacts | None = None,
    ) -> pd.DataFrame:
        """
        Return low/high experiments with baseline-relative changes.

        A matching baseline may be supplied to avoid rerunning it.
        Otherwise, one baseline simulation is performed first.
        """

        validate_case_inputs(base_case)

        if baseline_simulation is None:
            if self.logger is not None:
                self.logger.info(
                    "Running sensitivity reference simulation"
                )

            baseline = self.engine.run(base_case)
        else:
            self._verify_baseline(
                base_case,
                baseline_simulation,
            )
            baseline = baseline_simulation

        currency = "EUR" if base_case.tea is not None else "USD"

        if baseline.run_metadata.get("currency") != currency:
            raise SensitivityConsistencyError(
                "The baseline currency does not match the case."
            )

        profile_fingerprint = baseline.run_metadata.get(
            "profile_fingerprint"
        )
        model_fingerprint = baseline.run_metadata.get(
            "model_library_fingerprint"
        )

        if not profile_fingerprint or not model_fingerprint:
            raise ValueError(
                "The baseline must include fingerprints from "
                "the updated simulation engine."
            )

        lcomeoh_key = self._lcomeoh_key(base_case)
        npv_key = f"npv_{currency.lower()}"

        baseline_lcomeoh = float(
            baseline.kpis[lcomeoh_key]
        )
        baseline_npv = float(baseline.kpis[npv_key])

        total = len(EXPERIMENTS) * 2
        current = 0
        rows: list[dict[str, Any]] = []

        self._emit_progress(
            progress_callback,
            "sensitivity",
            0,
            total,
        )

        argument_names = {
            "electricity_price": "electricity_price_multiplier",
            "electrolyzer_power": "electrolyzer_power_multiplier",
            "storage_capacity": "storage_multiplier",
            "target_h2_feed": "target_h2_multiplier",
        }

        for parameter, low, high in EXPERIMENTS:
            for label, multiplier in (("low", low), ("high", high)):
                base_value, unit = self._parameter_value(
                    base_case,
                    parameter,
                )

                result: dict[str, Any] = {
                    "parameter": parameter,
                    "case_label": label,
                    "multiplier": multiplier,
                    "currency": currency,
                    "parameter_unit": unit,
                    "base_value": base_value,
                    "requested_value": base_value * multiplier,
                    "applied_value": float("nan"),
                    "effective_multiplier": float("nan"),
                    "variation_limited": False,
                    "no_effect_expected": False,
                    "evaluation_status": "failed",
                    "physical_simulation_reused": False,
                    "warning_count": 0,
                    "error_type": "",
                    "error_message": "",
                    "lcomeoh_metric": lcomeoh_key,
                    "baseline_lcomeoh": baseline_lcomeoh,
                    "baseline_npv": baseline_npv,
                }

                if self.logger is not None:
                    self.logger.info(
                        "Sensitivity %s/%s | %s | %s | multiplier=%s",
                        current + 1,
                        total,
                        parameter,
                        label,
                        multiplier,
                    )

                try:
                    case = self._clone_case(
                        base_case,
                        **{
                            argument_names[parameter]: multiplier
                        },
                    )
                    case.case_name = (
                        f"{base_case.case_name}"
                        f"_sensitivity_{parameter}_{label}"
                    )
                    case.provenance["sensitivity"] = {
                        "reference_case": base_case.case_name,
                        "parameter": parameter,
                        "label": label,
                        "requested_multiplier": multiplier,
                    }

                    applied_value, _ = self._parameter_value(
                        case,
                        parameter,
                    )

                    limited = (
                        parameter == "target_h2_feed"
                        and result["requested_value"] > applied_value
                    )
                    no_effect = (
                        applied_value == base_value
                        or (
                            parameter == "storage_capacity"
                            and not base_case.storage.enabled
                        )
                    )

                    result.update(
                        {
                            "applied_value": applied_value,
                            "effective_multiplier": (
                                applied_value / base_value
                                if base_value != 0
                                else float("nan")
                            ),
                            "variation_limited": limited,
                            "no_effect_expected": no_effect,
                        }
                    )

                    if no_effect:
                        kpis = dict(baseline.kpis)
                        warnings = list(baseline.warnings)
                        reused = True

                    elif parameter == "electricity_price":
                        kpis, warnings = self._reprice(
                            case,
                            baseline,
                        )
                        reused = True

                    else:
                        simulation = self.engine.run(case)

                        if (
                            simulation.run_metadata.get(
                                "profile_fingerprint"
                            ) != profile_fingerprint
                            or simulation.run_metadata.get(
                                "model_library_fingerprint"
                            ) != model_fingerprint
                        ):
                            raise SensitivityConsistencyError(
                                "The renewable profile or model "
                                "versions changed during sensitivity "
                                "analysis. Restart with consistent inputs."
                            )

                        if (
                            simulation.run_metadata.get("currency")
                            != currency
                        ):
                            raise SensitivityConsistencyError(
                                "A sensitivity experiment returned "
                                "an unexpected currency."
                            )

                        kpis = dict(simulation.kpis)
                        warnings = list(simulation.warnings)
                        reused = False

                    if limited:
                        warnings.append(
                            "The requested H2 target exceeded maximum "
                            "process intake. The applied target was "
                            "limited to that physical capacity."
                        )

                    if no_effect:
                        warnings.append(
                            "This variation does not change the active "
                            "case inputs or an enabled subsystem. "
                            "The baseline results were reused."
                        )

                    warnings = list(dict.fromkeys(warnings))

                    lcomeoh = float(kpis[lcomeoh_key])
                    npv = float(kpis[npv_key])

                    result.update(kpis)
                    result.update(
                        {
                            "evaluation_status": "completed",
                            "physical_simulation_reused": reused,
                            "warning_count": len(warnings),
                            "warning_messages": "\n".join(warnings),
                            "lcomeoh_value": lcomeoh,
                            "npv_value": npv,
                            "delta_lcomeoh": self._finite_difference(
                                lcomeoh,
                                baseline_lcomeoh,
                            ),
                            "delta_npv": self._finite_difference(
                                npv,
                                baseline_npv,
                            ),
                            "profile_fingerprint": profile_fingerprint,
                            "model_library_fingerprint": (
                                model_fingerprint
                            ),
                        }
                    )

                except SensitivityConsistencyError:
                    raise

                except Exception as exc:
                    result.update(
                        {
                            "evaluation_status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

                    if self.logger is not None:
                        self.logger.warning(
                            "Sensitivity failed for %s/%s: %s: %s",
                            parameter,
                            label,
                            type(exc).__name__,
                            exc,
                        )

                rows.append(result)
                current += 1

                self._emit_progress(
                    progress_callback,
                    "sensitivity",
                    current,
                    total,
                )

        results = pd.DataFrame(rows)

        results.attrs.update(
            {
                "analysis_type": "one_at_a_time",
                "reference_case": base_case.case_name,
                "currency": currency,
                "lcomeoh_metric": lcomeoh_key,
                "baseline_lcomeoh": baseline_lcomeoh,
                "baseline_npv": baseline_npv,
                "profile_fingerprint": profile_fingerprint,
                "model_library_fingerprint": model_fingerprint,
                "reoptimization_per_experiment": False,
                "process_capacity_varied": False,
            }
        )

        return results

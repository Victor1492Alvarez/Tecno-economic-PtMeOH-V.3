from __future__ import annotations

from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from domain.units import (
    KG_PER_T,
    MOLAR_MASS_H2_KG_PER_KMOL,
    SECONDS_PER_HOUR,
)
from infrastructure.model_registry import ModelRegistry
from infrastructure.surrogate_loader import (
    SurrogateBundle,
    load_surrogate_bundle,
)


SUMMARY_COLUMNS = [
    # Original columns retained for compatibility.
    "library",
    "model_name",
    "output_name",
    "prediction",
    "predictive_std",
    "input_column",
    "domain_min",
    "domain_max",
    "in_domain",
    "runtime_mode",
    "ready_for_runtime",
    "missing_files",

    # Additional diagnostics.
    "physical_h2_flow_kg_per_h",
    "model_input_value",
    "input_unit",
    "effective_input_unit",
    "input_unit_assumed",
    "output_unit",
    "output_min",
    "output_max",
    "output_in_range",
    "uncertainty_available",
    "negative_physical_output",
    "warnings",
]


RESERVED_OUTPUT_KEYS = {
    "all_models_in_domain",
    "all_input_domains_verified",
    "all_models_output_in_range",
    "all_output_ranges_verified",
    "input_out_of_domain_models_count",
    "input_domain_unknown_models_count",
    "output_out_of_range_models_count",
    "output_range_unknown_models_count",
    "negative_physical_outputs_count",
    "runtime_models_count",
    "mock_models_count",
    "total_models_count",
    "models_with_warnings_count",
    "prediction_warning_messages",
    "model_summary_df",
}


class MultiSurrogateManager:
    """
    Coordinate single-input, single-output surrogate models.

    Physical input to predict_all() is always H2 flow in kg/h.

    Declared model input units are applied before prediction.
    Predictions remain in each model's original output units.
    The simulation engine must explicitly convert those output units
    before using predictions in physical balances or the TEA.

    Extrapolation produces warnings, not automatic rejection.
    """

    def __init__(
        self,
        project_root: Path,
        library_name: str,
        allow_mock: bool = True,
        model_names: list[str] | None = None,
    ):
        self.project_root = Path(project_root)
        self.library_name = library_name
        self.allow_mock = allow_mock

        self.registry = ModelRegistry(self.project_root)

        discovered = self.registry.get_models_by_library(
            library_name
        )

        if model_names is None:
            selected = discovered
        else:
            if (
                not isinstance(model_names, list)
                or any(
                    not isinstance(name, str) or not name.strip()
                    for name in model_names
                )
            ):
                raise ValueError(
                    "model_names must be a list of nonempty strings."
                )

            selected = list(dict.fromkeys(model_names))
            unavailable = sorted(set(selected) - set(discovered))

            if unavailable:
                raise ValueError(
                    "The following models are not registered or "
                    "discovered in the selected library: "
                    + ", ".join(unavailable)
                    + "."
                )

        if not selected:
            raise ValueError(
                f"No models were found for library "
                f"{library_name!r}."
            )

        self.model_names = list(selected)

        self.bundles: dict[str, SurrogateBundle] = {
            name: load_surrogate_bundle(
                project_root=self.project_root,
                model_name=name,
                library_name=self.library_name,
                allow_mock=allow_mock,
            )
            for name in self.model_names
        }

        self._validate_output_names()

        initial_warnings = list(self.registry.warnings)

        for bundle in self.bundles.values():
            initial_warnings.extend(bundle.warnings)

        self.warnings = list(dict.fromkeys(initial_warnings))

        # Replaced on every call, rather than accumulated indefinitely.
        self.last_warnings: list[str] = []

    def _validate_output_names(self) -> None:
        """Prevent collisions with model outputs and manager diagnostics."""

        used = set(RESERVED_OUTPUT_KEYS)

        for name, bundle in self.bundles.items():
            output_name = bundle.output_column

            if not output_name.strip():
                raise ValueError(
                    f"Model {name!r} has an empty output name."
                )

            generated_names = {
                output_name,
                f"{output_name}__std",
            }

            collisions = used.intersection(generated_names)

            if collisions:
                raise ValueError(
                    f"Model {name!r} has conflicting output names: "
                    + ", ".join(sorted(collisions))
                    + ". Assign a unique Output Column "
                    "in the model metadata."
                )

            used.update(generated_names)

    @staticmethod
    def _validate_h2_flow(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                "Physical H2 flow must be a finite, "
                "nonnegative number in kg/h."
            )

        return float(value)

    @staticmethod
    def _convert_h2_input(
        h2_flow_kg_per_h: float,
        bundle: SurrogateBundle,
    ) -> tuple[float, str, bool]:
        """
        Return model input, effective unit, and assumed-unit status.

        Missing input units retain the legacy kg/h convention,
        accompanied by an explicit warning.

        Unknown declared units are not guessed.
        """

        declared_unit = bundle.input_unit.strip()

        if not declared_unit:
            return h2_flow_kg_per_h, "kg/h", True

        normalized = declared_unit.lower().replace(" ", "")

        if normalized == "kg/h":
            return h2_flow_kg_per_h, "kg/h", False

        if normalized == "t/h":
            return h2_flow_kg_per_h / KG_PER_T, "t/h", False

        if normalized == "kg/s":
            return (
                h2_flow_kg_per_h / SECONDS_PER_HOUR,
                "kg/s",
                False,
            )

        if normalized == "kmol/h":
            return (
                h2_flow_kg_per_h / MOLAR_MASS_H2_KG_PER_KMOL,
                "kmol/h",
                False,
            )

        raise ValueError(
            f"Model {bundle.model_name!r} declares unsupported "
            f"H2 input units {declared_unit!r}. "
            "Supported units are kg/h, t/h, kg/s, and kmol/h. "
            "No magnitude-based conversion will be attempted."
        )

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        """Convert nullable pandas diagnostic values safely."""

        if value is None or pd.isna(value):
            return None

        return bool(value)

    @staticmethod
    def _missing_files_text(bundle: SurrogateBundle) -> str:
        value = bundle.package_status.get("missing_files", [])

        if isinstance(value, str):
            return value

        return ", ".join(str(item) for item in value)

    def predict_all(
        self,
        h2_flow_kg_per_h: float,
    ) -> dict[str, Any]:
        """
        Predict every selected model for one physical operating point.

        Finite predictions outside declared ranges are retained.
        Model failures stop this evaluation with a contextual error.
        """

        physical_h2 = self._validate_h2_flow(
            h2_flow_kg_per_h
        )

        outputs: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        warning_messages = list(self.warnings)

        input_statuses: list[bool | None] = []
        output_statuses: list[bool | None] = []

        runtime_count = 0
        warning_model_count = 0
        negative_output_count = 0

        for name, bundle in self.bundles.items():
            model_input, effective_unit, unit_assumed = (
                self._convert_h2_input(
                    physical_h2,
                    bundle,
                )
            )

            try:
                prediction_df = bundle.predict([model_input])
            except Exception as exc:
                raise RuntimeError(
                    f"Prediction failed for model {name!r} "
                    f"in library {self.library_name!r}, "
                    f"at physical H2 flow {physical_h2:g} kg/h "
                    f"and model input {model_input:g} "
                    f"{effective_unit}. "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            if len(prediction_df) != 1:
                raise RuntimeError(
                    f"Model {name!r} returned "
                    f"{len(prediction_df)} rows for one input."
                )

            row = prediction_df.iloc[0]

            prediction = float(row["Prediction"])
            predictive_std = float(row["Predictive Std"])

            if not isfinite(prediction):
                raise RuntimeError(
                    f"Model {name!r} returned a non-finite prediction."
                )

            uncertainty_available = bool(
                row.get("Uncertainty Available", False)
            )

            if uncertainty_available and (
                not isfinite(predictive_std)
                or predictive_std < 0
            ):
                raise RuntimeError(
                    f"Model {name!r} returned invalid "
                    "predictive uncertainty."
                )

            input_status = self._optional_bool(
                row.get("Input In Domain")
            )
            output_status = self._optional_bool(
                row.get("Output In Range")
            )

            input_statuses.append(input_status)
            output_statuses.append(output_status)

            model_messages: list[str] = []

            warning_text = row.get("Warnings", "")

            if isinstance(warning_text, str) and warning_text:
                model_messages.append(warning_text)

            if unit_assumed:
                model_messages.append(
                    f"{name}: input units are missing. "
                    "The legacy kg/h convention was assumed. "
                    "Verify this assumption against the training data."
                )

            metadata = self.registry.get_model_metadata(name)
            category = metadata.get("category", "")

            # Negative thermal duties may be valid signed outputs.
            # Negative electrical consumption or product flow is flagged
            # without altering the raw prediction.
            negative_physical_output = (
                prediction < 0
                and (
                    category in {
                        "electrical_load",
                        "product_stream",
                    }
                    or name == "Model_Prod_MeOH"
                    or name.startswith("Model_Power_")
                )
            )

            if negative_physical_output:
                negative_output_count += 1
                model_messages.append(
                    f"{name}: a negative physical output "
                    f"({prediction:g} {bundle.output_unit}) "
                    "was predicted. The raw value is retained, "
                    "but it must not be silently used as valid "
                    "production or electricity consumption."
                )

            if not uncertainty_available:
                model_messages.append(
                    f"{name}: predictive uncertainty is unavailable; "
                    "it is not assumed to be zero."
                )

            if bundle.runtime_mode == "runtime":
                runtime_count += 1

            if model_messages:
                warning_model_count += 1
                warning_messages.extend(model_messages)

            output_name = bundle.output_column

            outputs[output_name] = prediction
            outputs[f"{output_name}__std"] = predictive_std

            records.append(
                {
                    "library": self.library_name,
                    "model_name": name,
                    "output_name": output_name,
                    "prediction": prediction,
                    "predictive_std": predictive_std,
                    "input_column": bundle.input_column,
                    "domain_min": bundle.domain_min,
                    "domain_max": bundle.domain_max,
                    "in_domain": input_status,
                    "runtime_mode": bundle.runtime_mode,
                    "ready_for_runtime": bool(
                        bundle.package_status.get(
                            "ready_for_runtime",
                            False,
                        )
                    ),
                    "missing_files": self._missing_files_text(
                        bundle
                    ),
                    "physical_h2_flow_kg_per_h": physical_h2,
                    "model_input_value": model_input,
                    "input_unit": bundle.input_unit,
                    "effective_input_unit": effective_unit,
                    "input_unit_assumed": unit_assumed,
                    "output_unit": bundle.output_unit,
                    "output_min": bundle.output_min,
                    "output_max": bundle.output_max,
                    "output_in_range": output_status,
                    "uncertainty_available": uncertainty_available,
                    "negative_physical_output": (
                        negative_physical_output
                    ),
                    "warnings": " | ".join(
                        dict.fromkeys(model_messages)
                    ),
                }
            )

        total_models = len(self.bundles)

        input_outside_count = sum(
            status is False for status in input_statuses
        )
        input_unknown_count = sum(
            status is None for status in input_statuses
        )
        output_outside_count = sum(
            status is False for status in output_statuses
        )
        output_unknown_count = sum(
            status is None for status in output_statuses
        )

        self.last_warnings = list(
            dict.fromkeys(warning_messages)
        )

        outputs.update(
            {
                # Legacy compatibility:
                # True means no known input bound was exceeded.
                # It does not mean every model domain is verified.
                "all_models_in_domain": input_outside_count == 0,

                "all_input_domains_verified": all(
                    status is True for status in input_statuses
                ),

                # Same convention for declared output ranges.
                "all_models_output_in_range": (
                    output_outside_count == 0
                ),
                "all_output_ranges_verified": all(
                    status is True for status in output_statuses
                ),

                "input_out_of_domain_models_count": (
                    input_outside_count
                ),
                "input_domain_unknown_models_count": (
                    input_unknown_count
                ),
                "output_out_of_range_models_count": (
                    output_outside_count
                ),
                "output_range_unknown_models_count": (
                    output_unknown_count
                ),
                "negative_physical_outputs_count": (
                    negative_output_count
                ),
                "runtime_models_count": runtime_count,
                "mock_models_count": total_models - runtime_count,
                "total_models_count": total_models,
                "models_with_warnings_count": warning_model_count,
                "prediction_warning_messages": list(
                    self.last_warnings
                ),
                "model_summary_df": pd.DataFrame(
                    records,
                    columns=SUMMARY_COLUMNS,
                ),
            }
        )

        return outputs

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from domain.units import MOLAR_MASS_H2_KG_PER_KMOL

from domain.validators import model_value_warnings
from infrastructure.model_registry import ModelRegistry


METADATA_KEYS = {
    "input_column",
    "output_column",
    "input_unit",
    "output_unit",
    "unit",
    "train_x_min",
    "train_x_max",
    "train_y_min",
    "train_y_max",
    "output_min",
    "output_max",
}


def _normalized_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _lookup(
    parameters: dict,
    metadata: dict,
    *names: str,
    default: Any = None,
) -> Any:
    """Read parameters first, then metadata, using normalized keys."""

    for source in (parameters, metadata):
        normalized = {
            _normalized_key(key): value
            for key, value in source.items()
        }

        for name in names:
            value = normalized.get(_normalized_key(name))

            if value is not None and not (
                isinstance(value, str) and not value.strip()
            ):
                return value

    return default


def _safe_parse_scalar(text: str) -> Any:
    value = text.strip()

    if not value:
        return value

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    try:
        return json.loads(value)
    except (ValueError, TypeError):
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _load_txt_parameters(txt_path: Path) -> dict:
    """Read JSON or simple key/value parameter files."""

    text = txt_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        return {}

    try:
        payload = json.loads(text)
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        return payload

    parameters: dict[str, Any] = {}

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue

        parameters[key.strip()] = _safe_parse_scalar(value)

    return parameters


def _load_py_metadata(py_path: Path) -> dict:
    """
    Read literal metadata assignments without executing Python code.

    Dynamically computed assignments are not evaluated.
    """

    tree = ast.parse(
        py_path.read_text(encoding="utf-8-sig"),
        filename=str(py_path),
    )
    metadata: dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue

        if value_node is None:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue

            key = _normalized_key(target.id)

            if key not in METADATA_KEYS:
                continue

            try:
                metadata[key] = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError):
                continue

    return metadata


def _input_array(values: Any) -> np.ndarray:
    """Accept one scalar or a one-dimensional sequence of real numbers."""

    raw = np.asarray(values)

    if raw.dtype.kind not in "iuf":
        raise ValueError(
            "Model inputs must contain real numeric values "
            "without missing entries."
        )

    if raw.ndim > 1:
        raise ValueError(
            "This loader expects a scalar or one-dimensional "
            "single-feature input."
        )

    array = raw.astype(float).reshape(-1)

    if array.size == 0:
        raise ValueError("Model input cannot be empty.")

    if not np.isfinite(array).all():
        raise ValueError(
            "Model input contains NaN or infinite values."
        )

    return array


def _numeric_vector(
    values: Any,
    expected_length: int,
    label: str,
) -> np.ndarray:
    """Validate a single-output prediction or uncertainty vector."""

    raw = np.asarray(values)

    if raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{label} must contain real numeric values."
        )

    if raw.ndim == 0:
        array = raw.reshape(1)
    elif raw.ndim == 1:
        array = raw
    elif raw.ndim == 2 and raw.shape[1] == 1:
        array = raw[:, 0]
    else:
        raise ValueError(
            f"{label} must contain exactly one output per input."
        )

    array = array.astype(float)

    if len(array) != expected_length:
        raise ValueError(
            f"{label} returned {len(array)} values for "
            f"{expected_length} inputs."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{label} contains NaN or infinite values."
        )

    return array


class MockSurrogate:
    """
    Deterministic demonstration predictor.

    This preserves the previous fallback formula.
    It is not a physically validated process model.
    """

    def __init__(
        self,
        model_name: str,
        library_name: str,
        input_column: str = "h2_flow_kg_per_h",
        output_column: str = "prediction",
    ):
        self.model_name = model_name
        self.library_name = library_name
        self.input_column = input_column
        self.output_column = output_column

    def predict(self, h2_flow: Any) -> pd.DataFrame:
        array = _input_array(h2_flow)

        seed = (
            sum(
                ord(character)
                for character in (
                    f"{self.library_name}:{self.model_name}"
                )
            )
            % 17
        ) + 1

        slope = 0.08 + 0.01 * seed

        prediction = (
            slope * array
            + 0.01 * np.sin(array / max(seed, 1))
        )

        return pd.DataFrame(
            {
                self.input_column: array,
                "Prediction": prediction,
                "Predictive Std": np.full(len(array), np.nan),
                "Uncertainty Available": False,
                "Model Name": self.model_name,
                "Output Column": self.output_column,
            }
        )


class RuntimeJoblibPredictor:
    """Adapt a single-input, single-output joblib predictor."""

    def __init__(
        self,
        model_object: Any,
        model_name: str,
        input_column: str,
        output_column: str,
    ):
        self.model_object = model_object
        self.model_name = model_name
        self.input_column = input_column
        self.output_column = output_column

        if input_column in {
            "Prediction",
            "Predictive Std",
            "Model Name",
            "Output Column",
            "Uncertainty Available",
        }:
            raise ValueError(
                "The model input column conflicts with a reserved "
                "prediction-result column."
            )

    def _predict_raw(self, array: np.ndarray) -> Any:
        payloads = (
            pd.DataFrame({self.input_column: array}),
            array.reshape(-1, 1),
            array,
        )
        failures: list[str] = []

        for payload in payloads:
            try:
                return self.model_object.predict(payload)
            except Exception as exc:
                failures.append(
                    f"{type(exc).__name__}: {exc}"
                )

        raise RuntimeError(
            f"Model {self.model_name!r} could not predict using "
            "any supported input representation. "
            + " | ".join(failures)
        )

    def predict(self, h2_flow: Any) -> pd.DataFrame:
        array = _input_array(h2_flow)
        raw = self._predict_raw(array)
        uncertainty = None

        if isinstance(raw, tuple) and len(raw) == 2:
            prediction, uncertainty = raw

        elif isinstance(raw, pd.DataFrame):
            if raw.columns.duplicated().any():
                raise ValueError(
                    f"Model {self.model_name!r} returned "
                    "duplicate output column names."
                )

            if "Prediction" in raw.columns:
                prediction = raw["Prediction"]
            elif self.output_column in raw.columns:
                prediction = raw[self.output_column]
            else:
                candidates = [
                    column
                    for column in raw.columns
                    if column not in {
                        self.input_column,
                        "Predictive Std",
                        "Model Name",
                        "Output Column",
                        "Uncertainty Available",
                    }
                ]

                if len(candidates) != 1:
                    raise ValueError(
                        f"Model {self.model_name!r} returned "
                        "an ambiguous prediction table."
                    )

                prediction = raw[candidates[0]]

            if "Predictive Std" in raw.columns:
                uncertainty = raw["Predictive Std"]

        else:
            prediction = raw

        prediction_values = _numeric_vector(
            prediction,
            len(array),
            f"Predictions from {self.model_name}",
        )

        uncertainty_available = uncertainty is not None

        if uncertainty_available:
            std_values = _numeric_vector(
                uncertainty,
                len(array),
                f"Predictive standard deviations from {self.model_name}",
            )

            if (std_values < 0).any():
                raise ValueError(
                    "Predictive standard deviations cannot be negative."
                )
        else:
            std_values = np.full(len(array), np.nan)

        return pd.DataFrame(
            {
                self.input_column: array,
                "Prediction": prediction_values,
                "Predictive Std": std_values,
                "Uncertainty Available": uncertainty_available,
                "Model Name": self.model_name,
                "Output Column": self.output_column,
            }
        )


class SurrogateBundle:
    """Predictor, metadata, package status, and range diagnostics."""

    def __init__(
        self,
        model_name: str,
        library_name: str,
        predictor: Any,
        metadata: dict,
        parameters: dict,
        package_status: dict,
    ):
        self.model_name = model_name
        self.library_name = library_name
        self.predictor = predictor
        self.metadata = metadata
        self.parameters = parameters
        self.package_status = package_status

        self.warnings: list[str] = list(
            package_status.get("warnings", [])
        )
        self._bounds: dict[str, float | None] = {}

        bound_names = {
            "domain_min": ("train_x_min",),
            "domain_max": ("train_x_max",),
            "output_min": ("output_min", "train_y_min"),
            "output_max": ("output_max", "train_y_max"),
        }

        for label, aliases in bound_names.items():
            raw_value = _lookup(
                parameters,
                metadata,
                *aliases,
            )

            if raw_value is None:
                self._bounds[label] = None
                continue

            try:
                if isinstance(raw_value, bool):
                    raise ValueError("Boolean bound")

                numeric = float(raw_value)

                if not np.isfinite(numeric):
                    raise ValueError("Non-finite bound")

                self._bounds[label] = numeric

            except (TypeError, ValueError, OverflowError):
                self._bounds[label] = None
                self.warnings.append(
                    f"{model_name}: invalid {label} metadata "
                    f"{raw_value!r}; this bound will not be checked."
                )

        for lower_key, upper_key in (
            ("domain_min", "domain_max"),
            ("output_min", "output_max"),
        ):
            lower = self._bounds[lower_key]
            upper = self._bounds[upper_key]

            if (
                lower is not None
                and upper is not None
                and lower > upper
            ):
                self._bounds[lower_key] = None
                self._bounds[upper_key] = None
                self.warnings.append(
                    f"{model_name}: reversed {lower_key}/"
                    f"{upper_key} bounds were disabled."
                )

    @property
    def input_column(self) -> str:
        return str(
            _lookup(
                self.parameters,
                self.metadata,
                "input_column",
                default="h2_flow_kg_per_h",
            )
        )

    @property
    def output_column(self) -> str:
        return str(
            _lookup(
                self.parameters,
                self.metadata,
                "output_column",
                default=self.model_name,
            )
        )

    @property
    def input_unit(self) -> str:
        return str(
            _lookup(
                self.parameters,
                self.metadata,
                "input_unit",
                default="",
            )
        )

    @property
    def output_unit(self) -> str:
        return str(
            _lookup(
                self.parameters,
                self.metadata,
                "output_unit",
                "unit",
                default="",
            )
        )

    @property
    def domain_min(self) -> float | None:
        return self._bounds["domain_min"]

    @property
    def domain_max(self) -> float | None:
        return self._bounds["domain_max"]

    @property
    def output_min(self) -> float | None:
        return self._bounds["output_min"]

    @property
    def output_max(self) -> float | None:
        return self._bounds["output_max"]

    @property
    def runtime_mode(self) -> str:
        return (
            "runtime"
            if self.package_status.get("ready_for_runtime")
            else "mock"
        )

    @staticmethod
    def _range_status(
        value: float,
        lower: float | None,
        upper: float | None,
    ) -> bool | None:
        """
        False means outside a known bound.
        True means inside a fully defined range.
        None means the available bounds cannot establish full validity.
        """

        if lower is not None and value < lower:
            return False

        if upper is not None and value > upper:
            return False

        if lower is not None and upper is not None:
            return True

        return None

    def predict(self, h2_flow: Any) -> pd.DataFrame:
        """
        Predict without clipping values to declared training ranges.

        Range warnings accompany results.
        Predictor failures are not replaced by mock predictions here.
        """

        array = _input_array(h2_flow)
        result = self.predictor.predict(array).copy()

        predictions = _numeric_vector(
            result["Prediction"],
            len(array),
            f"Predictions from {self.model_name}",
        )

        input_status: list[bool | None] = []
        output_status: list[bool | None] = []
        row_warnings: list[str] = []

        for input_value, output_value in zip(array, predictions):
            messages = list(self.warnings)

            messages.extend(
                model_value_warnings(
                    model_name=self.model_name,
                    variable_name=self.input_column,
                    value=float(input_value),
                    lower=self.domain_min,
                    upper=self.domain_max,
                    unit=self.input_unit,
                    value_kind="input",
                )
            )

            messages.extend(
                model_value_warnings(
                    model_name=self.model_name,
                    variable_name=self.output_column,
                    value=float(output_value),
                    lower=self.output_min,
                    upper=self.output_max,
                    unit=self.output_unit,
                    value_kind="output",
                )
            )

            input_status.append(
                self._range_status(
                    float(input_value),
                    self.domain_min,
                    self.domain_max,
                )
            )
            output_status.append(
                self._range_status(
                    float(output_value),
                    self.output_min,
                    self.output_max,
                )
            )
            row_warnings.append(
                " | ".join(dict.fromkeys(messages))
            )

        result["Input In Domain"] = pd.array(
            input_status,
            dtype="boolean",
        )
        result["Output In Range"] = pd.array(
            output_status,
            dtype="boolean",
        )
        result["Runtime Mode"] = self.runtime_mode
        result["Warnings"] = row_warnings

        return result


# PATCH: new class. Applies the packaged scaler before the Gaussian
# process regressor, as the model's own .py file does.
class ScaledGPRPredictor:
    """Apply the packaged scaler before a Gaussian-process regressor."""

    def __init__(self, scaler: Any, gpr: Any):
        self.scaler = scaler
        self.gpr = gpr

    def predict(self, values: Any):
        array = np.asarray(values, dtype=float)

        if array.ndim == 1:
            array = array.reshape(-1, 1)

        scaled = self.scaler.transform(array)

        return self.gpr.predict(scaled, return_std=True)


def _extract_predictor_candidate(model_object: Any) -> Any:
    if callable(getattr(model_object, "predict", None)):
        return model_object

    # PATCH: packaged dictionaries with a scaler and a GPR must be
    # evaluated with scaling; otherwise the GPR would receive raw inputs.
    if isinstance(model_object, dict):
        scaler = model_object.get("scaler")
        gpr = model_object.get("gpr")

        if (
            callable(getattr(scaler, "transform", None))
            and callable(getattr(gpr, "predict", None))
        ):
            return ScaledGPRPredictor(scaler, gpr)

    if isinstance(model_object, dict):
        preferred_keys = [
            "model",
            "predictor",
            "estimator",
            "pipeline",
            "regressor",
            "surrogate",
            "best_model",
        ]

        for key in preferred_keys:
            candidate = model_object.get(key)

            if callable(getattr(candidate, "predict", None)):
                return candidate

        for candidate in model_object.values():
            if callable(getattr(candidate, "predict", None)):
                return candidate

    return None


def load_surrogate_bundle(
    project_root: Path,
    model_name: str,
    library_name: str,
    allow_mock: bool = True,
) -> SurrogateBundle:
    """
    Load a trusted single-feature surrogate package.

    allow_mock=True preserves demonstration fallback.
    Set allow_mock=False when a real predictor is mandatory.

    joblib.load can execute code. Only trusted packages may be loaded.
    """

    registry = ModelRegistry(project_root)
    bundle_dir = registry.get_bundle_dir(
        library_name,
        model_name,
    )
    files = registry.get_bundle_files(
        library_name,
        model_name,
    )

    missing = [
        suffix
        for suffix, paths in files.items()
        if not paths
    ]
    ambiguous = [
        suffix
        for suffix, paths in files.items()
        if len(paths) > 1
    ]

    messages = list(registry.warnings)
    metadata = registry.get_model_metadata(model_name)
    parameters: dict[str, Any] = {}

    # Precedence:
    # TXT parameters > metadata.json > literal Python metadata > catalog.
    if len(files[".py"]) == 1:
        try:
            metadata.update(
                _load_py_metadata(files[".py"][0])
            )
        except (OSError, UnicodeError, SyntaxError) as exc:
            messages.append(
                f"{model_name}: Python metadata could not be read: "
                f"{type(exc).__name__}: {exc}."
            )

    metadata_path = bundle_dir / "metadata.json"

    if metadata_path.is_file():
        try:
            if metadata_path.resolve().parent != bundle_dir:
                raise ValueError(
                    "metadata.json resolves outside the model bundle."
                )

            payload = json.loads(
                metadata_path.read_text(encoding="utf-8-sig")
            )

            if not isinstance(payload, dict):
                raise ValueError(
                    "metadata.json must contain a JSON object."
                )

            metadata.update(payload)

        except (OSError, UnicodeError, ValueError) as exc:
            messages.append(
                f"{model_name}: JSON metadata could not be read: "
                f"{type(exc).__name__}: {exc}."
            )

    parameters_readable = True

    if len(files[".txt"]) == 1:
        try:
            parameters = _load_txt_parameters(files[".txt"][0])
        except (OSError, UnicodeError, ValueError) as exc:
            parameters_readable = False
            messages.append(
                f"{model_name}: parameter file could not be read: "
                f"{type(exc).__name__}: {exc}."
            )

    input_column = str(
        _lookup(
            parameters,
            metadata,
            "input_column",
            default="h2_flow_kg_per_h",
        )
    )
    output_column = str(
        _lookup(
            parameters,
            metadata,
            "output_column",
            default=model_name,
        )
    )

    if not _lookup(parameters, metadata, "input_unit"):
        messages.append(
            f"{model_name}: input units are not declared. "
            "The caller must verify compatibility; "
            "no automatic unit conversion is applied."
        )

    output_unit = _lookup(
        parameters,
        metadata,
        "output_unit",
        "unit",
        default="",
    )

    if not output_unit or "_or_" in str(output_unit):
        messages.append(
            f"{model_name}: output units are not fully specified. "
            "Verify them before using the prediction in physical "
            "balances or economic calculations."
        )

    package_status = {
        "ready_for_runtime": False,
        "missing_files": missing,
        "ambiguous_files": ambiguous,
        "runtime_error": None,
        "warnings": messages,
    }

    predictor = None

    if not missing and not ambiguous and parameters_readable:
        try:
            raw_model = joblib.load(files[".joblib"][0])
            model_object = _extract_predictor_candidate(raw_model)

            if model_object is None:
                raise TypeError(
                    "The joblib object does not contain a predictor "
                    "with a callable predict() method."
                )

            predictor = RuntimeJoblibPredictor(
                model_object=model_object,
                model_name=model_name,
                input_column=input_column,
                output_column=output_column,
            )
            package_status["ready_for_runtime"] = True

        except Exception as exc:
            package_status["runtime_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    if predictor is None:
        reasons: list[str] = []

        if missing:
            reasons.append(
                "missing files: " + ", ".join(missing)
            )

        if ambiguous:
            reasons.append(
                "ambiguous files: " + ", ".join(ambiguous)
            )

        if not parameters_readable:
            reasons.append("unreadable parameter file")

        if package_status["runtime_error"]:
            reasons.append(package_status["runtime_error"])

        reason = "; ".join(reasons) or "no usable predictor"

        if not allow_mock:
            raise RuntimeError(
                f"Model {model_name!r} cannot be loaded: {reason}."
            )

        messages.append(
            f"{model_name}: DEMONSTRATION MODE. "
            f"A mock predictor is being used because of {reason}. "
            "Its predictions are not validated process results."
        )

        predictor = MockSurrogate(
            model_name=model_name,
            library_name=library_name,
            input_column=input_column,
            output_column=output_column,
        )

    package_status["warnings"] = list(dict.fromkeys(messages))

    return SurrogateBundle(
        model_name=model_name,
        library_name=library_name,
        predictor=predictor,
        metadata=metadata,
        parameters=parameters,
        package_status=package_status,
    )

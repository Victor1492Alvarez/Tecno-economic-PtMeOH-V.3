from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_LIBRARIES = [
    "variable_h2_constant_co2",
    "variable_h2_variable_co2",
]

MODEL_GROUPS = [
    (
        "thermal_duty",
        "W",
        [
            "Model_Duty_RTR_Watt",
            "Model_Duty_FEHE_Watt",
            "Model_Duty_HX4_Watt",
            "Model_Duty_HX3_Watt",
            "Model_Duty_HX1_Watt",
        ],
    ),
    (
        "electrical_load",
        "W",
        [
            "Model_Power_C3_Watt",
            "Model_Power_C2_Watt",
            "Model_Power_C1_Watt",
        ],
    ),
    (
        "recycle_stream",
        "kmol/h_or_model_unit",
        [
            "Model_RECLP_CO2",
            "Model_RECLP_H2",
            "Model_RECLP_MeOH",
        ],
    ),
    (
        "purge_stream",
        "kmol/h_or_model_unit",
        [
            "Model_Purge_CO2",
            "Model_Purge_H2",
            "Model_Purge_MeOH",
        ],
    ),
    (
        "product_stream",
        "t/h_or_model_unit",
        ["Model_Prod_MeOH"],
    ),
]


def _build_embedded_catalog() -> dict:
    """Build a fallback catalog containing the original model names."""

    model_order: list[str] = []
    models: dict[str, dict] = {}

    for category, unit, names in MODEL_GROUPS:
        for name in names:
            model_order.append(name)
            models[name] = {
                "libraries": list(DEFAULT_LIBRARIES),
                "category": category,
                "unit": unit,
            }

    return {
        "libraries": list(DEFAULT_LIBRARIES),
        "model_order": model_order,
        "models": models,
    }


EMBEDDED_CATALOG = _build_embedded_catalog()


DISCOVERY_COLUMNS = [
    "library",
    "model_name",
    "bundle_dir",
    "joblib_found",
    "py_found",
    "txt_found",
    "ready_for_runtime",
    "ready_for_qa",
    "missing_files",
    "ambiguous_files",
    "package_warnings",
    "category",
    "unit",
]


def _safe_component(value: str, label: str) -> str:
    """Validate a single folder name, not an arbitrary relative path."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", ":", "\x00"))
    ):
        raise ValueError(
            f"{label} must be a nonempty folder name "
            "without path separators or traversal components."
        )

    return value


class ModelRegistry:
    """
    Discover model packages without importing or executing their code.

    The runtime package contract remains:
        one .joblib file
        one .py file, excluding __init__.py
        one .txt parameter file

    File presence does not prove that a predictor can be loaded,
    that its units are correct, or that its predictions are valid.
    Those checks belong to the loader and simulation layers.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.models_root = self.project_root / "models" / "packages"
        self.catalog_path = self._resolve_catalog_path()

        self.warnings: list[str] = []
        self.catalog_source = ""
        self._catalog_payload: dict = {}
        self._library_map: dict[str, list[str]] = {}

        self.refresh()

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            LOGGER.warning(message)

    def _candidate_catalog_paths(self) -> list[Path]:
        candidates = [
            self.project_root / "catalog.json",
            self.project_root / "models" / "catalog.json",
            self.project_root / "models" / "packages" / "catalog.json",
            self.project_root.parent / "catalog.json",
        ]

        unique: list[Path] = []
        seen: set[str] = set()

        for path in candidates:
            key = str(path.resolve())

            if key not in seen:
                seen.add(key)
                unique.append(path)

        return unique

    def _resolve_catalog_path(self) -> Path:
        for path in self._candidate_catalog_paths():
            if path.is_file():
                return path

        return self.project_root / "catalog.json"

    @staticmethod
    def _parse_library_map(
        payload: dict,
    ) -> dict[str, list[str]]:
        """Accept the three catalog layouts supported previously."""

        if not isinstance(payload, dict):
            raise ValueError("The catalog root must be a JSON object.")

        libraries = payload.get("libraries")

        if isinstance(libraries, dict):
            raw_map = libraries

        elif isinstance(libraries, list):
            models = payload.get("models")

            if not isinstance(models, dict):
                raise ValueError(
                    "A catalog with a libraries list "
                    "must also contain a models object."
                )

            raw_map = {}

            for library in libraries:
                _safe_component(library, "Library name")
                raw_map.setdefault(library, [])

            for model_name, metadata in models.items():
                _safe_component(model_name, "Model name")

                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Metadata for {model_name!r} "
                        "must be a JSON object."
                    )

                memberships = metadata.get("libraries", [])

                if not isinstance(memberships, list):
                    raise ValueError(
                        f"The libraries field for {model_name!r} "
                        "must be a list."
                    )

                for library in memberships:
                    _safe_component(library, "Library name")
                    raw_map.setdefault(library, []).append(model_name)

        else:
            # Legacy layout:
            # {"library_name": ["Model_A", "Model_B"]}
            raw_map = {
                key: value
                for key, value in payload.items()
                if isinstance(value, list)
                and key not in {"libraries", "model_order"}
            }

            if not raw_map:
                raise ValueError(
                    "The catalog has no recognized library definitions."
                )

        order = payload.get("model_order", [])

        if not isinstance(order, list) or any(
            not isinstance(name, str) for name in order
        ):
            raise ValueError("model_order must be a list of strings.")

        order_index = {
            name: index
            for index, name in enumerate(order)
        }

        parsed: dict[str, list[str]] = {}

        for library, names in raw_map.items():
            _safe_component(library, "Library name")

            if not isinstance(names, list):
                raise ValueError(
                    f"Model names for {library!r} must be a list."
                )

            for name in names:
                _safe_component(name, "Model name")

            # Preserve listed order unless model_order supplies a priority.
            unique_names = list(dict.fromkeys(names))
            original_order = {
                name: index
                for index, name in enumerate(unique_names)
            }

            parsed[library] = sorted(
                unique_names,
                key=lambda name: (
                    order_index.get(name, len(order)),
                    original_order[name],
                ),
            )

        return parsed

    def refresh(self) -> None:
        """Reload catalog definitions after a catalog change."""

        self.warnings.clear()
        self.catalog_path = self._resolve_catalog_path()

        if self.catalog_path.is_file():
            try:
                payload = json.loads(
                    self.catalog_path.read_text(encoding="utf-8")
                )
                library_map = self._parse_library_map(payload)
                self.catalog_source = str(self.catalog_path)

            except (OSError, UnicodeError, ValueError) as exc:
                self._warn(
                    f"The catalog could not be used: "
                    f"{self.catalog_path}. Reason: {exc}. "
                    "The embedded fallback catalog is being used."
                )
                payload = deepcopy(EMBEDDED_CATALOG)
                library_map = self._parse_library_map(payload)
                self.catalog_source = "embedded_fallback"
        else:
            self._warn(
                "No catalog.json was found. "
                "The embedded fallback catalog is being used."
            )
            payload = deepcopy(EMBEDDED_CATALOG)
            library_map = self._parse_library_map(payload)
            self.catalog_source = "embedded_fallback"

        self._catalog_payload = payload
        self._library_map = library_map

    def _load_catalog_payload(self) -> dict:
        """Return a copy of the currently loaded catalog."""

        return deepcopy(self._catalog_payload)

    def _catalog_library_map(self) -> dict[str, list[str]]:
        return deepcopy(self._library_map)

    def _checked_directory(self, path: Path) -> Path:
        """Prevent model directories from resolving outside models_root."""

        root = self.models_root.resolve()
        resolved = path.resolve()

        if not resolved.is_relative_to(root):
            raise ValueError(
                "The model directory resolves outside models/packages."
            )

        return resolved

    def get_bundle_dir(
        self,
        library_name: str,
        model_name: str,
    ) -> Path:
        library = _safe_component(library_name, "Library name")
        model = _safe_component(model_name, "Model name")

        return self._checked_directory(
            self.models_root / library / model
        )

    def _fs_library_map(self) -> dict[str, list[str]]:
        """Discover folders dynamically, including newly uploaded models."""

        discovered: dict[str, list[str]] = {}

        if not self.models_root.is_dir():
            return discovered

        for library_dir in sorted(self.models_root.iterdir()):
            if (
                not library_dir.is_dir()
                or library_dir.name.startswith((".", "__"))
            ):
                continue

            try:
                library = _safe_component(
                    library_dir.name,
                    "Library name",
                )
                self._checked_directory(library_dir)
            except ValueError as exc:
                self._warn(f"Skipped model library: {exc}")
                continue

            names: list[str] = []

            for model_dir in sorted(library_dir.iterdir()):
                if (
                    not model_dir.is_dir()
                    or model_dir.name.startswith((".", "__"))
                ):
                    continue

                try:
                    model = _safe_component(
                        model_dir.name,
                        "Model name",
                    )
                    self.get_bundle_dir(library, model)
                except ValueError as exc:
                    self._warn(f"Skipped model folder: {exc}")
                    continue

                names.append(model)

            discovered[library] = names

        return discovered

    def _merged_library_map(self) -> dict[str, list[str]]:
        catalog_map = self._catalog_library_map()
        filesystem_map = self._fs_library_map()

        merged: dict[str, list[str]] = {}

        for library in sorted(
            set(catalog_map) | set(filesystem_map)
        ):
            registered = catalog_map.get(library, [])
            additional = sorted(
                set(filesystem_map.get(library, []))
                - set(registered)
            )
            merged[library] = registered + additional

        return merged

    def get_library_names(self) -> list[str]:
        merged = self._merged_library_map()
        return sorted(merged) if merged else list(DEFAULT_LIBRARIES)

    def get_models_by_library(
        self,
        library_name: str,
    ) -> list[str]:
        library = _safe_component(library_name, "Library name")
        return list(self._merged_library_map().get(library, []))

    def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        """
        Return catalog metadata without importing model Python files.

        Folder-discovered models may have no catalog metadata.
        """

        model = _safe_component(model_name, "Model name")
        models = self._catalog_payload.get("models", {})

        if not isinstance(models, dict):
            return {}

        metadata = models.get(model, {})
        return deepcopy(metadata) if isinstance(metadata, dict) else {}

    def get_bundle_files(
        self,
        library_name: str,
        model_name: str,
    ) -> dict[str, list[Path]]:
        """List top-level runtime files without choosing ambiguously."""

        bundle_dir = self.get_bundle_dir(
            library_name,
            model_name,
        )

        files: dict[str, list[Path]] = {
            ".joblib": [],
            ".py": [],
            ".txt": [],
        }

        if not bundle_dir.is_dir():
            return files

        for path in sorted(bundle_dir.iterdir()):
            if not path.is_file() or path.name == "__init__.py":
                continue

            suffix = path.suffix.lower()

            if suffix not in files:
                continue

            # Runtime files must belong directly to the bundle.
            if path.resolve().parent != bundle_dir:
                self._warn(
                    f"Skipped external runtime file: {path}."
                )
                continue

            files[suffix].append(path)

        return files

    def _inspect_bundle(
        self,
        library_name: str,
        model_name: str,
    ) -> dict:
        bundle_dir = self.get_bundle_dir(
            library_name,
            model_name,
        )
        files = self.get_bundle_files(
            library_name,
            model_name,
        )

        missing = [
            suffix
            for suffix, paths in files.items()
            if len(paths) == 0
        ]
        ambiguous = [
            suffix
            for suffix, paths in files.items()
            if len(paths) > 1
        ]

        metadata = self.get_model_metadata(model_name)
        package_warnings: list[str] = []

        if ambiguous:
            package_warnings.append(
                "Multiple candidate files were found for: "
                + ", ".join(ambiguous)
                + ". Keep one runtime file of each required type."
            )

        unit = metadata.get("unit", "")

        if not unit or (
            isinstance(unit, str) and "_or_" in unit
        ):
            package_warnings.append(
                "The physical output unit is not fully specified "
                "in the catalog."
            )

        return {
            "library": library_name,
            "model_name": model_name,
            "bundle_dir": str(bundle_dir),
            "joblib_found": bool(files[".joblib"]),
            "py_found": bool(files[".py"]),
            "txt_found": bool(files[".txt"]),
            "ready_for_runtime": not missing and not ambiguous,
            "ready_for_qa": (
                len(files[".py"]) == 1
                and len(files[".txt"]) == 1
            ),
            "missing_files": ", ".join(missing),
            "ambiguous_files": ", ".join(ambiguous),
            "package_warnings": " ".join(package_warnings),
            "category": metadata.get("category", ""),
            "unit": unit,
        }

    def discover_packages(self) -> pd.DataFrame:
        rows: list[dict] = []

        for library, names in self._merged_library_map().items():
            if not names:
                rows.append(
                    {
                        "library": library,
                        "model_name": "",
                        "bundle_dir": str(
                            self.models_root / library
                        ),
                        "joblib_found": False,
                        "py_found": False,
                        "txt_found": False,
                        "ready_for_runtime": False,
                        "ready_for_qa": False,
                        "missing_files": "no registered models",
                        "ambiguous_files": "",
                        "package_warnings": "",
                        "category": "",
                        "unit": "",
                    }
                )
                continue

            for model_name in names:
                rows.append(
                    self._inspect_bundle(
                        library,
                        model_name,
                    )
                )

        return (
            pd.DataFrame(rows, columns=DISCOVERY_COLUMNS)
            .sort_values(["library", "model_name"])
            .reset_index(drop=True)
        )

    def get_library_fingerprint(
        self,
        library_name: str,
    ) -> str:
        """
        Hash catalog definitions and runtime-related package contents.

        Call when identifying a run or checking for model changes,
        not inside the timestep loop.

        Documentation PDFs and unrelated assets are excluded.
        """

        library = _safe_component(library_name, "Library name")
        model_names = self.get_models_by_library(library)
        digest = hashlib.sha256()

        def add_bytes(value: bytes) -> None:
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

        definition = {
            "library": library,
            "models": model_names,
            "metadata": {
                name: self.get_model_metadata(name)
                for name in model_names
            },
        }

        add_bytes(
            json.dumps(
                definition,
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        )

        relevant_suffixes = {
            ".joblib",
            ".py",
            ".txt",
            ".json",
            ".xlsx",
        }

        for model_name in model_names:
            bundle_dir = self.get_bundle_dir(
                library,
                model_name,
            )

            add_bytes(model_name.encode("utf-8"))
            add_bytes(
                b"present" if bundle_dir.is_dir() else b"missing"
            )

            if not bundle_dir.is_dir():
                continue

            for path in sorted(bundle_dir.iterdir()):
                if (
                    not path.is_file()
                    or path.suffix.lower() not in relevant_suffixes
                ):
                    continue

                if path.resolve().parent != bundle_dir:
                    raise ValueError(
                        "Cannot fingerprint a runtime asset "
                        f"outside its model bundle: {path}."
                    )

                add_bytes(path.name.encode("utf-8"))
                file_digest = hashlib.sha256()

                with path.open("rb") as source:
                    for chunk in iter(
                        lambda: source.read(1024 * 1024),
                        b"",
                    ):
                        file_digest.update(chunk)

                add_bytes(file_digest.digest())

        return digest.hexdigest()

    # Compatibility aliases used by earlier application versions.

    def catalog(self) -> pd.DataFrame:
        return self.discover_packages()

    def discover(self) -> pd.DataFrame:
        return self.discover_packages()

    def list_library_names(self) -> list[str]:
        return self.get_library_names()

    def list_models_by_library(
        self,
        library_name: str,
    ) -> list[str]:
        return self.get_models_by_library(library_name)

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
from typing import Any, Callable
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from application.case_runner import CaseRunner
from domain.data_models import (
    CO2FeedInputs,
    CoproductInputs,
    OptimizationInputs,
    SimulationArtifacts,
    TEACapexInputs,
    TEAInputs,
    TEAOpexInputs,
    TEAPriceInputs,
    UtilityInputs,
)
from domain.renewable_profile import (
    SUPPORTED_PROFILE_UNITS,
    infer_profile_meta_from_normalized,
    normalize_renewable_profile,
)
from domain.validators import (
    InputValidationError,
    inspect_case_inputs,
    validate_optimization_inputs,
)
from infrastructure.model_registry import ModelRegistry
from presentation.plotting import (
    PALETTE,
    capex_donut,
    cash_flow_chart,
    heatmap,
    line_profile,
    tornado,
    variable_cost_donut,
)


st.set_page_config(
    page_title="PtMeOH Sizing Tool — Thesis Edition",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent

PERSIST_ROOT = PROJECT_ROOT / "user_data"
PROFILE_STORE = PERSIST_ROOT / "renewable_profiles"
MODEL_ARCHIVE_STORE = PERSIST_ROOT / "model_archives"

EXCEL_MIME = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

MAX_MODEL_ZIP_BYTES = 128 * 1024 * 1024
MAX_MODEL_EXTRACTED_BYTES = 512 * 1024 * 1024
MAX_MODEL_ZIP_MEMBERS = 100
MAX_MODEL_COMPRESSION_RATIO = 250.0

# Enable only on deployments where access is restricted to trusted users.
# A trust-confirmation checkbox is not an authentication mechanism.
MODEL_UPLOADS_ENABLED = True

ProgressCallback = Callable[[str, int, int], None] | None


# ---------------------------------------------------------------------
# General helpers and session state.
# ---------------------------------------------------------------------


def slugify(text: str) -> str:
    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(text).strip(),
    )
    return cleaned.strip("_") or "asset"


def json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()

    return str(value)


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        default=json_default,
    )


def content_signature(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            default=json_default,
        ).encode("utf-8")
    ).hexdigest()


def initialize_session_state() -> None:
    defaults = {
        "simulation": None,
        "optimization": None,
        "sensitivities": None,
        "case_signature": None,
        "renewable_profile_df": None,
        "renewable_profile_meta": None,
        "renewable_profile_name": "default_synthetic_profile",
        "active_profile_origin": None,
        "telemetry_lines": [],
        "prepared_downloads": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_results_for_new_case(signature: str) -> None:
    if st.session_state.get("case_signature") == signature:
        return

    st.session_state["case_signature"] = signature
    st.session_state["simulation"] = None
    st.session_state["optimization"] = None
    st.session_state["sensitivities"] = None
    st.session_state["prepared_downloads"] = {}


def clear_active_profile() -> None:
    st.session_state["renewable_profile_df"] = None
    st.session_state["renewable_profile_meta"] = None
    st.session_state["renewable_profile_name"] = (
        "default_synthetic_profile"
    )
    st.session_state["active_profile_origin"] = None


def flash_message(kind: str, text: str) -> None:
    st.session_state["flash_message"] = (kind, text)


def render_flash_message() -> None:
    message = st.session_state.pop("flash_message", None)

    if message is None:
        return

    kind, text = message
    renderer = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }.get(kind, st.info)

    renderer(text)


def render_warnings(
    messages: list[str],
    title: str = "Warnings and assumptions",
) -> None:
    unique = list(
        dict.fromkeys(
            str(message)
            for message in messages
            if message
        )
    )

    if not unique:
        return

    st.warning(
        f"{len(unique)} warnings or assumptions apply. "
        "Review them before interpreting these results."
    )

    with st.expander(
        f"{title} ({len(unique)})",
        expanded=False,
    ):
        for message in unique:
            st.write(f"- {message}")


# ---------------------------------------------------------------------
# Renewable profile input and persistence.
# ---------------------------------------------------------------------


def read_tabular_file(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
            )

        if suffix == ".tsv":
            return pd.read_csv(uploaded_file, sep="\t")

        if suffix == ".xlsx":
            return pd.read_excel(
                uploaded_file,
                engine="openpyxl",
            )

        if suffix == ".xls":
            return pd.read_excel(
                uploaded_file,
                engine="xlrd",
            )

        if suffix == ".parquet":
            return pd.read_parquet(uploaded_file)

    except ImportError as exc:
        raise ValueError(
            f"The dependency required for {suffix} files "
            "is not installed. Use CSV/XLSX or install "
            "the updated project requirements."
        ) from exc

    raise ValueError(
        f"Unsupported renewable-profile file type: {suffix!r}."
    )


def choose_default_column(
    columns: list[str],
    keywords: list[str],
) -> int:
    lowered = [str(column).lower() for column in columns]

    for keyword in keywords:
        for index, column in enumerate(lowered):
            if keyword in column:
                return index

    return 0


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}_",
        dir=path.parent,
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)

        os.replace(temporary, path)

    finally:
        temporary.unlink(missing_ok=True)


def save_profile_assets(
    source_file,
    normalized_df: pd.DataFrame,
    profile_label: str,
    persist_enabled: bool,
    metadata: dict | None = None,
) -> None:
    if not persist_enabled:
        return

    csv_bytes = normalized_df.to_csv(
        index=False
    ).encode("utf-8")

    fingerprint = hashlib.sha256(csv_bytes).hexdigest()[:12]
    stem = f"{slugify(profile_label)}_{fingerprint}"

    normalized_path = PROFILE_STORE / f"{stem}__normalized.csv"

    atomic_write_bytes(
        normalized_path,
        csv_bytes,
    )

    atomic_write_bytes(
        normalized_path.with_suffix(".meta.json"),
        json_text(metadata or {}).encode("utf-8"),
    )

    if source_file is not None:
        suffix = Path(source_file.name).suffix.lower() or ".bin"

        atomic_write_bytes(
            PROFILE_STORE / f"{stem}__source{suffix}",
            source_file.getvalue(),
        )


def list_saved_profiles() -> list[Path]:
    if not PROFILE_STORE.is_dir():
        return []

    return sorted(PROFILE_STORE.glob("*__normalized.csv"))


def load_saved_profile(path: Path) -> tuple[pd.DataFrame, dict]:
    resolved = path.resolve()

    if resolved.parent != PROFILE_STORE.resolve():
        raise ValueError(
            "The saved profile must belong to the profile store."
        )

    frame = pd.read_csv(resolved)

    required = {"timestamp", "renewable_power_mw"}

    if not required.issubset(frame.columns):
        raise ValueError(
            "The saved profile is missing required columns."
        )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="raise",
    )
    frame["renewable_power_mw"] = pd.to_numeric(
        frame["renewable_power_mw"],
        errors="raise",
    )

    metadata_path = resolved.with_suffix(".meta.json")
    stored_meta: dict[str, Any] = {}

    if metadata_path.is_file():
        stored_meta = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if not isinstance(stored_meta, dict):
            raise ValueError(
                "The saved profile metadata must be a JSON object."
            )

    declared_step = stored_meta.get("time_step_h")

    if len(frame) == 1 and declared_step is None:
        raise ValueError(
            "This single-record saved profile has no interval "
            "metadata. Upload it again and specify its interval."
        )

    metadata = infer_profile_meta_from_normalized(
        df=frame,
        source_name=resolved.name,
        source_kind="saved",
        raw_rows=stored_meta.get("raw_rows", len(frame)),
        unit_mode=stored_meta.get("unit_mode", "MW"),
        expanded_to_hourly=stored_meta.get(
            "expanded_to_hourly",
            False,
        ),
        time_step_h=declared_step,
    )

    metadata["saved_source_metadata"] = stored_meta

    return frame, metadata


# ---------------------------------------------------------------------
# Trusted model-package installation.
# ---------------------------------------------------------------------


def install_model_zip(
    project_root: Path,
    library_name: str,
    model_name: str,
    uploaded_zip,
    persist_enabled: bool,
    unit_declaration: dict | None = None,
) -> dict:
    """
    Replace one complete trusted model package.

    This validates archive structure, not executable-code safety.
    Python and joblib packages must come from trusted sources.
    """

    if not MODEL_UPLOADS_ENABLED:
        raise PermissionError(
            "Model installation is disabled on this deployment."
        )

    archive_bytes = uploaded_zip.getvalue()

    if len(archive_bytes) > MAX_MODEL_ZIP_BYTES:
        raise ValueError(
            "The compressed model archive exceeds the upload limit."
        )

    registry = ModelRegistry(project_root)
    target_dir = registry.get_bundle_dir(
        library_name,
        model_name,
    )

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and not target_dir.is_dir():
        raise ValueError(
            "The target model path exists but is not a directory."
        )

    stage = Path(
        tempfile.mkdtemp(
            prefix=".model_stage_",
            dir=target_dir.parent,
        )
    )
    backup = target_dir.parent / f".model_backup_{uuid4().hex}"

    written: list[str] = []
    installation_warnings: list[str] = []
    ignored_files: list[str] = []
    backup_created = False

    allowed_suffixes = {
        ".joblib",
        ".py",
        ".txt",
        ".json",
        ".xlsx",
        ".pdf",
    }

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
            ]

            if not members:
                raise ValueError(
                    "The model archive contains no files."
                )

            if len(members) > MAX_MODEL_ZIP_MEMBERS:
                raise ValueError(
                    "The model archive contains too many files."
                )

            total_size = sum(
                member.file_size for member in members
            )

            if total_size > MAX_MODEL_EXTRACTED_BYTES:
                raise ValueError(
                    "The extracted model archive exceeds the size limit."
                )

            seen_names: set[str] = set()

            for member in members:
                relative_name = member.filename.replace("\\", "/")
                parts = relative_name.split("/")
                filename = parts[-1]

                if (
                    not filename
                    or any(
                        part.startswith((".", "__MACOSX"))
                        for part in parts
                        if part
                    )
                ):
                    continue
                    
                if Path(filename).suffix.lower() not in allowed_suffixes:
                    ignored_files.append(filename)
                    continue

                mode = member.external_attr >> 16

                if stat.S_ISLNK(mode):
                    raise ValueError(
                        "Symbolic links are not allowed in model archives."
                    )

                if member.flag_bits & 0x1:
                    raise ValueError(
                        "Encrypted model archives are not supported."
                    )

                ratio = member.file_size / max(
                    member.compress_size,
                    1,
                )

                if ratio > MAX_MODEL_COMPRESSION_RATIO:
                    raise ValueError(
                        "The model archive has an excessive "
                        "compression ratio."
                    )

                normalized_name = filename.casefold()

                if normalized_name in seen_names:
                    raise ValueError(
                        "Flattening the archive would overwrite "
                        f"a duplicate filename: {filename!r}."
                    )

                seen_names.add(normalized_name)

                with archive.open(member) as source:
                    with (stage / filename).open("wb") as destination:
                        shutil.copyfileobj(source, destination)

                written.append(filename)

        for suffix in (".joblib", ".py", ".txt"):
            candidates = [
                path
                for path in stage.iterdir()
                if path.suffix.lower() == suffix
                and path.name != "__init__.py"
            ]

            if len(candidates) != 1:
                raise ValueError(
                    "A model package must contain exactly one "
                    f"runtime {suffix} file. Found {len(candidates)}."
                )

        if unit_declaration is not None:
            metadata_path = stage / "metadata.json"
            metadata_payload: dict[str, Any] = {}

            if metadata_path.is_file():
                loaded = json.loads(
                    metadata_path.read_text(encoding="utf-8-sig")
                )

                if not isinstance(loaded, dict):
                    raise ValueError(
                        "metadata.json must contain a JSON object."
                    )

                metadata_payload = loaded

            for key in ("input_unit", "output_unit"):
                existing = metadata_payload.get(key)

                if existing and existing != unit_declaration[key]:
                    installation_warnings.append(
                        f"metadata.json declared {key}={existing!r}; "
                        f"the selected value "
                        f"{unit_declaration[key]!r} was applied."
                    )

            metadata_payload.update(
                {
                    "quantity_kind": unit_declaration["quantity_kind"],
                    "input_unit": unit_declaration["input_unit"],
                    "output_unit": unit_declaration["output_unit"],
                    "unit_declared_by": "user_at_install",
                    "unit_declared_utc": unit_declaration["declared_utc"],
                }
            )

            metadata_path.write_text(
                json_text(metadata_payload),
                encoding="utf-8",
            )
         
        if persist_enabled:
            archive_digest = hashlib.sha256(
                archive_bytes
            ).hexdigest()[:12]

            archive_path = (
                MODEL_ARCHIVE_STORE
                / slugify(library_name)
                / f"{slugify(model_name)}_{archive_digest}.zip"
            )
            atomic_write_bytes(
                archive_path,
                archive_bytes,
            )

        if target_dir.exists():
            os.replace(target_dir, backup)
            backup_created = True

        try:
            os.replace(stage, target_dir)
        except Exception:
            if backup_created and backup.exists():
                os.replace(backup, target_dir)
            raise

        if backup_created and backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                installation_warnings.append(
                    f"The previous package backup could not be "
                    f"removed automatically: {exc}"
                )

        return {
            "library": library_name,
            "model_name": model_name,
            "target_dir": str(target_dir),
            "written_files": sorted(written),
            "written_count": len(written),
            "ignored_files": sorted(ignored_files),
            "unit_declaration": unit_declaration,
            "archive_saved": persist_enabled,
            "warnings": installation_warnings,
        }

    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------
# Execution monitoring.
# ---------------------------------------------------------------------


def append_telemetry_line(
    message: str,
    max_lines: int = 250,
) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines = st.session_state.setdefault("telemetry_lines", [])
    lines.append(f"{timestamp} | {message}")
    st.session_state["telemetry_lines"] = lines[-max_lines:]


def get_telemetry_text(last_n: int = 60) -> str:
    lines = st.session_state.get("telemetry_lines", [])

    return (
        "\n".join(lines[-last_n:])
        if lines
        else "No telemetry messages yet."
    )


class StreamlitTelemetryHandler(logging.Handler):
    """Capture logs only from the thread performing this UI action."""

    def __init__(self, refresh: Callable[[], None]):
        super().__init__(level=logging.INFO)
        self.owner_thread = threading.get_ident()
        self.refresh = refresh

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.owner_thread:
            return

        try:
            append_telemetry_line(self.format(record))
            self.refresh()
        except Exception:
            # Telemetry must not interrupt the numerical calculation.
            pass


@contextmanager
def capture_runtime_logs(
    runner: CaseRunner,
    refresh: Callable[[], None],
):
    handler = StreamlitTelemetryHandler(refresh)
    handler.setFormatter(
        logging.Formatter(
            "%(name)s | %(levelname)s | %(message)s"
        )
    )

    logger = runner.logger
    logger.addHandler(handler)

    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def format_progress_label(
    stage: str,
    current: int,
    total: int,
    time_step_h: float,
) -> tuple[float, str]:
    safe_total = max(int(total), 1)
    safe_current = max(0, min(int(current), safe_total))
    fraction = safe_current / safe_total

    if stage == "simulation":
        elapsed_h = safe_current * time_step_h
        total_h = safe_total * time_step_h

        label = (
            f"Simulation | {fraction:.1%} | "
            f"{safe_current:,}/{safe_total:,} steps | "
            f"{elapsed_h:,.2f}/{total_h:,.2f} simulated h | "
            f"interval {time_step_h:g} h"
        )
    elif stage == "optimization":
        label = (
            f"Optimization | {fraction:.1%} | "
            f"{safe_current:,}/{safe_total:,} candidates"
        )
    elif stage == "sensitivity":
        label = (
            f"Sensitivity | {fraction:.1%} | "
            f"{safe_current:,}/{safe_total:,} experiments"
        )
    else:
        label = (
            f"{stage} | {fraction:.1%} | "
            f"{safe_current:,}/{safe_total:,}"
        )

    return fraction, label


# ---------------------------------------------------------------------
# Export helpers.
# ---------------------------------------------------------------------


def flatten_snapshot(
    snapshot: dict[str, Any],
    prefix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if not value:
                rows.append(
                    {"parameter": path, "value": "{}"}
                )
            else:
                for key, nested in value.items():
                    visit(
                        nested,
                        f"{path}.{key}" if path else str(key),
                    )
        else:
            rows.append(
                {
                    "parameter": path,
                    "value": (
                        json_text(value)
                        if isinstance(value, (list, tuple))
                        else value
                    ),
                }
            )

    visit(snapshot, prefix)

    return pd.DataFrame(
        rows,
        columns=["parameter", "value"],
    )


def _excel_cell(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (dict, list, tuple, set)):
        value = json_text(
            list(value) if isinstance(value, set) else value
        )

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, (pd.Period, Path)):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "Infinity" if value > 0 else "-Infinity"

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, str):
        # Prevent exported text from becoming an Excel formula.
        if value.startswith(("=", "+", "-", "@")):
            value = "'" + value

        return value[:32767]

    return value


def workbook_bytes(
    sheets: dict[str, pd.DataFrame],
) -> bytes:
    output = BytesIO()
    used_names: set[str] = set()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        for requested_name, frame in sheets.items():
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(
                    f"Export sheet {requested_name!r} "
                    "must be a DataFrame."
                )

            if len(frame) > 1_048_575:
                raise ValueError(
                    f"Sheet {requested_name!r} exceeds the Excel "
                    "row limit. Export that table as CSV instead."
                )

            if len(frame.columns) > 16_384:
                raise ValueError(
                    f"Sheet {requested_name!r} exceeds "
                    "the Excel column limit."
                )

            base_name = re.sub(
                r"[\[\]:*?/\\]",
                "_",
                requested_name,
            )[:31] or "Sheet"

            sheet_name = base_name
            counter = 1

            while sheet_name.casefold() in used_names:
                suffix = f"_{counter}"
                sheet_name = base_name[:31 - len(suffix)] + suffix
                counter += 1

            used_names.add(sheet_name.casefold())

            safe = frame.copy()
            safe.columns = [
                str(_excel_cell(column))
                for column in safe.columns
            ]

            for column in safe.columns:
                safe[column] = safe[column].map(_excel_cell)

            safe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    return output.getvalue()


def dataframe_to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "Sheet1",
) -> bytes:
    return workbook_bytes({sheet_name: df})


def simulation_export_sheets(
    simulation: SimulationArtifacts,
) -> dict[str, pd.DataFrame]:
    sheets = {
        "KPIs": pd.DataFrame(
            [
                {"metric": name, "value": value}
                for name, value in simulation.kpis.items()
            ]
        ),
        "Case inputs": flatten_snapshot(
            simulation.input_snapshot
        ),
        "Run metadata": flatten_snapshot(
            simulation.run_metadata
        ),
        "Warnings": pd.DataFrame(
            {"warning": simulation.warnings}
        ),
        "Model summary": simulation.model_summary,
    }

    diagnostics = simulation.surrogate_info.get(
        "diagnostic_summary"
    )

    if isinstance(diagnostics, pd.DataFrame):
        sheets["Model diagnostics"] = diagnostics

    if simulation.tea is not None:
        tea = simulation.tea

        sheets.update(
            {
                "CAPEX": tea.capex_breakdown,
                "Utilities": tea.utility_breakdown,
                "OPEX": tea.opex_breakdown,
                "Revenues": tea.revenue_breakdown,
                "Margins": tea.margins,
                "Cash flow": tea.cash_flow,
                "TEA assumptions": flatten_snapshot(
                    tea.assumptions
                ),
                "TEA status": flatten_snapshot(
                    {
                        **tea.status,
                        "simple_payback_years": (
                            tea.simple_payback_years
                        ),
                        "discounted_payback_years": (
                            tea.discounted_payback_years
                        ),
                    }
                ),
            }
        )

    return sheets


# End of app.py — Part 1 of 3.
# Append Part 2 directly below this line.
# ---------------------------------------------------------------------
# Application initialization.
# ---------------------------------------------------------------------


initialize_session_state()

st.title("PtMeOH Plant Sizing Tool — Thesis Edition")
st.caption(
    "Deterministic process simulation, modular electrolysis, "
    "hydrogen storage, surrogate diagnostics, and discounted "
    "techno-economic analysis."
)

render_flash_message()

try:
    runner_key = "_case_runner"

    if runner_key not in st.session_state:
        st.session_state[runner_key] = CaseRunner(PROJECT_ROOT)

    runner: CaseRunner = st.session_state[runner_key]
    registry = ModelRegistry(PROJECT_ROOT)

except Exception as exc:
    st.error(
        "The application could not initialize. Check the project "
        "structure, updated modules, and config/scenarios.json."
    )
    st.exception(exc)
    st.stop()


case = None
case_build_errors: list[str] = []
pre_warnings: list[str] = []

effective_profile_df = pd.DataFrame()
effective_profile_meta: dict[str, Any] = {}
profile_diag: dict[str, Any] = {}
case_payload: dict[str, Any] = {}

active_profile_df = None
active_profile_meta = None
active_profile_name = "default_synthetic_profile"

tea_configuration = None
electricity_price_usd_per_kwh = None
profile_selection_token = "synthetic"
profile_selection_ready = True


# ---------------------------------------------------------------------
# Sidebar: physical case definition.
# ---------------------------------------------------------------------


with st.sidebar:
    st.header("Case inputs")

    persist_assets = st.toggle(
        "Save uploaded profiles and model archives",
        value=True,
        key="persist_assets",
        help=(
            "Profiles and optional archive copies are saved to the "
            "server filesystem. Filesystem persistence depends on "
            "the deployment; this is not a database backup."
        ),
    )

    economic_mode = st.selectbox(
        "Economic methodology",
        ["Detailed TEA — EUR", "Legacy calculation — USD"],
        key="economic_mode",
    )
    use_eur = economic_mode == "Detailed TEA — EUR"

    scenario_names = list(runner.scenario_config)
    scenario_name = st.selectbox(
        "Legacy economic scenario",
        scenario_names,
        index=(
            scenario_names.index("moderate")
            if "moderate" in scenario_names
            else 0
        ),
        help=(
            "Applies to the legacy USD calculation. "
            "The detailed EUR TEA uses its own explicit inputs."
        ),
    )

    electrolyzer_power_mw = st.number_input(
        "Electrolyzer nominal power [MW]",
        min_value=0.001,
        value=82.0,
        step=1.0,
        format="%.3f",
    )
    module_count = st.number_input(
        "Electrolyzer module count",
        min_value=1,
        value=4,
        step=1,
    )

    storage_enabled = st.toggle(
        "Enable H2 storage",
        value=True,
    )
    storage_kg_h2 = st.number_input(
        "Usable H2 storage capacity [kg H2]",
        min_value=0.0,
        value=26000.0,
        step=100.0,
    )

    operating_mode = st.selectbox(
        "PtMeOH operating mode",
        ["quasi_base_load", "flexible"],
    )

    library_names = registry.get_library_names()

    surrogate_library = st.selectbox(
        "Surrogate model library",
        library_names,
    )

    target_h2_kg_per_h = st.number_input(
        "Target H2 feed to PtMeOH [kg/h]",
        min_value=0.0,
        value=1850.0,
        step=10.0,
    )
    max_h2_feed_kg_per_h = st.number_input(
        "Maximum PtMeOH H2 intake [kg/h]",
        min_value=0.001,
        value=2200.0,
        step=10.0,
    )

    with st.expander("Operating assumptions", expanded=False):
        min_load_fraction = st.number_input(
            "Minimum load per online module [%]",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
        ) / 100.0

        specific_energy = st.number_input(
            "Electrolysis specific energy [kWh/kg H2]",
            min_value=0.001,
            value=52.0,
            step=0.5,
        )

        initial_soc_fraction = st.number_input(
            "Initial H2 inventory [% of usable capacity]",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=5.0,
        ) / 100.0

        limit_storage_rates = st.checkbox(
            "Specify H2 storage rate limits",
            value=False,
        )

        if limit_storage_rates:
            max_charge_rate = st.number_input(
                "Maximum H2 charge rate [kg/h]",
                min_value=0.0,
                value=2500.0,
                step=100.0,
            )
            max_discharge_rate = st.number_input(
                "Maximum H2 discharge rate [kg/h]",
                min_value=0.0,
                value=2500.0,
                step=100.0,
            )
        else:
            max_charge_rate = None
            max_discharge_rate = None

        st.caption(
            "Without explicit rate limits, the tank uses the "
            "previous one-hour turnover convention. Initial and "
            "final inventories are reported; cyclic inventory "
            "is not automatically enforced."
        )

    with st.expander("Operating-mode explanation", expanded=False):
        st.write(
            "Quasi-base-load operation serves the H2 target, "
            "stores excess production, and discards remaining surplus."
        )
        st.write(
            "Flexible operation also allows remaining surplus to "
            "increase process feed above the target, up to maximum "
            "intake, after attempting storage."
        )


# ---------------------------------------------------------------------
# Sidebar: renewable profiles.
# ---------------------------------------------------------------------


with st.sidebar:
    st.subheader("Renewable profile database")

    profile_source = st.radio(
        "Renewable profile source",
        [
            "Synthetic default profile",
            "Upload renewable profile file",
            "Use saved renewable profile",
        ],
    )

    renewable_peak_power_mw = 145.0

    if profile_source == "Synthetic default profile":
        renewable_peak_power_mw = st.number_input(
            "Renewable peak power [MW]",
            min_value=0.001,
            value=145.0,
            step=1.0,
        )

        profile_selection_token = content_signature(
            {
                "source": "synthetic",
                "peak_mw": renewable_peak_power_mw,
            }
        )

        st.caption(
            "The synthetic profile is an illustrative deterministic "
            "profile, not a measured renewable-resource assessment."
        )

    elif profile_source == "Upload renewable profile file":
        profile_selection_ready = False

        uploaded_profile = st.file_uploader(
            "Upload renewable profile",
            type=[
                "csv",
                "txt",
                "tsv",
                "xlsx",
                "xls",
                "parquet",
            ],
            key="renewable_profile_upload",
        )

        if uploaded_profile is not None:
            try:
                raw_profile = read_tabular_file(uploaded_profile)

                if raw_profile.empty or len(raw_profile.columns) < 2:
                    raise ValueError(
                        "The profile must contain records and "
                        "at least two columns."
                    )

                columns = list(raw_profile.columns)

                timestamp_col = st.selectbox(
                    "Date / timestamp column",
                    columns,
                    index=choose_default_column(
                        [str(column) for column in columns],
                        ["timestamp", "datetime", "date", "time", "fecha"],
                    ),
                )
                renewable_col = st.selectbox(
                    "Renewable availability column",
                    columns,
                    index=choose_default_column(
                        [str(column) for column in columns],
                        [
                            "renewable",
                            "energy",
                            "power",
                            "generation",
                            "available",
                            "mw",
                        ],
                    ),
                )

                unit_mode = st.selectbox(
                    "Selected renewable column units",
                    list(SUPPORTED_PROFILE_UNITS),
                )

                expand_daily = st.checkbox(
                    "Expand daily profile to hourly resolution "
                    "when daily data is detected",
                    value=True,
                    key="expand_uploaded_daily",
                )

                single_row_interval = None

                if len(raw_profile) == 1:
                    single_row_interval = st.number_input(
                        "Single-record interval [h]",
                        min_value=0.001,
                        value=(
                            24.0
                            if unit_mode in {"MWh/day", "kWh/day"}
                            else 1.0
                        ),
                        step=0.5,
                    )

                st.caption(
                    "Expansion preserves daily energy using flat "
                    "hourly averages. It does not reconstruct "
                    "intraday variability."
                )
                st.dataframe(
                    raw_profile.head(12),
                    use_container_width=True,
                )

                profile_selection_token = content_signature(
                    {
                        "source": "upload",
                        "file_name": uploaded_profile.name,
                        "file_digest": hashlib.sha256(
                            uploaded_profile.getvalue()
                        ).hexdigest(),
                        "timestamp_column": str(timestamp_col),
                        "renewable_column": str(renewable_col),
                        "unit_mode": unit_mode,
                        "expand_daily": expand_daily,
                        "single_record_interval_h": single_row_interval,
                    }
                )

                if st.button(
                    "Load this renewable profile",
                    use_container_width=True,
                ):
                    normalized, metadata = normalize_renewable_profile(
                        raw_profile_df=raw_profile,
                        timestamp_col=timestamp_col,
                        renewable_col=renewable_col,
                        unit_mode=unit_mode,
                        expand_daily_to_hourly=expand_daily,
                        single_row_interval_h=single_row_interval,
                    )

                    metadata["source_name"] = uploaded_profile.name
                    metadata["source_kind"] = "uploaded"

                    save_profile_assets(
                        source_file=uploaded_profile,
                        normalized_df=normalized,
                        profile_label=uploaded_profile.name,
                        persist_enabled=persist_assets,
                        metadata=metadata,
                    )

                    st.session_state["renewable_profile_df"] = normalized
                    st.session_state["renewable_profile_meta"] = metadata
                    st.session_state["renewable_profile_name"] = (
                        uploaded_profile.name
                    )
                    st.session_state["active_profile_origin"] = (
                        profile_selection_token
                    )

                    flash_message(
                        "success",
                        f"Loaded {len(normalized):,} profile records "
                        f"with a {metadata['time_step_h']:g} h interval.",
                    )
                    st.rerun()

                profile_selection_ready = (
                    st.session_state.get("active_profile_origin")
                    == profile_selection_token
                )

            except Exception as exc:
                st.error(f"The profile could not be loaded: {exc}")
                profile_selection_token = "invalid_uploaded_profile"

        else:
            profile_selection_token = "no_uploaded_profile"

    else:
        profile_selection_ready = False
        saved_profiles = list_saved_profiles()

        if not saved_profiles:
            st.info("No saved renewable profiles were found.")
            profile_selection_token = "no_saved_profiles"

        else:
            saved_name = st.selectbox(
                "Saved renewable profile",
                [path.name for path in saved_profiles],
            )
            saved_path = next(
                path for path in saved_profiles
                if path.name == saved_name
            )

            expand_saved_daily = st.checkbox(
                "Expand daily profile to hourly resolution "
                "when daily data is detected",
                value=True,
                key="expand_saved_daily",
            )

            metadata_path = saved_path.with_suffix(".meta.json")

            profile_selection_token = content_signature(
                {
                    "source": "saved",
                    "file_digest": hashlib.sha256(
                        saved_path.read_bytes()
                    ).hexdigest(),
                    "metadata_digest": (
                        hashlib.sha256(
                            metadata_path.read_bytes()
                        ).hexdigest()
                        if metadata_path.is_file()
                        else None
                    ),
                    "expand_daily": expand_saved_daily,
                }
            )

            if st.button(
                "Load saved renewable profile",
                use_container_width=True,
            ):
                try:
                    saved_frame, saved_meta = load_saved_profile(
                        saved_path
                    )

                    normalized, metadata = normalize_renewable_profile(
                        raw_profile_df=saved_frame,
                        timestamp_col="timestamp",
                        renewable_col="renewable_power_mw",
                        unit_mode="MW",
                        expand_daily_to_hourly=expand_saved_daily,
                        single_row_interval_h=saved_meta["time_step_h"],
                    )

                    metadata["source_name"] = saved_name
                    metadata["source_kind"] = "saved"
                    metadata["saved_source_metadata"] = saved_meta

                    if saved_meta.get("expanded_to_hourly"):
                        metadata["expanded_to_hourly"] = True
                        metadata["looks_daily"] = True
                        metadata["original_step_h"] = 24.0
                        metadata["warnings"].append(
                            "This saved profile already contains "
                            "flat hourly values expanded from daily data."
                        )

                    st.session_state["renewable_profile_df"] = normalized
                    st.session_state["renewable_profile_meta"] = metadata
                    st.session_state["renewable_profile_name"] = saved_name
                    st.session_state["active_profile_origin"] = (
                        profile_selection_token
                    )

                    flash_message(
                        "success",
                        "The saved renewable profile was loaded.",
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"The saved profile could not be loaded: {exc}"
                    )

            profile_selection_ready = (
                st.session_state.get("active_profile_origin")
                == profile_selection_token
            )

    if profile_source != "Synthetic default profile":
        if profile_selection_ready:
            active_profile_df = st.session_state.get(
                "renewable_profile_df"
            )
            active_profile_meta = st.session_state.get(
                "renewable_profile_meta"
            )
            active_profile_name = st.session_state.get(
                "renewable_profile_name",
                "loaded_profile",
            )

            profile_selection_ready = (
                isinstance(active_profile_df, pd.DataFrame)
                and not active_profile_df.empty
            )

            if profile_selection_ready:
                renewable_peak_power_mw = float(
                    active_profile_df["renewable_power_mw"].max()
                )
                st.caption(
                    f"Active profile: {active_profile_name}"
                )
        else:
            st.info(
                "Load the selected profile before running. "
                "Changing its file, units, or expansion settings "
                "requires loading it again."
            )


# ---------------------------------------------------------------------
# Sidebar: model checklist and installation.
# ---------------------------------------------------------------------


catalog_df = registry.discover_packages()
filtered_catalog = catalog_df.loc[
    catalog_df["library"] == surrogate_library
].copy()

model_names = registry.get_models_by_library(
    surrogate_library
)

with st.sidebar:
    st.subheader("Surrogate model governance")

    require_runtime_models = st.checkbox(
        "Require real predictors for all selected models",
        value=True,
        help=(
            "Disable only for demonstration runs. "
            "Extrapolation is still allowed with warnings."
        ),
    )

    with st.expander("Detected model bundles", expanded=False):
        st.dataframe(
            filtered_catalog,
            use_container_width=True,
            hide_index=True,
        )

        for message in registry.warnings:
            st.warning(message)

        st.caption(
            "File readiness does not establish predictive validity. "
            "Verify model units, training ranges, and package versions."
        )

    with st.expander("Install a trusted model ZIP", expanded=False):
        if not MODEL_UPLOADS_ENABLED:
            st.info(
                "Installation is disabled on this deployment. "
                "An administrator can enable it with "
                "PTMEOH_ENABLE_MODEL_UPLOADS=true on an "
                "access-controlled deployment."
            )
        else:
            st.warning(
                "Python and joblib packages can execute code. "
                "Install only packages from a trusted source."
            )

            target_options = model_names + ["New model name"]

            target_selection = st.selectbox(
                "Target model bundle",
                target_options,
                key="install_target_model",
            )

            if target_selection == "New model name":
                target_model_name = st.text_input(
                    "Actual new model folder name",
                    value="",
                    help=(
                        "Use the real package name. Do not reuse a "
                        "purge or recycle model name for fresh CO2 feed."
                    ),
                ).strip()
            else:
                target_model_name = target_selection

            quantity_kind = st.radio(
                "Physical quantity predicted by this model",
                ["Select...", "Molar flow", "Energy duty"],
                key="install_quantity_kind",
            )

            input_unit_choice = None

            if quantity_kind == "Molar flow":
                output_unit_choice = st.radio(
                    "Unit (input H2 and output share it)",
                    ["mol/h", "kmol/h"],
                    horizontal=True,
                    key="install_unit_molar",
                )
                input_unit_choice = output_unit_choice

            elif quantity_kind == "Energy duty":
                output_unit_choice = st.radio(
                    "Duty unit",
                    ["W", "kW"],
                    horizontal=True,
                    key="install_unit_duty",
                )
                input_unit_choice = st.radio(
                    "H2 input unit of this model",
                    ["mol/h", "kmol/h"],
                    horizontal=True,
                    key="install_unit_duty_input",
                )
            else:
                output_unit_choice = None

            unit_choice = output_unit_choice
            
            model_zip = st.file_uploader(
                "Model ZIP",
                type=["zip"],
                key="trusted_model_zip",
            )

            trusted_source_confirmed = st.checkbox(
                "I am authorized to install this package and "
                "have verified its trusted source.",
                value=False,
            )

            if st.button(
                "Install or replace selected model package",
                disabled=not trusted_source_confirmed,
                use_container_width=True,
            ):
                try:
                    if model_zip is None:
                        raise ValueError("Select a model ZIP first.")

                    if not target_model_name:
                        raise ValueError(
                            "Enter the actual model folder name."
                        )

                    installation = install_model_zip(
                        PROJECT_ROOT,
                        surrogate_library,
                        target_model_name,
                        model_zip,
                        persist_assets,
                    )

                    runner.engine.clear_surrogate_cache()

                    flash_message(
                        "success",
                        f"Installed {installation['written_count']} "
                        f"files for {target_model_name}. "
                        + " ".join(installation["warnings"]),
                    )
                    st.rerun()

                except (BadZipFile, ValueError, PermissionError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)

    confirm_bundle = st.checkbox(
        "I confirm that the selected library and model packages "
        "are the intended ones for this case.",
        value=False,
        key=f"confirm_library_{surrogate_library}",
    )


# ---------------------------------------------------------------------
# Sidebar: economic inputs.
# ---------------------------------------------------------------------


eur_prices_confirmed = True

with st.sidebar:
    if use_eur:
        st.subheader("Detailed TEA — constant EUR")

        st.caption(
            "Enter prices directly in the selected base-year EUR. "
            "Zero is a valid price only if intentionally assumed. "
            "No USD-to-EUR conversion or future inflation is automatic."
        )

        with st.expander("Project and price inputs", expanded=True):
            project_start_year = st.number_input(
                "Investment year — project year 0",
                min_value=1900,
                value=2027,
                step=1,
            )
            project_years = st.number_input(
                "Operating lifetime [years]",
                min_value=1,
                value=25,
                step=1,
            )
            discount_rate = st.number_input(
                "Discount rate [%]",
                min_value=0.0,
                value=8.0,
                step=0.5,
            ) / 100.0

            electricity_eur = st.number_input(
                "Electricity price [EUR/kWh]",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
            )
            cooling_water_eur = st.number_input(
                "Cooling-water price [EUR/t]",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
            )
            co2_eur = st.number_input(
                "Captured CO2 price [EUR/t]",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            electrolysis_water_eur = st.number_input(
                "Electrolysis-water price [EUR/t]",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
            )
            methanol_eur = st.number_input(
                "Methanol selling price [EUR/t]",
                min_value=0.0,
                value=0.0,
                step=10.0,
            )

            eur_prices_confirmed = st.checkbox(
                "I have entered or intentionally accepted all EUR "
                "prices and cost assumptions.",
                value=False,
            )

        with st.expander("Inclusive plant CAPEX", expanded=False):
            reference_capex = st.number_input(
                "Reference total plant cost [EUR]",
                min_value=0.0,
                value=0.0,
                step=100000.0,
            )
            reference_capacity = st.number_input(
                "Reference nominal MeOH capacity [t/h]",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.4f",
            )
            design_capacity = st.number_input(
                "Design nominal MeOH capacity [t/h]",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.4f",
            )
            scaling_exponent = st.number_input(
                "Plant scaling exponent",
                min_value=0.001,
                value=0.7,
                step=0.05,
            )
            electrolyzer_eur_per_kw = st.number_input(
                "Electrolyzer cost [EUR/kW]",
                min_value=0.0,
                value=0.0,
                step=10.0,
            )
            h2_storage_eur_per_kg = st.number_input(
                "H2 storage cost [EUR/kg H2]",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )

            co2_storage_enabled = st.checkbox(
                "Include CO2 storage in CAPEX",
                value=False,
            )
            co2_storage_capacity = st.number_input(
                "CO2 storage capacity [kg CO2]",
                min_value=0.0,
                value=0.0,
                step=100.0,
                disabled=not co2_storage_enabled,
            )
            co2_storage_unit_cost = st.number_input(
                "CO2 storage cost [EUR/kg CO2]",
                min_value=0.0,
                value=0.0,
                step=1.0,
                disabled=not co2_storage_enabled,
            )

            st.caption(
                "Scaled CAPEX already includes all four components. "
                "Methanol synthesis is the residual after subtracting "
                "electrolyzer and enabled storage costs. "
                "A negative residual is an input inconsistency."
            )

        with st.expander("Utilities and coproduct assumptions", expanded=False):
            cooling_delta_t = st.number_input(
                "Cooling-water temperature rise [K]",
                min_value=0.001,
                value=10.0,
                step=1.0,
                help=(
                    "Editable modeling assumption. Verify against "
                    "the cooling-water conditions used in the thesis."
                ),
            )
            cooling_cp = st.number_input(
                "Cooling-water specific heat [J/(kg K)]",
                min_value=0.001,
                value=4180.0,
                step=10.0,
            )
            cooling_basis = st.selectbox(
                "Cooling-water cost basis",
                ["circulating", "makeup"],
            )
            makeup_fraction = st.number_input(
                "Makeup water [% of circulating water]",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.5,
                disabled=cooling_basis != "makeup",
                help="An explicit assumption, not a measured default.",
            ) / 100.0

            cooling_sign = st.selectbox(
                "Cooling-duty sign convention",
                ["negative_is_cooling", "positive_is_cooling"],
            )
            heater_sign = st.selectbox(
                "HX3 heating-duty sign convention",
                ["positive_is_heating", "negative_is_heating"],
            )
            heater_efficiency = st.number_input(
                "HX3 electrical heating efficiency [%]",
                min_value=0.001,
                max_value=100.0,
                value=100.0,
                step=1.0,
            ) / 100.0

            water_ratio = st.number_input(
                "Electrolysis water [kg/kg H2]",
                min_value=0.001,
                value=9.0,
                step=0.1,
            )
            oxygen_ratio = st.number_input(
                "Oxygen produced [kg/kg H2]",
                min_value=0.0,
                value=8.0,
                step=0.1,
            )

            oxygen_sales = st.checkbox(
                "Enable oxygen sales",
                value=False,
            )
            oxygen_saleable = st.number_input(
                "Saleable oxygen [% of production]",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=5.0,
                disabled=not oxygen_sales,
            ) / 100.0
            oxygen_price = st.number_input(
                "Oxygen selling price [EUR/t]",
                min_value=0.0,
                value=0.0,
                step=1.0,
                disabled=not oxygen_sales,
            )
            oxygen_conditioning = st.number_input(
                "Oxygen conditioning cost [EUR/t sold]",
                min_value=0.0,
                value=0.0,
                step=1.0,
                disabled=not oxygen_sales,
            )

        with st.expander("Fresh CO2 feed", expanded=False):
            co2_source = st.selectbox(
                "CO2 consumption calculation",
                ["stoichiometric", "surrogate"],
            )

            selected_co2_model = None

            if co2_source == "surrogate":
                if model_names:
                    selected_co2_model = st.selectbox(
                        "Actual fresh-CO2 feed model",
                        model_names,
                        help=(
                            "Select the model trained to predict fresh "
                            "CO2 feed from H2 input, not a recycle "
                            "or purge model."
                        ),
                    )
                else:
                    st.warning("No model names are available.")

            co2_output_unit = st.selectbox(
                "Fresh-CO2 model output unit",
                ["kg/h", "t/h", "kmol/h"],
                disabled=co2_source != "surrogate",
            )
            allow_co2_fallback = st.checkbox(
                "Explicitly allow stoichiometric CO2 fallback",
                value=False,
                disabled=co2_source != "surrogate",
            )
            co2_ratio = st.number_input(
                "Fallback CO2/MeOH mass ratio [kg/kg]",
                min_value=0.001,
                value=1.375,
                step=0.01,
                format="%.4f",
            )

        with st.expander("Fixed OPEX assumptions", expanded=False):
            st.caption(
                "Labor and supervision use EUR/year. All other "
                "values are fractions: 0.10 means 10%. "
                "Defaults reproduce the requested methodology."
            )

            default_opex = TEAOpexInputs()
            opex_table = pd.DataFrame(
                [
                    {"parameter": name, "value": float(value)}
                    for name, value in asdict(default_opex).items()
                ]
            )
            edited_opex = st.data_editor(
                opex_table,
                disabled=["parameter"],
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "value": st.column_config.NumberColumn(
                        "Value",
                        min_value=0.0,
                        format="%.6f",
                    ),
                },
                key="tea_opex_editor",
            )

        with st.expander("Annual operating basis", expanded=False):
            annual_hours = st.number_input(
                "Representative annual horizon [h]",
                min_value=1.0,
                value=8760.0,
                step=1.0,
                help=(
                    "Use 8784 for a complete leap-year horizon. "
                    "This is the represented year length, not an "
                    "additional full-load-hours multiplier."
                ),
            )
            annualize_profile = st.checkbox(
                "Explicitly annualize a shorter or longer profile",
                value=False,
            )
            st.caption(
                "Annualization scales operating quantities, not "
                "annual labor or other fixed charges. It assumes "
                "the profile is representative."
            )

        try:
            tea_configuration = TEAInputs(
                project_start_year=int(project_start_year),
                project_years=int(project_years),
                discount_rate=float(discount_rate),
                price_base_year=int(project_start_year),
                prices=TEAPriceInputs(
                    electricity_eur_per_kwh=electricity_eur,
                    cooling_water_eur_per_t=cooling_water_eur,
                    captured_co2_eur_per_t=co2_eur,
                    electrolysis_water_eur_per_t=electrolysis_water_eur,
                    methanol_eur_per_t=methanol_eur,
                    oxygen_eur_per_t=(
                        oxygen_price if oxygen_sales else None
                    ),
                ),
                capex=TEACapexInputs(
                    reference_plant_cost_eur=reference_capex,
                    reference_meoh_capacity_t_per_h=reference_capacity,
                    design_meoh_capacity_t_per_h=design_capacity,
                    scaling_exponent=scaling_exponent,
                    electrolyzer_eur_per_kw=electrolyzer_eur_per_kw,
                    h2_storage_eur_per_kg=h2_storage_eur_per_kg,
                    co2_storage_enabled=co2_storage_enabled,
                    co2_storage_capacity_kg=co2_storage_capacity,
                    co2_storage_eur_per_kg=(
                        co2_storage_unit_cost
                        if co2_storage_enabled
                        else None
                    ),
                ),
                opex=TEAOpexInputs(
                    **{
                        str(row["parameter"]): float(row["value"])
                        for row in edited_opex.to_dict("records")
                    }
                ),
                utilities=UtilityInputs(
                    cooling_water_cp_j_per_kg_k=cooling_cp,
                    cooling_water_delta_t_k=cooling_delta_t,
                    cooling_water_cost_basis=cooling_basis,
                    cooling_water_makeup_fraction=(
                        makeup_fraction
                        if cooling_basis == "makeup"
                        else None
                    ),
                    cooling_duty_sign=cooling_sign,
                    heater_duty_sign=heater_sign,
                    electric_heater_efficiency=heater_efficiency,
                    electrolysis_water_kg_per_kg_h2=water_ratio,
                    oxygen_kg_per_kg_h2=oxygen_ratio,
                ),
                co2_feed=CO2FeedInputs(
                    source=co2_source,
                    model_name=selected_co2_model,
                    output_unit=co2_output_unit,
                    kg_co2_per_kg_meoh=co2_ratio,
                    allow_stoichiometric_fallback=allow_co2_fallback,
                ),
                coproducts=CoproductInputs(
                    oxygen_sales_enabled=oxygen_sales,
                    oxygen_saleable_fraction=oxygen_saleable,
                    oxygen_conditioning_eur_per_t=oxygen_conditioning,
                ),
                annualize_profile=annualize_profile,
                annual_hours=annual_hours,
            )
        except (TypeError, ValueError) as exc:
            case_build_errors.append(
                f"The EUR TEA inputs could not be constructed: {exc}"
            )

    else:
        scenario_price = float(
            runner.scenario_config[scenario_name][
                "electricity_price_usd_per_mwh"
            ]
        ) / 1000.0

        electricity_price_usd_per_kwh = st.number_input(
            "Electricity price [USD/kWh]",
            min_value=0.0,
            value=scenario_price,
            step=0.001,
            format="%.4f",
            key=f"legacy_price_{scenario_name}",
        )

        st.caption(
            "Legacy mode retains the previous simplified USD "
            "economics and compressor-only downstream electricity "
            "accounting. Detailed EUR TEA features are not applied."
        )


# ---------------------------------------------------------------------
# Sidebar: optimization settings.
# ---------------------------------------------------------------------


with st.sidebar:
    with st.expander("Optimization settings", expanded=False):
        objective = st.selectbox(
            "Optimization objective",
            ["min_lcomeoh", "max_npv"],
        )

        lcomeoh_basis = st.selectbox(
            "LCOMeOH basis",
            ["gross", "net"] if use_eur else ["gross"],
            help=(
                "Net LCOMeOH credits enabled oxygen sales. "
                "It does not subtract methanol revenue."
            ),
        )

        minimum_utilization = st.number_input(
            "Minimum PtMeOH utilization [%]",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=5.0,
        ) / 100.0

        maximum_unmet = st.number_input(
            "Maximum unmet H2 [% of target demand]",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        ) / 100.0

        maximum_curtailment = st.number_input(
            "Maximum renewable curtailment [%]",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=5.0,
        ) / 100.0

        maximum_deficit = st.number_input(
            "Allowed electricity deficit per simulated period [MWh]",
            min_value=0.0,
            value=0.000001,
            step=0.000001,
            format="%.6f",
        )

        st.caption(
            "Extrapolation is warning-only. Electrical "
            "nonconvergence excludes a candidate from feasibility. "
            "The default grid retains the existing search ranges."
        )


# ---------------------------------------------------------------------
# Case construction and pre-run checks.
# ---------------------------------------------------------------------


if not profile_selection_ready:
    case_build_errors.append(
        "Load the selected renewable profile and its current settings."
    )

if use_eur and tea_configuration is None:
    case_build_errors.append(
        "A complete EUR TEA configuration is required."
    )

if profile_selection_ready and not case_build_errors:
    try:
        optimization_configuration = runner._default_optimization(
            power_mw=float(electrolyzer_power_mw),
            module_count=int(module_count),
            storage_enabled=bool(storage_enabled),
            storage_kg=float(storage_kg_h2),
            target_h2=float(target_h2_kg_per_h),
            maximum_h2=float(max_h2_feed_kg_per_h),
        )
        optimization_configuration.objective = objective
        optimization_configuration.lcomeoh_basis = lcomeoh_basis
        optimization_configuration.min_ptmeoh_utilization = (
            minimum_utilization
        )
        optimization_configuration.max_unmet_h2_fraction = maximum_unmet
        optimization_configuration.max_curtailment_fraction = (
            maximum_curtailment
        )
        optimization_configuration.max_power_deficit_mwh = maximum_deficit
        optimization_configuration.require_runtime_models = (
            require_runtime_models
        )

        case = runner.build_case(
            scenario_name=scenario_name,
            electrolyzer_power_mw=electrolyzer_power_mw,
            module_count=int(module_count),
            storage_enabled=bool(storage_enabled),
            storage_kg_h2=storage_kg_h2,
            operating_mode=operating_mode,
            surrogate_library=surrogate_library,
            target_h2_kg_per_h=target_h2_kg_per_h,
            max_h2_feed_kg_per_h=max_h2_feed_kg_per_h,
            renewable_peak_power_mw=renewable_peak_power_mw,
            renewable_profile_df=active_profile_df,
            electricity_price_usd_per_kwh=(
                electricity_price_usd_per_kwh
            ),
            tea_inputs=tea_configuration,
            optimization_inputs=optimization_configuration,
            profile_meta=active_profile_meta,
            # Uploaded/saved profiles were normalized above.
            # Preserve their effective interval here.
            expand_daily_to_hourly=False,
            initial_soc_fraction=initial_soc_fraction,
            max_charge_kg_per_h=max_charge_rate,
            max_discharge_kg_per_h=max_discharge_rate,
            min_load_fraction=min_load_fraction,
            specific_energy_kwh_per_kg_h2=specific_energy,
        )

        effective_profile_df = case.renewable_profile
        effective_profile_meta = case.provenance["profile_meta"]
        time_step_h=(
                active_profile_meta.get("time_step_h")
                if active_profile_meta is not None
                else None
        ),

        report = inspect_case_inputs(case)
        case_build_errors.extend(report.errors)
        pre_warnings.extend(report.warnings)
        pre_warnings.extend(
            effective_profile_meta.get("warnings", [])
        )

        validate_optimization_inputs(case)

        minimum_module_power = (
            case.electrolyzer.module_size_mw
            * case.electrolyzer.min_load_fraction
        )
        profile_power = effective_profile_df["renewable_power_mw"]

        profile_diag = {
            "peak_mw": float(profile_power.max()),
            "mean_mw": float(profile_power.mean()),
            "duration_h": (
                len(effective_profile_df) * case.time_step_h
            ),
            "profile_energy_mwh": float(
                profile_power.sum() * case.time_step_h
            ),
            "minimum_operating_module_power_mw": minimum_module_power,
            "below_minimum_module_fraction": float(
                (profile_power < minimum_module_power).mean()
            ),
            "zero_power_fraction": float(
                (profile_power <= 0).mean()
            ),
        }

        if profile_diag["below_minimum_module_fraction"] > 0.5:
            pre_warnings.append(
                "Renewable power is below the minimum power of "
                "one electrolyzer module during more than half "
                "the profile."
            )

        if use_eur:
            represented_h = profile_diag["duration_h"]

            if (
                not case.tea.annualize_profile
                and not np.isclose(
                    represented_h,
                    case.tea.annual_hours,
                    rtol=1e-6,
                    atol=1e-6,
                )
            ):
                case_build_errors.append(
                    f"The profile represents {represented_h:g} h, "
                    f"but the TEA annual horizon is "
                    f"{case.tea.annual_hours:g} h. "
                    "Provide a matching annual profile or explicitly "
                    "enable annualization."
                )

            # Detect inconsistent inclusive CAPEX before a long run.
            try:
                from domain.technoeconomics import TechnoEconomics

                TechnoEconomics(case)._capex()
            except (TypeError, ValueError, OverflowError) as exc:
                case_build_errors.append(str(exc))

        registry_fingerprint = registry.get_library_fingerprint(
            surrogate_library
        )
        profile_fingerprint = runner.engine._profile_fingerprint(
            effective_profile_df
        )

        case_payload = runner.engine._input_snapshot(
            case,
            profile_fingerprint,
            registry_fingerprint,
        )

    except InputValidationError as exc:
        case_build_errors.extend(exc.errors)
    except Exception as exc:
        case_build_errors.append(str(exc))


if not confirm_bundle:
    case_build_errors.append(
        "Confirm the intended model library and package selection."
    )

if use_eur and not eur_prices_confirmed:
    case_build_errors.append(
        "Confirm the EUR prices and economic assumptions."
    )

if require_runtime_models:
    if filtered_catalog.empty:
        case_build_errors.append(
            "No model packages were found for the selected library."
        )
    elif not filtered_catalog["ready_for_runtime"].all():
        case_build_errors.append(
            "Some selected model packages are missing or ambiguous. "
            "Install complete packages or explicitly select "
            "demonstration mode."
        )
else:
    pre_warnings.append(
        "Demonstration fallback is enabled. Missing or unusable "
        "predictors may be replaced by mock models."
    )

pre_warnings.extend(registry.warnings)

case_build_errors = list(dict.fromkeys(case_build_errors))
pre_warnings = list(dict.fromkeys(pre_warnings))

if case_payload:
    current_signature = content_signature(
        {
            "case": case_payload,
            "profile_selection": profile_selection_token,
        }
    )
else:
    current_signature = content_signature(
        {
            "invalid_case": True,
            "errors": case_build_errors,
            "profile_selection": profile_selection_token,
            "economic_mode": economic_mode,
        }
    )

reset_results_for_new_case(current_signature)

run_disabled = case is None or bool(case_build_errors)


# ---------------------------------------------------------------------
# Execution monitor and actions.
# ---------------------------------------------------------------------


st.subheader("Execution monitor")

progress_bar = st.progress(
    0.0,
    text="Idle",
)
progress_caption = st.empty()

with st.expander("Real-time telemetry", expanded=True):
    telemetry_placeholder = st.empty()
    telemetry_placeholder.code(
        get_telemetry_text(),
        language="text",
    )

_last_refresh_time = [0.0]


def refresh_runtime_panels(force: bool = False) -> None:
    now = datetime.now(timezone.utc).timestamp()

    if force or now - _last_refresh_time[0] >= 0.2:
        telemetry_placeholder.code(
            get_telemetry_text(),
            language="text",
        )
        _last_refresh_time[0] = now


def ui_progress(stage: str, current: int, total: int) -> None:
    step_h = case.time_step_h if case is not None else 1.0

    fraction, label = format_progress_label(
        stage,
        current,
        total,
        step_h,
    )
    progress_bar.progress(
        min(max(fraction, 0.0), 1.0),
        text=label,
    )
    progress_caption.caption(label)
    append_telemetry_line(label)
    refresh_runtime_panels()


def execute_action(action: str) -> None:
    if case is None or run_disabled:
        return

    keys_to_clear = {
        "simulation": ["simulation"],
        "optimization": ["optimization"],
        "sensitivity": ["sensitivities"],
        "all": ["simulation", "optimization", "sensitivities"],
    }

    for key in keys_to_clear[action]:
        st.session_state[key] = None

    st.session_state["prepared_downloads"] = {}
    st.session_state["telemetry_lines"] = []

    append_telemetry_line(f"Starting action: {action}")
    progress_caption.info(f"Running {action}...")
    refresh_runtime_panels(force=True)

    try:
        with capture_runtime_logs(
            runner,
            refresh_runtime_panels,
        ):
            if action == "simulation":
                st.session_state["simulation"] = runner.run_simulation(
                    case,
                    progress_callback=ui_progress,
                )

            elif action == "optimization":
                st.session_state["optimization"] = (
                    runner.run_optimization(
                        case,
                        progress_callback=ui_progress,
                    )
                )

            elif action == "sensitivity":
                st.session_state["sensitivities"] = (
                    runner.run_sensitivity(
                        case,
                        progress_callback=ui_progress,
                        baseline_simulation=st.session_state.get(
                            "simulation"
                        ),
                    )
                )

            else:
                outputs = runner.run_all(
                    case,
                    run_optimization=True,
                    run_sensitivity=True,
                    progress_callback=ui_progress,
                )
                st.session_state["simulation"] = outputs["simulation"]
                st.session_state["optimization"] = outputs["optimization"]
                st.session_state["sensitivities"] = outputs["sensitivities"]

        append_telemetry_line(f"Completed action: {action}")
        progress_bar.progress(
            1.0,
            text=f"{action.capitalize()} execution completed",
        )
        progress_caption.success(
            "Execution completed. Review feasibility, model "
            "warnings, and any failed candidate evaluations."
        )

    except Exception as exc:
        append_telemetry_line(
            f"FAILED | {type(exc).__name__}: {exc}"
        )
        progress_caption.error(
            f"{action.capitalize()} failed: {exc}"
        )
        st.exception(exc)

    finally:
        refresh_runtime_panels(force=True)


action_columns = st.columns(4)

for column, label, action in zip(
    action_columns,
    [
        "Run annual simulation",
        "Run optimization",
        "Run sensitivities",
        "Run all",
    ],
    ["simulation", "optimization", "sensitivity", "all"],
):
    with column:
        if st.button(
            label,
            use_container_width=True,
            disabled=run_disabled,
            key=f"action_{action}",
        ):
            execute_action(action)


if case_build_errors:
    st.error(
        "Execution is disabled until the blocking input issues "
        "listed in the Inputs tab are resolved."
    )


simulation = st.session_state.get("simulation")
optimization = st.session_state.get("optimization")
sensitivities = st.session_state.get("sensitivities")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Inputs",
        "Annual Simulation",
        "Techno-Economic Optimum",
        "Sensitivities",
    ]
)


# End of app.py — Part 2 of 3.
# Append Part 3 directly below this line.
# ---------------------------------------------------------------------
# Result-view helpers.
# ---------------------------------------------------------------------


def artifact_value(
    artifact: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(artifact, dict):
        return artifact.get(name, default)

    return getattr(artifact, name, default)


def numeric_text(
    value: Any,
    decimals: int = 3,
) -> str:
    if value is None:
        return "Not available"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(number):
        return "Undefined"

    if math.isinf(number):
        return "+Infinity" if number > 0 else "-Infinity"

    return f"{number:,.{decimals}f}"


def first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    return next(
        (
            column
            for column in candidates
            if column in frame.columns
        ),
        None,
    )


def result_time_step_h(
    result: SimulationArtifacts,
) -> float:
    metadata = artifact_value(result, "run_metadata", {}) or {}
    snapshot = artifact_value(result, "input_snapshot", {}) or {}

    candidates = [
        metadata.get("time_step_h"),
        snapshot.get("time_step_h"),
    ]

    for container_name in ("profile", "renewable_profile", "simulation"):
        nested = snapshot.get(container_name)

        if isinstance(nested, dict):
            candidates.append(nested.get("time_step_h"))

    for candidate in candidates:
        if candidate is None:
            continue

        value = float(candidate)

        if math.isfinite(value) and value > 0:
            return value

    frame = result.time_series

    if "timestamp" in frame.columns and len(frame) > 1:
        timestamps = pd.to_datetime(
            frame["timestamp"],
            errors="raise",
        ).sort_values()

        differences = (
            timestamps.diff()
            .dropna()
            .dt.total_seconds()
            .div(3600.0)
            .to_numpy(dtype=float)
        )

        if (
            len(differences)
            and (differences > 0).all()
            and np.allclose(
                differences,
                differences[0],
                rtol=1e-8,
                atol=1e-9,
            )
        ):
            return float(differences[0])

    raise ValueError(
        "The result interval could not be established from the "
        "stored metadata or timestamps. No one-hour default "
        "has been substituted."
    )


def kpi_table(kpis: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": metric, "value": value}
            for metric, value in kpis.items()
        ],
        columns=["metric", "value"],
    )


def render_metric_group(
    kpis: dict,
    specifications: list[tuple[str, str, int]],
    columns_per_row: int = 4,
) -> None:
    available = [
        specification
        for specification in specifications
        if specification[1] in kpis
    ]

    for start in range(0, len(available), columns_per_row):
        group = available[start:start + columns_per_row]
        columns = st.columns(len(group))

        for column, (label, key, decimals) in zip(columns, group):
            column.metric(
                label,
                numeric_text(kpis[key], decimals),
            )


def render_tea_results(
    result: SimulationArtifacts,
    key_prefix: str,
) -> None:
    tea = artifact_value(result, "tea")

    if tea is None:
        st.caption(
            "Detailed EUR tables are not attached to this result. "
            "Legacy USD results remain available in the KPI table."
        )
        return

    st.subheader("Detailed techno-economic results")

    render_metric_group(
        tea.kpis,
        [
            ("Total CAPEX [EUR]", "total_capex_eur", 0),
            (
                "Variable OPEX [EUR/year]",
                "annual_variable_opex_eur",
                0,
            ),
            (
                "Gross LCOMeOH [EUR/t]",
                "lcomeoh_gross_eur_per_t_meoh",
                2,
            ),
            (
                "Net LCOMeOH [EUR/t]",
                "lcomeoh_net_eur_per_t_meoh",
                2,
            ),
            ("NPV [EUR]", "npv_eur", 0),
        ],
    )

    payback_columns = st.columns(2)

    for column, label, value in (
        (
            payback_columns[0],
            "Simple payback [years]",
            tea.simple_payback_years,
        ),
        (
            payback_columns[1],
            "Discounted payback [years]",
            tea.discounted_payback_years,
        ),
    ):
        column.metric(
            label,
            (
                "Not reached / undefined"
                if value is None
                else numeric_text(value, 2)
            ),
        )

    st.caption(
        "A missing payback is not a zero-year payback. "
        "Inspect the economic status and cash-flow table for "
        "the applicable interpretation."
    )

    left, right = st.columns(2)

    with left:
        try:
            st.plotly_chart(
                capex_donut(tea),
                use_container_width=True,
                key=f"{key_prefix}_capex",
            )
        except (ValueError, TypeError) as exc:
            st.warning(f"CAPEX chart unavailable: {exc}")

    with right:
        try:
            st.plotly_chart(
                variable_cost_donut(tea),
                use_container_width=True,
                key=f"{key_prefix}_variable_cost",
            )
        except (ValueError, TypeError) as exc:
            st.warning(f"Variable-cost chart unavailable: {exc}")

    discounted = st.checkbox(
        "Display discounted cash flows",
        value=False,
        key=f"{key_prefix}_discounted",
    )

    try:
        st.plotly_chart(
            cash_flow_chart(
                tea,
                discounted=discounted,
            ),
            use_container_width=True,
            key=f"{key_prefix}_cash_flow",
        )
    except (ValueError, TypeError) as exc:
        st.warning(f"Cash-flow chart unavailable: {exc}")

    tables = [
        ("CAPEX breakdown", tea.capex_breakdown),
        ("Utility consumption and cost", tea.utility_breakdown),
        ("OPEX breakdown", tea.opex_breakdown),
        ("Revenues", tea.revenue_breakdown),
        ("Operating margins", tea.margins),
        ("Project cash flow", tea.cash_flow),
        ("All economic KPIs", kpi_table(tea.kpis)),
    ]

    for title, frame in tables:
        with st.expander(title, expanded=False):
            st.dataframe(
                frame,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Economic assumptions and status", expanded=False):
        st.write("Assumptions")
        st.json(json_text(tea.assumptions))
        st.write("Calculation status")
        st.json(json_text(tea.status))


# ---------------------------------------------------------------------
# Process visualization.
# ---------------------------------------------------------------------


PROCESS_COLUMNS = {
    "renewable_power_mw": [
        "renewable_power_mw",
    ],
    "h2_produced_kg_per_h": [
        "h2_produced_kg_per_h",
    ],
    "h2_to_ptmeoh_kg_per_h": [
        "h2_to_ptmeoh_kg_per_h",
    ],
    "tank_soc_kg_h2": [
        "tank_soc_kg_h2",
    ],
    "methanol_t_per_h": [
        "methanol_t_per_h",
    ],
    "h2_purge_kg_per_h": [
        "h2_purge_kg_per_h",
        "purge_h2_kg_per_h",
        "Model_Purge_H2_kg_per_h",
    ],
    "h2_recycle_kg_per_h": [
        "h2_recycle_kg_per_h",
        "recycle_h2_kg_per_h",
        "Model_RECLP_H2_kg_per_h",
    ],
}


def prepare_process_frame(
    time_series: pd.DataFrame,
) -> pd.DataFrame:
    if "timestamp" not in time_series.columns:
        raise ValueError(
            "The simulation result has no timestamp column."
        )

    source = time_series.copy()
    source["timestamp"] = pd.to_datetime(
        source["timestamp"],
        errors="raise",
    )
    source = source.sort_values("timestamp").reset_index(drop=True)

    if source["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps prevent a traceable process view."
        )

    frame = source[["timestamp"]].copy()

    for canonical, candidates in PROCESS_COLUMNS.items():
        selected = first_existing_column(source, candidates)

        if selected is not None:
            frame[canonical] = pd.to_numeric(
                source[selected],
                errors="raise",
            )

    return frame


def aggregate_process_window(
    frame: pd.DataFrame,
    time_step_h: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolution: str,
) -> pd.DataFrame:
    """
    Aggregate interval-average flows using interval overlap.

    Production is integrated in tonnes.
    Rates and power are duration-weighted means.
    Inventory is the last reported end-of-step inventory in each bin.
    """

    records: list[dict[str, Any]] = []
    interval = pd.Timedelta(hours=time_step_h)

    for row in frame.to_dict("records"):
        interval_start = row["timestamp"]
        interval_end = interval_start + interval

        left = max(interval_start, start)
        right = min(interval_end, end)

        if right <= left:
            continue

        while left < right:
            if resolution == "Monthly":
                bin_start = left.normalize().replace(day=1)
                next_boundary = bin_start + pd.offsets.MonthBegin(1)

            elif resolution == "Daily":
                bin_start = left.normalize()
                next_boundary = bin_start + pd.DateOffset(days=1)

            else:
                bin_start = left
                next_boundary = right

            segment_end = min(right, next_boundary)
            duration_h = (
                segment_end - left
            ).total_seconds() / 3600.0

            record = {
                "plot_time": bin_start,
                "duration_h": duration_h,
            }

            for column in frame.columns:
                if column == "timestamp":
                    continue

                value = row[column]

                if pd.isna(value) or not np.isfinite(float(value)):
                    record[column] = np.nan
                elif column == "tank_soc_kg_h2":
                    record[column] = float(value)
                else:
                    record[column] = float(value) * duration_h

            records.append(record)
            left = segment_end

    if not records:
        return pd.DataFrame()

    segments = pd.DataFrame(records)
    output: list[dict[str, Any]] = []

    for plot_time, group in segments.groupby(
        "plot_time",
        sort=True,
    ):
        duration_h = float(group["duration_h"].sum())
        aggregated = {
            "plot_time": plot_time,
            "duration_h": duration_h,
        }

        for column in frame.columns:
            if column == "timestamp":
                continue

            values = group[column]

            # Preserve missing/invalid information instead of turning
            # incomplete production or rates into an apparent zero.
            if values.isna().any():
                value = np.nan
            elif column == "tank_soc_kg_h2":
                value = float(values.iloc[-1])
            elif column == "methanol_t_per_h":
                value = float(values.sum())
            else:
                value = float(values.sum()) / duration_h

            output_column = (
                "methanol_t_per_period"
                if column == "methanol_t_per_h"
                else column
            )
            aggregated[output_column] = value

        output.append(aggregated)

    return pd.DataFrame(output)


def render_process_views(
    result: SimulationArtifacts,
    key_prefix: str,
) -> None:
    time_series = result.time_series

    if time_series.empty:
        st.info("No process time series is available.")
        return

    try:
        step_h = result_time_step_h(result)
        frame = prepare_process_frame(time_series)
    except (ValueError, TypeError, KeyError) as exc:
        st.warning(f"Integrated process view unavailable: {exc}")
        return

    if frame.empty:
        st.info("No process records are available.")
        return

    st.subheader("Integrated process visualization")

    controls = st.columns(3)

    with controls[0]:
        view = st.radio(
            "Resolution",
            ["Annual", "Daily", "Weekly detail"],
            horizontal=True,
            key=f"{key_prefix}_resolution",
        )

    with controls[1]:
        h2_unit = st.radio(
            "H2 flow units",
            ["kg/h", "t/h"],
            horizontal=True,
            key=f"{key_prefix}_h2_unit",
        )

    profile_start = frame["timestamp"].min()
    profile_end = (
        frame["timestamp"].max()
        + pd.Timedelta(hours=step_h)
    )

    start = profile_start
    end = profile_end
    resolution = "Monthly"

    if view != "Annual":
        first_month = profile_start.normalize().replace(day=1)
        month_starts = pd.date_range(
            start=first_month,
            end=profile_end - pd.Timedelta(nanoseconds=1),
            freq="MS",
        )
        month_labels = [
            value.strftime("%Y-%m")
            for value in month_starts
        ]

        with controls[2]:
            selected_month = st.selectbox(
                "Month",
                month_labels,
                key=f"{key_prefix}_month",
            )

        month_start = month_starts[
            month_labels.index(selected_month)
        ]
        month_end = month_start + pd.offsets.MonthBegin(1)

        start = max(profile_start, month_start)
        end = min(profile_end, month_end)
        resolution = "Daily"

        if view == "Weekly detail":
            week_starts = []
            candidate = (
                month_start
                - pd.DateOffset(days=month_start.weekday())
            )

            while candidate < month_end:
                if candidate + pd.DateOffset(days=7) > start:
                    week_starts.append(candidate)
                candidate += pd.DateOffset(days=7)

            week_labels = [
                (
                    f"{week_start:%Y-%m-%d} to "
                    f"{week_start + pd.DateOffset(days=6):%Y-%m-%d}"
                )
                for week_start in week_starts
            ]

            selected_week = st.selectbox(
                "Week intersecting the selected month",
                week_labels,
                key=f"{key_prefix}_week",
            )
            week_start = week_starts[
                week_labels.index(selected_week)
            ]

            start = max(start, week_start)
            end = min(
                end,
                week_start + pd.DateOffset(days=7),
            )
            resolution = "Native"

    window = aggregate_process_window(
        frame,
        step_h,
        start,
        end,
        resolution,
    )

    if window.empty:
        st.info("The selected window has no process records.")
        return

    st.caption(
        f"Window: {start} to {end}, end exclusive. "
        f"Native interval: {step_h:g} h. "
        "Power and flows are duration-weighted means; "
        "MeOH bars are integrated tonnes per displayed period. "
        "Tank inventory is the last reported end-of-step value "
        "in each displayed period."
    )

    h2_factor = 1.0 if h2_unit == "kg/h" else 0.001

    graph1_options = {
        "Renewable available": "renewable_power_mw",
        "Electrolyzer H2 production": "h2_produced_kg_per_h",
        "H2 to PtMeOH": "h2_to_ptmeoh_kg_per_h",
        "Tank H2 inventory": "tank_soc_kg_h2",
    }
    graph1_options = {
        label: column
        for label, column in graph1_options.items()
        if column in window.columns
    }

    selected_graph1 = st.multiselect(
        "Graph 1 visible series",
        list(graph1_options),
        default=list(graph1_options),
        key=f"{key_prefix}_graph1_series",
    )

    if selected_graph1:
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                "Renewable availability",
                "Hydrogen flows",
                "Hydrogen storage inventory",
            ),
        )

        for label in selected_graph1:
            column = graph1_options[label]

            if column == "renewable_power_mw":
                row_number = 1
                factor = 1.0
            elif column == "tank_soc_kg_h2":
                row_number = 3
                factor = 1.0
            else:
                row_number = 2
                factor = h2_factor

            figure.add_trace(
                go.Scatter(
                    x=window["plot_time"],
                    y=window[column] * factor,
                    mode="lines+markers",
                    name=label,
                    connectgaps=False,
                ),
                row=row_number,
                col=1,
            )

        figure.update_yaxes(title_text="MW", row=1, col=1)
        figure.update_yaxes(
            title_text=h2_unit,
            row=2,
            col=1,
        )
        figure.update_yaxes(
            title_text="kg H2",
            row=3,
            col=1,
        )
        figure.update_layout(
            height=750,
            title="Renewable power, H2 production, feed, and inventory",
            paper_bgcolor=PALETTE["bg"],
            plot_bgcolor="white",
            hovermode="x unified",
            legend=dict(orientation="h"),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"{key_prefix}_process_h2",
        )

    graph2_options = {
        "MeOH production": "methanol_t_per_period",
        "H2 to PtMeOH": "h2_to_ptmeoh_kg_per_h",
        "H2 purge": "h2_purge_kg_per_h",
        "H2 recycle": "h2_recycle_kg_per_h",
    }
    graph2_options = {
        label: column
        for label, column in graph2_options.items()
        if column in window.columns
    }

    selected_graph2 = st.multiselect(
        "Graph 2 visible series",
        list(graph2_options),
        default=list(graph2_options),
        key=f"{key_prefix}_graph2_series",
    )

    if selected_graph2:
        figure = make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        for label in selected_graph2:
            column = graph2_options[label]

            if column == "methanol_t_per_period":
                trace = go.Bar(
                    x=window["plot_time"],
                    y=window[column],
                    name=label,
                    marker_color=PALETTE["orange"],
                )
                secondary = False
            else:
                trace = go.Scatter(
                    x=window["plot_time"],
                    y=window[column] * h2_factor,
                    mode="lines+markers",
                    name=label,
                    connectgaps=False,
                )
                secondary = True

            figure.add_trace(
                trace,
                secondary_y=secondary,
            )

        figure.update_yaxes(
            title_text="MeOH [t per displayed period]",
            secondary_y=False,
        )
        figure.update_yaxes(
            title_text=f"H2 flow [{h2_unit}]",
            secondary_y=True,
        )
        figure.update_layout(
            title="Methanol production and process-side H2 flows",
            paper_bgcolor=PALETTE["bg"],
            plot_bgcolor="white",
            hovermode="x unified",
            legend=dict(orientation="h"),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"{key_prefix}_process_meoh",
        )

    raw_surrogate_columns = [
        column
        for column in (
            "Model_Purge_H2",
            "Model_RECLP_H2",
        )
        if column in time_series.columns
    ]

    if raw_surrogate_columns:
        with st.expander(
            "Raw purge/recycle surrogate outputs",
            expanded=False,
        ):
            st.info(
                "These raw model outputs are shown in their native "
                "units. They are not labeled kg/h unless a "
                "unit-normalized result column is available."
            )
            st.plotly_chart(
                line_profile(
                    time_series,
                    raw_surrogate_columns,
                    "Raw surrogate predictions",
                    "Native model output units",
                ),
                use_container_width=True,
                key=f"{key_prefix}_raw_surrogates",
            )


# ---------------------------------------------------------------------
# Runtime diagnostics and exports.
# ---------------------------------------------------------------------


def render_runtime_diagnostics(
    result: SimulationArtifacts,
    key_prefix: str,
) -> None:
    time_series = result.time_series

    with st.expander(
        "Surrogate and runtime diagnostics",
        expanded=False,
    ):
        st.write(result.surrogate_info)

        if not result.model_summary.empty:
            st.dataframe(
                result.model_summary,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander(
        "Potentially problematic timesteps",
        expanded=False,
    ):
        flagged = pd.Series(
            False,
            index=time_series.index,
        )

        for column in (
            "unmet_h2_kg_per_h",
            "curtailed_power_mw",
            "power_deficit_mw",
        ):
            if column in time_series.columns:
                flagged |= (
                    pd.to_numeric(
                        time_series[column],
                        errors="coerce",
                    ) > 0
                )

        if "surrogate_all_models_in_domain" in time_series.columns:
            flagged |= (
                time_series["surrogate_all_models_in_domain"] == 0
            )

        columns = [
            column
            for column in (
                "timestamp",
                "renewable_power_mw",
                "power_to_electrolyzer_mw",
                "h2_produced_kg_per_h",
                "h2_to_ptmeoh_kg_per_h",
                "unmet_h2_kg_per_h",
                "curtailed_power_mw",
                "tank_soc_kg_h2",
                "surrogate_all_models_in_domain",
                "power_deficit_mw",
            )
            if column in time_series.columns
        ]

        st.caption(
            f"{int(flagged.sum()):,} flagged records. "
            "Preview limited to the first 200."
        )
        st.dataframe(
            time_series.loc[flagged, columns].head(200),
            use_container_width=True,
        )

    with st.expander(
        "Traceable time-series preview",
        expanded=False,
    ):
        st.dataframe(
            time_series.head(96),
            use_container_width=True,
        )
        st.caption(
            "The complete time series is available in the CSV export."
        )


def render_simulation_downloads(
    result: SimulationArtifacts,
    key_prefix: str,
) -> None:
    with st.expander("Export review files", expanded=False):
        st.caption(
            "Exports use the completed result's stored input "
            "snapshot and metadata. The workbook contains summary "
            "and economic tables; the CSV contains the full "
            "simulation time series."
        )

        cache_key = f"{key_prefix}_{current_signature}"
        prepared = st.session_state["prepared_downloads"]

        if st.button(
            "Prepare review files",
            key=f"{key_prefix}_prepare_downloads",
        ):
            payload = {
                "time_series": result.time_series.to_csv(
                    index=False
                ).encode("utf-8"),
            }

            try:
                payload["workbook"] = workbook_bytes(
                    simulation_export_sheets(result)
                )
            except Exception as exc:
                st.error(
                    f"The workbook could not be generated: {exc}. "
                    "The time-series CSV remains available."
                )

            prepared[cache_key] = payload

        payload = prepared.get(cache_key)

        if payload:
            if "workbook" in payload:
                st.download_button(
                    "Download simulation review workbook",
                    data=payload["workbook"],
                    file_name=f"{key_prefix}_review.xlsx",
                    mime=EXCEL_MIME,
                    key=f"{key_prefix}_download_workbook",
                )

            st.download_button(
                "Download complete time series",
                data=payload["time_series"],
                file_name=f"{key_prefix}_time_series.csv",
                mime="text/csv",
                key=f"{key_prefix}_download_csv",
            )


# ---------------------------------------------------------------------
# Tab 1: inputs and pre-run review.
# ---------------------------------------------------------------------


with tab1:
    st.subheader("Pre-run gates")

    if case_build_errors:
        for message in case_build_errors:
            st.error(message)
    else:
        st.success("No blocking pre-run input issues were detected.")

    render_warnings(
        pre_warnings,
        title="Pre-run warnings and assumptions",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Case input snapshot")

        if case_payload:
            st.json(json_text(case_payload))
        else:
            st.info(
                "A validated case snapshot is not yet available."
            )

        st.subheader("Model bundle checklist")
        st.dataframe(
            filtered_catalog,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Renewable profile metadata")
        st.json(json_text(effective_profile_meta))

        st.subheader("Profile-operability diagnostics")
        st.json(json_text(profile_diag))

        if not effective_profile_df.empty:
            st.dataframe(
                effective_profile_df.head(24),
                use_container_width=True,
            )

    with st.expander("Recent execution telemetry", expanded=False):
        st.code(get_telemetry_text(250), language="text")
        st.caption(
            "The interface retains the latest 250 telemetry entries. "
            "This is a bounded execution monitor, not a permanent "
            "or complete audit log."
        )


# ---------------------------------------------------------------------
# Tab 2: base-case simulation.
# ---------------------------------------------------------------------


with tab2:
    if simulation is None:
        st.info(
            "Run the annual simulation to generate base-case results."
        )
    else:
        render_warnings(
            simulation.warnings,
            title="Base-case warnings",
        )

        st.subheader("Physical and operating KPIs")

        render_metric_group(
            simulation.kpis,
            [
                ("Annual MeOH [t/year]", "annual_methanol_t", 3),
                (
                    "Electrolyzer full-load hours [h/year]",
                    "electrolyzer_full_load_hours_h",
                    1,
                ),
                (
                    "PtMeOH utilization [-]",
                    "ptmeoh_utilization_factor",
                    4,
                ),
                (
                    "Renewable utilization [-]",
                    "renewable_utilization_fraction",
                    4,
                ),
                (
                    "Unmet H2 [t/year]",
                    "h2_not_supplied_t",
                    4,
                ),
                (
                    "Curtailment fraction [-]",
                    "curtailment_fraction",
                    4,
                ),
                (
                    "Out-of-domain fraction [-]",
                    "surrogate_out_of_domain_fraction",
                    4,
                ),
                (
                    "Runtime model fraction [-]",
                    "runtime_models_fraction",
                    4,
                ),
                ("Tank empty [h]", "tank_empty_hours", 1),
                ("Tank full [h]", "tank_full_hours", 1),
                ("Curtailment [h]", "curtailed_hours", 1),
                (
                    "Total electricity [MWh/year]",
                    "annual_total_electricity_mwh",
                    3,
                ),
                (
                    "Legacy LCOMeOH [USD/t]",
                    "lcomeoh_usd_per_t_meoh",
                    2,
                ),
                ("Legacy NPV [USD]", "npv_usd", 0),
            ],
        )

        with st.expander("All simulation KPIs", expanded=False):
            st.dataframe(
                kpi_table(simulation.kpis),
                use_container_width=True,
                hide_index=True,
            )

        render_process_views(
            simulation,
            key_prefix="base",
        )
        render_runtime_diagnostics(
            simulation,
            key_prefix="base",
        )
        render_tea_results(
            simulation,
            key_prefix="base",
        )
        render_simulation_downloads(
            simulation,
            key_prefix="base",
        )


# ---------------------------------------------------------------------
# Optimization display helpers.
# ---------------------------------------------------------------------


def optimization_metric(
    results: pd.DataFrame,
    selected_objective: str,
    selected_basis: str,
    currency: str,
) -> tuple[str, bool]:
    currency_lower = currency.lower()

    if selected_objective == "max_npv":
        candidates = [f"npv_{currency_lower}"]
        ascending = False

    elif currency == "EUR" and selected_basis == "net":
        candidates = ["lcomeoh_net_eur_per_t_meoh"]
        ascending = True

    elif currency == "EUR":
        candidates = [
            "lcomeoh_gross_eur_per_t_meoh",
            "lcomeoh_eur_per_t_meoh",
        ]
        ascending = True

    else:
        candidates = ["lcomeoh_usd_per_t_meoh"]
        ascending = True

    metric = first_existing_column(results, candidates)

    if metric is None:
        raise ValueError(
            "The result table does not contain the selected "
            f"{selected_objective}/{selected_basis}/{currency} "
            "ranking metric. No different objective or currency "
            "has been substituted."
        )

    return metric, ascending


def stored_case_name(result: SimulationArtifacts) -> str | None:
    for container in (
        artifact_value(result, "input_snapshot", {}),
        artifact_value(result, "run_metadata", {}),
    ):
        if isinstance(container, dict):
            name = container.get("case_name")

            if name:
                return str(name)

    return None


# ---------------------------------------------------------------------
# Tab 3: optimization.
# ---------------------------------------------------------------------


with tab3:
    if optimization is None:
        st.info(
            "Run optimization to evaluate the design grid."
        )
    else:
        results = artifact_value(
            optimization,
            "results",
            pd.DataFrame(),
        )

        if not isinstance(results, pd.DataFrame):
            st.error(
                "The optimizer did not return a DataFrame "
                "in its results field."
            )

        elif results.empty:
            st.warning(
                "The optimizer returned no evaluated design rows."
            )

        else:
            try:
                result_currency = "EUR" if use_eur else "USD"

                metric, ascending = optimization_metric(
                    results,
                    objective,
                    lcomeoh_basis,
                    result_currency,
                )

                if "feasible" not in results.columns:
                    raise ValueError(
                        "The result table has no feasibility column. "
                        "No recommendation can be made."
                    )

                completed = pd.Series(
                    True,
                    index=results.index,
                )

                if "evaluation_status" in results.columns:
                    completed = (
                        results["evaluation_status"] == "completed"
                    )

                metric_values = pd.to_numeric(
                    results[metric],
                    errors="coerce",
                )

                eligible = results.loc[
                    completed
                    & results["feasible"].isin([True, 1])
                    & np.isfinite(metric_values)
                ].copy()

                sort_columns = [metric]
                sort_directions = [ascending]

                if "warning_count" in eligible.columns:
                    sort_columns.append("warning_count")
                    sort_directions.append(True)

                capex_column = f"total_capex_{result_currency.lower()}"

                if capex_column in eligible.columns:
                    sort_columns.append(capex_column)
                    sort_directions.append(True)

                eligible = eligible.sort_values(
                    sort_columns,
                    ascending=sort_directions,
                    kind="stable",
                )

                counts = st.columns(3)
                counts[0].metric(
                    "Returned candidate rows",
                    f"{len(results):,}",
                )
                counts[1].metric(
                    "Completed evaluations",
                    f"{int(completed.sum()):,}",
                )
                counts[2].metric(
                    "Feasible with finite objective",
                    f"{len(eligible):,}",
                )

                st.caption(
                    f"Ranking metric: {metric}. "
                    f"Order: {'ascending' if ascending else 'descending'}. "
                    "Ties use warning count and CAPEX when available. "
                    "Recommendations are limited to completed, "
                    "feasible evaluations with a finite objective."
                )

                if eligible.empty:
                    st.warning(
                        "No feasible candidate with a finite selected "
                        "objective was found. No design is recommended. "
                        "Review the constraints, failure details, "
                        "and raw candidate results."
                    )

                else:
                    recommended = eligible.iloc[0]

                    st.subheader("Recommended feasible configuration")
                    st.dataframe(
                        recommended.rename("value").to_frame(),
                        use_container_width=True,
                    )

                    st.subheader("Ranked feasible shortlist")
                    st.dataframe(
                        eligible.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                    optimized_simulation = artifact_value(
                        optimization,
                        "best_simulation",
                    )

                    recommended_name = recommended.get("case_name")

                    if optimized_simulation is None:
                        st.info(
                            "The optimizer did not attach a best-design "
                            "simulation. Its detailed economic charts "
                            "cannot be displayed without that artifact."
                        )

                    elif (
                        recommended_name is not None
                        and stored_case_name(optimized_simulation)
                        == str(recommended_name)
                    ):
                        render_warnings(
                            optimized_simulation.warnings,
                            title="Recommended-design warnings",
                        )
                        render_tea_results(
                            optimized_simulation,
                            key_prefix="optimum",
                        )
                        render_simulation_downloads(
                            optimized_simulation,
                            key_prefix="optimum",
                        )

                    else:
                        st.warning(
                            "The attached best-design simulation could "
                            "not be matched to the recommended row "
                            "using its stored case name. Detailed "
                            "economic charts are withheld to avoid "
                            "showing a different design's economics."
                        )

                st.subheader("Design-space heatmap")

                feasible_heatmap = st.checkbox(
                    "Show feasible candidates only",
                    value=True,
                    key="optimization_feasible_heatmap",
                )
                heatmap_aggregation = st.selectbox(
                    "Heatmap aggregation",
                    ["best", "mean", "min", "max"],
                    key="optimization_heatmap_aggregation",
                )

                st.caption(
                    "Each cell projects the search onto power and "
                    "storage. Different H2 targets and module counts "
                    "may occupy the same cell; the heatmap is not "
                    "a single fixed-target design surface."
                )

                try:
                    st.plotly_chart(
                        heatmap(
                            results,
                            z_col=metric,
                            feasible_only=feasible_heatmap,
                            aggregation=heatmap_aggregation,
                        ),
                        use_container_width=True,
                        key="optimization_heatmap",
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    st.warning(f"Heatmap unavailable: {exc}")

                st.download_button(
                    "Download complete optimization results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="ptmeoh_optimization_results.csv",
                    mime="text/csv",
                    key="optimization_results_download",
                )

            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))

            with st.expander(
                "Raw optimization results and failed evaluations",
                expanded=False,
            ):
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                )

                if "evaluation_status" in results.columns:
                    failed = results.loc[
                        results["evaluation_status"] != "completed"
                    ]

                    if not failed.empty:
                        st.write("Unsuccessful evaluations")
                        st.dataframe(
                            failed,
                            use_container_width=True,
                            hide_index=True,
                        )


# ---------------------------------------------------------------------
# Tab 4: sensitivity.
# ---------------------------------------------------------------------


with tab4:
    if sensitivities is None:
        st.info(
            "Run sensitivities to evaluate the low/high perturbations."
        )

    elif not isinstance(sensitivities, pd.DataFrame):
        st.error(
            "The sensitivity runner did not return a DataFrame."
        )

    elif sensitivities.empty:
        st.warning(
            "The sensitivity runner returned no experiment rows."
        )

    else:
        st.caption(
            "The chart ranks the absolute LCOMeOH span between "
            "successful low and high cases. It is not an elasticity "
            "ranking: perturbation ranges differ by parameter. "
            "Review requested and applied values, bounds, physical "
            "result reuse, and failed cases in the table."
        )

        try:
            st.plotly_chart(
                tornado(sensitivities),
                use_container_width=True,
                key="sensitivity_tornado",
            )
        except (ValueError, TypeError, KeyError) as exc:
            st.warning(f"Sensitivity chart unavailable: {exc}")

        st.dataframe(
            sensitivities,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download complete sensitivity results",
            data=sensitivities.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="ptmeoh_sensitivity_results.csv",
            mime="text/csv",
            key="sensitivity_results_download",
        )


# End of app.py — Part 3 of 3.

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from domain.data_models import TEAArtifacts


PALETTE = {
    "teal": "#0b7285",
    "green": "#2f9e44",
    "cyan": "#4dabf7",
    "slate": "#6b7f86",
    "dark": "#18333a",
    "bg": "#f4f8f8",
    "orange": "#da7101",
    "purple": "#a12c7b",
    "red": "#c92a2a",
}

SERIES_COLORS = [
    PALETTE["teal"],
    PALETTE["green"],
    PALETTE["cyan"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["slate"],
]


def _style(
    figure: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    figure.update_layout(
        title=title,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor="white",
        font=dict(color=PALETTE["dark"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=30, r=25, t=90, b=45),
        hovermode="closest",
    )

    if x_title is not None:
        figure.update_xaxes(title=x_title)

    if y_title is not None:
        figure.update_yaxes(title=y_title)

    return figure


def _empty_figure(
    title: str,
    message: str,
) -> go.Figure:
    figure = _style(go.Figure(), title)

    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)

    return figure


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Chart data must be a pandas.DataFrame.")

    if frame.columns.duplicated().any():
        raise ValueError(
            "Chart data contains duplicate column names."
        )

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Chart data is missing required columns: "
            + ", ".join(missing)
            + "."
        )


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = frame[column]

    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
    ):
        raise ValueError(
            f"Chart column {column!r} must be numeric."
        )

    return values.astype(float)


def _currency(
    frame: pd.DataFrame,
    metric: str,
) -> str:
    """Determine currency from the metric and declared result metadata."""

    metric_currency = None

    if "_eur" in metric:
        metric_currency = "EUR"
    elif "_usd" in metric:
        metric_currency = "USD"

    declared_currency = None

    if "currency" in frame.columns:
        currencies = (
            frame["currency"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
        )

        if len(currencies) > 1:
            raise ValueError(
                "A chart cannot combine results in different currencies."
            )

        if len(currencies) == 1:
            declared_currency = currencies[0]

    if declared_currency is None:
        declared_currency = frame.attrs.get("currency")

        if declared_currency is not None:
            declared_currency = str(declared_currency).upper()

    if (
        metric_currency is not None
        and declared_currency is not None
        and metric_currency != declared_currency
    ):
        raise ValueError(
            "The metric currency conflicts with the result metadata."
        )

    currency = metric_currency or declared_currency

    if currency not in {"EUR", "USD"}:
        raise ValueError(
            "The result currency could not be determined."
        )

    return currency


def _default_lcomeoh_column(frame: pd.DataFrame) -> str:
    for column in (
        "lcomeoh_eur_per_t_meoh",
        "lcomeoh_usd_per_t_meoh",
        "lcomeoh_gross_eur_per_t_meoh",
    ):
        if column in frame.columns:
            return column

    raise ValueError(
        "No supported LCOMeOH column is available."
    )


# ---------------------------------------------------------------------
# Existing profile chart.
# ---------------------------------------------------------------------


def line_profile(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    y_title: str,
) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_figure(
            title,
            "No profile data is available.",
        )

    _require_columns(df, ["timestamp"])

    figure = go.Figure()

    for index, column in enumerate(columns):
        if column not in df.columns:
            continue

        figure.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[column],
                mode="lines",
                name=column,
                connectgaps=False,
                line=dict(
                    color=SERIES_COLORS[
                        index % len(SERIES_COLORS)
                    ],
                    width=2,
                ),
            )
        )

    if not figure.data:
        return _empty_figure(
            title,
            "None of the requested series is available.",
        )

    figure = _style(
        figure,
        title,
        x_title="Timestamp",
        y_title=y_title,
    )
    figure.update_layout(hovermode="x unified")

    return figure


# ---------------------------------------------------------------------
# Design-space heatmap.
# ---------------------------------------------------------------------


def heatmap(
    results: pd.DataFrame,
    z_col: str | None = None,
    *,
    feasible_only: bool = False,
    aggregation: Literal["best", "mean", "min", "max"] = "best",
    module_count: int | None = None,
    target_h2_kg_per_h: float | None = None,
) -> go.Figure:
    """
    Display a two-dimensional projection of the evaluated design grid.

    "best" selects minimum cost or maximum NPV in each cell.
    Multiple designs can occupy a cell because module count and H2
    target are additional search dimensions.

    Aggregation is stated in the chart title.
    """

    if not isinstance(results, pd.DataFrame) or results.empty:
        return _empty_figure(
            "Design heatmap",
            "No optimization results are available.",
        )

    data = results.copy()

    if "evaluation_status" in data.columns:
        data = data.loc[
            data["evaluation_status"] == "completed"
        ].copy()

    if data.empty:
        return _empty_figure(
            "Design heatmap",
            "No successfully evaluated candidates are available.",
        )

    metric = z_col or _default_lcomeoh_column(data)

    _require_columns(
        data,
        [
            "storage_kg_h2",
            "electrolyzer_power_mw",
            metric,
        ],
    )

    currency = _currency(data, metric)

    if feasible_only:
        _require_columns(data, ["feasible"])
        data = data.loc[data["feasible"] == 1].copy()

    if module_count is not None:
        _require_columns(data, ["module_count"])
        data = data.loc[
            data["module_count"] == module_count
        ].copy()

    if target_h2_kg_per_h is not None:
        _require_columns(data, ["target_h2_kg_per_h"])
        data = data.loc[
            np.isclose(
                _numeric(data, "target_h2_kg_per_h"),
                target_h2_kg_per_h,
                rtol=1e-9,
                atol=1e-9,
            )
        ].copy()

    if data.empty:
        return _empty_figure(
            "Design heatmap",
            "No candidates match the selected filters.",
        )

    metric_values = _numeric(data, metric)
    storage_values = _numeric(data, "storage_kg_h2")
    power_values = _numeric(data, "electrolyzer_power_mw")

    finite = (
        np.isfinite(metric_values)
        & np.isfinite(storage_values)
        & np.isfinite(power_values)
    )
    omitted_count = int((~finite).sum())

    data = data.loc[finite].copy()

    if data.empty:
        return _empty_figure(
            "Design heatmap",
            "No finite metric values are available.",
        )

    if aggregation not in {"best", "mean", "min", "max"}:
        raise ValueError(
            "Heatmap aggregation must be best, mean, min, or max."
        )

    is_npv = metric.startswith("npv_")

    aggregate_function = (
        "max" if is_npv else "min"
    ) if aggregation == "best" else aggregation

    pivot = data.pivot_table(
        index="storage_kg_h2",
        columns="electrolyzer_power_mw",
        values=metric,
        aggfunc=aggregate_function,
    ).sort_index().sort_index(axis=1)

    counts = (
        data.groupby(
            ["storage_kg_h2", "electrolyzer_power_mw"]
        )
        .size()
        .unstack("electrolyzer_power_mw")
        .reindex(
            index=pivot.index,
            columns=pivot.columns,
        )
        .fillna(0)
    )

    metric_label = (
        f"NPV [{currency}]"
        if is_npv
        else f"LCOMeOH [{currency}/t MeOH]"
    )
    scope = (
        "feasible candidates"
        if feasible_only
        else "all completed candidates"
    )

    title = (
        f"Design heatmap — {metric_label}"
        f"<br><sup>{aggregate_function} per cell; {scope}. "
        "Cells may represent different H2 targets/module counts."
        "</sup>"
    )

    figure = go.Figure(
        go.Heatmap(
            x=pivot.columns.to_list(),
            y=pivot.index.to_list(),
            z=pivot.to_numpy(dtype=float),
            customdata=counts.to_numpy(dtype=int),
            colorscale=[
                [0.0, "#d9f0ef"],
                [0.5, "#8dd3d2"],
                [1.0, PALETTE["teal"]],
            ],
            colorbar=dict(title=metric_label),
            hoverongaps=False,
            hovertemplate=(
                "Electrolyzer: %{x:g} MW"
                "<br>H2 storage: %{y:g} kg"
                "<br>Value: %{z:,.3f}"
                "<br>Candidates in cell: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    figure = _style(
        figure,
        title,
        x_title="Electrolyzer nominal power [MW]",
        y_title="Usable H2 storage [kg H2]",
    )

    if omitted_count:
        figure.add_annotation(
            text=(
                f"{omitted_count} non-finite candidate values "
                "were omitted."
            ),
            x=0,
            y=-0.22,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
        )
        figure.update_layout(
            margin=dict(l=30, r=25, t=100, b=95)
        )

    return figure


# ---------------------------------------------------------------------
# Existing sensitivity-span chart, now currency-aware.
# ---------------------------------------------------------------------


def tornado(sens: pd.DataFrame) -> go.Figure:
    """
    Preserve the existing low-to-high LCOMeOH impact-span chart.

    This is not an elasticity ranking or a baseline-centered tornado.
    Both successful low and high cases are required for each parameter.
    """

    if not isinstance(sens, pd.DataFrame) or sens.empty:
        return _empty_figure(
            "Sensitivity impact",
            "No sensitivity results are available.",
        )

    data = sens.copy()

    if "evaluation_status" in data.columns:
        data = data.loc[
            data["evaluation_status"] == "completed"
        ].copy()

    if data.empty:
        return _empty_figure(
            "Sensitivity impact",
            "No successfully evaluated sensitivity cases are available.",
        )

    _require_columns(data, ["parameter", "case_label"])

    if "lcomeoh_value" in data.columns:
        metric = "lcomeoh_value"
    else:
        metric = _default_lcomeoh_column(data)

    _require_columns(data, [metric])
    currency = _currency(data, metric)

    if "lcomeoh_metric" in data.columns:
        bases = data["lcomeoh_metric"].dropna().unique()

        if len(bases) > 1:
            raise ValueError(
                "The sensitivity chart cannot mix "
                "different LCOMeOH definitions."
            )

    spans: list[dict] = []
    omitted: list[str] = []

    for parameter, group in data.groupby("parameter"):
        low = group.loc[group["case_label"] == "low"]
        high = group.loc[group["case_label"] == "high"]

        if len(low) != 1 or len(high) != 1:
            omitted.append(str(parameter))
            continue

        low_value = float(low[metric].iloc[0])
        high_value = float(high[metric].iloc[0])

        if not np.isfinite([low_value, high_value]).all():
            omitted.append(str(parameter))
            continue

        spans.append(
            {
                "parameter": str(parameter),
                "impact": abs(high_value - low_value),
                "low_value": low_value,
                "high_value": high_value,
            }
        )

    if not spans:
        return _empty_figure(
            "Sensitivity impact",
            "No complete low/high pairs with finite values are available.",
        )

    chart_data = pd.DataFrame(spans).sort_values("impact")

    figure = go.Figure(
        go.Bar(
            x=chart_data["impact"],
            y=chart_data["parameter"],
            orientation="h",
            marker_color=PALETTE["green"],
            customdata=chart_data[
                ["low_value", "high_value"]
            ].to_numpy(),
            hovertemplate=(
                "%{y}"
                "<br>Impact span: %{x:,.3f}"
                "<br>Low-input case: %{customdata[0]:,.3f}"
                "<br>High-input case: %{customdata[1]:,.3f}"
                "<extra></extra>"
            ),
        )
    )

    figure = _style(
        figure,
        "Sensitivity impact — low/high LCOMeOH span",
        x_title=f"Absolute LCOMeOH span [{currency}/t MeOH]",
        y_title="Parameter",
    )

    if omitted:
        figure.add_annotation(
            text=(
                "Incomplete or non-finite pairs omitted: "
                + ", ".join(omitted)
            ),
            x=0,
            y=-0.25,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
        )
        figure.update_layout(
            margin=dict(l=30, r=25, t=90, b=110)
        )

    return figure


# ---------------------------------------------------------------------
# Detailed EUR TEA charts.
# ---------------------------------------------------------------------


def _donut(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    center_label: str,
) -> go.Figure:
    _require_columns(
        frame,
        [label_column, value_column],
    )

    if frame.empty:
        return _empty_figure(
            title,
            "No cost breakdown is available.",
        )

    values = _numeric(frame, value_column)

    if not np.isfinite(values).all():
        raise ValueError(
            "A cost donut cannot contain missing or infinite values."
        )

    if (values < 0).any():
        raise ValueError(
            "A cost donut cannot represent negative cost components."
        )

    positive = values > 0
    data = frame.loc[positive].copy()
    total = float(values.sum())

    if not np.isfinite(total):
        raise ValueError(
            "The cost-breakdown total is not finite."
        )

    if total == 0:
        return _empty_figure(
            title,
            "All cost components are zero.",
        )

    figure = go.Figure(
        go.Pie(
            labels=data[label_column],
            values=data[value_column],
            hole=0.58,
            sort=False,
            marker=dict(colors=SERIES_COLORS),
            textinfo="percent",
            hovertemplate=(
                "%{label}"
                "<br>%{value:,.2f} EUR"
                "<br>%{percent}"
                "<extra></extra>"
            ),
        )
    )

    figure = _style(figure, title)
    figure.update_layout(
        showlegend=True,
        annotations=[
            dict(
                text=(
                    f"{center_label}"
                    f"<br>{total:,.0f} EUR"
                ),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
            )
        ],
    )

    return figure


def capex_donut(tea: TEAArtifacts) -> go.Figure:
    """Display the inclusive CAPEX breakdown without adding it twice."""

    if tea.currency != "EUR":
        raise ValueError(
            "The detailed CAPEX chart expects EUR TEA results."
        )

    _require_columns(
        tea.capex_breakdown,
        ["component", "cost_eur"],
    )

    component_total = float(
        _numeric(tea.capex_breakdown, "cost_eur").sum()
    )
    expected_total = tea.kpis.get("total_capex_eur")

    if expected_total is None or not np.isclose(
        component_total,
        float(expected_total),
        rtol=1e-8,
        atol=1e-6,
    ):
        raise ValueError(
            "The CAPEX breakdown does not match total plant CAPEX."
        )

    return _donut(
        tea.capex_breakdown,
        label_column="component",
        value_column="cost_eur",
        title="Inclusive plant CAPEX distribution",
        center_label="Total CAPEX",
    )


def variable_cost_donut(tea: TEAArtifacts) -> go.Figure:
    """Display each variable-cost item exactly once."""

    if tea.currency != "EUR":
        raise ValueError(
            "The variable-cost chart expects EUR TEA results."
        )

    _require_columns(
        tea.opex_breakdown,
        ["group", "item", "annual_cost_eur"],
    )

    variable = tea.opex_breakdown.loc[
        tea.opex_breakdown["group"] == "Variable"
    ].copy()

    variable_total = float(
        _numeric(variable, "annual_cost_eur").sum()
    )
    expected_total = tea.kpis.get("annual_variable_opex_eur")

    if expected_total is None or not np.isclose(
        variable_total,
        float(expected_total),
        rtol=1e-8,
        atol=1e-6,
    ):
        raise ValueError(
            "The variable-cost breakdown does not match "
            "total variable OPEX."
        )

    return _donut(
        variable,
        label_column="item",
        value_column="annual_cost_eur",
        title="Annual variable-cost distribution",
        center_label="Variable OPEX/year",
    )


def cash_flow_chart(
    tea: TEAArtifacts,
    *,
    discounted: bool = False,
) -> go.Figure:
    """
    Plot annual net cash flow and its cumulative value.

    Year 0 includes the investment.
    Depreciation is not plotted as an additional cash payment.
    """

    if tea.currency != "EUR":
        raise ValueError(
            "The cash-flow chart expects EUR TEA results."
        )

    annual_column = (
        "discounted_cash_flow_eur"
        if discounted
        else "net_cash_flow_eur"
    )
    cumulative_column = (
        "cumulative_discounted_cash_flow_eur"
        if discounted
        else "cumulative_cash_flow_eur"
    )

    flow = tea.cash_flow

    _require_columns(
        flow,
        [
            "project_year",
            "calendar_year",
            annual_column,
            cumulative_column,
        ],
    )

    if flow.empty:
        return _empty_figure(
            "Project cash flow",
            "No cash-flow table is available.",
        )

    data = flow.sort_values("project_year").copy()

    if data["project_year"].duplicated().any():
        raise ValueError(
            "The cash-flow table contains duplicate project years."
        )

    for column in (
        "project_year",
        "calendar_year",
        annual_column,
        cumulative_column,
    ):
        if not np.isfinite(_numeric(data, column)).all():
            raise ValueError(
                f"The cash-flow chart contains non-finite "
                f"values in {column!r}."
            )

    annual = _numeric(data, annual_column)
    cumulative = _numeric(data, cumulative_column)

    if not np.allclose(
        annual.cumsum(),
        cumulative,
        rtol=1e-8,
        atol=1e-6,
    ):
        raise ValueError(
            "Cumulative cash flow does not match "
            "the annual cash-flow series."
        )

    basis = "Discounted" if discounted else "Undiscounted"

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=data["calendar_year"],
            y=annual,
            name=f"{basis} annual net cash flow",
            marker_color=[
                PALETTE["green"] if value >= 0 else PALETTE["red"]
                for value in annual
            ],
            customdata=data[["project_year"]].to_numpy(),
            hovertemplate=(
                "Calendar year: %{x}"
                "<br>Project year: %{customdata[0]}"
                "<br>Annual net cash flow: %{y:,.2f} EUR"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data["calendar_year"],
            y=cumulative,
            mode="lines+markers",
            name=f"{basis} cumulative cash flow",
            line=dict(color=PALETTE["teal"], width=3),
            hovertemplate=(
                "Calendar year: %{x}"
                "<br>Cumulative cash flow: %{y:,.2f} EUR"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_color=PALETTE["slate"],
    )

    figure = _style(
        figure,
        f"{basis} project cash flow",
        x_title="Calendar year",
        y_title=f"Cash flow [EUR, base year {tea.price_base_year}]",
    )
    figure.update_layout(
        hovermode="x unified",
        barmode="relative",
    )
    figure.update_xaxes(
        tickformat="d",
    )

    return figure

from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from domain.data_models import CaseInputs, TEAArtifacts
from domain.units import KG_PER_T, KW_PER_MW
from domain.validators import inspect_tea_inputs


class TechnoEconomics:
    """
    Economic evaluation of simulated operating quantities.

    With case.tea:
        Detailed TEA in constant base-year EUR.

    Without case.tea:
        Previous USD methodology retained for compatibility.

    All simulation quantities must already use the units declared
    in their column names.
    """

    def __init__(self, case: CaseInputs):
        self.case = case
        self.last_artifacts: TEAArtifacts | None = None

    @staticmethod
    def _crf(rate: float, years: int) -> float:
        if years <= 0:
            raise ValueError(
                "Project lifetime must be greater than zero."
            )

        if rate < 0 or not isfinite(rate):
            raise ValueError(
                "Discount rate must be finite and nonnegative."
            )

        if rate == 0:
            return 1.0 / years

        return rate / (
            1.0 - np.exp(-years * np.log1p(rate))
        )

    @staticmethod
    def _series(
        df: pd.DataFrame,
        column: str,
        allow_negative: bool = False,
    ) -> pd.Series:
        """Read an explicitly named numeric simulation column."""

        if column not in df.columns:
            raise ValueError(
                f"The TEA requires simulation column {column!r}."
            )

        values = df[column]

        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
            or values.isna().any()
        ):
            raise ValueError(
                f"Simulation column {column!r} must contain "
                "real numeric values without missing entries."
            )

        values = values.astype(float)

        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(
                f"Simulation column {column!r} contains "
                "non-finite values."
            )

        if not allow_negative and (values < 0).any():
            raise ValueError(
                f"Simulation column {column!r} contains negative "
                "physical quantities. Raw predictions must be "
                "reviewed before economic evaluation."
            )

        return values

    @staticmethod
    def _payback(cash_flows: np.ndarray) -> float | None:
        """
        Find the first recovery of the initial investment.

        Fractional years use linear interpolation within the
        recovery year.
        """

        cumulative = float(cash_flows[0])

        if cumulative >= 0:
            return 0.0

        for year in range(1, len(cash_flows)):
            previous = cumulative
            cumulative += float(cash_flows[year])

            if cumulative >= 0 and cash_flows[year] > 0:
                return (
                    year - 1
                    + (-previous / float(cash_flows[year]))
                )

        return None

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Preserve the original dictionary-based entry point.

        EUR results include detailed tables in "tea_artifacts".
        """

        self.last_artifacts = None

        if self.case.tea is None:
            return self._compute_legacy(df)

        artifacts = self.compute_detailed(df)
        self.last_artifacts = artifacts

        return {
            **artifacts.kpis,
            "tea_artifacts": artifacts,
            "currency": "EUR",
        }

    def _capex(self) -> tuple[float, pd.DataFrame]:
        tea = self.case.tea

        if tea is None:
            raise ValueError("EUR TEA configuration is missing.")

        settings = tea.capex

        plant_cost = (
            settings.reference_plant_cost_eur
            * (
                settings.design_meoh_capacity_t_per_h
                / settings.reference_meoh_capacity_t_per_h
            ) ** settings.scaling_exponent
        )

        electrolyzer_cost = (
            self.case.electrolyzer.nominal_power_mw
            * KW_PER_MW
            * settings.electrolyzer_eur_per_kw
        )

        h2_storage_cost = (
            self.case.storage.usable_capacity_kg_h2
            * settings.h2_storage_eur_per_kg
            if self.case.storage.enabled
            else 0.0
        )

        co2_storage_cost = (
            settings.co2_storage_capacity_kg
            * settings.co2_storage_eur_per_kg
            if settings.co2_storage_enabled
            else 0.0
        )

        synthesis_cost = (
            plant_cost
            - electrolyzer_cost
            - h2_storage_cost
            - co2_storage_cost
        )

        if not all(
            isfinite(value)
            for value in (
                plant_cost,
                electrolyzer_cost,
                h2_storage_cost,
                co2_storage_cost,
                synthesis_cost,
            )
        ):
            raise ValueError(
                "The CAPEX calculation produced non-finite values."
            )

        if synthesis_cost < 0:
            raise ValueError(
                "The inclusive scaled plant CAPEX is lower than "
                "the sum of electrolyzer and storage costs. "
                "Review the reference plant cost, capacity basis, "
                "or equipment unit costs. Synthesis CAPEX will "
                "not be silently clipped to zero."
            )

        breakdown = pd.DataFrame(
            [
                {
                    "component": "PEM electrolyzer",
                    "cost_eur": electrolyzer_cost,
                },
                {
                    "component": "H2 storage",
                    "cost_eur": h2_storage_cost,
                },
                {
                    "component": "CO2 storage",
                    "cost_eur": co2_storage_cost,
                },
                {
                    "component": "Methanol synthesis",
                    "cost_eur": synthesis_cost,
                },
            ]
        )

        return float(plant_cost), breakdown

    def compute_detailed(self, df: pd.DataFrame) -> TEAArtifacts:
        """Calculate detailed EUR economics from normalized simulation data."""

        tea = self.case.tea

        if tea is None:
            raise ValueError(
                "compute_detailed() requires case.tea."
            )

        report = inspect_tea_inputs(tea)
        report.raise_if_invalid()
        warnings = list(report.warnings)

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(
                "A nonempty simulation table is required for the TEA."
            )

        if df.columns.duplicated().any():
            raise ValueError(
                "The simulation table contains duplicate column names."
            )

        dt = float(self.case.time_step_h)

        if not isfinite(dt) or dt <= 0:
            raise ValueError(
                "The simulation time step must be finite and positive."
            )

        duration_h = len(df) * dt

        if tea.annualize_profile:
            annual_factor = tea.annual_hours / duration_h

            warnings.append(
                f"Profile operating quantities were scaled by "
                f"{annual_factor:.6g} to represent "
                f"{tea.annual_hours:g} annual hours. "
                "This assumes the profile is representative. "
                "Annual fixed costs were not scaled."
            )
        else:
            if not np.isclose(
                duration_h,
                tea.annual_hours,
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    f"The simulated profile represents "
                    f"{duration_h:g} h, but the configured annual "
                    f"horizon is {tea.annual_hours:g} h. "
                    "Provide the full annual horizon or explicitly "
                    "enable profile annualization."
                )

            annual_factor = 1.0

        def annual_total(
            column: str,
            allow_negative: bool = False,
        ) -> float:
            values = self._series(
                df,
                column,
                allow_negative=allow_negative,
            )
            total = float(values.sum() * dt * annual_factor)

            if not isfinite(total):
                raise ValueError(
                    f"Annual integration of {column!r} "
                    "produced a non-finite value."
                )

            return total

        annual_meoh_t = annual_total("methanol_t_per_h")
        annual_h2_t = (
            annual_total("h2_produced_kg_per_h") / KG_PER_T
        )

        # All five electrical consumers must be explicitly provided.
        electrical_columns = {
            "Electrolyzer": "power_to_electrolyzer_mw",
            "Compressor C1": "power_c1_mw",
            "Compressor C2": "power_c2_mw",
            "Compressor C3": "power_c3_mw",
            "Electric heater HX3": "power_hx3_mw",
        }

        electrical_series = {
            equipment: self._series(df, column)
            for equipment, column in electrical_columns.items()
        }

        component_power = sum(electrical_series.values())
        reported_power = self._series(
            df,
            "total_internal_power_mw",
        )

        if not np.allclose(
            component_power.to_numpy(),
            reported_power.to_numpy(),
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError(
                "Total internal power does not match the sum of "
                "electrolyzer, compressors, and HX3 electricity."
            )

        prices = tea.prices
        utility_rows: list[dict[str, Any]] = []

        for equipment, values in electrical_series.items():
            energy_mwh = float(
                values.sum() * dt * annual_factor
            )

            utility_rows.append(
                {
                    "equipment": equipment,
                    "utility": "Electricity",
                    "annual_quantity": energy_mwh * 1000.0,
                    "quantity_unit": "kWh/year",
                    "unit_price_eur": prices.electricity_eur_per_kwh,
                    "annual_cost_eur": (
                        energy_mwh
                        * 1000.0
                        * prices.electricity_eur_per_kwh
                    ),
                }
            )

        annual_power_mwh = float(
            reported_power.sum() * dt * annual_factor
        )

        annual_downstream_power_mwh = float(
            sum(
                values.sum()
                for equipment, values in electrical_series.items()
                if equipment != "Electrolyzer"
            )
            * dt
            * annual_factor
        )

        # Normalized signed thermal duties must be provided in W.
        utilities = tea.utilities
        cp = utilities.cooling_water_cp_j_per_kg_k
        delta_t = utilities.cooling_water_delta_t_k

        water_basis_factor = (
            utilities.cooling_water_makeup_fraction
            if utilities.cooling_water_cost_basis == "makeup"
            else 1.0
        )

        annual_circulating_water_t = 0.0
        annual_charged_cooling_water_t = 0.0

        for model_name in utilities.cooling_model_names:
            column = f"{model_name}__w"
            signed_duty = self._series(
                df,
                column,
                allow_negative=True,
            )

            if utilities.cooling_duty_sign == "negative_is_cooling":
                cooling_duty_w = (-signed_duty).clip(lower=0.0)
                opposite_sign = signed_duty > 0
            else:
                cooling_duty_w = signed_duty.clip(lower=0.0)
                opposite_sign = signed_duty < 0

            if opposite_sign.any():
                warnings.append(
                    f"{model_name}: some duties have the opposite "
                    "sign to the configured cooling convention. "
                    "Those duties do not create cooling-water demand. "
                    "Verify the model sign convention."
                )

            # W / [J/(kg K) * K] gives kg/s.
            water_kg_per_s = cooling_duty_w / (cp * delta_t)

            circulating_t = float(
                water_kg_per_s.sum()
                * 3600.0
                * dt
                * annual_factor
                / KG_PER_T
            )
            charged_t = circulating_t * water_basis_factor

            annual_circulating_water_t += circulating_t
            annual_charged_cooling_water_t += charged_t

            utility_rows.append(
                {
                    "equipment": model_name,
                    "utility": "Cooling water",
                    "annual_quantity": charged_t,
                    "quantity_unit": "t/year",
                    "unit_price_eur": prices.cooling_water_eur_per_t,
                    "annual_cost_eur": (
                        charged_t * prices.cooling_water_eur_per_t
                    ),
                }
            )

        utility_breakdown = pd.DataFrame(utility_rows)

        electricity_cost = float(
            utility_breakdown.loc[
                utility_breakdown["utility"] == "Electricity",
                "annual_cost_eur",
            ].sum()
        )
        cooling_water_cost = float(
            utility_breakdown.loc[
                utility_breakdown["utility"] == "Cooling water",
                "annual_cost_eur",
            ].sum()
        )

        # Raw materials and coproducts.
        if tea.co2_feed.source == "surrogate":
            if "co2_feed_kg_per_h" in df.columns:
                annual_co2_t = (
                    annual_total("co2_feed_kg_per_h") / KG_PER_T
                )
                co2_source = "surrogate"

            elif tea.co2_feed.allow_stoichiometric_fallback:
                annual_co2_t = (
                    annual_meoh_t
                    * tea.co2_feed.kg_co2_per_kg_meoh
                )
                co2_source = "stoichiometric_fallback"

                warnings.append(
                    "The normalized CO2 model series is unavailable. "
                    "The explicitly permitted stoichiometric "
                    "fallback was used."
                )
            else:
                raise ValueError(
                    "Surrogate-based CO2 consumption requires "
                    "co2_feed_kg_per_h in the simulation results."
                )
        else:
            annual_co2_t = (
                annual_meoh_t
                * tea.co2_feed.kg_co2_per_kg_meoh
            )
            co2_source = "stoichiometric"

        annual_electrolysis_water_t = (
            annual_h2_t
            * utilities.electrolysis_water_kg_per_kg_h2
        )

        annual_oxygen_produced_t = (
            annual_h2_t * utilities.oxygen_kg_per_kg_h2
        )

        coproducts = tea.coproducts

        annual_oxygen_sold_t = (
            annual_oxygen_produced_t
            * coproducts.oxygen_saleable_fraction
            if coproducts.oxygen_sales_enabled
            else 0.0
        )

        oxygen_conditioning_cost = (
            annual_oxygen_sold_t
            * coproducts.oxygen_conditioning_eur_per_t
        )

        co2_cost = annual_co2_t * prices.captured_co2_eur_per_t
        electrolysis_water_cost = (
            annual_electrolysis_water_t
            * prices.electrolysis_water_eur_per_t
        )

        raw_material_cost = co2_cost + electrolysis_water_cost

        variable_costs = {
            "Captured CO2": co2_cost,
            "Electrolysis water": electrolysis_water_cost,
            "Electricity": electricity_cost,
            "Cooling water": cooling_water_cost,
            "Oxygen conditioning": oxygen_conditioning_cost,
        }
        variable_total = float(sum(variable_costs.values()))

        total_capex, capex_breakdown = self._capex()
        factors = tea.opex

        maintenance = (
            total_capex * factors.maintenance_fraction_of_capex
        )
        annual_taxes = (
            total_capex * factors.annual_taxes_fraction_of_capex
        )

        fixed_costs = {
            "Operating labor": factors.operating_labor_eur_per_year,
            "Operating supervision": (
                factors.operating_supervision_eur_per_year
            ),
            "Maintenance and repair": maintenance,
            "Operating supplies": (
                maintenance
                * factors.supplies_fraction_of_maintenance
            ),
            "Royalties": (
                variable_total
                * factors.royalties_fraction_of_variable_cost
            ),
            "Annual CAPEX-based taxes": annual_taxes,
            "Insurance": (
                total_capex * factors.insurance_fraction_of_capex
            ),
            "Plant overhead": (
                (
                    factors.operating_labor_eur_per_year
                    + factors.operating_supervision_eur_per_year
                    + maintenance
                )
                * factors.overhead_fraction
            ),
            "Administration": (
                factors.operating_labor_eur_per_year
                * factors.administration_fraction_of_labor
            ),
            "Working capital interest charge": (
                total_capex
                * factors.working_capital_interest_fraction_of_capex
            ),
        }

        fixed_total = float(sum(fixed_costs.values()))
        cash_opex = variable_total + fixed_total
        depreciation = total_capex / tea.project_years
        accounting_opex = cash_opex + depreciation

        opex_rows = [
            {
                "group": group,
                "item": item,
                "annual_cost_eur": float(cost),
                "is_cash": True,
            }
            for group, costs in (
                ("Variable", variable_costs),
                ("Fixed", fixed_costs),
            )
            for item, cost in costs.items()
        ]
        opex_rows.append(
            {
                "group": "Non-cash",
                "item": "Straight-line depreciation",
                "annual_cost_eur": depreciation,
                "is_cash": False,
            }
        )

        methanol_revenue = (
            annual_meoh_t * prices.methanol_eur_per_t
        )
        oxygen_revenue = (
            annual_oxygen_sold_t * prices.oxygen_eur_per_t
            if coproducts.oxygen_sales_enabled
            else 0.0
        )
        total_revenue = methanol_revenue + oxygen_revenue
        operating_cash_flow = total_revenue - cash_opex

        revenue_breakdown = pd.DataFrame(
            [
                {
                    "product": "Methanol",
                    "annual_quantity_t": annual_meoh_t,
                    "price_eur_per_t": prices.methanol_eur_per_t,
                    "annual_revenue_eur": methanol_revenue,
                },
                {
                    "product": "Oxygen",
                    "annual_quantity_t": annual_oxygen_sold_t,
                    "price_eur_per_t": (
                        prices.oxygen_eur_per_t
                        if coproducts.oxygen_sales_enabled
                        else 0.0
                    ),
                    "annual_revenue_eur": oxygen_revenue,
                },
            ]
        )

        # Annual taxes are already included in cash_opex.
        # Remove them only when displaying a before-tax subtotal.
        profit_before_annual_taxes = (
            total_revenue
            - (cash_opex - annual_taxes)
            - depreciation
        )
        net_accounting_profit = (
            profit_before_annual_taxes - annual_taxes
        )

        margin_values = {
            "gross_margin_eur": (
                total_revenue - raw_material_cost
            ),
            "variable_contribution_margin_eur": (
                total_revenue - variable_total
            ),
            "main_product_cash_margin_eur": (
                methanol_revenue - cash_opex
            ),
            "accounting_profit_before_annual_taxes_eur": (
                profit_before_annual_taxes
            ),
            "net_accounting_profit_eur": net_accounting_profit,
            "operating_cash_flow_eur": operating_cash_flow,
        }

        margins = pd.DataFrame(
            [
                {
                    "metric": name,
                    "annual_value_eur": value,
                }
                for name, value in margin_values.items()
            ]
        )

        # Year 0 is construction/investment.
        # All operating-year quantities are constant in base-year EUR.
        years = np.arange(tea.project_years + 1)
        operating = years > 0
        discount = np.exp(
            -years * np.log1p(tea.discount_rate)
        )

        capex_series = np.where(
            operating,
            0.0,
            total_capex,
        )
        opex_series = np.where(
            operating,
            cash_opex,
            0.0,
        )
        depreciation_series = np.where(
            operating,
            depreciation,
            0.0,
        )
        meoh_series = np.where(
            operating,
            annual_meoh_t,
            0.0,
        )
        main_revenue_series = np.where(
            operating,
            methanol_revenue,
            0.0,
        )
        coproduct_revenue_series = np.where(
            operating,
            oxygen_revenue,
            0.0,
        )

        revenue_series = (
            main_revenue_series + coproduct_revenue_series
        )
        cash_flow = (
            revenue_series - opex_series - capex_series
        )
        discounted_cash_flow = cash_flow * discount
        gross_cost_series = capex_series + opex_series
        net_cost_series = (
            gross_cost_series - coproduct_revenue_series
        )

        flow_table = pd.DataFrame(
            {
                "project_year": years,
                "calendar_year": tea.project_start_year + years,
                "capex_eur": capex_series,
                "cash_opex_eur": opex_series,
                "depreciation_eur": depreciation_series,
                "methanol_production_t": meoh_series,
                "methanol_revenue_eur": main_revenue_series,
                "coproduct_revenue_eur": coproduct_revenue_series,
                "total_revenue_eur": revenue_series,
                "discount_factor": discount,
                "gross_cost_eur": gross_cost_series,
                "net_cost_after_coproducts_eur": net_cost_series,
                "discounted_gross_cost_eur": (
                    gross_cost_series * discount
                ),
                "discounted_net_cost_eur": (
                    net_cost_series * discount
                ),
                "discounted_production_t": meoh_series * discount,
                "net_cash_flow_eur": cash_flow,
                "discounted_cash_flow_eur": discounted_cash_flow,
                "cumulative_cash_flow_eur": np.cumsum(cash_flow),
                "cumulative_discounted_cash_flow_eur": (
                    np.cumsum(discounted_cash_flow)
                ),
            }
        )

        discounted_production = float(
            flow_table["discounted_production_t"].sum()
        )
        discounted_gross_cost = float(
            flow_table["discounted_gross_cost_eur"].sum()
        )
        discounted_net_cost = float(
            flow_table["discounted_net_cost_eur"].sum()
        )

        if discounted_production > 0:
            gross_lcomeoh = (
                discounted_gross_cost / discounted_production
            )
            net_lcomeoh = (
                discounted_net_cost / discounted_production
            )
        else:
            gross_lcomeoh = float("inf")
            net_lcomeoh = float("inf")
            warnings.append(
                "Methanol production is zero. LCOMeOH is undefined "
                "and is represented as infinity."
            )

        selected_lcomeoh = (
            net_lcomeoh
            if self.case.optimization.lcomeoh_basis == "net"
            else gross_lcomeoh
        )

        simple_payback = self._payback(cash_flow)
        discounted_payback = self._payback(
            discounted_cash_flow
        )

        if simple_payback is None:
            warnings.append(
                "The investment is not recovered within "
                f"{tea.project_years} years on an undiscounted basis."
            )

        if discounted_payback is None:
            warnings.append(
                "The investment is not recovered within "
                f"{tea.project_years} years on a discounted basis."
            )

        if "power_deficit_mw" in df.columns:
            deficit_mwh = annual_total("power_deficit_mw")

            if deficit_mwh > 1e-6:
                warnings.append(
                    "The simulation includes an electricity deficit. "
                    "Costs include the reported internal electrical "
                    "demand, but this TEA does not establish that "
                    "the missing electricity can physically be supplied."
                )

        warnings.append(
            "Financial results inherit the validity and extrapolation "
            "limitations of the simulation and surrogate models."
        )

        kpis = {
            "annual_meoh_t": annual_meoh_t,
            "annual_h2_t": annual_h2_t,
            "annual_co2_t": annual_co2_t,
            "annual_electrolysis_water_t": (
                annual_electrolysis_water_t
            ),
            "annual_circulating_cooling_water_t": (
                annual_circulating_water_t
            ),
            "annual_charged_cooling_water_t": (
                annual_charged_cooling_water_t
            ),
            "annual_oxygen_produced_t": annual_oxygen_produced_t,
            "annual_oxygen_sold_t": annual_oxygen_sold_t,
            "annual_power_mwh": annual_power_mwh,
            "annual_downstream_power_mwh": (
                annual_downstream_power_mwh
            ),
            "total_capex_eur": total_capex,
            "annual_variable_opex_eur": variable_total,
            "annual_fixed_opex_eur": fixed_total,
            "annual_opex_eur": cash_opex,
            "annual_cash_opex_eur": cash_opex,
            "annual_depreciation_eur": depreciation,
            "annual_accounting_opex_eur": accounting_opex,
            "annual_taxes_eur": annual_taxes,
            "annual_raw_material_cost_eur": raw_material_cost,
            "annual_electricity_cost_eur": electricity_cost,
            "annual_cooling_water_cost_eur": cooling_water_cost,
            "annual_methanol_revenue_eur": methanol_revenue,
            "annual_oxygen_revenue_eur": oxygen_revenue,
            "annual_total_revenue_eur": total_revenue,
            "annual_operating_cash_flow_eur": operating_cash_flow,
            "discounted_gross_cost_eur": discounted_gross_cost,
            "discounted_net_cost_eur": discounted_net_cost,
            "discounted_methanol_production_t": (
                discounted_production
            ),
            "lcomeoh_gross_eur_per_t_meoh": gross_lcomeoh,
            "lcomeoh_net_eur_per_t_meoh": net_lcomeoh,
            "lcomeoh_eur_per_t_meoh": selected_lcomeoh,
            "npv_eur": float(discounted_cash_flow.sum()),
            **margin_values,
        }

        # Infinity is intentional only for undefined zero-production
        # levelized costs. Other non-finite results are calculation errors.
        for name, value in kpis.items():
            if name.startswith("lcomeoh") and annual_meoh_t == 0:
                continue

            if not isfinite(float(value)):
                raise ValueError(
                    f"The TEA produced a non-finite result for {name}."
                )

        if not np.isfinite(
            flow_table.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "The financial cash-flow table contains "
                "non-finite values."
            )

        artifacts = TEAArtifacts(
            currency="EUR",
            price_base_year=tea.price_base_year,
            kpis=kpis,
            status={
                "simple_payback_recovered": (
                    simple_payback is not None
                ),
                "discounted_payback_recovered": (
                    discounted_payback is not None
                ),
                "methanol_production_positive": (
                    annual_meoh_t > 0
                ),
                "co2_source": co2_source,
                "lcomeoh_basis": (
                    self.case.optimization.lcomeoh_basis
                ),
            },
            capex_breakdown=capex_breakdown,
            utility_breakdown=utility_breakdown,
            opex_breakdown=pd.DataFrame(opex_rows),
            revenue_breakdown=revenue_breakdown,
            margins=margins,
            cash_flow=flow_table,
            simple_payback_years=simple_payback,
            discounted_payback_years=discounted_payback,
            assumptions={
                "tea_inputs": asdict(tea),
                "profile_duration_h": duration_h,
                "annualization_factor": annual_factor,
                "annual_operating_horizon_h": tea.annual_hours,
                "first_operating_year": (
                    tea.project_start_year + 1
                ),
                "price_basis": "constant base-year EUR",
                "future_price_escalation": False,
                "corporate_income_tax_modeled": False,
                "annual_capex_taxes_counted_once": True,
                "depreciation_is_cash_expenditure": False,
                "plant_capex_is_inclusive": True,
                "residual_value_eur": 0.0,
                "replacement_investment_eur": 0.0,
                "annual_production_repeated_over_project": True,
                "payback_interpolation": (
                    "linear within recovery year"
                ),
                "legacy_scenario_multipliers_applied": False,
            },
            warnings=list(dict.fromkeys(warnings)),
        )

        self.last_artifacts = artifacts
        return artifacts

    def _compute_legacy(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Preserve the previous USD economic methodology.

        The legacy LCOH allocates whole-plant costs to H2 production.
        It is retained for compatibility, not relabeled as a
        subsystem-only hydrogen cost.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "Simulation results must be a pandas.DataFrame."
            )

        dt = float(self.case.time_step_h)

        def total(column: str) -> float:
            if column not in df.columns:
                return 0.0
            return float(self._series(df, column).sum() * dt)

        annual_meoh_t = total("methanol_t_per_h")
        annual_power_mwh = total("total_internal_power_mw")
        annual_downstream_power_mwh = total(
            "downstream_aux_power_mw"
        )
        annual_h2_t = (
            total("h2_produced_kg_per_h") / KG_PER_T
        )

        electrolyzer_capex = (
            self.case.electrolyzer.nominal_power_mw
            * KW_PER_MW
            * self.case.electrolyzer.capex_usd_per_kw
        )
        storage_capex = (
            self.case.storage.usable_capacity_kg_h2
            * self.case.storage.capex_usd_per_kg_h2
            if self.case.storage.enabled
            else 0.0
        )
        downstream_capex = 0.15 * electrolyzer_capex

        economic = self.case.economic

        total_capex = (
            electrolyzer_capex
            + storage_capex
            + downstream_capex
        ) * economic.capex_multiplier

        fixed_opex = (
            electrolyzer_capex
            * self.case.electrolyzer.fixed_opex_fraction
            + storage_capex * 0.01
            + downstream_capex * 0.03
        )

        electricity_cost = (
            annual_power_mwh
            * economic.electricity_price_usd_per_mwh
        )
        co2_cost = (
            annual_meoh_t
            * 1.375
            * economic.co2_price_usd_per_t
        )

        annual_opex = (
            fixed_opex + electricity_cost + co2_cost
        ) * economic.opex_multiplier

        annualized_capex = total_capex * self._crf(
            economic.discount_rate,
            economic.project_years,
        )

        annual_revenue = (
            annual_meoh_t * economic.methanol_price_usd_per_t
        )
        annual_cashflow = annual_revenue - annual_opex

        npv = -total_capex

        for year in range(1, economic.project_years + 1):
            npv += annual_cashflow / (
                (1.0 + economic.discount_rate) ** year
            )

        return {
            "annual_meoh_t": annual_meoh_t,
            "annual_power_mwh": annual_power_mwh,
            "annual_downstream_power_mwh": (
                annual_downstream_power_mwh
            ),
            "total_capex_usd": total_capex,
            "annual_opex_usd": annual_opex,
            "lcoh_usd_per_t_h2": (
                (annualized_capex + annual_opex) / annual_h2_t
                if annual_h2_t > 1e-9
                else float("inf")
            ),
            "lcomeoh_usd_per_t_meoh": (
                (annualized_capex + annual_opex) / annual_meoh_t
                if annual_meoh_t > 1e-9
                else float("inf")
            ),
            "npv_usd": npv,
        }

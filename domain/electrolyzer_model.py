from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from domain.units import KW_PER_MW


@dataclass
class ElectrolyzerState:
    """Instantaneous electrolyzer operating state."""

    # Original fields retained for compatibility.
    power_to_electrolyzer_mw: float
    module_count_online: int
    h2_produced_kg_per_h: float

    # Additional operational diagnostics.
    available_power_mw: float = 0.0
    unused_available_power_mw: float = 0.0
    power_per_online_module_mw: float = 0.0
    online_module_load_fraction: float = 0.0
    plant_load_fraction: float = 0.0

    below_minimum_module_load: bool = False
    minimum_load_limited: bool = False
    nominal_capacity_limited: bool = False


class ElectrolyzerModel:
    """
    Modular electrolyzer with constant specific energy consumption.

    Modules are identical and independently dispatchable.
    Online modules share the accepted electrical power equally.

    The model does not introduce startup energy, ramp limits,
    degradation, standby consumption, or part-load efficiency curves.
    """

    def __init__(
        self,
        nominal_power_mw: float,
        module_size_mw: float,
        min_load_fraction: float,
        specific_energy_kwh_per_kg_h2: float,
    ):
        self.nominal_power_mw = self._positive(
            nominal_power_mw,
            "Electrolyzer nominal power [MW]",
        )
        self.module_size_mw = self._positive(
            module_size_mw,
            "Electrolyzer module size [MW]",
        )
        self.min_load_fraction = self._nonnegative(
            min_load_fraction,
            "Electrolyzer minimum load fraction",
        )
        self.specific_energy_kwh_per_kg_h2 = self._positive(
            specific_energy_kwh_per_kg_h2,
            "Specific energy consumption [kWh/kg H2]",
        )

        if self.min_load_fraction > 1.0:
            raise ValueError(
                "Electrolyzer minimum load fraction must not exceed 1."
            )

        module_ratio = (
            self.nominal_power_mw / self.module_size_mw
        )

        if not math.isfinite(module_ratio):
            raise ValueError(
                "The nominal-power-to-module-size ratio must be finite."
            )

        module_count = int(round(module_ratio))

        if module_count < 1 or not math.isclose(
            module_ratio,
            module_count,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Nominal electrolyzer power must correspond to an "
                "integer number of identical modules. "
                "Set module_size_mw = nominal_power_mw / module_count."
            )

        self.total_modules = module_count

        # Normalize negligible rounding differences while preserving
        # the installed plant capacity exactly.
        self.module_size_mw = (
            self.nominal_power_mw / self.total_modules
        )

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        """Validate a finite, nonnegative scalar."""

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a finite, nonnegative number."
            )

        return float(value)

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        """Validate a finite, strictly positive scalar."""

        numeric = cls._nonnegative(value, name)

        if numeric <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

        return numeric

    @property
    def minimum_module_power_mw(self) -> float:
        """Minimum electrical power required by one online module."""

        return self.module_size_mw * self.min_load_fraction

    @property
    def minimum_plant_operating_power_mw(self) -> float:
        """Minimum power needed to operate at least one module."""

        return self.minimum_module_power_mw

    def step(
        self,
        available_power_mw: float,
    ) -> ElectrolyzerState:
        """
        Dispatch modules for the available electrical power.

        Available power must already exclude other plant loads
        allocated by the simulation engine.

        This method is stateless and can be called repeatedly during
        an internal power-balance iteration.
        """

        available = self._nonnegative(
            available_power_mw,
            "Available electrolyzer power [MW]",
        )

        usable_power = min(
            available,
            self.nominal_power_mw,
        )
        minimum_module_power = self.minimum_module_power_mw

        below_minimum = (
            usable_power > 0.0
            and usable_power < minimum_module_power
        )

        if usable_power <= 0.0 or below_minimum:
            return ElectrolyzerState(
                power_to_electrolyzer_mw=0.0,
                module_count_online=0,
                h2_produced_kg_per_h=0.0,
                available_power_mw=available,
                unused_available_power_mw=available,
                below_minimum_module_load=below_minimum,
                minimum_load_limited=below_minimum,
                nominal_capacity_limited=(
                    available > self.nominal_power_mw
                ),
            )

        # Initially choose the smallest number of modules that could
        # absorb all usable power without exceeding nominal module power.
        module_count_online = min(
            self.total_modules,
            max(
                1,
                math.ceil(
                    usable_power / self.module_size_mw
                ),
            ),
        )

        # High minimum-load fractions can create operating gaps.
        # Reduce the module count when the initial combination cannot
        # satisfy the minimum load of every online module.
        if (
            minimum_module_power > 0.0
            and usable_power
            < module_count_online * minimum_module_power
        ):
            module_count_online = min(
                module_count_online,
                math.floor(
                    usable_power / minimum_module_power
                ),
            )

        if module_count_online <= 0:
            return ElectrolyzerState(
                power_to_electrolyzer_mw=0.0,
                module_count_online=0,
                h2_produced_kg_per_h=0.0,
                available_power_mw=available,
                unused_available_power_mw=available,
                below_minimum_module_load=True,
                minimum_load_limited=True,
                nominal_capacity_limited=(
                    available > self.nominal_power_mw
                ),
            )

        accepted_power = min(
            usable_power,
            module_count_online * self.module_size_mw,
        )

        power_per_module = (
            accepted_power / module_count_online
        )

        h2_produced = (
            accepted_power
            * KW_PER_MW
            / self.specific_energy_kwh_per_kg_h2
        )

        if not math.isfinite(h2_produced):
            raise ValueError(
                "Electrolyzer H2 production is not finite. "
                "Check power and specific energy consumption."
            )

        tolerance_mw = max(
            1e-12,
            self.nominal_power_mw * 1e-12,
        )

        return ElectrolyzerState(
            power_to_electrolyzer_mw=accepted_power,
            module_count_online=module_count_online,
            h2_produced_kg_per_h=h2_produced,
            available_power_mw=available,
            unused_available_power_mw=max(
                available - accepted_power,
                0.0,
            ),
            power_per_online_module_mw=power_per_module,
            online_module_load_fraction=(
                power_per_module / self.module_size_mw
            ),
            plant_load_fraction=(
                accepted_power / self.nominal_power_mw
            ),
            below_minimum_module_load=False,
            minimum_load_limited=(
                usable_power - accepted_power > tolerance_mw
            ),
            nominal_capacity_limited=(
                available - self.nominal_power_mw > tolerance_mw
            ),
        )

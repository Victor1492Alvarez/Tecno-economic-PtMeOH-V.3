from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class H2StorageState:
    """Storage result for one simulation interval."""

    time_step_h: float

    starting_soc_kg_h2: float
    ending_soc_kg_h2: float

    requested_charge_kg_per_h: float
    requested_discharge_kg_per_h: float

    actual_charge_kg_per_h: float
    actual_discharge_kg_per_h: float

    unaccepted_charge_kg_per_h: float
    unserved_discharge_kg_per_h: float

    charged_mass_kg_h2: float
    discharged_mass_kg_h2: float

    charge_capacity_limited: bool
    charge_rate_limited: bool
    discharge_inventory_limited: bool
    discharge_rate_limited: bool

    is_full: bool
    is_empty: bool

    balance_residual_kg_h2: float


class H2StorageModel:
    """
    Usable hydrogen inventory with charge and discharge rate limits.

    Flow rates are expressed in kg/h.
    Inventory and capacity are expressed in kg.
    The time step is expressed in hours.

    No leakage, storage losses, or compression energy are assumed.
    Charging and discharging within the same interval are not allowed.

    preview() calculates a possible transition without changing inventory.
    step() calculates and commits one transition.
    """

    def __init__(
        self,
        capacity_kg_h2: float,
        initial_soc_fraction: float,
        max_charge_kg_per_h: float | None = None,
        max_discharge_kg_per_h: float | None = None,
    ):
        self.capacity_kg_h2 = self._nonnegative(
            capacity_kg_h2,
            "Usable H2 capacity [kg]",
        )

        fraction = self._nonnegative(
            initial_soc_fraction,
            "Initial H2 inventory fraction",
        )

        if fraction > 1.0:
            raise ValueError(
                "Initial H2 inventory fraction must not exceed 1."
            )

        self.initial_soc_fraction = fraction

        # Preserve the previous default: a numerical rate limit equal
        # to the usable capacity, corresponding to a one-hour turnover.
        self.max_charge_kg_per_h = (
            self.capacity_kg_h2
            if max_charge_kg_per_h is None
            else self._nonnegative(
                max_charge_kg_per_h,
                "Maximum H2 charge rate [kg/h]",
            )
        )

        self.max_discharge_kg_per_h = (
            self.capacity_kg_h2
            if max_discharge_kg_per_h is None
            else self._nonnegative(
                max_discharge_kg_per_h,
                "Maximum H2 discharge rate [kg/h]",
            )
        )

        self.soc_kg_h2 = (
            self.capacity_kg_h2 * self.initial_soc_fraction
        )

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        """Validate a finite, nonnegative scalar."""

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
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
    def inventory_tolerance_kg(self) -> float:
        """Tolerance used only for numerical boundary checks."""

        return max(1e-9, self.capacity_kg_h2 * 1e-12)

    @property
    def available_capacity_kg_h2(self) -> float:
        """Remaining usable storage capacity."""

        inventory = self._validated_inventory(self.soc_kg_h2)
        return max(self.capacity_kg_h2 - inventory, 0.0)

    @property
    def soc_fraction(self) -> float:
        """Current inventory as a fraction of usable capacity."""

        inventory = self._validated_inventory(self.soc_kg_h2)

        if self.capacity_kg_h2 == 0:
            return 0.0

        return inventory / self.capacity_kg_h2

    def _validated_inventory(self, value: float) -> float:
        """
        Validate inventory without silently clipping physical errors.

        Only floating-point deviations within the inventory tolerance
        are normalized to the nearest boundary.
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
        ):
            raise ValueError(
                "H2 inventory must be a finite number."
            )

        inventory = float(value)
        tolerance = self.inventory_tolerance_kg

        if inventory < -tolerance:
            raise ValueError(
                "H2 inventory cannot be negative."
            )

        if inventory > self.capacity_kg_h2 + tolerance:
            raise ValueError(
                "H2 inventory exceeds the usable storage capacity."
            )

        return min(
            max(inventory, 0.0),
            self.capacity_kg_h2,
        )

    def reset(
        self,
        initial_soc_fraction: float | None = None,
    ) -> None:
        """
        Reset inventory to the configured or explicitly supplied fraction.

        An explicitly supplied fraction updates the reset configuration.
        """

        if initial_soc_fraction is not None:
            fraction = self._nonnegative(
                initial_soc_fraction,
                "Initial H2 inventory fraction",
            )

            if fraction > 1.0:
                raise ValueError(
                    "Initial H2 inventory fraction must not exceed 1."
                )

            self.initial_soc_fraction = fraction

        self.soc_kg_h2 = (
            self.capacity_kg_h2 * self.initial_soc_fraction
        )

    def set_inventory(self, inventory_kg_h2: float) -> None:
        """Set inventory explicitly after validating its physical bounds."""

        self.soc_kg_h2 = self._validated_inventory(
            inventory_kg_h2
        )

    def preview(
        self,
        requested_charge_kg_per_h: float = 0.0,
        requested_discharge_kg_per_h: float = 0.0,
        time_step_h: float = 1.0,
        starting_soc_kg_h2: float | None = None,
    ) -> H2StorageState:
        """
        Calculate a storage transition without modifying the tank.

        An explicit starting inventory supports repeated internal
        calculations from the same initial state.

        Unaccepted charging is not automatically classified as
        curtailment. The process dispatcher must decide how to use
        or account for that hydrogen.
        """

        dt = self._positive(
            time_step_h,
            "Time step [h]",
        )

        requested_charge = self._nonnegative(
            requested_charge_kg_per_h,
            "Requested H2 charge rate [kg/h]",
        )
        requested_discharge = self._nonnegative(
            requested_discharge_kg_per_h,
            "Requested H2 discharge rate [kg/h]",
        )

        if requested_charge > 0 and requested_discharge > 0:
            raise ValueError(
                "Simultaneous H2 charging and discharging are not "
                "supported within the same interval."
            )

        starting_inventory = self._validated_inventory(
            self.soc_kg_h2
            if starting_soc_kg_h2 is None
            else starting_soc_kg_h2
        )

        available_capacity = max(
            self.capacity_kg_h2 - starting_inventory,
            0.0,
        )

        charge_capacity_limit = available_capacity / dt
        discharge_inventory_limit = starting_inventory / dt

        charge_rate = min(
            requested_charge,
            self.max_charge_kg_per_h,
            charge_capacity_limit,
        )
        discharge_rate = min(
            requested_discharge,
            self.max_discharge_kg_per_h,
            discharge_inventory_limit,
        )

        charged_mass = charge_rate * dt
        discharged_mass = discharge_rate * dt

        ending_inventory = self._validated_inventory(
            starting_inventory
            + charged_mass
            - discharged_mass
        )

        balance_residual = (
            starting_inventory
            + charged_mass
            - discharged_mass
            - ending_inventory
        )

        tolerance = self.inventory_tolerance_kg

        if abs(balance_residual) > tolerance:
            raise RuntimeError(
                "The H2 storage inventory balance did not close: "
                f"residual={balance_residual:.12g} kg."
            )

        # Express the comparison tolerance on the same flow-rate basis.
        flow_tolerance = tolerance / dt

        return H2StorageState(
            time_step_h=dt,
            starting_soc_kg_h2=starting_inventory,
            ending_soc_kg_h2=ending_inventory,
            requested_charge_kg_per_h=requested_charge,
            requested_discharge_kg_per_h=requested_discharge,
            actual_charge_kg_per_h=charge_rate,
            actual_discharge_kg_per_h=discharge_rate,
            unaccepted_charge_kg_per_h=max(
                requested_charge - charge_rate,
                0.0,
            ),
            unserved_discharge_kg_per_h=max(
                requested_discharge - discharge_rate,
                0.0,
            ),
            charged_mass_kg_h2=charged_mass,
            discharged_mass_kg_h2=discharged_mass,
            charge_capacity_limited=(
                requested_charge > charge_rate + flow_tolerance
                and charge_capacity_limit
                <= charge_rate + flow_tolerance
            ),
            charge_rate_limited=(
                requested_charge > charge_rate + flow_tolerance
                and self.max_charge_kg_per_h
                <= charge_rate + flow_tolerance
            ),
            discharge_inventory_limited=(
                requested_discharge
                > discharge_rate + flow_tolerance
                and discharge_inventory_limit
                <= discharge_rate + flow_tolerance
            ),
            discharge_rate_limited=(
                requested_discharge
                > discharge_rate + flow_tolerance
                and self.max_discharge_kg_per_h
                <= discharge_rate + flow_tolerance
            ),
            is_full=(
                self.capacity_kg_h2 - ending_inventory
                <= tolerance
            ),
            is_empty=(
                ending_inventory <= tolerance
            ),
            balance_residual_kg_h2=balance_residual,
        )

    def step(
        self,
        requested_charge_kg_per_h: float = 0.0,
        requested_discharge_kg_per_h: float = 0.0,
        time_step_h: float = 1.0,
    ) -> H2StorageState:
        """Calculate and commit exactly one simulation interval."""

        state = self.preview(
            requested_charge_kg_per_h=requested_charge_kg_per_h,
            requested_discharge_kg_per_h=requested_discharge_kg_per_h,
            time_step_h=time_step_h,
        )

        self.soc_kg_h2 = state.ending_soc_kg_h2

        return state

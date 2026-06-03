"""
EV Battery Prediction System – Telemetry Simulator
────────────────────────────────────────────────────
Physics-inspired 16-cell battery pack simulator with realistic
degradation curves, fault injection, and scenario management.
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from config import (
    CAPACITY_AH,
    DEGRADATION_RATE_BASE,
    FAST_CHARGE_DEGRADATION,
    MAX_CYCLES,
    NOMINAL_VOLTAGE,
    NUM_CELLS,
    THERMAL_RUNAWAY_TEMP,
    VOLTAGE_RANGE,
)


# ─── Fault Scenarios ──────────────────────────────────────────
class Scenario(Enum):
    NORMAL = auto()
    FAST_CHARGING = auto()
    THERMAL_RUNAWAY = auto()
    CELL_IMBALANCE = auto()
    SHORT_CIRCUIT = auto()
    DEEP_DISCHARGE = auto()


# ─── Single Battery Cell ──────────────────────────────────────
@dataclass
class BatteryCell:
    """Models a single Li-ion cylindrical cell (e.g. 21700 NMC)."""

    cell_id: int
    soh: float = 100.0          # State of Health %
    soc: float = 80.0           # State of Charge %
    temperature: float = 25.0   # °C
    voltage: float = 3.7        # V
    current: float = 0.0        # A (positive = discharge)
    internal_resistance: float = 20.0  # mΩ

    # Internal state
    _cycle_count: int = 0
    _capacity: float = CAPACITY_AH
    _ambient_temp: float = 25.0
    _fault_active: bool = False
    _fault_type: Optional[Scenario] = None

    def _voltage_from_soc(self) -> float:
        """OCV (Open-Circuit Voltage) model: polynomial fit for NMC cell."""
        s = self.soc / 100.0
        # Realistic OCV curve: V = a0 + a1*s + a2*s^2 + a3*s^3 - IR drop
        ocv = (
            3.0
            + 0.8 * s
            + 0.25 * s ** 2
            + 0.15 * s ** 3
        )
        # IR voltage drop
        ir_drop = (self.current * self.internal_resistance / 1000.0)
        v = ocv - ir_drop
        return max(VOLTAGE_RANGE[0], min(VOLTAGE_RANGE[1], v))

    def _update_resistance(self) -> None:
        """Internal resistance increases as the cell ages and heats up."""
        age_factor = 1.0 + (100.0 - self.soh) * 0.015
        temp_factor = 1.0 + max(0, (self.temperature - 25.0)) * 0.005
        base_r = 18.0 + random.gauss(0, 0.3)
        self.internal_resistance = base_r * age_factor * temp_factor

    def _update_temperature(self, dt: float) -> None:
        """Thermal model: Joule heating + ambient cooling."""
        # I²R heating
        power_heat = (self.current ** 2) * (self.internal_resistance / 1000.0)
        # Newton's cooling law
        cooling = 0.08 * (self.temperature - self._ambient_temp)
        dT = (power_heat * 3.0 - cooling) * dt
        self.temperature += dT + random.gauss(0, 0.15)
        self.temperature = max(self._ambient_temp - 5, self.temperature)

    def _update_soc(self, dt: float) -> None:
        """Coulomb counting for SoC."""
        # dSoC = -I * dt / (capacity * 3600) * 100%
        effective_cap = self._capacity * (self.soh / 100.0)
        if effective_cap > 0:
            d_soc = -(self.current * dt) / (effective_cap * 3.6) * 100.0
            self.soc = max(0.0, min(100.0, self.soc + d_soc))

    def _degrade(self, scenario: Scenario) -> None:
        """Apply degradation per cycle based on scenario."""
        rate = DEGRADATION_RATE_BASE
        if scenario == Scenario.FAST_CHARGING:
            rate = FAST_CHARGE_DEGRADATION
        elif scenario == Scenario.THERMAL_RUNAWAY:
            rate *= 5.0
        elif scenario == Scenario.SHORT_CIRCUIT:
            rate *= 3.0
        elif scenario == Scenario.DEEP_DISCHARGE:
            rate *= 2.0

        # Temperature-accelerated aging (Arrhenius-inspired)
        if self.temperature > 35:
            rate *= 1.0 + (self.temperature - 35) * 0.02

        self.soh = max(0.0, self.soh - rate + random.gauss(0, rate * 0.1))

    def step(self, dt: float, scenario: Scenario) -> dict:
        """Advance the cell by one timestep. Returns snapshot dict."""
        self._update_resistance()
        self._update_temperature(dt)
        self._update_soc(dt)
        self.voltage = self._voltage_from_soc()
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "voltage": round(self.voltage, 4),
            "current": round(self.current, 4),
            "temperature": round(self.temperature, 2),
            "internal_resistance": round(self.internal_resistance, 2),
            "soc": round(self.soc, 2),
            "soh": round(self.soh, 2),
        }


# ─── Battery Pack (16 cells) ─────────────────────────────────
class BatteryPack:
    """
    A 16-cell battery pack with scenario-based fault injection
    and full-cycle simulation support.
    """

    def __init__(self, num_cells: int = NUM_CELLS):
        self.num_cells = num_cells
        self.cells: list[BatteryCell] = [
            BatteryCell(cell_id=i) for i in range(num_cells)
        ]
        self.cycle = 0
        self.scenario = Scenario.NORMAL
        self._affected_cells: list[int] = []
        self.history: list[list[dict]] = []

    # ── Scenario setters ────────────────────────────────────
    def set_scenario(self, scenario: Scenario,
                     affected_cells: Optional[list[int]] = None) -> None:
        """
        Switch the active scenario.

        Parameters
        ----------
        scenario : Scenario
            The driving scenario for the next simulation steps.
        affected_cells : list[int] | None
            Cell IDs targeted by the fault (random if None).
        """
        self.scenario = scenario
        if affected_cells is not None:
            self._affected_cells = affected_cells
        elif scenario != Scenario.NORMAL:
            count = random.randint(1, 3)
            self._affected_cells = random.sample(
                range(self.num_cells), min(count, self.num_cells)
            )
        else:
            self._affected_cells = []

    # ── Current profile ─────────────────────────────────────
    def _generate_current_profile(self) -> list[float]:
        """Generate per-cell current based on scenario and SoC."""
        currents = []
        for cell in self.cells:
            if self.scenario == Scenario.FAST_CHARGING and cell.soc < 90:
                # High charging current (negative)
                base = -random.uniform(3.0, 5.0)
            elif self.scenario == Scenario.NORMAL:
                # Normal drive cycle oscillation
                phase = math.sin(self.cycle * 0.1 + cell.cell_id * 0.5)
                base = phase * random.uniform(0.5, 2.5)
            elif self.scenario == Scenario.DEEP_DISCHARGE:
                base = random.uniform(2.0, 4.0)  # heavy discharge
            else:
                base = random.uniform(-1.0, 2.0)
            currents.append(base)
        return currents

    # ── Fault injection ─────────────────────────────────────
    def _inject_faults(self) -> None:
        """Apply scenario-specific faults to affected cells."""
        for cid in self._affected_cells:
            cell = self.cells[cid]
            if self.scenario == Scenario.THERMAL_RUNAWAY:
                # Rapid temperature escalation
                cell.temperature += random.uniform(3.0, 8.0)
                cell.temperature = min(THERMAL_RUNAWAY_TEMP + 10, cell.temperature)
                cell.internal_resistance *= 1.1
            elif self.scenario == Scenario.SHORT_CIRCUIT:
                cell.current = random.uniform(8.0, 15.0)
                cell.voltage = max(VOLTAGE_RANGE[0], cell.voltage - 0.3)
                cell.temperature += random.uniform(2.0, 5.0)
            elif self.scenario == Scenario.CELL_IMBALANCE:
                cell.soc += random.uniform(-5.0, -2.0)
                cell.soc = max(0, cell.soc)
                cell.internal_resistance *= random.uniform(1.05, 1.2)
            elif self.scenario == Scenario.DEEP_DISCHARGE:
                cell.soc = max(0, cell.soc - random.uniform(1.0, 3.0))

    # ── Step ────────────────────────────────────────────────
    def step(self, dt: float = 1.0) -> list[dict]:
        """
        Advance the entire pack by one timestep.

        Returns a list of 16 cell snapshot dicts.
        """
        currents = self._generate_current_profile()
        for i, cell in enumerate(self.cells):
            cell.current = currents[i]

        # Inject faults before physics step
        if self.scenario != Scenario.NORMAL:
            self._inject_faults()

        snapshots = []
        for cell in self.cells:
            snap = cell.step(dt, self.scenario)
            snapshots.append(snap)

        self.history.append(snapshots)
        return snapshots

    def advance_cycle(self) -> list[dict]:
        """Complete one charge-discharge cycle (multiple steps)."""
        self.cycle += 1
        snapshots = self.step(dt=1.0)

        # Apply end-of-cycle degradation
        for cell in self.cells:
            cell._degrade(self.scenario)

        return snapshots

    def get_pack_summary(self) -> dict:
        """Aggregate pack-level metrics."""
        cells = [c.snapshot() for c in self.cells]
        voltages = [c["voltage"] for c in cells]
        temps = [c["temperature"] for c in cells]
        sohs = [c["soh"] for c in cells]

        return {
            "cycle": self.cycle,
            "scenario": self.scenario.name,
            "total_voltage": round(sum(voltages), 2),
            "avg_voltage": round(np.mean(voltages), 4),
            "min_voltage": round(min(voltages), 4),
            "max_voltage": round(max(voltages), 4),
            "avg_temperature": round(np.mean(temps), 2),
            "max_temperature": round(max(temps), 2),
            "avg_soh": round(np.mean(sohs), 2),
            "min_soh": round(min(sohs), 2),
            "num_cells": self.num_cells,
            "affected_cells": self._affected_cells,
        }

    def reset(self) -> None:
        """Reset pack to factory state."""
        self.__init__(self.num_cells)

    def get_cell_history_df(self, cell_id: int,
                            last_n: int = 200) -> list[dict]:
        """Return recent history for a specific cell."""
        history = []
        start = max(0, len(self.history) - last_n)
        for step_idx in range(start, len(self.history)):
            snap = self.history[step_idx][cell_id].copy()
            snap["step"] = step_idx
            history.append(snap)
        return history

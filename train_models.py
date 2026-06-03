"""
train_models.py - Training Script for EV Battery ML Models

Generates synthetic battery degradation data for 16 cells over 2000 charge
cycles, then trains:
    1. AnomalyDetector (Isolation Forest) on the normal operating subset.
    2. RULPredictor (LSTM) on sliding-window sequences for SoH / RUL.

Run:
    python train_models.py
"""

import time
import numpy as np
import pandas as pd

from ml_models import AnomalyDetector, RULPredictor


# ──────────────────────────────────────────────
# Simulation parameters
# ──────────────────────────────────────────────
NUM_CELLS = 16
MAX_CYCLES = 2000
SEQUENCE_LENGTH = 50          # sliding-window length
FAULT_PROBABILITY = 0.05      # ~5 % of rows get fault injection
RNG_SEED = 42


# ══════════════════════════════════════════════
# 1.  Synthetic Data Generation
# ══════════════════════════════════════════════
def generate_battery_data(
    num_cells: int = NUM_CELLS,
    max_cycles: int = MAX_CYCLES,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate synthetic battery degradation data for multiple cells.

    For each cell the simulation produces *max_cycles* rows representing
    successive charge/discharge cycles.  The generated columns are:

    * ``cell_id`` — identifier of the cell (0 … num_cells-1)
    * ``cycle`` — cycle number (0 … max_cycles-1)
    * ``soh`` — State of Health (%), degrades ~0.03-0.05 % per cycle
    * ``soc`` — State of Charge (%), oscillates 0-100
    * ``voltage`` — terminal voltage (V)
    * ``current`` — charge/discharge current (A)
    * ``temperature`` — cell temperature (°C)
    * ``internal_resistance`` — cell internal resistance (mΩ)
    * ``is_fault`` — boolean flag for injected fault scenarios

    Args:
        num_cells: Number of battery cells to simulate.
        max_cycles: Number of charge cycles per cell.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the generated data (num_cells × max_cycles rows).
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for cell_id in range(num_cells):
        # ── per-cell initial conditions ─────
        soh = 100.0
        degradation_rate = rng.uniform(0.03, 0.05)  # % per cycle

        for cycle in range(max_cycles):
            # — SoH degrades each cycle with slight noise ─────────
            soh -= degradation_rate + rng.normal(0, 0.005)
            soh = max(soh, 0.0)

            # — SoC oscillates following charge/discharge patterns ─
            soc = 50.0 + 50.0 * np.sin(2 * np.pi * cycle / 80) + rng.normal(0, 3)
            soc = np.clip(soc, 0.0, 100.0)

            # — Voltage follows discharge curve V(soc) ────────────
            voltage = 3.0 + 1.2 * (soc / 100.0) + rng.normal(0, 0.02)

            # — Current with charge/discharge patterns ────────────
            current = rng.uniform(-5, 5)

            # — Temperature ───────────────────────────────────────
            temperature = 25.0 + abs(current) * 2.0 + rng.normal(0, 1.0)

            # — Internal resistance increases as SoH drops ───────
            internal_resistance = 20.0 + (100.0 - soh) * 1.5

            # — ~5 % fault injection ──────────────────────────────
            is_fault = False
            if rng.random() < FAULT_PROBABILITY:
                is_fault = True
                fault_type = rng.choice(["temp_spike", "voltage_drop", "resistance_jump"])
                if fault_type == "temp_spike":
                    temperature = rng.uniform(60, 85)
                elif fault_type == "voltage_drop":
                    voltage = rng.uniform(2.0, 2.8)
                elif fault_type == "resistance_jump":
                    internal_resistance += rng.uniform(30, 60)

            records.append(
                {
                    "cell_id": cell_id,
                    "cycle": cycle,
                    "voltage": round(voltage, 4),
                    "current": round(current, 4),
                    "temperature": round(temperature, 2),
                    "internal_resistance": round(internal_resistance, 4),
                    "soc": round(soc, 2),
                    "soh": round(soh, 4),
                    "is_fault": is_fault,
                }
            )

    df = pd.DataFrame(records)
    return df


# ══════════════════════════════════════════════
# 2.  Sliding-Window Sequence Builder
# ══════════════════════════════════════════════
def create_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences and corresponding targets from the
    full dataset.

    Each sequence contains ``seq_len`` consecutive rows of the 6 input
    features (voltage, current, temperature, internal_resistance, soc,
    soh).  The target for each window is taken from the **last** row:

    * ``soh`` — State of Health value at the window's end.
    * ``rul`` — Remaining Useful Life scaled to 0-1
      (``(MAX_CYCLES - cycle) / MAX_CYCLES``).

    Args:
        df: Full training DataFrame produced by
            :func:`generate_battery_data`.
        seq_len: Length of each sliding window (default 50).

    Returns:
        Tuple ``(sequences, targets)`` where:
        * *sequences*: ``np.ndarray`` of shape ``(N, seq_len, 6)``
        * *targets*: ``np.ndarray`` of shape ``(N, 2)`` — ``[soh, rul]``
    """
    feature_cols = ["voltage", "current", "temperature", "internal_resistance", "soc", "soh"]

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for _, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle").reset_index(drop=True)
        values = cell_df[feature_cols].values.astype(np.float64)
        cycles = cell_df["cycle"].values

        for i in range(len(values) - seq_len):
            seq = values[i : i + seq_len]
            last_row = cell_df.iloc[i + seq_len - 1]

            soh_target = last_row["soh"]
            rul_target = (MAX_CYCLES - cycles[i + seq_len - 1]) / MAX_CYCLES

            sequences.append(seq)
            targets.append([soh_target, rul_target])

    return np.array(sequences, dtype=np.float64), np.array(targets, dtype=np.float64)


# ══════════════════════════════════════════════
# 3.  Main Training Pipeline
# ══════════════════════════════════════════════
def main() -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("  EV Battery ML Training Pipeline")
    print("=" * 60)

    t_start = time.perf_counter()

    # ── Step 1: Generate data ─────────────────
    print("\n>> Generating synthetic battery data ...")
    df = generate_battery_data()
    print(f"  Generated {len(df):,} rows  ({NUM_CELLS} cells x {MAX_CYCLES} cycles)")
    print(f"  Fault rows: {df['is_fault'].sum():,}  ({df['is_fault'].mean() * 100:.1f} %)")

    # ── Step 2: Train Anomaly Detector ────────
    print("\n>> Training Anomaly Detector (Isolation Forest) ...")
    normal_df = df[~df["is_fault"]].copy()
    print(f"  Using {len(normal_df):,} normal samples for training")

    detector = AnomalyDetector()
    detector.fit(normal_df)
    detector.save()

    # Quick evaluation on full dataset
    scored_df = detector.predict(df)
    n_anomalies = scored_df["is_anomaly"].sum()
    print(f"  Anomalies detected across full dataset: {n_anomalies:,} / {len(df):,}"
          f"  ({n_anomalies / len(df) * 100:.1f} %)")

    # ── Step 3: Build sequences ───────────────
    print(f"\n>> Creating training sequences (window={SEQUENCE_LENGTH}) ...")
    sequences, targets = create_sequences(df)
    print(f"  Sequences: {sequences.shape[0]:,}  shape={sequences.shape}")
    print(f"  Targets:   {targets.shape}")

    # ── Step 4: Train RUL Predictor (LSTM) ────
    print("\n>> Training RUL Predictor (LSTM) ...")
    predictor = RULPredictor()
    losses = predictor.fit(sequences, targets, epochs=50, batch_size=32)
    predictor.save()

    # ── Step 5: Quick inference sanity check ──
    print("\n>> Sanity-check inference ...")
    sample_seq = sequences[0]
    prediction = predictor.predict(sample_seq)
    actual_soh = targets[0, 0]
    actual_rul = targets[0, 1]
    print(f"  Sample prediction:  SoH={prediction['predicted_soh']:.2f}  "
          f"RUL={prediction['predicted_rul']:.4f}")
    print(f"  Actual values:      SoH={actual_soh:.2f}  RUL={actual_rul:.4f}")

    # ── Summary ───────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Total samples generated : {len(df):,}")
    print(f"  Normal samples (IF)     : {len(normal_df):,}")
    print(f"  Training sequences      : {sequences.shape[0]:,}")
    print(f"  LSTM final loss         : {losses[-1]:.6f}")
    print(f"  Anomalies flagged       : {n_anomalies:,} ({n_anomalies / len(df) * 100:.1f} %)")
    print(f"  Elapsed time            : {elapsed:.1f} s")
    print("  Models saved to         : models/")
    print("=" * 60)


if __name__ == "__main__":
    main()

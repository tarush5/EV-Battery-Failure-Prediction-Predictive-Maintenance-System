"""
EV Battery Failure Prediction & Predictive Maintenance System
─────────────────────────────────────────────────────────────
Global configuration: paths, thresholds, hyperparameters, and constants.
"""

import os
from pathlib import Path

# ─── Project Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
try:
    MODEL_DIR.mkdir(exist_ok=True)
except Exception:
    pass

# ─── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'ev_battery.db'}"
)

# ─── Battery Pack Configuration ────────────────────────────────
NUM_CELLS = 16
CELL_ROWS = 4
CELL_COLS = 4
NOMINAL_VOLTAGE = 3.7          # V per cell
VOLTAGE_RANGE = (2.8, 4.2)    # V
TEMPERATURE_RANGE = (15.0, 70.0)  # °C
RESISTANCE_RANGE = (10.0, 200.0)  # mΩ
CAPACITY_AH = 5.0             # Amp-hours

# ─── Sensor Alert Thresholds ──────────────────────────────────
TEMP_WARNING = 45.0
TEMP_CRITICAL = 55.0
VOLTAGE_LOW_WARNING = 3.0
VOLTAGE_HIGH_WARNING = 4.25
RESISTANCE_WARNING = 100.0
SOH_WARNING = 80.0
SOH_CRITICAL = 60.0
RUL_WARNING = 300   # cycles
RUL_CRITICAL = 100  # cycles

# ─── ML Model Configuration ──────────────────────────────────
FEATURE_COLUMNS = [
    "voltage", "current", "temperature",
    "internal_resistance", "soc", "soh",
]
ANOMALY_FEATURES = [
    "voltage", "current", "temperature",
    "internal_resistance", "soc",
]
SEQUENCE_LENGTH = 50           # timesteps for LSTM input window
LSTM_INPUT_SIZE = len(FEATURE_COLUMNS)
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 50
BATCH_SIZE = 32

# Isolation Forest
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS = 200
IF_RANDOM_STATE = 42

# Model file paths
IF_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_rul.pth"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"

# ─── Alert Severity Levels ────────────────────────────────────
SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_EMERGENCY = "EMERGENCY"

# ─── Simulation Constants ─────────────────────────────────────
MAX_CYCLES = 2000              # total battery lifecycle
DEGRADATION_RATE_BASE = 0.003  # % SoH loss per cycle (base)
FAST_CHARGE_DEGRADATION = 0.008
THERMAL_RUNAWAY_TEMP = 65.0

# ─── Dashboard Theme Colors ──────────────────────────────────
COLORS = {
    "bg_primary":    "#0a0e17",
    "bg_secondary":  "#111827",
    "bg_card":       "#1a2332",
    "accent_cyan":   "#06d6a0",
    "accent_blue":   "#4361ee",
    "accent_purple": "#7209b7",
    "accent_pink":   "#f72585",
    "accent_orange": "#ff9f1c",
    "text_primary":  "#e2e8f0",
    "text_secondary":"#94a3b8",
    "success":       "#06d6a0",
    "warning":       "#ff9f1c",
    "danger":        "#ef4444",
    "critical":      "#f72585",
}

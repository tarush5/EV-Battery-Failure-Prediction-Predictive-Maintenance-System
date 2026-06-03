"""
ml_models.py - Machine Learning Models for EV Battery Failure Prediction

This module implements two core ML models:
1. AnomalyDetector: Isolation Forest-based anomaly detection for battery sensor data.
2. RULPredictor: LSTM-based Remaining Useful Life and State of Health prediction.

Dependencies: torch, numpy, pandas, scikit-learn, joblib
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# Constants (hardcoded, not imported from config)
# ──────────────────────────────────────────────
_IF_N_ESTIMATORS = 200
_IF_CONTAMINATION = 0.05
_IF_RANDOM_STATE = 42

_MODEL_DIR = "models"
_IF_MODEL_PATH = os.path.join(_MODEL_DIR, "isolation_forest.joblib")
_SCALER_PATH = os.path.join(_MODEL_DIR, "feature_scaler.joblib")
_LSTM_MODEL_PATH = os.path.join(_MODEL_DIR, "lstm_rul.pth")

_ANOMALY_FEATURES = ["voltage", "current", "temperature", "internal_resistance", "soc"]
_LSTM_INPUT_SIZE = 6  # voltage, current, temperature, internal_resistance, soc, soh


# ═══════════════════════════════════════════════
# Anomaly Detection — Isolation Forest
# ═══════════════════════════════════════════════
class AnomalyDetector:
    """Isolation Forest-based anomaly detector for battery sensor readings.

    Detects anomalous battery behaviour by training an Isolation Forest on
    five sensor features: voltage, current, temperature, internal_resistance,
    and state-of-charge (soc). Data is standardised before fitting.

    Attributes:
        model: sklearn IsolationForest instance.
        scaler: sklearn StandardScaler fitted to training data.
        is_fitted: Whether the model has been trained or loaded.
    """

    def __init__(self) -> None:
        """Initialise the AnomalyDetector, loading a saved model if available."""
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=_IF_N_ESTIMATORS,
            contamination=_IF_CONTAMINATION,
            random_state=_IF_RANDOM_STATE,
        )
        self.is_fitted: bool = False

        # Attempt to load a previously saved model
        if not self.load():
            print("[AnomalyDetector] No saved model found -- call fit() to train.")

    # ── Training ──────────────────────────────
    def fit(self, df: pd.DataFrame) -> None:
        """Train the Isolation Forest on a DataFrame of sensor readings.

        Args:
            df: pandas DataFrame containing at least the columns
                ``voltage``, ``current``, ``temperature``,
                ``internal_resistance``, and ``soc``.

        Raises:
            KeyError: If required feature columns are missing from *df*.
        """
        missing = set(_ANOMALY_FEATURES) - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        X = df[_ANOMALY_FEATURES].values.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        print(f"[AnomalyDetector] Trained on {len(df)} samples.")

    # ── Inference ─────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score each row for anomalousness and flag anomalies.

        Args:
            df: pandas DataFrame with the same feature columns used in
                training.

        Returns:
            A **copy** of *df* with two additional columns:

            * ``anomaly_score`` — Isolation Forest decision-function score
              (lower = more anomalous).
            * ``is_anomaly`` — Boolean flag (``True`` when the model labels
              the sample as an outlier).

        Raises:
            RuntimeError: If the model has not been fitted or loaded.
            KeyError: If required feature columns are missing from *df*.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        missing = set(_ANOMALY_FEATURES) - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        result = df.copy()
        X = result[_ANOMALY_FEATURES].values.astype(np.float64)
        X_scaled = self.scaler.transform(X)

        result["anomaly_score"] = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)  # 1 = normal, -1 = anomaly
        result["is_anomaly"] = predictions == -1

        return result

    # ── Persistence ───────────────────────────
    def save(self) -> None:
        """Save the trained model and scaler to disk.

        Creates the ``models/`` directory if it does not exist.
        """
        os.makedirs(_MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, _IF_MODEL_PATH)
        joblib.dump(self.scaler, _SCALER_PATH)
        print(f"[AnomalyDetector] Saved model  -> {_IF_MODEL_PATH}")
        print(f"[AnomalyDetector] Saved scaler -> {_SCALER_PATH}")

    def load(self) -> bool:
        """Load a previously saved model and scaler from disk.

        Returns:
            ``True`` if both files were found and loaded successfully,
            ``False`` otherwise.
        """
        if os.path.exists(_IF_MODEL_PATH) and os.path.exists(_SCALER_PATH):
            self.model = joblib.load(_IF_MODEL_PATH)
            self.scaler = joblib.load(_SCALER_PATH)
            self.is_fitted = True
            print(f"[AnomalyDetector] Loaded model from {_IF_MODEL_PATH}")
            return True
        return False


# ═══════════════════════════════════════════════
# LSTM Network for SoH / RUL Prediction
# ═══════════════════════════════════════════════
class BatteryLSTM(nn.Module):
    """LSTM neural network for predicting battery State-of-Health and
    Remaining Useful Life.

    Architecture:
        input → LSTM(hidden_size, num_layers, dropout) → Linear → 2 outputs

    The two outputs correspond to:
        0. ``predicted_soh`` — predicted State of Health (%)
        1. ``predicted_rul`` — predicted Remaining Useful Life (scaled)

    Args:
        input_size: Number of input features per timestep (default 6).
        hidden_size: Number of LSTM hidden units (default 128).
        num_layers: Number of stacked LSTM layers (default 2).
        dropout: Dropout probability between LSTM layers (default 0.2).
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 2)  # → [soh_pred, rul_pred]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch, 2)`` with columns
            ``[predicted_soh, predicted_rul]``.
        """
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output from the last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        out = self.fc(last_hidden)  # (batch, 2)
        return out


# ═══════════════════════════════════════════════
# RUL Predictor — high-level training / inference
# ═══════════════════════════════════════════════
class RULPredictor:
    """High-level wrapper around :class:`BatteryLSTM` for training and
    inference of Remaining Useful Life and State of Health.

    Handles device placement (CPU), training loop, and model persistence.

    Attributes:
        model: The underlying :class:`BatteryLSTM` network.
        device: ``torch.device`` the model runs on (always CPU here).
        is_fitted: Whether the model has been trained or loaded.
    """

    def __init__(self) -> None:
        """Initialise the RULPredictor, loading saved weights if available."""
        self.device = torch.device("cpu")
        self.model = BatteryLSTM(input_size=_LSTM_INPUT_SIZE).to(self.device)
        self.is_fitted: bool = False

        if not self.load():
            print("[RULPredictor] No saved model found -- call fit() to train.")

    # ── Training ──────────────────────────────
    def fit(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        """Train the LSTM on windowed battery sequences.

        Args:
            sequences: Input array of shape ``(N, 50, 6)`` — *N* windows,
                each 50 timesteps of 6 features.
            targets: Target array of shape ``(N, 2)`` where each row is
                ``[soh, rul]``.
            epochs: Number of full training passes (default 50).
            batch_size: Mini-batch size (default 32).

        Returns:
            List of mean loss values, one per epoch.
        """
        self.model.train()

        X = torch.tensor(sequences, dtype=torch.float32, device=self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            batch_losses: list[float] = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            mean_loss = float(np.mean(batch_losses))
            epoch_losses.append(mean_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [RULPredictor] Epoch {epoch:>3}/{epochs}  loss={mean_loss:.6f}")

        self.is_fitted = True
        return epoch_losses

    # ── Inference ─────────────────────────────
    def predict(self, sequence: np.ndarray) -> dict:
        """Predict SoH and RUL from a single sequence window.

        Args:
            sequence: Array of shape ``(50, 6)`` — 50 timesteps of
                6 features (voltage, current, temperature,
                internal_resistance, soc, soh).

        Returns:
            Dictionary with keys ``predicted_soh`` and ``predicted_rul``
            (both floats).

        Raises:
            RuntimeError: If the model has not been fitted or loaded.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                sequence, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, 50, 6)
            output = self.model(x)  # (1, 2)

        return {
            "predicted_soh": float(output[0, 0].item()),
            "predicted_rul": float(output[0, 1].item()),
        }

    # ── Persistence ───────────────────────────
    def save(self) -> None:
        """Save model weights to disk.

        Creates the ``models/`` directory if it does not exist.
        """
        os.makedirs(_MODEL_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), _LSTM_MODEL_PATH)
        print(f"[RULPredictor] Saved model -> {_LSTM_MODEL_PATH}")

    def load(self) -> bool:
        """Load model weights from disk.

        Returns:
            ``True`` if the weights file was found and loaded,
            ``False`` otherwise.
        """
        if os.path.exists(_LSTM_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(_LSTM_MODEL_PATH, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self.is_fitted = True
            print(f"[RULPredictor] Loaded model from {_LSTM_MODEL_PATH}")
            return True
        return False

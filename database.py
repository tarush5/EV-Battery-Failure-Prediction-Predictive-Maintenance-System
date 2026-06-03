"""
EV Battery Prediction System – Database Manager
─────────────────────────────────────────────────
SQLAlchemy-based storage for sensor readings, anomalies, alerts,
and model predictions.  Supports PostgreSQL and SQLite (fallback).
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL


# ─── ORM Base ──────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─── Table Definitions ────────────────────────────────────────
class SensorReading(Base):
    """Raw telemetry from each battery cell."""

    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cell_id = Column(Integer, nullable=False, index=True)
    cycle = Column(Integer, nullable=False)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    internal_resistance = Column(Float, nullable=False)
    soc = Column(Float, nullable=False)
    soh = Column(Float, nullable=False)


class AnomalyLog(Base):
    """Detected anomalies from the Isolation Forest model."""

    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cell_id = Column(Integer, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    feature_snapshot = Column(Text)  # JSON of sensor values at detection
    severity = Column(String(20), nullable=False)


class Alert(Base):
    """System alerts for operator notification."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cell_id = Column(Integer, nullable=True)  # None = pack-level alert
    severity = Column(String(20), nullable=False)
    category = Column(String(50), nullable=False)  # THERMAL, VOLTAGE, ANOMALY …
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)


class ModelPrediction(Base):
    """RUL / SoH predictions from the LSTM model."""

    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cell_id = Column(Integer, nullable=False, index=True)
    predicted_soh = Column(Float)
    predicted_rul = Column(Float)  # remaining cycles
    confidence = Column(Float)


# ─── Database Manager ─────────────────────────────────────────
class DatabaseManager:
    """Handles connections, schema creation, and CRUD operations."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or DATABASE_URL
        self._connect()

    # ── Connection ──────────────────────────────────────────
    def _connect(self):
        """Create engine with fallback to SQLite if PostgreSQL is unavailable."""
        try:
            self.engine = create_engine(self.url, echo=False, pool_pre_ping=True)
            # Quick connectivity check
            with self.engine.connect() as conn:
                conn.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )
            self.db_type = "postgresql" if "postgresql" in self.url else "sqlite"
        except (OperationalError, Exception):
            # Fallback to local SQLite
            from config import BASE_DIR

            fallback = f"sqlite:///{BASE_DIR / 'ev_battery.db'}"
            self.engine = create_engine(fallback, echo=False)
            self.db_type = "sqlite"

        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    # ── Insert helpers ──────────────────────────────────────
    def insert_readings(self, readings: list[dict]) -> None:
        """Bulk-insert sensor readings."""
        with self.SessionLocal() as session:
            session.bulk_insert_mappings(SensorReading, readings)
            session.commit()

    def insert_anomaly(self, cell_id: int, score: float,
                       snapshot: str, severity: str) -> None:
        with self.SessionLocal() as session:
            session.add(AnomalyLog(
                cell_id=cell_id,
                anomaly_score=score,
                feature_snapshot=snapshot,
                severity=severity,
            ))
            session.commit()

    def insert_alert(self, cell_id: Optional[int], severity: str,
                     category: str, message: str) -> None:
        with self.SessionLocal() as session:
            session.add(Alert(
                cell_id=cell_id,
                severity=severity,
                category=category,
                message=message,
            ))
            session.commit()

    def insert_prediction(self, cell_id: int, soh: float,
                          rul: float, confidence: float) -> None:
        with self.SessionLocal() as session:
            session.add(ModelPrediction(
                cell_id=cell_id,
                predicted_soh=soh,
                predicted_rul=rul,
                confidence=confidence,
            ))
            session.commit()

    # ── Query helpers ───────────────────────────────────────
    def get_recent_readings(self, cell_id: int,
                            limit: int = 200) -> pd.DataFrame:
        """Return recent sensor readings for a given cell."""
        with self.SessionLocal() as session:
            rows = (
                session.query(SensorReading)
                .filter(SensorReading.cell_id == cell_id)
                .order_by(SensorReading.timestamp.desc())
                .limit(limit)
                .all()
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": r.timestamp, "cell_id": r.cell_id,
            "cycle": r.cycle, "voltage": r.voltage,
            "current": r.current, "temperature": r.temperature,
            "internal_resistance": r.internal_resistance,
            "soc": r.soc, "soh": r.soh,
        } for r in rows]).sort_values("timestamp")

    def get_all_readings_df(self, limit: int = 5000) -> pd.DataFrame:
        """Return all recent readings across all cells."""
        with self.SessionLocal() as session:
            rows = (
                session.query(SensorReading)
                .order_by(SensorReading.timestamp.desc())
                .limit(limit)
                .all()
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": r.timestamp, "cell_id": r.cell_id,
            "cycle": r.cycle, "voltage": r.voltage,
            "current": r.current, "temperature": r.temperature,
            "internal_resistance": r.internal_resistance,
            "soc": r.soc, "soh": r.soh,
        } for r in rows]).sort_values("timestamp")

    def get_active_alerts(self, limit: int = 50) -> list[dict]:
        """Return recent unacknowledged alerts."""
        with self.SessionLocal() as session:
            rows = (
                session.query(Alert)
                .filter(Alert.acknowledged == False)  # noqa: E712
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
        return [{
            "id": a.id, "timestamp": a.timestamp,
            "cell_id": a.cell_id, "severity": a.severity,
            "category": a.category, "message": a.message,
        } for a in rows]

    def get_predictions(self, cell_id: int,
                        limit: int = 100) -> pd.DataFrame:
        """Return prediction history for a cell."""
        with self.SessionLocal() as session:
            rows = (
                session.query(ModelPrediction)
                .filter(ModelPrediction.cell_id == cell_id)
                .order_by(ModelPrediction.timestamp.desc())
                .limit(limit)
                .all()
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": r.timestamp, "cell_id": r.cell_id,
            "predicted_soh": r.predicted_soh,
            "predicted_rul": r.predicted_rul,
            "confidence": r.confidence,
        } for r in rows]).sort_values("timestamp")

    def acknowledge_alert(self, alert_id: int) -> None:
        with self.SessionLocal() as session:
            alert = session.query(Alert).get(alert_id)
            if alert:
                alert.acknowledged = True
                session.commit()

    def clear_all(self) -> None:
        """Reset all tables — use during development only."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

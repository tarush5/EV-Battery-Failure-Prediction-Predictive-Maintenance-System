"""
EV Battery Failure Prediction & Predictive Maintenance System
══════════════════════════════════════════════════════════════
Main Streamlit Dashboard — Premium dark-themed interface with
real-time 3D battery visualization, ML prognostics, and alerts.
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    COLORS,
    NUM_CELLS,
    SEQUENCE_LENGTH,
    SEVERITY_CRITICAL,
    SEVERITY_EMERGENCY,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SOH_CRITICAL,
    SOH_WARNING,
    TEMP_CRITICAL,
    TEMP_WARNING,
    VOLTAGE_HIGH_WARNING,
    VOLTAGE_LOW_WARNING,
    RUL_CRITICAL,
    RUL_WARNING,
    MAX_CYCLES,
)
from database import DatabaseManager
from telemetry_sim import BatteryPack, Scenario
from ml_models import AnomalyDetector, RULPredictor

# ─── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="EV Battery Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Premium Dark Theme CSS ───────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ── Global Reset ──────────────────────── */
        .stApp {{
            background: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            font-family: 'Inter', sans-serif;
        }}
        .stApp > header {{
            background: transparent;
        }}

        /* ── Sidebar ───────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
            border-right: 1px solid rgba(6, 214, 160, 0.1);
        }}
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {COLORS['accent_cyan']};
        }}

        /* ── Cards / Containers ────────────────── */
        .glass-card {{
            background: linear-gradient(135deg,
                rgba(26, 35, 50, 0.9) 0%,
                rgba(17, 24, 39, 0.8) 100%);
            border: 1px solid rgba(6, 214, 160, 0.15);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        /* ── Metric Cards Row ──────────────────── */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: linear-gradient(135deg,
                rgba(26, 35, 50, 0.95) 0%,
                rgba(17, 24, 39, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 20px 24px;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(6, 214, 160, 0.3);
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            border-radius: 14px 14px 0 0;
        }}
        .metric-card:nth-child(1)::before {{ background: linear-gradient(90deg, {COLORS['accent_cyan']}, {COLORS['accent_blue']}); }}
        .metric-card:nth-child(2)::before {{ background: linear-gradient(90deg, {COLORS['accent_blue']}, {COLORS['accent_purple']}); }}
        .metric-card:nth-child(3)::before {{ background: linear-gradient(90deg, {COLORS['accent_orange']}, {COLORS['danger']}); }}
        .metric-card:nth-child(4)::before {{ background: linear-gradient(90deg, {COLORS['critical']}, {COLORS['accent_purple']}); }}

        .metric-label {{
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: {COLORS['text_secondary']};
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
        }}
        .metric-sub {{
            font-size: 11px;
            color: {COLORS['text_secondary']};
            margin-top: 4px;
        }}

        /* ── Alert Badge ───────────────────────── */
        .alert-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .alert-info    {{ background: rgba(67, 97, 238, 0.2); color: {COLORS['accent_blue']}; border: 1px solid rgba(67, 97, 238, 0.3); }}
        .alert-warning {{ background: rgba(255, 159, 28, 0.2); color: {COLORS['warning']}; border: 1px solid rgba(255, 159, 28, 0.3); }}
        .alert-critical{{ background: rgba(239, 68, 68, 0.2); color: {COLORS['danger']}; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .alert-emergency{{ background: rgba(247, 37, 133, 0.2); color: {COLORS['critical']}; border: 1px solid rgba(247, 37, 133, 0.3); }}

        /* ── Alert Table ───────────────────────── */
        .alert-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
        }}
        .alert-table th {{
            text-align: left;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {COLORS['text_secondary']};
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        .alert-table td {{
            padding: 10px 12px;
            font-size: 13px;
            background: rgba(17, 24, 39, 0.5);
        }}
        .alert-table tr td:first-child {{ border-radius: 8px 0 0 8px; }}
        .alert-table tr td:last-child {{ border-radius: 0 8px 8px 0; }}

        /* ── Section Headings ──────────────────── */
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: {COLORS['text_primary']};
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-title .icon {{
            font-size: 20px;
        }}

        /* ── Tab Styling ───────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(26, 35, 50, 0.6);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.06);
            color: {COLORS['text_secondary']};
            padding: 8px 20px;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(6, 214, 160, 0.15), rgba(67, 97, 238, 0.15));
            border-color: {COLORS['accent_cyan']};
            color: {COLORS['accent_cyan']} !important;
        }}

        /* ── Plotly chart bg ────────────────────── */
        .js-plotly-plot .plotly .main-svg {{
            background: transparent !important;
        }}

        /* ── Selectbox / Slider ─────────────────── */
        div[data-baseweb="select"] {{
            background: rgba(26, 35, 50, 0.8);
            border-radius: 8px;
        }}

        /* ── Scrollbar ─────────────────────────── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(6, 214, 160, 0.3);
            border-radius: 3px;
        }}

        /* ── Header Banner ─────────────────────── */
        .header-banner {{
            background: linear-gradient(135deg,
                rgba(6, 214, 160, 0.08) 0%,
                rgba(67, 97, 238, 0.08) 50%,
                rgba(114, 9, 183, 0.08) 100%);
            border: 1px solid rgba(6, 214, 160, 0.12);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .header-title {{
            font-size: 26px;
            font-weight: 900;
            background: linear-gradient(135deg, {COLORS['accent_cyan']}, {COLORS['accent_blue']}, {COLORS['accent_purple']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}
        .header-sub {{
            font-size: 13px;
            color: {COLORS['text_secondary']};
            margin-top: 4px;
        }}
        .header-status {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: {COLORS['accent_cyan']};
        }}
        .pulse-dot {{
            width: 8px; height: 8px;
            background: {COLORS['accent_cyan']};
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(1.5); }}
        }}
    </style>
    """, unsafe_allow_html=True)


# ─── Plotly Chart Helpers ─────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_secondary"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.06)",
        font=dict(size=11),
    ),
)


def make_line_chart(df, x_col, y_cols, title, colors_list=None):
    """Create a premium Plotly line chart."""
    fig = go.Figure()
    default_colors = [
        COLORS["accent_cyan"], COLORS["accent_blue"],
        COLORS["accent_purple"], COLORS["accent_orange"],
        COLORS["critical"],
    ]
    colors_list = colors_list or default_colors

    for i, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col] if x_col in df.columns else df.index,
                y=df[col],
                name=col.replace("_", " ").title(),
                mode="lines",
                line=dict(
                    color=colors_list[i % len(colors_list)],
                    width=2,
                ),
                fill="tozeroy" if len(y_cols) == 1 else None,
                fillcolor=f"rgba({int(colors_list[i % len(colors_list)].lstrip('#')[0:2], 16)}, "
                          f"{int(colors_list[i % len(colors_list)].lstrip('#')[2:4], 16)}, "
                          f"{int(colors_list[i % len(colors_list)].lstrip('#')[4:6], 16)}, 0.08)"
                if len(y_cols) == 1 else None,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text_primary"])),
        height=300,
        **PLOTLY_LAYOUT,
    )
    return fig


def make_gauge(value, title, min_val, max_val, thresholds=None):
    """Create a premium Plotly gauge indicator."""
    steps = []
    if thresholds:
        prev = min_val
        colors_gauge = [
            "rgba(6, 214, 160, 0.3)",
            "rgba(255, 159, 28, 0.3)",
            "rgba(239, 68, 68, 0.3)",
        ]
        for i, thresh in enumerate(thresholds):
            steps.append(dict(range=[prev, thresh], color=colors_gauge[min(i, 2)]))
            prev = thresh
        steps.append(dict(range=[prev, max_val], color=colors_gauge[-1]))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=13, color=COLORS["text_secondary"])),
        number=dict(font=dict(size=28, color=COLORS["text_primary"], family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickcolor=COLORS["text_secondary"]),
            bar=dict(color=COLORS["accent_cyan"]),
            bgcolor="rgba(26, 35, 50, 0.5)",
            borderwidth=0,
            steps=steps,
        ),
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_secondary"]),
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


# ─── Alert Generation ────────────────────────────────────────
def generate_alerts(cell_data, db):
    """Check cell data against thresholds and create alerts."""
    alerts_generated = []
    for cell in cell_data:
        cid = cell["cell_id"]

        # Temperature alerts
        if cell["temperature"] >= TEMP_CRITICAL:
            msg = f"Cell {cid}: CRITICAL temperature {cell['temperature']:.1f}°C — risk of thermal runaway!"
            db.insert_alert(cid, SEVERITY_EMERGENCY, "THERMAL", msg)
            alerts_generated.append({"severity": SEVERITY_EMERGENCY, "message": msg})
        elif cell["temperature"] >= TEMP_WARNING:
            msg = f"Cell {cid}: Elevated temperature {cell['temperature']:.1f}°C"
            db.insert_alert(cid, SEVERITY_WARNING, "THERMAL", msg)
            alerts_generated.append({"severity": SEVERITY_WARNING, "message": msg})

        # Voltage alerts
        if cell["voltage"] <= VOLTAGE_LOW_WARNING:
            msg = f"Cell {cid}: Low voltage {cell['voltage']:.3f}V — possible deep discharge"
            db.insert_alert(cid, SEVERITY_CRITICAL, "VOLTAGE", msg)
            alerts_generated.append({"severity": SEVERITY_CRITICAL, "message": msg})
        elif cell["voltage"] >= VOLTAGE_HIGH_WARNING:
            msg = f"Cell {cid}: Over-voltage {cell['voltage']:.3f}V"
            db.insert_alert(cid, SEVERITY_WARNING, "VOLTAGE", msg)
            alerts_generated.append({"severity": SEVERITY_WARNING, "message": msg})

        # SoH alerts
        if cell["soh"] <= SOH_CRITICAL:
            msg = f"Cell {cid}: SoH critically low at {cell['soh']:.1f}% — replace cell"
            db.insert_alert(cid, SEVERITY_CRITICAL, "DEGRADATION", msg)
            alerts_generated.append({"severity": SEVERITY_CRITICAL, "message": msg})
        elif cell["soh"] <= SOH_WARNING:
            msg = f"Cell {cid}: SoH below threshold at {cell['soh']:.1f}%"
            db.insert_alert(cid, SEVERITY_WARNING, "DEGRADATION", msg)
            alerts_generated.append({"severity": SEVERITY_WARNING, "message": msg})

        # Anomaly alert
        if cell.get("is_anomaly", False):
            msg = f"Cell {cid}: Anomaly detected (score: {cell.get('anomaly_score', 0):.3f})"
            db.insert_alert(cid, SEVERITY_CRITICAL, "ANOMALY", msg)
            alerts_generated.append({"severity": SEVERITY_CRITICAL, "message": msg})

    return alerts_generated


# ─── Initialize Session State ─────────────────────────────────
def init_state():
    if "pack" not in st.session_state:
        st.session_state.pack = BatteryPack()
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()
    if "anomaly_detector" not in st.session_state:
        st.session_state.anomaly_detector = AnomalyDetector()
    if "rul_predictor" not in st.session_state:
        st.session_state.rul_predictor = RULPredictor()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_cell" not in st.session_state:
        st.session_state.selected_cell = 0
    if "alerts_log" not in st.session_state:
        st.session_state.alerts_log = []
    if "cycle_count" not in st.session_state:
        st.session_state.cycle_count = 0
    if "auto_run" not in st.session_state:
        st.session_state.auto_run = False


# ─── Run One Simulation Step ──────────────────────────────────
def run_step():
    """Execute one cycle: simulate → detect anomalies → predict → alert."""
    pack = st.session_state.pack
    db = st.session_state.db
    ad = st.session_state.anomaly_detector
    rp = st.session_state.rul_predictor

    # 1. Advance one cycle
    snapshots = pack.advance_cycle()
    st.session_state.cycle_count = pack.cycle

    # 2. Anomaly detection
    df = pd.DataFrame(snapshots)
    if ad.is_fitted:
        df = ad.predict(df)
        for _, row in df.iterrows():
            snapshots[int(row["cell_id"])]["anomaly_score"] = row["anomaly_score"]
            snapshots[int(row["cell_id"])]["is_anomaly"] = bool(row["is_anomaly"])
    else:
        for s in snapshots:
            s["anomaly_score"] = 0.0
            s["is_anomaly"] = False

    # 3. RUL prediction (if model fitted and enough history)
    if rp.is_fitted and len(pack.history) >= SEQUENCE_LENGTH:
        for cell_id in range(NUM_CELLS):
            cell_hist = pack.get_cell_history_df(cell_id, last_n=SEQUENCE_LENGTH)
            if len(cell_hist) >= SEQUENCE_LENGTH:
                seq = np.array([
                    [h["voltage"], h["current"], h["temperature"],
                     h["internal_resistance"], h["soc"], h["soh"]]
                    for h in cell_hist[-SEQUENCE_LENGTH:]
                ])
                pred = rp.predict(seq)
                # Scale RUL back from 0-1 to actual cycles
                pred_rul = pred["predicted_rul"] * MAX_CYCLES
                pred_soh = pred["predicted_soh"]
                snapshots[cell_id]["predicted_rul"] = round(max(0, pred_rul), 1)
                snapshots[cell_id]["predicted_soh"] = round(pred_soh, 2)
                db.insert_prediction(cell_id, pred_soh, pred_rul, confidence=0.85)

    # 4. Store readings
    readings = [{
        "cell_id": s["cell_id"], "cycle": pack.cycle,
        "voltage": s["voltage"], "current": s["current"],
        "temperature": s["temperature"],
        "internal_resistance": s["internal_resistance"],
        "soc": s["soc"], "soh": s["soh"],
    } for s in snapshots]
    db.insert_readings(readings)

    # 5. Generate alerts
    new_alerts = generate_alerts(snapshots, db)
    st.session_state.alerts_log = new_alerts + st.session_state.alerts_log[:100]

    # 6. Update history
    st.session_state.history.append(snapshots)
    if len(st.session_state.history) > 300:
        st.session_state.history = st.session_state.history[-300:]

    return snapshots


# ─── Render Header ────────────────────────────────────────────
def render_header():
    st.markdown(f"""
    <div class="header-banner">
        <div>
            <div class="header-title">⚡ EV Battery Prediction System</div>
            <div class="header-sub">
                Real-time Failure Prediction & Predictive Maintenance • {NUM_CELLS}-Cell Pack
            </div>
        </div>
        <div class="header-status">
            <div class="pulse-dot"></div>
            Cycle {st.session_state.cycle_count} •
            {datetime.now().strftime('%H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Render Metric Cards ─────────────────────────────────────
def render_metrics(cell_data):
    avg_soh = np.mean([c["soh"] for c in cell_data])
    avg_v = np.mean([c["voltage"] for c in cell_data])
    max_temp = max(c["temperature"] for c in cell_data)
    anomaly_count = sum(1 for c in cell_data if c.get("is_anomaly", False))

    soh_color = COLORS["success"] if avg_soh > SOH_WARNING else (
        COLORS["warning"] if avg_soh > SOH_CRITICAL else COLORS["danger"]
    )
    temp_color = COLORS["success"] if max_temp < TEMP_WARNING else (
        COLORS["warning"] if max_temp < TEMP_CRITICAL else COLORS["danger"]
    )
    anomaly_color = COLORS["success"] if anomaly_count == 0 else COLORS["danger"]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Pack Health (SoH)</div>
            <div class="metric-value" style="color: {soh_color}">{avg_soh:.1f}%</div>
            <div class="metric-sub">Average across {NUM_CELLS} cells</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Voltage</div>
            <div class="metric-value" style="color: {COLORS['accent_blue']}">{avg_v:.3f}V</div>
            <div class="metric-sub">Per cell • Pack: {avg_v * NUM_CELLS:.1f}V</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Peak Temperature</div>
            <div class="metric-value" style="color: {temp_color}">{max_temp:.1f}°C</div>
            <div class="metric-sub">{'⚠️ Above threshold' if max_temp >= TEMP_WARNING else '✅ Normal range'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Anomalies</div>
            <div class="metric-value" style="color: {anomaly_color}">{anomaly_count}</div>
            <div class="metric-sub">{'🔴 Cells require attention' if anomaly_count > 0 else '🟢 All cells nominal'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Render Alert Log ─────────────────────────────────────────
def render_alerts():
    alerts = st.session_state.alerts_log[:20]
    if not alerts:
        st.info("✅ No active alerts — all systems nominal.")
        return

    severity_class = {
        SEVERITY_INFO: "alert-info",
        SEVERITY_WARNING: "alert-warning",
        SEVERITY_CRITICAL: "alert-critical",
        SEVERITY_EMERGENCY: "alert-emergency",
    }

    rows_html = ""
    for i, a in enumerate(alerts):
        cls = severity_class.get(a["severity"], "alert-info")
        rows_html += f"""
        <tr>
            <td><span class="alert-badge {cls}">{a['severity']}</span></td>
            <td style="color: {COLORS['text_primary']}">{a['message']}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title"><span class="icon">🚨</span> Alert Log</div>
        <table class="alert-table">
            <thead><tr><th>Severity</th><th>Message</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ─── Cell Detail Panel ────────────────────────────────────────
def render_cell_detail(cell_data, cell_id):
    """Render detailed diagnostics for a selected cell."""
    cell = cell_data[cell_id]

    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 16px;">
        <div class="section-title">
            <span class="icon">🔋</span> Cell {cell_id} Diagnostics
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voltage", f"{cell['voltage']:.3f} V")
    c2.metric("Temperature", f"{cell['temperature']:.1f} °C")
    c3.metric("SoC", f"{cell['soc']:.1f}%")
    c4.metric("SoH", f"{cell['soh']:.1f}%")

    if cell.get("predicted_rul") is not None:
        c5, c6 = st.columns(2)
        c5.metric("Predicted RUL", f"{cell['predicted_rul']:.0f} cycles")
        c6.metric("Predicted SoH", f"{cell.get('predicted_soh', 0):.1f}%")

    # Cell history charts
    if len(st.session_state.history) > 5:
        hist = []
        for step_idx, step_data in enumerate(st.session_state.history):
            row = step_data[cell_id].copy()
            row["step"] = step_idx
            hist.append(row)
        df_hist = pd.DataFrame(hist)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = make_line_chart(df_hist, "step", ["voltage"], f"Cell {cell_id} — Voltage Profile")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = make_line_chart(
                df_hist, "step", ["temperature"],
                f"Cell {cell_id} — Temperature",
                colors_list=[COLORS["accent_orange"]]
            )
            st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig = make_line_chart(
                df_hist, "step", ["soc", "soh"],
                f"Cell {cell_id} — SoC & SoH"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_d:
            fig = make_line_chart(
                df_hist, "step", ["internal_resistance"],
                f"Cell {cell_id} — Internal Resistance",
                colors_list=[COLORS["accent_purple"]]
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════
def main():
    inject_css()
    init_state()

    # ── Sidebar Controls ────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.divider()

        # Scenario selector
        scenario_map = {
            "🟢 Normal Driving": Scenario.NORMAL,
            "⚡ Fast Charging (DC)": Scenario.FAST_CHARGING,
            "🔥 Thermal Runaway": Scenario.THERMAL_RUNAWAY,
            "⚖️ Cell Imbalance": Scenario.CELL_IMBALANCE,
            "💥 Short Circuit": Scenario.SHORT_CIRCUIT,
            "📉 Deep Discharge": Scenario.DEEP_DISCHARGE,
        }
        selected_scenario = st.selectbox(
            "Driving Scenario",
            list(scenario_map.keys()),
            index=0,
        )
        scenario = scenario_map[selected_scenario]

        # Affected cells for fault scenarios
        affected_cells = None
        if scenario != Scenario.NORMAL:
            affected_cells = st.multiselect(
                "Affected Cells",
                list(range(NUM_CELLS)),
                default=[0, 5],
            )

        st.session_state.pack.set_scenario(scenario, affected_cells)

        st.divider()

        # Simulation controls
        st.markdown("### 🎮 Simulation")
        steps = st.slider("Steps per click", 1, 50, 5)

        col_run, col_reset = st.columns(2)
        with col_run:
            run_btn = st.button("▶ Run Steps", use_container_width=True, type="primary")
        with col_reset:
            if st.button("🔄 Reset Pack", use_container_width=True):
                st.session_state.pack.reset()
                st.session_state.history = []
                st.session_state.alerts_log = []
                st.session_state.cycle_count = 0
                st.session_state.db.clear_all()
                st.rerun()

        # Auto-run toggle
        auto_run = st.toggle("🔁 Auto-run (continuous)", value=st.session_state.auto_run)
        st.session_state.auto_run = auto_run

        st.divider()

        # Cell selector
        st.markdown("### 🔍 Cell Inspector")
        st.session_state.selected_cell = st.slider(
            "Select Cell", 0, NUM_CELLS - 1,
            st.session_state.selected_cell,
        )

        st.divider()

        # Model status
        st.markdown("### 🤖 Model Status")
        ad_status = "✅ Loaded" if st.session_state.anomaly_detector.is_fitted else "❌ Not trained"
        rp_status = "✅ Loaded" if st.session_state.rul_predictor.is_fitted else "❌ Not trained"
        st.markdown(f"**Isolation Forest:** {ad_status}")
        st.markdown(f"**LSTM (RUL):** {rp_status}")

        if not st.session_state.anomaly_detector.is_fitted:
            st.caption("Run `python train_models.py` to train models.")

        st.divider()

        # Pack info
        pack_summary = st.session_state.pack.get_pack_summary()
        st.markdown("### 📊 Pack Info")
        st.json(pack_summary, expanded=False)

    # ── Run simulation ──────────────────────────
    cell_data = None
    if run_btn or st.session_state.auto_run:
        for _ in range(steps):
            cell_data = run_step()

    # Get latest data (from run or from history)
    if cell_data is None and st.session_state.history:
        cell_data = st.session_state.history[-1]
    elif cell_data is None:
        # Initial state — run one step
        cell_data = run_step()

    # ── Main Layout ─────────────────────────────
    render_header()
    render_metrics(cell_data)

    # ── 3D Visualization + Analytics Tabs ───────
    try:
        from components_3d import render_battery_3d
        render_battery_3d(cell_data, st.session_state.selected_cell)
    except Exception as e:
        # Fallback if 3D component has issues
        st.warning(f"3D visualization unavailable: {e}")
        # Show a simple table instead
        st.dataframe(
            pd.DataFrame(cell_data),
            use_container_width=True,
            hide_index=True,
        )

    # ── Tabbed Analytics ────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Real-time Sensors",
        "🧠 ML Prognostics",
        "🚨 Alert Log",
        "🔍 Cell Inspector",
    ])

    with tab1:
        if len(st.session_state.history) > 3:
            # Aggregate history for charts
            all_steps = []
            for step_idx, step_data in enumerate(st.session_state.history):
                for cell in step_data:
                    row = cell.copy()
                    row["step"] = step_idx
                    all_steps.append(row)
            df_all = pd.DataFrame(all_steps)

            # Average across cells per step
            df_avg = df_all.groupby("step").agg({
                "voltage": "mean", "current": "mean",
                "temperature": "mean", "soc": "mean",
                "soh": "mean", "internal_resistance": "mean",
            }).reset_index()

            col1, col2 = st.columns(2)
            with col1:
                fig = make_line_chart(df_avg, "step", ["voltage"], "Pack Average Voltage")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = make_line_chart(
                    df_avg, "step", ["temperature"],
                    "Pack Average Temperature",
                    colors_list=[COLORS["accent_orange"]]
                )
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                fig = make_line_chart(df_avg, "step", ["soc", "soh"], "Pack SoC & SoH")
                st.plotly_chart(fig, use_container_width=True)
            with col4:
                fig = make_line_chart(
                    df_avg, "step", ["current"],
                    "Pack Average Current",
                    colors_list=[COLORS["accent_blue"]]
                )
                st.plotly_chart(fig, use_container_width=True)

            # Per-cell heatmap
            st.markdown('<div class="section-title"><span class="icon">🌡️</span> Cell Temperature Heatmap</div>',
                        unsafe_allow_html=True)
            last_step = st.session_state.history[-1]
            temps = [c["temperature"] for c in last_step]
            temp_matrix = np.array(temps).reshape(4, 4)
            fig = go.Figure(go.Heatmap(
                z=temp_matrix,
                colorscale=[
                    [0, COLORS["accent_cyan"]],
                    [0.4, "#ffd166"],
                    [0.7, COLORS["accent_orange"]],
                    [1, COLORS["danger"]],
                ],
                text=[[f"Cell {r*4+c}<br>{temp_matrix[r][c]:.1f}°C" for c in range(4)] for r in range(4)],
                texttemplate="%{text}",
                textfont=dict(size=11, color="white"),
                showscale=True,
                colorbar=dict(title="°C"),
            ))
            fig.update_layout(
                height=300,
                title=dict(text="Cell Grid Temperature Map (4×4)", font=dict(size=14, color=COLORS["text_primary"])),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Run a few simulation steps to see real-time charts.")

    with tab2:
        if len(st.session_state.history) > 3:
            st.markdown('<div class="section-title"><span class="icon">🧠</span> ML Model Predictions</div>',
                        unsafe_allow_html=True)

            # Gauges for selected cell
            sel = st.session_state.selected_cell
            sel_data = cell_data[sel]

            g1, g2, g3 = st.columns(3)
            with g1:
                fig = make_gauge(sel_data["soh"], f"Cell {sel} SoH", 0, 100, [SOH_CRITICAL, SOH_WARNING, 100])
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                rul_val = sel_data.get("predicted_rul", MAX_CYCLES - st.session_state.cycle_count)
                fig = make_gauge(rul_val, f"Cell {sel} RUL (cycles)", 0, MAX_CYCLES, [RUL_CRITICAL, RUL_WARNING, MAX_CYCLES])
                st.plotly_chart(fig, use_container_width=True)
            with g3:
                fig = make_gauge(sel_data["temperature"], f"Cell {sel} Temp", 0, 80, [TEMP_WARNING, TEMP_CRITICAL, 80])
                st.plotly_chart(fig, use_container_width=True)

            # Anomaly scatter
            st.markdown('<div class="section-title"><span class="icon">🔍</span> Anomaly Detection Results</div>',
                        unsafe_allow_html=True)
            anom_data = pd.DataFrame(cell_data)
            if "anomaly_score" in anom_data.columns:
                colors = [COLORS["danger"] if a else COLORS["accent_cyan"]
                          for a in anom_data.get("is_anomaly", [False] * len(anom_data))]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Cell {i}" for i in range(NUM_CELLS)],
                    y=anom_data["anomaly_score"],
                    marker=dict(color=colors),
                    text=anom_data["anomaly_score"].round(3),
                    textposition="auto",
                ))
                fig.update_layout(
                    title=dict(text="Isolation Forest Anomaly Scores (lower = more anomalous)",
                               font=dict(size=14, color=COLORS["text_primary"])),
                    height=300,
                    yaxis_title="Anomaly Score",
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

            # SoH degradation across all cells
            if len(st.session_state.history) > 10:
                st.markdown('<div class="section-title"><span class="icon">📉</span> SoH Degradation Trends</div>',
                            unsafe_allow_html=True)
                fig = go.Figure()
                for cid in range(NUM_CELLS):
                    soh_series = [step[cid]["soh"] for step in st.session_state.history]
                    fig.add_trace(go.Scatter(
                        y=soh_series,
                        name=f"Cell {cid}",
                        mode="lines",
                        line=dict(width=1.5),
                        opacity=0.7,
                    ))
                fig.update_layout(
                    title=dict(text="All Cells — State of Health Over Time",
                               font=dict(size=14, color=COLORS["text_primary"])),
                    height=350,
                    yaxis_title="SoH (%)",
                    xaxis_title="Simulation Step",
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Run more simulation steps to see ML predictions.")

    with tab3:
        render_alerts()

    with tab4:
        render_cell_detail(cell_data, st.session_state.selected_cell)

    # ── Auto-rerun ──────────────────────────────
    if st.session_state.auto_run:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()

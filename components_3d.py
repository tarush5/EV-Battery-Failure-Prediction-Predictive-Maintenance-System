"""
components_3d.py
================
Provides immersive 3D battery-pack visualisation and glassmorphism metric
cards for the EV Battery Predictive Maintenance dashboard.

Functions
---------
render_battery_3d(cell_data, selected_cell)
    Embeds a full Three.js WebGL scene showing 16 cylindrical cells inside a
    transparent casing with thermal colour-mapping, anomaly pulse rings,
    current-flow particles, orbit controls, and an info HUD overlay.

render_battery_stats_cards(cell_data)
    Renders a row of four glassmorphism metric cards (Pack Health, Avg Voltage,
    Peak Temperature, Active Anomalies) with animated gradient borders.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# 3D Battery Scene
# ---------------------------------------------------------------------------

def render_battery_3d(
    cell_data: list[dict],
    selected_cell: Optional[int] = None,
) -> None:
    """Render an interactive Three.js 3D battery pack visualisation.

    Parameters
    ----------
    cell_data : list[dict]
        A list of 16 dicts, each containing keys ``cell_id``, ``voltage``,
        ``current``, ``temperature``, ``internal_resistance``, ``soc``,
        ``soh``, ``anomaly_score``, and ``is_anomaly``.
    selected_cell : int | None, optional
        Index (0-15) of the currently selected cell.  When set the
        corresponding cylinder receives a cyan highlight and elevation.
    """

    cell_json = json.dumps(cell_data)
    sel_json = json.dumps(selected_cell)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
  html,body{{width:100%;height:100%;overflow:hidden;background:#0a0e17;font-family:'Segoe UI',system-ui,sans-serif;}}
  #canvas-container{{position:relative;width:100%;height:600px;}}
  canvas{{display:block;width:100%!important;height:100%!important;}}

  /* ---------- HUD ---------- */
  #hud{{
    position:absolute;top:12px;right:12px;
    display:flex;flex-direction:column;gap:8px;
    z-index:10;pointer-events:none;
  }}
  .hud-card{{
    background:rgba(15,20,35,0.55);
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border:1px solid rgba(100,220,255,0.15);
    border-radius:10px;padding:10px 16px;
    color:#e0e6f0;font-size:13px;
    display:flex;align-items:center;gap:8px;
    text-shadow:0 0 6px rgba(0,200,255,0.3);
  }}
  .hud-card .val{{font-size:17px;font-weight:700;color:#67e8f9;}}
  .hud-card .lbl{{opacity:0.7;font-size:11px;}}

  /* ---------- Legend ---------- */
  #legend{{
    position:absolute;bottom:12px;left:12px;
    background:rgba(15,20,35,0.55);
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border:1px solid rgba(100,220,255,0.12);
    border-radius:10px;padding:10px 14px;
    color:#cbd5e1;font-size:11px;
    display:flex;gap:14px;align-items:center;z-index:10;pointer-events:none;
  }}
  .legend-swatch{{width:14px;height:14px;border-radius:3px;display:inline-block;margin-right:4px;vertical-align:middle;}}
</style>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.163.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.163.0/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="canvas-container">
  <!-- HUD filled by JS -->
  <div id="hud"></div>
  <div id="legend">
    <span><span class="legend-swatch" style="background:#06d6a0"></span>≤25°C</span>
    <span><span class="legend-swatch" style="background:#ffd166"></span>35°C</span>
    <span><span class="legend-swatch" style="background:#ff9f1c"></span>45°C</span>
    <span><span class="legend-swatch" style="background:#ef4444"></span>55°C</span>
    <span><span class="legend-swatch" style="background:#f72585"></span>≥60°C</span>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

/* ------------------------------------------------------------------ */
/* Data injection                                                      */
/* ------------------------------------------------------------------ */
const cellData   = {cell_json};
const selectedCell = {sel_json};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
function tempToColor(t) {{
  // piecewise HSL lerp through the 5-stop gradient
  const stops = [
    {{ t: 25, h: 160, s: 95, l: 45 }},  // #06d6a0
    {{ t: 35, h: 44,  s: 100,l: 60 }},  // #ffd166
    {{ t: 45, h: 33,  s: 100,l: 55 }},  // #ff9f1c
    {{ t: 55, h: 0,   s: 84, l: 60 }},  // #ef4444
    {{ t: 60, h: 330, s: 92, l: 56 }},  // #f72585
  ];
  if (t <= stops[0].t) return new THREE.Color().setHSL(stops[0].h/360, stops[0].s/100, stops[0].l/100);
  if (t >= stops[stops.length-1].t) return new THREE.Color().setHSL(stops[stops.length-1].h/360, stops[stops.length-1].s/100, stops[stops.length-1].l/100);
  for (let i = 0; i < stops.length - 1; i++) {{
    if (t >= stops[i].t && t <= stops[i+1].t) {{
      const f = (t - stops[i].t) / (stops[i+1].t - stops[i].t);
      const h = THREE.MathUtils.lerp(stops[i].h, stops[i+1].h, f);
      const s = THREE.MathUtils.lerp(stops[i].s, stops[i+1].s, f);
      const l = THREE.MathUtils.lerp(stops[i].l, stops[i+1].l, f);
      return new THREE.Color().setHSL(h/360, s/100, l/100);
    }}
  }}
  return new THREE.Color(0x06d6a0);
}}

/* ------------------------------------------------------------------ */
/* Scene setup                                                         */
/* ------------------------------------------------------------------ */
const container = document.getElementById('canvas-container');
const renderer  = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, 600);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
container.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e17);
scene.fog = new THREE.FogExp2(0x0a0e17, 0.035);

const camera = new THREE.PerspectiveCamera(42, container.clientWidth / 600, 0.1, 200);
camera.position.set(8, 7, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping   = true;
controls.dampingFactor   = 0.06;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.6;
controls.minDistance = 6;
controls.maxDistance = 25;
controls.target.set(0, 0.4, 0);

/* ------------------------------------------------------------------ */
/* Lighting                                                            */
/* ------------------------------------------------------------------ */
scene.add(new THREE.AmbientLight(0x8899bb, 0.4));

const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
dir1.position.set(5, 10, 7);
dir1.castShadow = false;
scene.add(dir1);

const dir2 = new THREE.DirectionalLight(0x6ea8fe, 0.5);
dir2.position.set(-6, 4, -5);
scene.add(dir2);

/* ------------------------------------------------------------------ */
/* Star field background particles                                     */
/* ------------------------------------------------------------------ */
const starCount = 600;
const starGeo   = new THREE.BufferGeometry();
const starPos   = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {{
  starPos[i*3]   = (Math.random() - 0.5) * 80;
  starPos[i*3+1] = (Math.random() - 0.5) * 80;
  starPos[i*3+2] = (Math.random() - 0.5) * 80;
}}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
const starMat = new THREE.PointsMaterial({{ color: 0xaaccff, size: 0.12, transparent: true, opacity: 0.7 }});
const stars   = new THREE.Points(starGeo, starMat);
scene.add(stars);

/* ------------------------------------------------------------------ */
/* Battery casing (transparent box with wireframe edges)               */
/* ------------------------------------------------------------------ */
const casingW = 4.2, casingH = 2.6, casingD = 4.2;
const casingGeo = new THREE.BoxGeometry(casingW, casingH, casingD, 1, 1, 1);
const casingMat = new THREE.MeshPhysicalMaterial({{
  color: 0x88bbee,
  metalness: 0.1,
  roughness: 0.05,
  transmission: 0.9,
  ior: 1.5,
  thickness: 0.5,
  transparent: true,
  opacity: 0.18,
  side: THREE.DoubleSide,
}});
const casing = new THREE.Mesh(casingGeo, casingMat);
casing.position.y = casingH / 2 - 0.05;
scene.add(casing);

// Glowing wireframe edges
const edgesGeo = new THREE.EdgesGeometry(casingGeo);
const edgesMat = new THREE.LineBasicMaterial({{ color: 0x22d3ee, transparent: true, opacity: 0.45, linewidth: 1 }});
const edgesMesh = new THREE.LineSegments(edgesGeo, edgesMat);
edgesMesh.position.copy(casing.position);
scene.add(edgesMesh);

/* ------------------------------------------------------------------ */
/* Ground grid                                                         */
/* ------------------------------------------------------------------ */
const gridHelper = new THREE.GridHelper(20, 40, 0x1a2744, 0x111a2e);
gridHelper.position.y = -0.06;
scene.add(gridHelper);

/* ------------------------------------------------------------------ */
/* 16 Cylindrical Cells                                                */
/* ------------------------------------------------------------------ */
const cellRadius = 0.3;
const cellHeight = 1.8;
const cellGeo = new THREE.CylinderGeometry(cellRadius, cellRadius, cellHeight, 24, 1);
const cells   = [];      // mesh references
const cellMats = [];

// Torus ring for anomaly pulse
const torusGeo = new THREE.TorusGeometry(cellRadius + 0.12, 0.04, 8, 32);
const anomalyRings = [];

// Sprite labels
function makeLabel(text) {{
  const canvas = document.createElement('canvas');
  canvas.width = 64; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#e0e6f0';
  ctx.font = 'bold 36px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true, depthTest: false }});
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.45, 0.45, 1);
  return sprite;
}}

const spacing = 1.05;
const offset  = spacing * 1.5;

cellData.forEach((cd, idx) => {{
  const row = Math.floor(idx / 4);
  const col = idx % 4;
  const x = col * spacing - offset;
  const z = row * spacing - offset;

  const color = tempToColor(cd.temperature);
  const mat = new THREE.MeshStandardMaterial({{
    color: color,
    emissive: color.clone().multiplyScalar(0.35),
    emissiveIntensity: 0.9,
    metalness: 0.35,
    roughness: 0.45,
  }});
  cellMats.push(mat);

  const mesh = new THREE.Mesh(cellGeo, mat);
  mesh.position.set(x, cellHeight / 2, z);

  // Selected cell highlight
  if (selectedCell === idx) {{
    mesh.position.y += 0.25;
    mat.emissive.set(0x22d3ee);
    mat.emissiveIntensity = 1.6;
  }}

  scene.add(mesh);
  cells.push(mesh);

  // Anomaly ring
  if (cd.is_anomaly) {{
    const ringMat = new THREE.MeshBasicMaterial({{ color: 0xf72585, transparent: true, opacity: 0.85 }});
    const ring = new THREE.Mesh(torusGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(x, cellHeight + 0.15, z);
    scene.add(ring);
    anomalyRings.push(ring);

    // Point light near anomaly
    const pl = new THREE.PointLight(0xf72585, 0.6, 3);
    pl.position.set(x, cellHeight + 0.5, z);
    scene.add(pl);
  }}

  // Label
  const label = makeLabel(String(cd.cell_id));
  label.position.set(x, cellHeight + 0.35 + (selectedCell === idx ? 0.25 : 0), z);
  scene.add(label);
}});

/* ------------------------------------------------------------------ */
/* Current flow particles                                              */
/* ------------------------------------------------------------------ */
const maxParticlesPerCell = 30;
const totalParticles = 16 * maxParticlesPerCell;
const particleGeo = new THREE.BufferGeometry();
const pPositions = new Float32Array(totalParticles * 3);
const pColors    = new Float32Array(totalParticles * 3);
const pVelocities = new Float32Array(totalParticles);  // y-velocity
const pCellIdx    = new Int16Array(totalParticles);     // which cell
const pActive     = new Uint8Array(totalParticles);

const chargingColor    = new THREE.Color(0x22d3ee);
const dischargingColor = new THREE.Color(0xff9f1c);

let pIdx = 0;
cellData.forEach((cd, idx) => {{
  const row = Math.floor(idx / 4);
  const col = idx % 4;
  const cx  = col * spacing - offset;
  const cz  = row * spacing - offset;
  const absCur = Math.abs(cd.current);
  const active = Math.min(maxParticlesPerCell, Math.round((absCur / 10) * maxParticlesPerCell));
  const isCharging = cd.current < 0;
  const speed = (absCur / 10) * 0.06;
  const baseColor = isCharging ? chargingColor : dischargingColor;

  for (let j = 0; j < maxParticlesPerCell; j++) {{
    const i = pIdx++;
    pCellIdx[i] = idx;
    if (j < active) {{
      pActive[i] = 1;
      const angle  = Math.random() * Math.PI * 2;
      const r      = Math.random() * cellRadius * 0.65;
      pPositions[i*3]     = cx + Math.cos(angle) * r;
      pPositions[i*3 + 1] = Math.random() * cellHeight;
      pPositions[i*3 + 2] = cz + Math.sin(angle) * r;
      pVelocities[i] = isCharging ? speed : -speed;
      pColors[i*3]     = baseColor.r;
      pColors[i*3 + 1] = baseColor.g;
      pColors[i*3 + 2] = baseColor.b;
    }} else {{
      pActive[i] = 0;
      pPositions[i*3] = 0; pPositions[i*3+1] = -100; pPositions[i*3+2] = 0;
      pColors[i*3] = 0; pColors[i*3+1] = 0; pColors[i*3+2] = 0;
    }}
  }}
}});

particleGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
particleGeo.setAttribute('color',    new THREE.BufferAttribute(pColors, 3));
const particleMat = new THREE.PointsMaterial({{
  size: 0.045,
  vertexColors: true,
  transparent: true,
  opacity: 0.85,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
}});
const particleSystem = new THREE.Points(particleGeo, particleMat);
scene.add(particleSystem);

/* ------------------------------------------------------------------ */
/* HUD                                                                 */
/* ------------------------------------------------------------------ */
const hudEl = document.getElementById('hud');
const avgTemp    = (cellData.reduce((s,c) => s + c.temperature, 0) / cellData.length).toFixed(1);
const packV      = cellData.reduce((s,c) => s + c.voltage, 0).toFixed(2);
const anomCount  = cellData.filter(c => c.is_anomaly).length;
hudEl.innerHTML = `
  <div class="hud-card"><span class="val">${{avgTemp}}°C</span><span class="lbl">Avg Temp</span></div>
  <div class="hud-card"><span class="val">${{packV}} V</span><span class="lbl">Pack Voltage</span></div>
  <div class="hud-card"><span class="val">${{anomCount}}</span><span class="lbl">Anomalies</span></div>
`;

/* ------------------------------------------------------------------ */
/* Animation loop                                                      */
/* ------------------------------------------------------------------ */
const clock = new THREE.Clock();

function animate() {{
  requestAnimationFrame(animate);
  const t  = clock.getElapsedTime();
  const dt = clock.getDelta();

  // Star twinkle
  starMat.opacity = 0.5 + 0.25 * Math.sin(t * 0.4);

  // Anomaly ring pulse
  anomalyRings.forEach((ring, i) => {{
    const s = 1.0 + 0.25 * Math.sin(t * 4 + i);
    ring.scale.set(s, s, 1);
    ring.material.opacity = 0.55 + 0.4 * Math.abs(Math.sin(t * 4 + i));
  }});

  // Temperature flash for ≥60°C
  cellData.forEach((cd, idx) => {{
    if (cd.temperature >= 60) {{
      const flash = 0.8 + 0.5 * Math.abs(Math.sin(t * 6 + idx));
      cellMats[idx].emissiveIntensity = flash;
    }}
  }});

  // Current-flow particle update
  const posArr = particleGeo.attributes.position.array;
  for (let i = 0; i < totalParticles; i++) {{
    if (!pActive[i]) continue;
    posArr[i*3 + 1] += pVelocities[i];
    if (posArr[i*3 + 1] > cellHeight) posArr[i*3 + 1] = 0;
    if (posArr[i*3 + 1] < 0) posArr[i*3 + 1] = cellHeight;
  }}
  particleGeo.attributes.position.needsUpdate = true;

  // Edge glow pulse
  edgesMat.opacity = 0.3 + 0.15 * Math.sin(t * 1.5);

  controls.update();
  renderer.render(scene, camera);
}}

animate();

/* ------------------------------------------------------------------ */
/* Resize handling                                                     */
/* ------------------------------------------------------------------ */
window.addEventListener('resize', () => {{
  const w = container.clientWidth;
  camera.aspect = w / 600;
  camera.updateProjectionMatrix();
  renderer.setSize(w, 600);
}});
</script>
</body>
</html>
"""

    components.html(html_content, height=650, scrolling=False)


# ---------------------------------------------------------------------------
# Stats Cards
# ---------------------------------------------------------------------------

def render_battery_stats_cards(cell_data: list[dict]) -> None:
    """Render a row of four glassmorphism metric cards summarising pack health.

    Cards displayed:
      1. **Pack Health** — average State of Health across all cells.
      2. **Avg Voltage** — arithmetic mean cell voltage.
      3. **Peak Temperature** — hottest cell temperature.
      4. **Active Anomalies** — number of cells flagged as anomalous.

    Parameters
    ----------
    cell_data : list[dict]
        Same 16-element list of cell dictionaries used by
        :func:`render_battery_3d`.
    """

    avg_soh  = sum(c["soh"] for c in cell_data) / max(len(cell_data), 1)
    avg_volt = sum(c["voltage"] for c in cell_data) / max(len(cell_data), 1)
    peak_tmp = max(c["temperature"] for c in cell_data)
    anomalies = sum(1 for c in cell_data if c["is_anomaly"])

    cards = [
        {
            "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
            "value": f"{avg_soh:.1f}%",
            "label": "Pack Health (SoH)",
            "gradient": "linear-gradient(135deg, #06b6d4, #34d399)",
        },
        {
            "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#facc15" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>',
            "value": f"{avg_volt:.2f} V",
            "label": "Avg Voltage",
            "gradient": "linear-gradient(135deg, #f59e0b, #facc15)",
        },
        {
            "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg>',
            "value": f"{peak_tmp:.1f}°C",
            "label": "Peak Temperature",
            "gradient": "linear-gradient(135deg, #ef4444, #f97316)",
        },
        {
            "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fb7185" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            "value": str(anomalies),
            "label": "Active Anomalies",
            "gradient": "linear-gradient(135deg, #e11d48, #f472b6)",
        },
    ]

    cards_json = json.dumps(cards)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:transparent;font-family:'Segoe UI',system-ui,sans-serif;}}

  .cards-row{{
    display:flex;gap:16px;flex-wrap:wrap;justify-content:center;padding:8px 4px;
  }}

  .stat-card{{
    position:relative;
    flex:1 1 180px;max-width:240px;
    background:rgba(15,20,40,0.6);
    backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
    border-radius:16px;
    padding:22px 20px;
    color:#e2e8f0;
    overflow:hidden;
    transition:transform 0.25s ease, box-shadow 0.25s ease;
  }}
  .stat-card:hover{{
    transform:translateY(-4px);
    box-shadow:0 8px 30px rgba(0,200,255,0.12);
  }}

  /* Animated gradient border */
  .stat-card::before{{
    content:'';
    position:absolute;inset:0;
    border-radius:16px;padding:1.5px;
    background:var(--grad);
    -webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);
    mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);
    -webkit-mask-composite:xor;mask-composite:exclude;
    animation:borderSpin 4s linear infinite;
    background-size:300% 300%;
  }}
  @keyframes borderSpin{{
    0%{{background-position:0% 50%}}
    50%{{background-position:100% 50%}}
    100%{{background-position:0% 50%}}
  }}

  .card-icon{{margin-bottom:10px;}}
  .card-value{{font-size:26px;font-weight:800;letter-spacing:-0.5px;}}
  .card-label{{font-size:12px;opacity:0.6;margin-top:4px;text-transform:uppercase;letter-spacing:0.5px;}}
</style>
</head>
<body>
<div class="cards-row" id="cards-row"></div>
<script>
  const cards = {cards_json};
  const row   = document.getElementById('cards-row');
  cards.forEach(c => {{
    const div = document.createElement('div');
    div.className = 'stat-card';
    div.style.setProperty('--grad', c.gradient);
    div.innerHTML = `
      <div class="card-icon">${{c.icon}}</div>
      <div class="card-value">${{c.value}}</div>
      <div class="card-label">${{c.label}}</div>
    `;
    row.appendChild(div);
  }});
</script>
</body>
</html>
"""
    components.html(html, height=170, scrolling=False)

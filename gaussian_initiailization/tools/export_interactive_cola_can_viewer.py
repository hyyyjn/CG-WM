from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a standalone WebGL viewer for a learned cola-can Gaussian body.")
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--fit_trajectory", default=None, type=Path)
    parser.add_argument("--output_html", required=True, type=Path)
    parser.add_argument("--foreground_threshold", default=0.99, type=float)
    parser.add_argument("--opacity_threshold", default=0.0, type=float)
    parser.add_argument("--max_points", default=45000, type=int)
    parser.add_argument("--stage1_centroid", default="0,0,0.1")
    parser.add_argument("--title", default="ContactGaussian-WM Cola Can Viewer")
    return parser.parse_args()


def parse_vec3(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).replace(" ", ",").split(",") if value.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected x,y,z vector, got {raw!r}")
    return values


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_gaussian_points(path: Path, foreground_threshold: float, opacity_threshold: float, max_points: int):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("plyfile is required to export the interactive viewer.") from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    required = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "foreground_logit", "opacity")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing fields required for viewer export: {missing}")

    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)
    foreground = 1.0 / (1.0 + np.exp(-np.asarray(vertices["foreground_logit"]).astype(np.float32)))
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertices["opacity"]).astype(np.float32)))
    mask = foreground >= float(foreground_threshold)
    if opacity_threshold > 0.0:
        mask &= opacity >= float(opacity_threshold)

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No Gaussian primitives passed the requested thresholds.")

    if max_points > 0 and idx.size > max_points:
        scores = foreground[idx] * np.maximum(opacity[idx], 1e-4)
        keep_order = np.argsort(scores)[-int(max_points) :]
        idx = idx[keep_order]

    xyz = xyz[idx]
    fdc = np.stack([vertices["f_dc_0"][idx], vertices["f_dc_1"][idx], vertices["f_dc_2"][idx]], axis=-1).astype(np.float32)
    rgb = np.clip(fdc * SH_C0 + 0.5, 0.0, 1.0)
    alpha = np.clip(opacity[idx], 0.72, 1.0).astype(np.float32)
    colors = np.concatenate([rgb, alpha[:, None]], axis=-1)

    return {
        "positions": np.round(xyz, 6).reshape(-1).tolist(),
        "colors": np.round(colors, 5).reshape(-1).tolist(),
        "count": int(xyz.shape[0]),
        "source_count": int(vertices.shape[0]),
        "foreground_count": int(np.flatnonzero(mask).size),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
    }


def normalize_trajectory(payload: dict, *, label: str) -> dict:
    states = payload.get("states") or []
    frames = []
    for i, state in enumerate(states):
        position = state.get("predicted_position", state.get("target_position", state.get("position")))
        quaternion = state.get(
            "predicted_quaternion_wxyz",
            state.get("target_quaternion_wxyz", state.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])),
        )
        target = state.get("target_position", position)
        if position is None:
            continue
        frames.append(
            {
                "i": int(state.get("frame_index", i)),
                "t": float(state.get("time", i / 30.0)),
                "p": [float(v) for v in position],
                "q": [float(v) for v in quaternion],
                "target": [float(v) for v in target],
                "contact": float(state.get("contact_gate", 0.0)),
            }
        )
    if not frames:
        raise ValueError(f"{label} trajectory contains no usable states.")
    return {"label": label, "frames": frames}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, system-ui, sans-serif; }
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #111; }
    canvas { width: 100vw; height: 100vh; display: block; background: linear-gradient(#f8f8f4, #e9edf0); cursor: grab; }
    canvas:active { cursor: grabbing; }
    .panel {
      position: fixed; left: 16px; top: 16px; width: min(420px, calc(100vw - 32px));
      background: rgba(13, 16, 20, 0.82); border: 1px solid rgba(255,255,255,0.14);
      border-radius: 8px; color: #f5f7fb; padding: 12px; backdrop-filter: blur(8px);
      box-shadow: 0 18px 60px rgba(0,0,0,0.22);
    }
    .row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    .row:first-child { margin-top: 0; }
    button, select {
      background: #232a33; color: #fff; border: 1px solid rgba(255,255,255,0.18);
      border-radius: 6px; height: 30px; padding: 0 10px; font: inherit;
    }
    button:hover, select:hover { background: #303945; }
    input[type="range"] { flex: 1; }
    label { font-size: 12px; color: #cfd7e3; white-space: nowrap; }
    .stat { margin-left: auto; font-variant-numeric: tabular-nums; color: #ffffff; font-size: 12px; }
    .title { font-weight: 700; letter-spacing: 0; }
    .hint { color: #aeb9c8; font-size: 12px; line-height: 1.45; }
  </style>
</head>
<body>
<canvas id="view"></canvas>
<div class="panel">
  <div class="row"><div class="title">ContactGaussian-WM Cola Can</div><div class="stat" id="pointStat"></div></div>
  <div class="row">
    <button id="play">Pause</button>
    <button id="reset">Reset Cam</button>
    <button id="view34">3/4</button>
    <button id="viewSide">Side</button>
    <button id="viewTop">Top</button>
    <select id="traj"></select>
    <label><input id="follow" type="checkbox" checked /> follow</label>
  </div>
  <div class="row">
    <label>frame</label>
    <input id="frame" type="range" min="0" max="1" value="0" step="1" />
    <div class="stat" id="frameStat"></div>
  </div>
  <div class="row">
    <label>point size</label>
    <input id="size" type="range" min="2" max="80" value="30" step="1" />
    <div class="stat" id="sizeStat"></div>
  </div>
  <div class="hint">Drag to orbit, wheel to zoom, right-drag to pan. This viewer uses the learned foreground Gaussian centers/colors and the optimized rigid trajectory for realtime inspection.</div>
</div>
<script>
const DATA = __DATA__;

const canvas = document.getElementById('view');
const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
if (!gl) alert('WebGL is required for this viewer.');

const vs = `
attribute vec3 aPosition;
attribute vec4 aColor;
uniform mat4 uViewProj;
uniform vec3 uTranslation;
uniform vec4 uQuat;
uniform vec3 uCentroid;
uniform float uPointSize;
varying vec4 vColor;
vec3 quatRotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
void main() {
  vec3 local = aPosition - uCentroid;
  vec3 world = quatRotate(uQuat, local) + uTranslation;
  vec4 clip = uViewProj * vec4(world, 1.0);
  gl_Position = clip;
  gl_PointSize = clamp(uPointSize / max(0.18, clip.w), 1.0, 96.0);
  vColor = aColor;
}`;
const fs = `
precision mediump float;
varying vec4 vColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float soft = smoothstep(0.25, 0.02, r2);
  gl_FragColor = vec4(vColor.rgb, vColor.a * soft);
}`;

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vs));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);

const objectAttribs = {};
function bufferAttrib(name, data, size) {
  const loc = gl.getAttribLocation(program, name);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  objectAttribs[name] = { loc, buf, size };
}
bufferAttrib('aPosition', new Float32Array(DATA.positions), 3);
bufferAttrib('aColor', new Float32Array(DATA.colors), 4);

function bindObjectAttribs() {
  for (const name of Object.keys(objectAttribs)) {
    const attr = objectAttribs[name];
    gl.bindBuffer(gl.ARRAY_BUFFER, attr.buf);
    gl.enableVertexAttribArray(attr.loc);
    gl.vertexAttribPointer(attr.loc, attr.size, gl.FLOAT, false, 0, 0);
  }
}

const U = {
  viewProj: gl.getUniformLocation(program, 'uViewProj'),
  translation: gl.getUniformLocation(program, 'uTranslation'),
  quat: gl.getUniformLocation(program, 'uQuat'),
  centroid: gl.getUniformLocation(program, 'uCentroid'),
  pointSize: gl.getUniformLocation(program, 'uPointSize'),
};

const playBtn = document.getElementById('play');
const resetBtn = document.getElementById('reset');
const view34Btn = document.getElementById('view34');
const viewSideBtn = document.getElementById('viewSide');
const viewTopBtn = document.getElementById('viewTop');
const trajSelect = document.getElementById('traj');
const frameSlider = document.getElementById('frame');
const followBox = document.getElementById('follow');
const sizeSlider = document.getElementById('size');
const frameStat = document.getElementById('frameStat');
const sizeStat = document.getElementById('sizeStat');
const pointStat = document.getElementById('pointStat');

DATA.trajectories.forEach((traj, i) => {
  const opt = document.createElement('option');
  opt.value = String(i);
  opt.textContent = traj.label;
  trajSelect.appendChild(opt);
});
pointStat.textContent = `${DATA.count.toLocaleString()} gaussians`;

let trajIndex = 0;
let frameIndex = 0;
let playing = true;
let yaw = -0.70, pitch = 0.34, radius = 0.92;
let pan = [0, 0, 0];
let drag = null;
let lastTime = performance.now();

function activeTraj() { return DATA.trajectories[trajIndex]; }
function activeFrame() { return activeTraj().frames[Math.max(0, Math.min(frameIndex, activeTraj().frames.length - 1))]; }
function setFrameMax() {
  frameSlider.max = String(activeTraj().frames.length - 1);
  frameIndex = Math.min(frameIndex, activeTraj().frames.length - 1);
  frameSlider.value = String(frameIndex);
}
setFrameMax();

playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; };
function setCamera(nextYaw, nextPitch, nextRadius) {
  yaw = nextYaw; pitch = nextPitch; radius = nextRadius; pan = [0, 0, 0];
}
resetBtn.onclick = () => setCamera(-0.70, 0.34, 0.92);
view34Btn.onclick = () => setCamera(-0.72, 0.38, 0.92);
viewSideBtn.onclick = () => setCamera(-1.57, 0.22, 0.95);
viewTopBtn.onclick = () => setCamera(-0.70, 1.12, 1.20);
trajSelect.onchange = () => { trajIndex = Number(trajSelect.value); frameIndex = 0; setFrameMax(); };
frameSlider.oninput = () => { frameIndex = Number(frameSlider.value); playing = false; playBtn.textContent = 'Play'; };

canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, button: e.button }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', e => { drag = null; canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.button === 2) {
    pan[0] -= dx * radius * 0.0012;
    pan[2] += dy * radius * 0.0012;
  } else {
    yaw += dx * 0.006;
    pitch = Math.max(-1.35, Math.min(1.15, pitch + dy * 0.005));
  }
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  radius = Math.max(0.16, Math.min(3.6, radius * Math.exp(e.deltaY * 0.001)));
}, { passive: false });

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
}

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/n, v[1]/n, v[2]/n];
}
function cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function lookAt(eye, target) {
  const z = normalize([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]);
  const x = normalize(cross([0,0,1], z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ];
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [
    f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0
  ];
}
function mul(a, b) {
  const out = new Array(16).fill(0);
  for (let c=0; c<4; c++) for (let r=0; r<4; r++) {
    out[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
  }
  return out;
}

function drawGrid(viewProj, target) {
  const grid = [];
  const extent = 1.65, step = 0.1, z = 0;
  for (let v=-extent; v<=extent+1e-6; v+=step) {
    grid.push(-extent, v, z, extent, v, z, v, -extent, z, v, extent, z);
  }
  const floor = [-extent, -extent, z, extent, -extent, z, -extent, extent, z, extent, extent, z];
  if (!drawGrid.buf) {
    drawGrid.buf = gl.createBuffer();
    drawGrid.floorBuf = gl.createBuffer();
    drawGrid.program = gl.createProgram();
    const gvs = `attribute vec3 p; uniform mat4 m; void main(){ gl_Position=m*vec4(p,1.0); }`;
    const gfs = `precision mediump float; uniform vec4 c; void main(){ gl_FragColor=c; }`;
    gl.attachShader(drawGrid.program, compile(gl.VERTEX_SHADER, gvs));
    gl.attachShader(drawGrid.program, compile(gl.FRAGMENT_SHADER, gfs));
    gl.linkProgram(drawGrid.program);
  }
  gl.useProgram(drawGrid.program);
  const colorLoc = gl.getUniformLocation(drawGrid.program, 'c');
  const matrixLoc = gl.getUniformLocation(drawGrid.program, 'm');
  const loc = gl.getAttribLocation(drawGrid.program, 'p');
  gl.uniformMatrix4fv(matrixLoc, false, new Float32Array(viewProj));

  gl.bindBuffer(gl.ARRAY_BUFFER, drawGrid.floorBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(floor), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  gl.uniform4f(colorLoc, 0.78, 0.80, 0.78, 0.32);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, floor.length/3);

  gl.bindBuffer(gl.ARRAY_BUFFER, drawGrid.buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(grid), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  gl.uniform4f(colorLoc, 0.18, 0.20, 0.21, 0.26);
  gl.drawArrays(gl.LINES, 0, grid.length/3);
  gl.useProgram(program);
}

function render(now) {
  resize();
  const dt = Math.min(0.05, (now - lastTime) / 1000);
  lastTime = now;
  const frames = activeTraj().frames || [];
  if (frames.length === 0) {
    requestAnimationFrame(render);
    return;
  }
  if (playing) {
    frameIndex = (frameIndex + dt * 20) % frames.length;
    frameSlider.value = String(Math.floor(frameIndex));
  }
  if (!Number.isFinite(frameIndex)) frameIndex = 0;
  let safeIndex = Math.floor(frameIndex);
  safeIndex = ((safeIndex % frames.length) + frames.length) % frames.length;
  const f = frames[safeIndex] || frames[0];
  const targetBase = followBox.checked ? [f.p[0], f.p[1], Math.max(0.18, f.p[2])] : [0,0,0.35];
  const target = [targetBase[0] + pan[0], targetBase[1] + pan[1], targetBase[2] + pan[2]];
  const eye = [
    target[0] + radius * Math.sin(yaw) * Math.cos(pitch),
    target[1] - radius * Math.cos(yaw) * Math.cos(pitch),
    target[2] + radius * Math.sin(pitch) + 0.12
  ];
  const proj = perspective(38 * Math.PI / 180, canvas.width / canvas.height, 0.01, 50);
  const view = lookAt(eye, target);
  const viewProj = mul(proj, view);

  gl.clearColor(0.94, 0.95, 0.94, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  drawGrid(viewProj, target);

  gl.useProgram(program);
  bindObjectAttribs();
  gl.uniformMatrix4fv(U.viewProj, false, new Float32Array(viewProj));
  gl.uniform3fv(U.translation, new Float32Array(f.p));
  gl.uniform4fv(U.quat, new Float32Array([f.q[1], f.q[2], f.q[3], f.q[0]]));
  gl.uniform3fv(U.centroid, new Float32Array(DATA.centroid));
  gl.uniform1f(U.pointSize, Number(sizeSlider.value));
  gl.drawArrays(gl.POINTS, 0, DATA.count);

  frameStat.textContent = `${safeIndex} / ${frames.length - 1}`;
  sizeStat.textContent = `${Number(sizeSlider.value).toFixed(1)}`;
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    points = load_gaussian_points(
        args.stage1_ply.resolve(),
        foreground_threshold=float(args.foreground_threshold),
        opacity_threshold=float(args.opacity_threshold),
        max_points=int(args.max_points),
    )
    trajectories = [normalize_trajectory(read_json(args.trajectory.resolve()), label="GT pose")]
    if args.fit_trajectory is not None:
        trajectories.append(normalize_trajectory(read_json(args.fit_trajectory.resolve()), label="optimized physics"))

    data = {
        **points,
        "centroid": parse_vec3(args.stage1_centroid),
        "trajectories": trajectories,
        "stage1_ply": str(args.stage1_ply.resolve()),
    }
    html = HTML_TEMPLATE.replace("__TITLE__", args.title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_html": str(args.output_html.resolve()),
                "stage1_ply": str(args.stage1_ply.resolve()),
                "gaussians": int(points["count"]),
                "trajectories": [traj["label"] for traj in trajectories],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

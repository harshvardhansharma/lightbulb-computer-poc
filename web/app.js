// Lightbulb Computer - projector app.
// Camera (phone via Continuity Camera / DroidCam) -> hand tracking -> homography
// -> projected cursor + widgets. Voice -> Claude -> widget actions.

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { solveHomography, applyH, invertH } from "./homography.js";

const $ = (s) => document.querySelector(s);
const video = $("#video"), surface = $("#surface"), overlay = $("#overlay");
const octx = overlay.getContext("2d");
const store = {
  get(k, d) { try { const v = localStorage.getItem("lb." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lb." + k, JSON.stringify(v)); } catch {} },
};

const state = {
  H: null,              // camera(normalized) -> projector(normalized)
  calibPts: [],         // camera-space points found during calibration (debug)
  landmarker: null,
  hands: [],            // latest raw landmark sets
  lastVideoTime: -1,
  cursor: null,         // {x, y} projector px, smoothed
  cursorCam: null,      // {x, y} camera normalized (raw fingertip)
  lastCursorAt: 0,
  pinching: false,
  grab: null,           // {widget, mode: 'move'|'pan', lastX, lastY, offX, offY, map}
  dwell: null,          // {el, since, x, y, fired}
  highlights: [],       // {x, y, r, label, until}
  thinking: false,
  wakeWord: store.get("wakeWord", true),
  debug: false,
};

const status = (t) => ($("#hud-status").textContent = t);
const voiceHud = (t) => ($("#hud-voice").textContent = t || "");

// ---------------------------------------------------------------- overlay sizing
function resize() {
  const dpr = devicePixelRatio || 1;
  overlay.width = innerWidth * dpr; overlay.height = innerHeight * dpr;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener("resize", resize); resize();

// ---------------------------------------------------------------- camera
async function listCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("#camera-select");
  sel.innerHTML = "";
  for (const d of devs) sel.append(new Option(d.label || d.deviceId.slice(0, 8), d.deviceId));
  return devs;
}

async function startCamera(deviceId) {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  const id = stream.getVideoTracks()[0].getSettings().deviceId;
  state.cameraId = id;
  $("#camera-select").value = id;
  state.H = store.get("H:" + id, null);
  state.calibPts = store.get("calibPts:" + id, []);
  if (state.H && checkFit(state.H)) state.H = null; // discard a stored bad calibration
  updateRoi();
  status(`camera: ${stream.getVideoTracks()[0].label} · ${state.H ? "calibrated" : "NOT calibrated - press C"}`);
}

async function initCamera() {
  // Ask once so labels are visible, then prefer the phone camera.
  const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
  tmp.getTracks().forEach((t) => t.stop());
  const devs = await listCameras();
  const saved = store.get("camera", null);
  const phone = devs.find((d) => /iphone|continuity|droidcam|camo|epoccam|ivcam/i.test(d.label));
  const pick = devs.find((d) => d.deviceId === saved) || phone || devs[0];
  await startCamera(pick?.deviceId);
  // Only an explicit pick is remembered; otherwise the phone camera wins.
  $("#camera-select").onchange = (e) => { store.set("camera", e.target.value); startCamera(e.target.value); };
  // Continuity Camera often connects after the page loads: switch to it when it appears.
  navigator.mediaDevices.ondevicechange = async () => {
    const devs = await listCameras();
    const phone = devs.find((d) => /iphone|continuity|droidcam|camo|epoccam|ivcam/i.test(d.label));
    if (phone && !store.get("camera", null) && phone.deviceId !== state.cameraId) startCamera(phone.deviceId);
    else $("#camera-select").value = state.cameraId;
  };
}

// Grayscale snapshot of the camera at low res, for calibration blob finding.
const snapCanvas = document.createElement("canvas");
const snapCtx = snapCanvas.getContext("2d", { willReadFrequently: true });
function grayFrame(w = 320) {
  const h = Math.round((w * video.videoHeight) / video.videoWidth);
  snapCanvas.width = w; snapCanvas.height = h;
  snapCtx.drawImage(video, 0, 0, w, h);
  const d = snapCtx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  return { g, w, h };
}
async function avgFrames(n, gap = 60, width = 320) {
  let acc = null, w, h;
  for (let k = 0; k < n; k++) {
    const f = grayFrame(width); w = f.w; h = f.h;
    if (!acc) acc = new Float32Array(f.g.length);
    for (let i = 0; i < acc.length; i++) acc[i] += f.g[i] / n;
    await sleep(gap);
  }
  return { g: acc, w, h };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- calibration
// Flash white dots at known projector positions, find them in the camera by
// differencing against an all-black frame, then fit a homography.
async function calibrate() {
  if (!video.videoWidth) return status("no camera");
  const calib = $("#calib"), dot = $("#calib-dot"), msg = $("#calib-msg");
  // Big squares: they must be clearly visible to a camera that may be far away, in daylight.
  const size = Math.round(Math.min(innerWidth, innerHeight) * 0.16);
  Object.assign(dot.style, { width: size + "px", height: size + "px", margin: `${-size / 2}px 0 0 ${-size / 2}px`, display: "none" });
  calib.hidden = false;
  msg.textContent = "Calibrating… step out of view and keep still";
  await sleep(2000);
  msg.textContent = "";

  const grid = [0.12, 0.5, 0.88];
  const src = [], dst = [];
  state.calibLog = [];
  for (const py of grid) for (const px of grid) {
    dot.style.left = px * innerWidth + "px"; dot.style.top = py * innerHeight + "px";
    // Difference each "on" frame against an "off" frame taken just before it,
    // so daylight drift, auto-exposure and people moving don't pollute it.
    let acc = null, w, h;
    for (let rep = 0; rep < 2; rep++) {
      dot.style.display = "none"; await sleep(450);
      const off = await avgFrames(2, 40, 640);
      dot.style.display = "block"; await sleep(650);
      const on = await avgFrames(2, 40, 640);
      w = on.w; h = on.h;
      if (!acc) acc = new Float32Array(on.g.length);
      for (let i = 0; i < acc.length; i++) acc[i] += Math.max(0, on.g[i] - off.g[i]) / 2;
    }
    dot.style.display = "none";
    const blob = findBlob(acc, w, h);
    state.calibLog.push({ proj: [px, py], ...blob });
    if (blob.cam) { src.push(blob.cam); dst.push([px, py]); }
  }

  const fit = fitRobust(src, dst);
  const reason = !fit ? (src.length < 4 ? `only ${src.length}/9 squares seen` : "detected squares don't agree with each other") : checkFit(fit.H);
  if (reason) {
    capture("calibration-failed").catch(() => {});
    msg.textContent = `Calibration failed: ${reason}. Make the projection fill more of the camera view (press D), reduce daylight, keep people out of frame, then press C.`;
    await sleep(6000); calib.hidden = true;
    return;
  }
  const { H, used, err } = fit;
  const id = state.cameraId;
  state.H = H; state.calibPts = used.map((i) => src[i]);
  store.set("H:" + id, H); store.set("calibPts:" + id, state.calibPts);
  updateRoi();
  msg.textContent = `Calibrated with ${used.length}/9 points · avg error ${err.toFixed(1)}px`;
  state.calibErr = err;
  capture("calibration").catch(() => {});
  await sleep(1500);
  calib.hidden = true;
  status(`calibrated (${used.length}/9 pts, ${err.toFixed(1)}px)`);
}

// Brightest compact blob in a difference image.
function findBlob(diff, w, h) {
  let max = 0;
  for (let i = 0; i < diff.length; i++) if (diff[i] > max) max = diff[i];
  const out = { maxDiff: +max.toFixed(1), cam: null };
  if (max < 20) return { ...out, why: "too faint" };
  const thr = max * 0.5;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sw = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = diff[y * w + x] - thr;
    if (d > 0) { sx += x * d; sy += y * d; sxx += x * x * d; syy += y * y * d; sw += d; }
  }
  const cx = sx / sw, cy = sy / sw;
  const spread = Math.sqrt(Math.max(0, sxx / sw - cx * cx) + Math.max(0, syy / sw - cy * cy)) / w;
  out.spread = +spread.toFixed(4);
  if (spread > 0.06) return { ...out, why: "not compact (noise or motion)" };
  return { ...out, cam: [(cx + 0.5) / w, (cy + 0.5) / h] };
}

// Robust fit: try every 4-point subset (only 126 for 9 points), keep the model
// most points agree with (within 25 px), then refit on those inliers.
function fitRobust(src, dst) {
  if (src.length < 4) return null;
  const errOf = (H, i) => {
    const [u, v] = applyH(H, src[i][0], src[i][1]);
    return Math.hypot((u - dst[i][0]) * innerWidth, (v - dst[i][1]) * innerHeight);
  };
  const idx = src.map((_, i) => i);
  let bestInliers = [];
  const n = src.length;
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) for (let c = b + 1; c < n; c++) for (let d = c + 1; d < n; d++) {
    const sub = [a, b, c, d];
    const H = solveHomography(sub.map((i) => src[i]), sub.map((i) => dst[i]));
    if (!H) continue;
    const inl = idx.filter((i) => errOf(H, i) < 25);
    if (inl.length > bestInliers.length) bestInliers = inl;
  }
  if (bestInliers.length < 5) return null; // 4 points always fit exactly; demand agreement
  const H = solveHomography(bestInliers.map((i) => src[i]), bestInliers.map((i) => dst[i]));
  if (!H) return null;
  const err = bestInliers.reduce((s, i) => s + errOf(H, i), 0) / bestInliers.length;
  return { H, used: bestInliers, err };
}

// The projection must look like a sane, convex quad in the camera.
function checkFit(H) {
  const inv = invertH(H);
  if (!inv) return "degenerate fit";
  const q = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => applyH(inv, u, v));
  let area = 0, sign = 0;
  for (let i = 0; i < 4; i++) {
    const [a, b, c] = [q[i], q[(i + 1) % 4], q[(i + 2) % 4]];
    area += a[0] * b[1] - b[0] * a[1];
    const cross = Math.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]));
    if (sign && cross !== sign) return "projection outline is twisted";
    sign = cross;
  }
  if (Math.abs(area / 2) < 0.01) return "projection too small or collapsed in the camera view";
  return null;
}

// Hands are small when the camera is far away: detect on a zoomed crop around
// the projection (plus margin) instead of the whole frame.
function updateRoi() {
  const inv = state.H && invertH(state.H);
  if (!inv) { state.roi = null; return; }
  const q = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => applyH(inv, u, v));
  let x0 = Math.min(...q.map((p) => p[0])), x1 = Math.max(...q.map((p) => p[0]));
  let y0 = Math.min(...q.map((p) => p[1])), y1 = Math.max(...q.map((p) => p[1]));
  const m = 0.35 * Math.max(x1 - x0, y1 - y0);
  x0 = Math.max(0, x0 - m); x1 = Math.min(1, x1 + m); y0 = Math.max(0, y0 - m); y1 = Math.min(1, y1 + m);
  state.roi = (x1 - x0) * (y1 - y0) > 0.6 ? null : { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// camera-normalized -> projector px
function camToScreen(x, y) {
  if (!state.H) return [x * innerWidth, y * innerHeight]; // uncalibrated: assume aligned
  const [u, v] = applyH(state.H, x, y);
  return [u * innerWidth, v * innerHeight];
}

// ---------------------------------------------------------------- hands
async function initHands() {
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  state.landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "models/hand_landmarker.task", delegate: "GPU" },
    runningMode: "IMAGE", // crops change every frame, so no temporal tracking inside MediaPipe
    numHands: 2,
    minHandDetectionConfidence: 0.35,
    minTrackingConfidence: 0.5,
  });
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Far-away hands are too small for the palm detector on the full frame, so:
//  - search: scan zoomed tiles (plus the full frame) one per video frame until a hand shows up;
//  - track: then keep detecting on a tight zoomed crop centered on the hand.
const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d");
let searchIdx = 0, missed = 0;

function detectIn(r) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const sw = r.w * vw, sh = r.h * vh;
  cropCanvas.width = Math.round(Math.min(768, Math.max(sw, sw * 2)));
  cropCanvas.height = Math.round((cropCanvas.width * sh) / sw);
  cropCtx.drawImage(video, r.x * vw, r.y * vh, sw, sh, 0, 0, cropCanvas.width, cropCanvas.height);
  const hands = state.landmarker.detect(cropCanvas).landmarks || [];
  return hands.map((h) => h.map((p) => ({ x: r.x + p.x * r.w, y: r.y + p.y * r.h, z: p.z })));
}

function searchTiles() {
  // Tiles over the projection area + margin (or the whole frame if uncalibrated).
  const area = state.roi || { x: 0, y: 0, w: 1, h: 1 };
  const tw = Math.min(area.w, 0.4), th = Math.min(area.h, 0.5);
  const tiles = [{ x: 0, y: 0, w: 1, h: 1 }];
  const nx = Math.max(1, Math.ceil((area.w - tw) / (tw / 2)) + 1), ny = Math.max(1, Math.ceil((area.h - th) / (th / 2)) + 1);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    tiles.push({
      x: area.x + (nx === 1 ? 0 : ((area.w - tw) * i) / (nx - 1)),
      y: area.y + (ny === 1 ? 0 : ((area.h - th) * j) / (ny - 1)),
      w: tw, h: th,
    });
  }
  return tiles;
}

function detectHands() {
  const aspect = video.videoWidth / video.videoHeight;
  let hands;
  if (state.track) {
    const t = state.track;
    const h = clamp(t.size * 4, 0.3, 1), w = Math.min(1, h / aspect * 1.4);
    state.crop = { x: clamp(t.x - w / 2, 0, 1 - w), y: clamp(t.y - h / 2, 0, 1 - h), w, h };
    hands = detectIn(state.crop);
    if (!hands.length && ++missed > 4) { state.track = null; missed = 0; }
  } else {
    const tiles = searchTiles();
    state.crop = tiles[searchIdx++ % tiles.length];
    hands = detectIn(state.crop);
  }
  if (hands.length) {
    missed = 0;
    const h = hands[0];
    const xs = h.map((p) => p.x), ys = h.map((p) => p.y);
    state.track = {
      x: (Math.min(...xs) + Math.max(...xs)) / 2,
      y: (Math.min(...ys) + Math.max(...ys)) / 2,
      size: Math.max(Math.max(...xs) - Math.min(...xs), (Math.max(...ys) - Math.min(...ys))),
    };
  }
  return hands;
}

function processHands(now) {
  // Primary hand = the first one pointing or pinching.
  const hands = state.hands;
  let lm = null;
  for (const h of hands) {
    const size = dist(h[0], h[9]);
    // Far from the camera the landmarks bunch up and thumb/index always look
    // touching, so below this palm size (camera px) the hand is a pointer only.
    const palmPx = Math.hypot((h[0].x - h[9].x) * video.videoWidth, (h[0].y - h[9].y) * video.videoHeight);
    const canPinch = palmPx >= 40;
    const pointing = !canPinch || dist(h[8], h[0]) > dist(h[6], h[0]) * 1.1;
    const pinchRatio = canPinch ? dist(h[4], h[8]) / size : 1;
    if (pointing || pinchRatio < 0.45) { lm = h; lm.pinchRatio = pinchRatio; break; }
  }
  if (!lm) {
    if (now - state.lastCursorAt > 350) { state.cursor = null; state.cursorCam = null; releaseGrab(); }
    return;
  }
  // Hysteresis so pinch doesn't flicker.
  const wasPinching = state.pinching;
  state.pinching = wasPinching ? lm.pinchRatio < 0.5 : lm.pinchRatio < 0.3;

  // While pinching use the thumb/index midpoint (stable); else the index tip.
  const p = state.pinching
    ? { x: (lm[4].x + lm[8].x) / 2, y: (lm[4].y + lm[8].y) / 2 }
    : lm[8];
  const [sx, sy] = camToScreen(p.x, p.y);
  state.cursorCam = { x: lm[8].x, y: lm[8].y };
  if (!state.cursor) state.cursor = { x: sx, y: sy };
  else {
    // Speed-adaptive smoothing: steady when still, responsive when moving.
    const speed = Math.hypot(sx - state.cursor.x, sy - state.cursor.y);
    const a = Math.min(0.85, 0.2 + speed / 120);
    state.cursor.x += (sx - state.cursor.x) * a;
    state.cursor.y += (sy - state.cursor.y) * a;
  }
  state.lastCursorAt = now;

  if (state.pinching && !wasPinching) startGrab(state.cursor.x, state.cursor.y);
  else if (state.pinching) moveGrab(state.cursor.x, state.cursor.y);
  else if (wasPinching) releaseGrab();

  updateHover(now);
}

// ---------------------------------------------------------------- pinch grab
function startGrab(x, y) {
  const el = document.elementFromPoint(x, y);
  const widget = el?.closest(".widget");
  if (!widget) return;
  const mapEl = el.closest(".leaflet");
  const r = widget.getBoundingClientRect();
  state.grab = mapEl
    ? { widget, mode: "pan", map: widget._map, lastX: x, lastY: y }
    : { widget, mode: "move", offX: x - r.left, offY: y - r.top };
  widget.classList.add("grabbed");
  surface.append(widget); // bring to front
}
function moveGrab(x, y) {
  const g = state.grab;
  if (!g) return;
  if (g.mode === "move") {
    g.widget.style.left = x - g.offX + "px";
    g.widget.style.top = y - g.offY + "px";
  } else {
    g.map.panBy([g.lastX - x, g.lastY - y], { animate: false });
    g.lastX = x; g.lastY = y;
  }
}
function releaseGrab() {
  if (state.grab) state.grab.widget.classList.remove("grabbed");
  state.grab = null;
}

// ---------------------------------------------------------------- dwell to click
const DWELL_MS = 850;
function updateHover(now) {
  document.querySelectorAll(".hover").forEach((e) => e.classList.remove("hover"));
  if (!state.cursor || state.pinching) { state.dwell = null; return; }
  const el = document.elementFromPoint(state.cursor.x, state.cursor.y);
  el?.closest(".widget")?.classList.add("hover");
  const btn = el?.closest(".btn");
  if (!btn) { state.dwell = null; return; }
  btn.classList.add("hover");
  const d = state.dwell;
  if (!d || d.el !== btn || Math.hypot(d.x - state.cursor.x, d.y - state.cursor.y) > 45) {
    state.dwell = { el: btn, since: now, x: state.cursor.x, y: state.cursor.y, fired: false };
  } else if (!d.fired && now - d.since > DWELL_MS) {
    d.fired = true;
    blip(880);
    btn.click();
  }
}

// ---------------------------------------------------------------- render loop
function loop() {
  const now = performance.now();
  if (state.landmarker && video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    state.hands = detectHands();
    processHands(now);
  }
  draw(now);
  if (state.debug) drawDebug();
  requestAnimationFrame(loop);
}

function draw(now) {
  octx.clearRect(0, 0, innerWidth, innerHeight);

  // Highlights projected back onto real objects.
  state.highlights = state.highlights.filter((h) => h.until > now);
  for (const h of state.highlights) {
    const pulse = 1 + 0.12 * Math.sin(now / 180);
    octx.lineWidth = 6; octx.strokeStyle = "#ffd35c";
    octx.beginPath(); octx.arc(h.x, h.y, h.r * pulse, 0, Math.PI * 2); octx.stroke();
    octx.lineWidth = 2;
    octx.beginPath(); octx.arc(h.x, h.y, h.r * pulse + 14, 0, Math.PI * 2); octx.stroke();
    if (h.label) {
      octx.font = "600 24px -apple-system, sans-serif";
      octx.fillStyle = "#ffd35c"; octx.textAlign = "center";
      octx.fillText(h.label, h.x, h.y - h.r - 26);
    }
  }

  // Cursor - projected onto the fingertip itself.
  const c = state.cursor;
  if (c) {
    octx.lineWidth = 4;
    octx.strokeStyle = state.pinching ? "#ffd35c" : "#ffffff";
    octx.beginPath(); octx.arc(c.x, c.y, state.pinching ? 14 : 22, 0, Math.PI * 2); octx.stroke();
    const d = state.dwell;
    if (d && !d.fired) {
      const t = Math.min(1, (now - d.since) / DWELL_MS);
      octx.lineWidth = 7; octx.strokeStyle = "#ffd35c";
      octx.beginPath(); octx.arc(c.x, c.y, 32, -Math.PI / 2, -Math.PI / 2 + t * Math.PI * 2); octx.stroke();
    }
  }

  // Thinking indicator.
  if (state.thinking) {
    const x = c ? c.x : innerWidth / 2, y = c ? c.y - 60 : innerHeight / 2;
    for (let i = 0; i < 3; i++) {
      const a = 0.3 + 0.7 * Math.max(0, Math.sin(now / 200 - i * 0.8));
      octx.fillStyle = `rgba(255,211,92,${a})`;
      octx.beginPath(); octx.arc(x - 24 + i * 24, y, 7, 0, Math.PI * 2); octx.fill();
    }
  }
}

const dbg = $("#debug-canvas"), dctx = dbg.getContext("2d");
function drawDebug() {
  if (!video.videoWidth) return;
  dbg.width = video.videoWidth / 2; dbg.height = video.videoHeight / 2;
  drawAnnotated(dctx, dbg.width, dbg.height);
}

// Camera frame + what the app thinks: projection outline (green), calibration
// dots (cyan), hand landmarks (magenta), fingertip (yellow), and where the
// projected cursor should land according to the calibration (white cross).
function drawAnnotated(dctx, W, H) {
  dctx.drawImage(video, 0, 0, W, H);
  // Projection area in camera space.
  const inv = state.H && invertH(state.H);
  if (inv) {
    dctx.strokeStyle = "#0f0"; dctx.lineWidth = 2; dctx.beginPath();
    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]].forEach(([u, v], i) => {
      const [x, y] = applyH(inv, u, v);
      i ? dctx.lineTo(x * W, y * H) : dctx.moveTo(x * W, y * H);
    });
    dctx.stroke();
  }
  if (state.crop) {
    dctx.strokeStyle = state.track ? "#0f0" : "#888"; dctx.lineWidth = 1;
    dctx.strokeRect(state.crop.x * W, state.crop.y * H, state.crop.w * W, state.crop.h * H);
  }
  if (state.roi) {
    dctx.strokeStyle = "#f90"; dctx.lineWidth = 2; dctx.setLineDash([6, 4]);
    dctx.strokeRect(state.roi.x * W, state.roi.y * H, state.roi.w * W, state.roi.h * H);
    dctx.setLineDash([]);
  }
  dctx.fillStyle = "#0ff";
  for (const [x, y] of state.calibPts) dctx.fillRect(x * W - 3, y * H - 3, 6, 6);
  for (const h of state.hands) {
    dctx.fillStyle = "#f0f";
    for (const p of h) dctx.fillRect(p.x * W - 2, p.y * H - 2, 4, 4);
    dctx.fillStyle = "#ff0"; dctx.beginPath(); dctx.arc(h[8].x * W, h[8].y * H, 5, 0, 7); dctx.fill();
  }
  if (inv && state.cursor) {
    const [x, y] = applyH(inv, state.cursor.x / innerWidth, state.cursor.y / innerHeight);
    dctx.strokeStyle = "#fff"; dctx.lineWidth = 2; dctx.beginPath();
    dctx.moveTo(x * W - 10, y * H); dctx.lineTo(x * W + 10, y * H);
    dctx.moveTo(x * W, y * H - 10); dctx.lineTo(x * W, y * H + 10); dctx.stroke();
  }
}

// ---------------------------------------------------------------- debug capture
// S = one annotated frame, R = burst of 12 over ~6 s. Saved by the server to ./captures.
async function capture(label = "snap") {
  if (!video.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  const raw = c.getContext("2d");
  raw.drawImage(video, 0, 0);
  const rawJpeg = c.toDataURL("image/jpeg", 0.85).split(",")[1];
  drawAnnotated(raw, c.width, c.height);
  const meta = {
    label, t: new Date().toISOString(), ua: navigator.userAgent,
    video: [video.videoWidth, video.videoHeight], screen: [innerWidth, innerHeight],
    camera: video.srcObject?.getVideoTracks()[0]?.label,
    H: state.H, calibPts: state.calibPts, calibLog: state.calibLog, roi: state.roi,
    cursor: state.cursor, pinching: state.pinching,
    hands: state.hands.map((h) => h.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)])),
  };
  const res = await fetch("/api/debug", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, raw: rawJpeg, annotated: c.toDataURL("image/jpeg", 0.85).split(",")[1], meta }),
  });
  const { saved } = await res.json();
  voiceHud(`📸 saved ${saved}`);
}
async function captureBurst() {
  // Countdown so you can walk into position after pressing R.
  for (let n = 3; n > 0; n--) { voiceHud(`📸 burst in ${n}…`); blip(500); await sleep(1000); }
  for (let i = 0; i < 12; i++) { blip(1200, 0.03); await capture(`burst-${i}`); await sleep(500); }
}

// ---------------------------------------------------------------- widgets
let placeIdx = 0;
function placeFor(w, h) {
  // Near the finger if the user pointed recently, else the next free slot.
  if (state.cursor && performance.now() - state.lastCursorAt < 2000) {
    return [clamp(state.cursor.x - w / 2, 10, innerWidth - w - 10), clamp(state.cursor.y + 40, 10, innerHeight - h - 10)];
  }
  const slots = [[0.04, 0.06], [0.37, 0.06], [0.68, 0.06], [0.04, 0.52], [0.37, 0.52], [0.68, 0.52]];
  const rects = [...surface.children].map((e) => e.getBoundingClientRect());
  for (let k = 0; k < slots.length; k++) {
    const [fx, fy] = slots[(placeIdx + k) % slots.length];
    const x = fx * innerWidth, y = fy * innerHeight;
    if (!rects.some((r) => x < r.right && x + w > r.left && y < r.bottom && y + h > r.top)) { placeIdx += k + 1; return [x, y]; }
  }
  const [fx, fy] = slots[placeIdx++ % slots.length];
  return [fx * innerWidth + 20, fy * innerHeight + 20];
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function makeWidget(kind, title, html, summary) {
  const el = document.createElement("div");
  el.className = `widget ${kind}`;
  el.innerHTML = `<div class="widget-head"><span class="title">${esc(title)}</span><span class="btn small close">✕</span></div>${html}`;
  el.dataset.summary = summary || `${kind}${title ? ": " + title : ""}`;
  el.querySelector(".close").onclick = () => removeWidget(el);
  surface.append(el);
  const r = el.getBoundingClientRect();
  const [x, y] = placeFor(r.width, r.height);
  el.style.left = x + "px"; el.style.top = y + "px";
  enableMouseDrag(el);
  return el;
}
function removeWidget(el) {
  el._cleanup?.();
  el.remove();
}
function widgetUnderPointer() {
  if (!state.cursor) return null;
  return document.elementFromPoint(state.cursor.x, state.cursor.y)?.closest(".widget");
}

function enableMouseDrag(el) {
  el.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".btn, .leaflet")) return;
    const r = el.getBoundingClientRect(), ox = e.clientX - r.left, oy = e.clientY - r.top;
    surface.append(el);
    const move = (ev) => { el.style.left = ev.clientX - ox + "px"; el.style.top = ev.clientY - oy + "px"; };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  });
}

const tickers = new Set();
setInterval(() => tickers.forEach((f) => f()), 250);

const W = {
  clock() {
    const el = makeWidget("clock", "", `<div class="time"></div><div class="date"></div>`, "clock");
    const t = () => {
      const d = new Date();
      el.querySelector(".time").textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      el.querySelector(".date").textContent = d.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
    };
    t(); tickers.add(t); el._cleanup = () => tickers.delete(t);
  },

  timer({ label, seconds }) {
    const el = makeWidget("timer", label || "Timer", `<div class="remain"></div>
      <div class="row"><span class="btn pause">Pause</span><span class="btn plus">+1 min</span></div>`);
    let end = Date.now() + seconds * 1000, paused = null, rang = false;
    const fmt = (ms) => {
      const s = Math.max(0, Math.ceil(ms / 1000));
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = String(s % 60).padStart(2, "0");
      return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
    };
    const t = () => {
      const left = paused ?? end - Date.now();
      el.querySelector(".remain").textContent = fmt(left);
      el.dataset.summary = `timer "${label}" ${fmt(left)} left`;
      if (left <= 0 && !rang) { rang = true; el.classList.add("alarm"); alarm(el); speak(`${label || "Timer"} is done`); }
    };
    el.querySelector(".pause").onclick = (e) => {
      if (paused == null) { paused = end - Date.now(); e.target.textContent = "Resume"; }
      else { end = Date.now() + paused; paused = null; e.target.textContent = "Pause"; }
    };
    el.querySelector(".plus").onclick = () => {
      if (paused != null) paused += 60000; else end = Math.max(end, Date.now()) + 60000;
      rang = false; el.classList.remove("alarm");
    };
    t(); tickers.add(t); el._cleanup = () => tickers.delete(t);
  },

  answer_card({ title, body }) {
    makeWidget("answer", title || "Answer", `<div class="body">${esc(body)}</div>`, `answer card "${title}"`);
  },

  note({ text }) {
    makeWidget("note", "Note", `<div class="body">${esc(text)}</div>`, `note "${text.slice(0, 40)}"`);
  },

  recipe({ title, ingredients, steps }) {
    const el = makeWidget("recipe", title, `
      <ul>${ingredients.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>
      <div class="step-n"></div><div class="step"></div>
      <div class="row"><span class="btn prev">◀</span><span class="btn next">Next ▶</span><span class="btn tstart" hidden></span></div>`);
    let i = 0;
    const show = () => {
      const s = steps[i];
      el.querySelector(".step-n").textContent = `Step ${i + 1} of ${steps.length}`;
      el.querySelector(".step").textContent = s.text;
      const tb = el.querySelector(".tstart");
      tb.hidden = !s.timer_seconds;
      tb.textContent = `⏱ ${Math.round(s.timer_seconds / 60) || 1} min`;
      el.dataset.summary = `recipe "${title}" on step ${i + 1}/${steps.length}`;
      el.querySelector("ul").style.display = i === 0 ? "" : "none";
    };
    el.querySelector(".prev").onclick = () => { i = Math.max(0, i - 1); show(); };
    el.querySelector(".next").onclick = () => { i = Math.min(steps.length - 1, i + 1); show(); };
    el.querySelector(".tstart").onclick = () => W.timer({ label: `Step ${i + 1}`, seconds: steps[i].timer_seconds });
    show();
  },

  async map({ place, zoom }) {
    const el = makeWidget("map", place, `<div class="leaflet"></div>
      <div class="row"><span class="btn zin">＋</span><span class="btn zout">－</span></div>`, `map of ${place}`);
    const map = L.map(el.querySelector(".leaflet"), { zoomControl: false, attributionControl: true }).setView([20, 0], 2);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(map);
    el._map = map;
    el.querySelector(".zin").onclick = () => map.zoomIn();
    el.querySelector(".zout").onclick = () => map.zoomOut();
    try {
      const r = await (await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(place)}`)).json();
      if (r[0]) map.setView([+r[0].lat, +r[0].lon], zoom || 12);
    } catch (e) { console.warn("geocode failed", e); }
  },

  async weather({ place }) {
    const el = makeWidget("weather", place, `<div class="temp">…</div><div class="desc"></div><div class="days"></div>`, `weather for ${place}`);
    try {
      const g = await (await fetch(`https://geocoding-api.open-meteo.com/v1/search?count=1&name=${encodeURIComponent(place.split(",")[0])}`)).json();
      const loc = g.results?.[0];
      if (!loc) throw new Error("place not found");
      const f = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current=temperature_2m,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=4`)).json();
      el.querySelector(".title").textContent = `${loc.name}, ${loc.country_code}`;
      el.querySelector(".temp").textContent = `${Math.round(f.current.temperature_2m)}°`;
      el.querySelector(".desc").textContent = wmo(f.current.weather_code);
      el.querySelector(".days").innerHTML = f.daily.time.slice(1).map((d, i) =>
        `<div>${new Date(d).toLocaleDateString([], { weekday: "short" })}<br>${Math.round(f.daily.temperature_2m_max[i + 1])}° / ${Math.round(f.daily.temperature_2m_min[i + 1])}°</div>`).join("");
    } catch (e) {
      el.querySelector(".temp").textContent = "—";
      el.querySelector(".desc").textContent = String(e.message || e);
    }
  },

  highlight({ items }) {
    const now = performance.now();
    for (const it of items) {
      const [x, y] = camToScreen(it.x, it.y);
      const [x2] = camToScreen(it.x + (it.radius || 0.03), it.y);
      state.highlights.push({ x, y, r: clamp(Math.abs(x2 - x), 24, 220), label: it.label, until: now + 15000 });
    }
  },

  clear({ target }) {
    if (target === "all") [...surface.children].forEach(removeWidget), (state.highlights = []);
    else { const w = widgetUnderPointer() || surface.lastElementChild; if (w) removeWidget(w); }
  },

  say({ text }) { speak(text); },
};

function wmo(c) {
  return c === 0 ? "Clear" : c <= 3 ? "Partly cloudy" : c <= 48 ? "Fog" : c <= 57 ? "Drizzle" : c <= 67 ? "Rain"
    : c <= 77 ? "Snow" : c <= 82 ? "Showers" : c <= 86 ? "Snow showers" : "Thunderstorm";
}

// ---------------------------------------------------------------- sound + speech
let actx;
function blip(freq = 660, dur = 0.08) {
  try {
    actx ||= new AudioContext();
    const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = freq; g.gain.value = 0.08;
    o.connect(g).connect(actx.destination); o.start(); o.stop(actx.currentTime + dur);
  } catch {}
}
function alarm(el) {
  let n = 0;
  const id = setInterval(() => { if (!el.isConnected || !el.classList.contains("alarm") || n++ > 20) return clearInterval(id); blip(1000, 0.15); }, 500);
}
function speak(text) {
  if (!text) return;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

// ---------------------------------------------------------------- commands
async function handleCommand(text) {
  text = text.trim();
  if (!text) return;
  $("#help").hidden = true;
  voiceHud(`“${text}”`);
  const t = text.toLowerCase().replace(/[.!?]$/, "");

  // Instant local paths for the most common things.
  if (/^(re)?calibrat/.test(t)) return calibrate();
  if (/^(clear|remove|close) (all|everything)/.test(t)) return W.clear({ target: "all" });
  if (/^(close|remove|dismiss|hide) (this|that|it)$/.test(t)) return W.clear({ target: "pointed" });
  if (/^(show )?(the )?(clock|time)$/.test(t)) return W.clock();
  const tm = t.match(/(\d+)\s*(second|sec|minute|min|hour)s?\b.*timer|timer (?:for )?(\d+)\s*(second|sec|minute|min|hour)/);
  if (tm && t.split(" ").length <= 7) {
    const n = +(tm[1] || tm[3]), unit = tm[2] || tm[4];
    return W.timer({ label: "Timer", seconds: n * (unit.startsWith("h") ? 3600 : unit.startsWith("m") ? 60 : 1) });
  }

  // Everything else: ask Claude, with what the camera sees.
  state.thinking = true;
  try {
    const pointing = state.cursorCam && performance.now() - state.lastCursorAt < 1500 ? state.cursorCam : null;
    const res = await fetch("/api/assist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        image: video.videoWidth ? snapshotJpeg(pointing) : null,
        pointer: !!pointing,
        widgets: [...surface.children].map((w) => w.dataset.summary),
      }),
    });
    const data = await res.json();
    if (data.error) { W.answer_card({ title: "Error", body: data.error }); return; }
    for (const a of data.actions) {
      try { await W[a.name]?.(a.input); } catch (e) { console.error(a, e); }
    }
  } catch (e) {
    W.answer_card({ title: "Error", body: String(e) });
  } finally {
    state.thinking = false;
  }
}

function snapshotJpeg(pointer) {
  const c = document.createElement("canvas");
  const s = Math.min(1, 1280 / video.videoWidth);
  c.width = video.videoWidth * s; c.height = video.videoHeight * s;
  const x = c.getContext("2d");
  x.drawImage(video, 0, 0, c.width, c.height);
  if (pointer) {
    x.strokeStyle = "#ff00ff"; x.lineWidth = 5;
    x.beginPath(); x.arc(pointer.x * c.width, pointer.y * c.height, 28, 0, Math.PI * 2); x.stroke();
  }
  return c.toDataURL("image/jpeg", 0.8).split(",")[1];
}

// ---------------------------------------------------------------- voice
const WAKE = /\b(?:hey |ok |okay )?(?:light ?bulbs?|lite ?bulb|light bold|lightbulb computer)\b[,.!]?\s*/i;

// Voice events go to the server log too, so problems can be diagnosed remotely.
const PAGE_ID = (/Chrome\//.test(navigator.userAgent) ? "chrome" : /Safari\//.test(navigator.userAgent) ? "safari" : "browser")
  + "-" + Math.random().toString(36).slice(2, 6);
function vlog(...a) {
  console.log("[voice]", ...a);
  fetch("/api/log", { method: "POST", body: JSON.stringify({ t: new Date().toISOString(), msg: `${PAGE_ID} ${document.visibilityState}: ${a.join(" ")}` }) }).catch(() => {});
}

// Mic level meter (debug view only): Safari allows one mic capture per page, so a
// second getUserMedia stream would cut off the speech recognizer's audio.
let meter = null;
async function setMicMeter(on) {
  if (!on) {
    if (meter) { clearInterval(meter.timer); meter.stream.getTracks().forEach((t) => t.stop()); meter = null; }
    $("#hud-mic").style.width = "0";
    return;
  }
  if (meter) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    vlog("mic meter on:", stream.getAudioTracks()[0].label);
    actx ||= new AudioContext();
    const an = actx.createAnalyser(); an.fftSize = 512;
    actx.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize);
    const timer = setInterval(() => {
      if (actx.state === "suspended") actx.resume().catch(() => {});
      an.getByteTimeDomainData(buf);
      let peak = 0;
      for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
      $("#hud-mic").style.width = Math.round(Math.min(1, (peak / 128) * 3) * 60) + "px";
    }, 100);
    meter = { stream, timer };
  } catch (e) { vlog("mic meter failed:", e.name, e.message); }
}

// Same recognizer setup as the version that worked in Safari; only passive
// logging and a manual restart (V) on top - no watchdog, no second mic stream.
const voice = {};
function initVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { voiceHud("voice: unsupported in this browser - press Enter to type"); return; }
  const rec = new SR();
  rec.continuous = true; rec.interimResults = true; rec.lang = "en-US";
  let armedUntil = 0;
  rec.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i], txt = r[0].transcript.trim();
      const hasWake = WAKE.test(txt);
      if (!r.isFinal) {
        if (!state.wakeWord || hasWake || Date.now() < armedUntil) voiceHud("🎙 " + txt);
        continue;
      }
      vlog(`final: "${txt}"`);
      if (!state.wakeWord) { handleCommand(txt); continue; }
      if (hasWake) {
        const cmd = txt.split(WAKE).pop().trim();
        if (cmd) handleCommand(cmd);
        else { armedUntil = Date.now() + 6000; blip(660); voiceHud("🎙 listening…"); }
      } else if (Date.now() < armedUntil) { armedUntil = 0; handleCommand(txt); }
      else voiceHud("");
    }
  };
  rec.onstart = () => vlog("start");
  rec.onspeechstart = () => vlog("speechstart");
  rec.onerror = (e) => {
    vlog(`error: ${e.error} ${e.message || ""}`);
    if (e.error === "not-allowed") voiceHud("mic blocked - allow microphone for localhost, then press V");
  };
  rec.onend = () => { vlog("end"); setTimeout(() => { try { rec.start(); } catch {} }, 250); };
  rec.start();
  voice.restart = () => { vlog("manual restart"); try { rec.start(); } catch { try { rec.stop(); } catch {} } };
  voiceHud(state.wakeWord ? "say “Lightbulb, …”" : "listening (no wake word)");
}

// ---------------------------------------------------------------- keyboard
async function projectorFullscreen() {
  try {
    if ("getScreenDetails" in window) {
      const sd = await window.getScreenDetails();
      const ext = sd.screens.find((s) => !s.isPrimary) || sd.currentScreen;
      return await document.documentElement.requestFullscreen({ screen: ext });
    }
  } catch (e) { console.warn(e); }
  document.documentElement.requestFullscreen();
}

const cmdForm = $("#cmd"), cmdInput = $("#cmd-input");
cmdForm.onsubmit = (e) => { e.preventDefault(); const v = cmdInput.value; cmdInput.value = ""; cmdForm.hidden = true; handleCommand(v); };
addEventListener("keydown", (e) => {
  if (!cmdForm.hidden) { if (e.key === "Escape") cmdForm.hidden = true; return; }
  const k = e.key.toLowerCase();
  if (k === "/" || e.code === "Slash" || k === "enter") { e.preventDefault(); cmdForm.hidden = false; cmdInput.focus(); }
  else if (k === "p") projectorFullscreen();
  else if (k === "c") calibrate();
  else if (k === "d") { state.debug = !state.debug; $("#debug").hidden = !state.debug; setMicMeter(state.debug); document.body.classList.toggle("mouse", state.debug); }
  else if (k === "h") $("#help").hidden = !$("#help").hidden;
  else if (k === "m") document.body.classList.toggle("mouse");
  else if (k === "s") capture();
  else if (k === "v") voice.restart?.();
  else if (k === "r") captureBurst();
  else if (k === "w") { state.wakeWord = !state.wakeWord; store.set("wakeWord", state.wakeWord); voiceHud(state.wakeWord ? "wake word ON" : "wake word OFF - always listening"); }
  else if (k === "escape") { $("#help").hidden = true; }
});

// ---------------------------------------------------------------- boot
(async function boot() {
  $("#help").hidden = false;
  requestAnimationFrame(loop);
  status("waiting for camera permission…");
  initVoice();
  initHands().catch((e) => status("hand model failed: " + e.message));
  initCamera().catch((e) => status("camera error: " + e.message));
})();

// Handy from the DevTools console: lb.handleCommand("show me a map of Kyoto")
window.lb = { state, W, handleCommand, calibrate, capture, detectHands, processHands, updateRoi, findBlob, fitRobust, checkFit };

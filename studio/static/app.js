// photostyle Studio front end. No build step.
// The WebGL shader is an exact port of photostyle.render.GlobalRenderer:
//   per-channel piecewise-linear curves -> 3x3 matrix + bias -> vignette -> clamp.
// Every slider change re-renders on the GPU; Python is only asked to predict
// parameters, render full-resolution exports, and train styles.
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r;
};

// ------------------------------------------------------------------ state
const S = {
  photos: [], styles: [], id: null, img: null, style: null,
  params: null,              // server EditParams payload
  strength: 1,
  d: { warmth: 0, tint: 0, sat: 0, vig: 0 },   // user deltas (colour, vignette)
  knots: null,               // user curve override: [3][K] or null
  mode: "split", split: 0.5, holdBefore: false, ch: 0,
  undo: [], redo: [],
};

// ------------------------------------------------------------------ renderer maths (mirrors Python)
function layout(p) {
  const K = p.knots, curve = p.renderer === "shared" ? K - 2 : 3 * K;
  return { curve: [0, curve], dM: [curve, curve + 9], bias: [curve + 9, curve + 12], vig: [curve + 12, curve + 13] };
}
function modelGroups(p, strength) {
  const L = layout(p), th = p.theta.map((v) => v * strength), g = {};
  for (const k in L) g[k] = th.slice(L[k][0], L[k][1]);
  return g;
}
function knotsFromCurve(p, cp) {
  const K = p.knots, out = [];
  if (p.renderer === "shared") {
    const y = []; for (let i = 0; i < K; i++) y.push(i / (K - 1) + (i > 0 && i < K - 1 ? cp[i - 1] : 0));
    return [y, y.slice(), y.slice()];
  }
  for (let c = 0; c < 3; c++) {
    const base = c * K, y = [cp[base]];
    for (let j = 1; j < K; j++) y.push(y[j - 1] + Math.exp(cp[base + j]) / (K - 1));
    out.push(y);
  }
  return out;
}
function curveFromKnots(p, kn) {       // inverse of knotsFromCurve (per-channel monotone)
  const K = p.knots, cp = [];
  for (let c = 0; c < 3; c++) {
    cp.push(kn[c][0]);
    for (let j = 1; j < K; j++) cp.push(Math.log(Math.max(1e-4, (kn[c][j] - kn[c][j - 1]) * (K - 1))));
  }
  return cp;
}
function effective() {                 // effective renderer inputs for WebGL and export
  const p = S.params, g = modelGroups(p, S.strength);
  const knots = S.knots || knotsFromCurve(p, g.curve);
  const dM = g.dM.slice(), bias = g.bias.slice(), s = S.d.sat / 100 * 0.5;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) dM[i * 3 + j] += s * ((i === j ? 1 : 0) - 1 / 3);
  const w = S.d.warmth / 100, t = S.d.tint / 100;
  bias[0] += 0.05 * w + 0.03 * t; bias[1] += -0.03 * t; bias[2] += -0.05 * w + 0.03 * t;
  const vig = g.vig[0] + S.d.vig / 100;
  return { knots, dM, bias, vig, g };
}
function exportParams() {              // EditParams for the server (overrides are absolute)
  const p = S.params, e = effective(), ov = { dM: e.dM, bias: e.bias, vignette: [e.vig] };
  if (S.knots && p.renderer !== "shared") ov.curve = curveFromKnots(p, S.knots);
  return { style: p.style, renderer: p.renderer, knots: p.knots, theta: p.theta, strength: S.strength,
           overrides: ov, ood_score: p.ood_score, version: 1 };
}

// ------------------------------------------------------------------ WebGL
const VS = `#version 300 es
in vec2 pos; out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }`;
const FS = `#version 300 es
precision highp float;
uniform sampler2D img; uniform float knots[51]; uniform int K;
uniform mat3 M; uniform vec3 bias; uniform float vig; uniform float split; uniform int mode;
in vec2 uv; out vec4 o;
float curve(int ch, float x) {
  float t = clamp(x, 0.0, 1.0) * float(K - 1);
  int i = min(int(floor(t)), K - 2); float u = t - float(i);
  float a = knots[ch * K + i], b = knots[ch * K + i + 1]; return a + u * (b - a);
}
void main() {
  vec3 c = texture(img, vec2(uv.x, 1.0 - uv.y)).rgb;
  vec3 e = vec3(curve(0, c.r), curve(1, c.g), curve(2, c.b));
  e = M * e + bias;
  vec2 p = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  e *= clamp(1.0 - vig * dot(p, p) / 2.0, 0.0, 1.0);
  e = clamp(e, 0.0, 1.0);
  if (mode == 1 || (mode == 0 && uv.x < split)) e = c;
  o = vec4(e, 1.0);
}`;
function makeGL(canvas) {
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, premultipliedAlpha: false });
  const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  const pr = gl.createProgram();
  gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
  gl.useProgram(pr);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(pr, "pos"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture();
  const u = (n) => gl.getUniformLocation(pr, n);
  return { gl, tex, u, canvas };
}
function setImage(R, img) {
  const { gl, tex } = R;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
}
function draw(R, e, K, mode = 2, split = 0.5) {
  const { gl, u } = R;
  gl.viewport(0, 0, R.canvas.width, R.canvas.height);
  const flat = new Float32Array(51); e.knots.flat().forEach((v, i) => (flat[i] = v));
  gl.uniform1fv(u("knots"), flat); gl.uniform1i(u("K"), K);
  const M = [1 + e.dM[0], e.dM[1], e.dM[2], e.dM[3], 1 + e.dM[4], e.dM[5], e.dM[6], e.dM[7], 1 + e.dM[8]];
  gl.uniformMatrix3fv(u("M"), true, new Float32Array(M));      // row-major -> transpose
  gl.uniform3fv(u("bias"), new Float32Array(e.bias)); gl.uniform1f(u("vig"), e.vig);
  gl.uniform1f(u("split"), split); gl.uniform1i(u("mode"), mode);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
const main = makeGL($("#view"));
const thumbCanvas = document.createElement("canvas");
const thumbR = makeGL(thumbCanvas);
window.__studio = { S, effective, exportParams, draw, main };   // for automated tests

function render() {
  if (!S.img || !S.params) return;
  const mode = S.holdBefore || S.mode === "before" ? 1 : S.mode === "after" ? 2 : 0;
  draw(main, effective(), S.params.knots, mode, S.split);
  placeDivider(); drawCurves(); drawHist(); provenance();
}

// ------------------------------------------------------------------ UI pieces
function placeDivider() {
  const c = $("#view"), d = $("#divider"), r = c.getBoundingClientRect(), w = $("#canvasWrap").getBoundingClientRect();
  d.hidden = S.mode !== "split" || S.holdBefore;
  d.style.left = `${r.left - w.left + S.split * r.width}px`;
}
function drawCurves() {
  const cv = $("#curves"), x = cv.getContext("2d"), W = cv.width, H = cv.height, e = effective();
  x.clearRect(0, 0, W, H); x.strokeStyle = "#444"; x.beginPath();
  for (let i = 1; i < 4; i++) { x.moveTo(i * W / 4, 0); x.lineTo(i * W / 4, H); x.moveTo(0, i * H / 4); x.lineTo(W, i * H / 4); }
  x.stroke();
  const col = ["#ff6b6b", "#6bdc6b", "#6b9bff"], K = S.params.knots;
  const ghost = knotsFromCurve(S.params, modelGroups(S.params, S.strength).curve);
  for (let c = 0; c < 3; c++) {
    const kn = e.knots[c], on = c === S.ch;
    if (S.knots) { x.setLineDash([3, 3]); x.strokeStyle = "#777"; x.beginPath();
      ghost[c].forEach((v, i) => { const px = i / (K - 1) * W, py = H - v * H; i ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke(); x.setLineDash([]); }
    x.strokeStyle = on ? col[c] : col[c] + "55"; x.lineWidth = on ? 2 : 1; x.beginPath();
    kn.forEach((v, i) => { const px = i / (K - 1) * W, py = H - v * H; i ? x.lineTo(px, py) : x.moveTo(px, py); });
    x.stroke();
    if (on) kn.forEach((v, i) => { x.fillStyle = col[c]; x.fillRect(i / (K - 1) * W - 3, H - v * H - 3, 6, 6); });
  }
}
function drawHist() {
  const cv = $("#hist"), x = cv.getContext("2d"), W = cv.width, H = cv.height, gl = main.gl;
  const w = main.canvas.width, h = main.canvas.height, px = new Uint8Array(w * h * 4);
  draw(main, effective(), S.params.knots, 2); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const after = [new Array(64).fill(0), new Array(64).fill(0), new Array(64).fill(0)], lum = new Array(64).fill(0);
  for (let i = 0; i < px.length; i += 4 * 7) for (let c = 0; c < 3; c++) after[c][px[i + c] >> 2]++;
  if (!S.srcPx) { const t = document.createElement("canvas"); t.width = w; t.height = h; const tx = t.getContext("2d");
    tx.drawImage(S.img, 0, 0, w, h); S.srcPx = tx.getImageData(0, 0, w, h).data; }
  for (let i = 0; i < S.srcPx.length; i += 4 * 7) lum[(0.2126 * S.srcPx[i] + 0.7152 * S.srcPx[i + 1] + 0.0722 * S.srcPx[i + 2]) >> 2]++;
  const mx = Math.max(...lum, ...after.flat());
  x.clearRect(0, 0, W, H); x.fillStyle = "#666";
  lum.forEach((v, i) => x.fillRect(i * W / 64, H - v / mx * H, W / 64, v / mx * H));
  ["#ff6b6b", "#6bdc6b", "#6b9bff"].forEach((c, k) => { x.strokeStyle = c; x.beginPath();
    after[k].forEach((v, i) => { const X = (i + 0.5) * W / 64, Y = H - v / mx * H; i ? x.lineTo(X, Y) : x.moveTo(X, Y); }); x.stroke(); });
  const mode = S.holdBefore || S.mode === "before" ? 1 : S.mode === "after" ? 2 : 0;
  draw(main, effective(), S.params.knots, mode, S.split);
}
function provenance() {
  const user = { curve: !!S.knots, color: S.d.warmth || S.d.tint || S.d.sat, vignette: S.d.vig };
  $$(".dot").forEach((d) => d.classList.toggle("user", !!user[d.dataset.group]));
}
function showParamsInfo() {
  const p = S.params;
  $("#explain").textContent = p.explain || "";
  const ood = $("#ood");
  ood.hidden = !(p.ood_score > 1.5);
  ood.textContent = `This photo is unlike the photos "${p.style}" learned from (score ${p.ood_score.toFixed(2)}). ` +
                    "Try a lower strength or another style.";
}

// ------------------------------------------------------------------ history
const snap = () => JSON.stringify({ d: S.d, knots: S.knots, strength: S.strength });
function commit() { S.undo.push(S._last); S.redo = []; S._last = snap(); }
function restore(s) { const o = JSON.parse(s); S.d = o.d; S.knots = o.knots; S.strength = o.strength; syncControls(); render(); }
function syncControls() {
  $("#strength").value = Math.round(S.strength * 100); $("#strengthOut").textContent = `${Math.round(S.strength * 100)}%`;
  $("#warmth").value = S.d.warmth; $("#tint").value = S.d.tint; $("#sat").value = S.d.sat; $("#vig").value = S.d.vig;
}

// ------------------------------------------------------------------ data flow
async function loadStyles() {
  S.styles = await (await api("/api/styles")).json();
  $("#styleSel").innerHTML = S.styles.map((s) => `<option value="${s.name}">${s.name}</option>`).join("");
  if (!S.style && S.styles.length) S.style = S.styles[0].name;
  $("#styleSel").value = S.style || "";
}
async function loadPhotos() {
  S.photos = await (await api("/api/photos")).json();
  $("#library").innerHTML = S.photos.map((p) => `<img src="/api/photo/${p.id}?edge=240" data-id="${p.id}" alt="${p.name}" tabindex="0">`).join("");
  $$("#library img").forEach((im) => (im.onclick = () => selectPhoto(im.dataset.id)));
  $("#empty").hidden = S.photos.length > 0;
}
async function selectPhoto(id) {
  S.id = id; S.srcPx = null;
  $$("#library img").forEach((im) => im.classList.toggle("sel", im.dataset.id === id));
  $("#photoName").textContent = S.photos.find((p) => p.id === id)?.name || "";
  const img = new Image(); img.src = `/api/photo/${id}?edge=1600`; await img.decode();
  S.img = img; main.canvas.width = img.naturalWidth; main.canvas.height = img.naturalHeight; setImage(main, img);
  await predict(); styleStrip();
}
async function predict() {
  if (!S.id || !S.style) return;
  S.params = await (await api("/api/predict", { id: S.id, style: S.style })).json();
  S.d = { warmth: 0, tint: 0, sat: 0, vig: 0 }; S.knots = null; S.strength = S.params.strength ?? 1; S.undo = []; S.redo = []; S._last = snap();
  syncControls(); showParamsInfo(); render();
}
let stripGen = 0;
async function styleStrip() {
  const gen = ++stripGen;   // a newer call (another photo, a new style) supersedes this one
  const strip = $("#styleStrip"); strip.innerHTML = "";
  const w = 140, h = Math.round(w * S.img.naturalHeight / S.img.naturalWidth);
  thumbCanvas.width = w; thumbCanvas.height = h; setImage(thumbR, S.img);
  const items = [{ name: "original" }, ...S.styles];
  for (const [i, st] of items.entries()) {
    if (gen !== stripGen) return;
    const div = document.createElement("div"); div.className = "thumb" + (st.name === S.style ? " sel" : "");
    const c = document.createElement("canvas"); c.width = w; c.height = h; div.append(c);
    const lab = document.createElement("span"); lab.textContent = `${i ? i + " · " : ""}${st.name}`; div.append(lab);
    strip.append(div);
    if (st.name === "original") { c.getContext("2d").drawImage(S.img, 0, 0, w, h); div.onclick = () => { S.mode = "before"; render(); }; continue; }
    const p = await (await api("/api/predict", { id: S.id, style: st.name })).json();
    if (gen !== stripGen) return;
    const saved = [S.params, S.knots, S.d, S.strength];
    S.params = p; S.knots = null; S.d = { warmth: 0, tint: 0, sat: 0, vig: 0 }; S.strength = p.strength ?? 1;
    draw(thumbR, effective(), p.knots, 2); c.getContext("2d").drawImage(thumbCanvas, 0, 0);
    [S.params, S.knots, S.d, S.strength] = saved;
    div.onclick = () => { S.style = st.name; $("#styleSel").value = st.name; $$(".thumb").forEach((t) => t.classList.remove("sel")); div.classList.add("sel"); S.mode = "split"; predict(); };
  }
}
async function download(fmt, id = S.id, params = exportParams()) {
  const r = await api("/api/export", { id, params, format: fmt });
  const name = (r.headers.get("content-disposition") || "").match(/filename="?([^";]+)/)?.[1] || `export.${fmt}`;
  const a = document.createElement("a"); a.href = URL.createObjectURL(await r.blob()); a.download = name; a.click();
}
async function pollJob(job, onProg) {
  for (;;) {
    const j = await (await api(`/api/jobs/${job}`)).json(); onProg(j);
    if (j.status !== "running") return j;
    await new Promise((r) => setTimeout(r, 800));
  }
}

// ------------------------------------------------------------------ events
$("#upload").onchange = async (ev) => {
  const fd = new FormData(); [...ev.target.files].forEach((f) => fd.append("files", f));
  const res = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  await loadPhotos(); if (res.length) selectPhoto(res[0].id);
};
$("#styleSel").onchange = (e) => { S.style = e.target.value; predict(); styleStrip(); };
$("#strength").oninput = (e) => { S.strength = e.target.value / 100; $("#strengthOut").textContent = `${e.target.value}%`; render(); };
for (const k of ["warmth", "tint", "sat", "vig"]) $(`#${k}`).oninput = (e) => { S.d[k] = +e.target.value; render(); };
$$("#panel input[type=range]").forEach((i) => (i.onchange = commit));
$$(".seg").forEach((b) => (b.onclick = () => { S.mode = b.dataset.mode; $$(".seg").forEach((x) => x.classList.toggle("on", x === b)); render(); }));
$$(".tab").forEach((b) => (b.onclick = () => { S.ch = +b.dataset.ch; $$(".tab").forEach((x) => x.classList.toggle("on", x === b)); drawCurves(); }));
$("#btnReset").onclick = () => { S.d = { warmth: 0, tint: 0, sat: 0, vig: 0 }; S.knots = null; S.strength = S.params.strength ?? 1; commit(); syncControls(); render(); };
$("#btnExport").onclick = () => S.params && download($("#exportFmt").value);
$("#btnRemember").onclick = async () => {
  if (!S.params) return;
  const r = await (await api("/api/remember", { id: S.id, style: S.style, params: exportParams() })).json();
  $("#explain").textContent = `Remembered (${r.stored} correction${r.stored > 1 ? "s" : ""} for ${S.style}).`;
};
$("#btnPersonalise").onclick = async () => {
  const { job } = await (await api("/api/personalise", { style: S.style })).json();
  const j = await pollJob(job, (j) => ($("#explain").textContent = `Personalising… ${j.step}/${j.total}`));
  $("#explain").textContent = j.status === "done" ? `Saved style "${S.style.replace(/_personal$/, "")}_personal".` : j.error;
  await loadStyles(); styleStrip();
};
// curve editor: drag the nearest knot of the active channel
(() => {
  const cv = $("#curves"); let drag = -1;
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height]; };
  cv.onpointerdown = (e) => {
    if (!S.params || S.params.renderer === "shared") return;
    const [x] = pos(e), K = S.params.knots; drag = Math.round(x * (K - 1));
    if (!S.knots) S.knots = effective().knots.map((r) => r.slice());
    cv.setPointerCapture(e.pointerId);
  };
  cv.onpointermove = (e) => {
    if (drag < 0) return;
    const [, y] = pos(e), kn = S.knots[S.ch], K = kn.length;
    const lo = drag > 0 ? kn[drag - 1] + 1e-3 : -0.2, hi = drag < K - 1 ? kn[drag + 1] - 1e-3 : 1.2;   // stay monotone
    kn[drag] = Math.min(hi, Math.max(lo, y)); render();
  };
  cv.onpointerup = () => { if (drag >= 0) commit(); drag = -1; };
})();
// compare divider
(() => {
  const d = $("#divider"); let on = false;
  d.onpointerdown = (e) => { on = true; d.setPointerCapture(e.pointerId); };
  d.onpointermove = (e) => { if (!on) return; const r = $("#view").getBoundingClientRect(); S.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); render(); };
  d.onpointerup = () => (on = false);
  d.onkeydown = (e) => { if (e.key === "ArrowLeft") S.split = Math.max(0, S.split - 0.05); if (e.key === "ArrowRight") S.split = Math.min(1, S.split + 0.05); render(); };
})();
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if (e.key === "\\") { S.holdBefore = true; render(); }
  else if (e.key === "[" || e.key === "]") { S.strength = Math.max(0, Math.min(1.5, S.strength + (e.key === "]" ? 0.05 : -0.05))); syncControls(); render(); commit(); }
  else if (e.key.toLowerCase() === "r" && !e.ctrlKey && !e.metaKey) $("#btnReset").click();
  else if (e.key.toLowerCase() === "e") $("#btnExport").click();
  else if (e.key.toLowerCase() === "y") { const m = ["split", "after", "before"]; S.mode = m[(m.indexOf(S.mode) + 1) % 3]; $$(".seg").forEach((x) => x.classList.toggle("on", x.dataset.mode === S.mode)); render(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    if (e.shiftKey) { if (S.redo.length) { S.undo.push(snap()); restore(S.redo.pop()); } }
    else if (S.undo.length) { S.redo.push(snap()); restore(S.undo.pop()); }
    e.preventDefault();
  } else if (/^[1-9]$/.test(e.key) && S.styles[+e.key - 1]) { S.style = S.styles[+e.key - 1].name; $("#styleSel").value = S.style; predict(); styleStrip(); }
  else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const i = S.photos.findIndex((p) => p.id === S.id), j = i + (e.key === "ArrowRight" ? 1 : -1);
    if (S.photos[j]) selectPhoto(S.photos[j].id);
  }
});
document.addEventListener("keyup", (e) => { if (e.key === "\\") { S.holdBefore = false; render(); } });
window.addEventListener("resize", placeDivider);

// wizard (Phase 14)
$("#btnWizard").onclick = () => $("#wizard").showModal();
$$("input[name=mode]").forEach((r) => (r.onchange = () => { const p = $("input[name=mode]:checked").value === "paired"; $("#wizPaired").hidden = !p; $("#wizUnpaired").hidden = p; }));
$("#wizStart").onclick = async () => {
  const mode = $("input[name=mode]:checked").value, fd = new FormData(), name = $("#wizName").value.trim();
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) { $("#wizCheck").textContent = "Name: letters, digits, - or _."; return; }
  fd.append("name", name); fd.append("mode", mode);
  const add = (field, sel) => [...$(sel).files].forEach((f) => fd.append(field, f));
  if (mode === "paired") {
    add("before", "#wizBefore"); add("after", "#wizAfter");
    const b = new Set([...$("#wizBefore").files].map((f) => f.name.replace(/\.[^.]+$/, "")));
    const n = [...$("#wizAfter").files].filter((f) => b.has(f.name.replace(/\.[^.]+$/, ""))).length;
    $("#wizCheck").textContent = `${n} matched pairs` + (n < 20 ? " — 20+ recommended; 50+ is better." : ".");
    if (n < 3) return;
  } else {
    add("examples", "#wizExamples"); add("inputs", "#wizInputs");
    const n = $("#wizExamples").files.length;
    $("#wizCheck").textContent = `${n} examples` + (n < 20 ? " — 20+ recommended." : ".");
    if (n < 3) return;
  }
  const prog = $("#wizProg"); prog.hidden = false;
  const { job } = await (await fetch("/api/learn", { method: "POST", body: fd })).json();
  const j = await pollJob(job, (j) => { prog.max = j.total || 1; prog.value = j.step || 0; });
  $("#wizCheck").textContent = j.status === "done" ? `Style "${name}" saved. It is now in the style strip.` : j.error;
  await loadStyles(); if (S.id) styleStrip();
};

// batch (Phase 15: series consistency)
$("#btnBatch").onclick = () => $("#batch").showModal();
$("#consistency").oninput = (e) => ($("#consOut").textContent = `${e.target.value}%`);
async function batchPreview() {
  const ps = await (await api("/api/batch_predict", { ids: S.photos.map((p) => p.id), style: S.style, consistency: $("#consistency").value / 100 })).json();
  const grid = $("#batchGrid"); grid.innerHTML = "";
  for (const [i, ph] of S.photos.entries()) {
    const img = new Image(); img.src = `/api/photo/${ph.id}?edge=400`; await img.decode();
    thumbCanvas.width = img.naturalWidth; thumbCanvas.height = img.naturalHeight; setImage(thumbR, img);
    const saved = [S.params, S.knots, S.d, S.strength];
    S.params = ps[i]; S.knots = null; S.d = { warmth: 0, tint: 0, sat: 0, vig: 0 }; S.strength = ps[i].strength ?? 1;
    draw(thumbR, effective(), ps[i].knots, 2);
    [S.params, S.knots, S.d, S.strength] = saved;
    const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext("2d").drawImage(thumbCanvas, 0, 0); grid.append(c);
  }
  return ps;
}
$("#batchRun").onclick = batchPreview;
$("#batchExport").onclick = async () => {
  const ps = await batchPreview();
  for (const [i, ph] of S.photos.entries()) await download("jpg", ph.id, { ...ps[i], overrides: {} });
};

(async () => { await loadStyles(); await loadPhotos(); if (S.photos.length) selectPhoto(S.photos[0].id); })();

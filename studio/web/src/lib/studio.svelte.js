// Studio state and actions (Svelte 5 runes). Heavy objects (images, WebGL) live in `view`,
// outside the reactive proxy. All model calls go through the backend (server or browser).
//
// An edit has three layers (A6.3): the look (a predicted Phase-A5 look, applied first, scaled by
// Strength), then the photo's own adjustments `S.p` (every slider, curve and crop), which belong to
// the photo and survive a change of look. Undo covers all three.
import { untrack } from "svelte";
import { api, loadImage, pollJob, json } from "./api.js";
import { B, initBackend } from "./backend.js";
import { effectiveFrom, mixEffective } from "./render.js";
import { draw, makeGL, outputSize, pixelScale, setImage, wbGains, combine, scaleLook, DEFAULTS, SPEC } from "./pipeline.js";
import * as store from "./store.js";

export const ZERO_SCENE = () => ({ haze: 0, clarity_near: 0, clarity_far: 0, sky_exposure: 0, sky_warmth: 0, sky_saturation: 0 });
const NO_LOOK = "";
// Starting adjustments for a raw file: a linear decode has none of the S-shaped tone curve that
// cameras and raw converters apply by default; contrast +20 is a moderate stand-in, shown on the
// slider so it can be changed.
const RAW_BASE = { contrast: 20 };

export const S = $state({
  ready: false,
  backend: "server",
  caps: { scene: false, batch: false, create: false, remember: false },
  formats: [["jpg", "JPEG"]],
  accept: "image/*",
  model: { loaded: 0, total: 0, ready: true, error: "" },   // browser backend: backbone download
  photos: [], styles: [], id: null, style: null,
  params: null,              // the look's predicted params for this photo (null: no look)
  byStyle: {},               // style name -> params for the current photo (thumbnails, peek)
  peek: null,                // style name previewed while hovering the style list
  strength: 1,
  p: {},                     // the photo's adjustments: sliders by name, curves {rgb,r,g,b}, crop
  scene: ZERO_SCENE(),       // scene tools (server-rendered, applied before the grade)
  sceneBusy: false, busy: false,
  mode: "split", split: 0.5, holdBefore: false, ch: "rgb",
  tool: null,                // null | "crop" | "wb" (eyedropper) | "mask" (edit the selected mask on the photo)
  maskSel: -1,               // selected mask (index into S.p.masks)
  showMask: false,           // tint the selected mask's area (O)
  brush: { size: 6, hardness: 50, flow: 60, erase: false },   // size in % of the long edge
  clip: 0,                   // clipping warning: 0 off, 1 shadows/highlights
  zoom: 0,                   // 0 = fit, else scale (1 = 100 %)
  pan: [0, 0],
  readout: null,             // {x, y, rgb} under the pointer
  message: "",
  imgVersion: 0,             // bumps when a new photo is on the GPU
  frame: 0,                  // bumps after every draw (histogram)
  undo: [], redo: [], last: "",
  presets: [],               // the user's own presets (localStorage)
  auto: null,                // the base correction for this photo ({slider: value}) when on
  autoOn: false,
  depth: { loading: false, progress: 0, ready: false, show: false, error: "" },
  clipboard: null,           // copied settings
  snapshots: [],             // named states of this photo
});

export const view = { img: null, sceneImg: null, main: null, thumb: null, thumbCanvas: null, tween: null, shown: null, depth: null,
                      brushes: new Map() };   // mask id -> {canvas, version, data} (painted masks, source space)
const edits = new Map();     // photo id -> {p, style, strength}: each photo keeps its own edit

export function initGL(canvas) {
  view.main = makeGL(canvas);
  view.thumbCanvas = document.createElement("canvas");
  view.thumb = makeGL(view.thumbCanvas);
  /** @type {any} */ (window).__studio = { S, effective, exportParams, setSlider, setStyle, setAuto, exportCurrent, get main() { return view.main; } };   // automated tests
}

export async function boot() {
  const b = await initBackend((m) => Object.assign(S.model, m));
  Object.assign(S, { backend: b.kind, caps: b.caps, formats: b.formats, accept: b.accept });
  if (b.kind === "browser") S.model.ready = false;
  S.presets = store.loadPresets(); S.clipboard = store.loadClip();
  if (b.caps.persist) {
    for (const [id, e] of await store.loadEdits()) edits.set(id, e);
    await b.restore?.();
  }
  await loadStyles(); await loadPhotos();
  S.ready = true;
  if (S.photos.length) await selectPhoto(S.photos[0].id);
}

const lookOf = (params, strength) => (params && params.kind !== "sliders" ? effectiveFrom(params, strength) : null);
/** The sliders the pipeline runs: auto correction, then a slider look at its strength, then yours. */
const layered = (params, strength, p) => (params?.kind === "sliders" || (S.autoOn && S.auto)
  ? combine(S.autoOn ? S.auto : null, params?.kind === "sliders" ? scaleLook(params.sliders, strength) : null, p) : p);
/** Everything the renderer needs for the current edit. */
export const effective = () => ({ p: layered(S.params, S.strength, S.p), look: lookOf(S.params, S.strength), K: S.params?.knots ?? 9, brushes: brushList(), depth: view.depth });
export const modeCode = () => (S.holdBefore || S.mode === "before" ? 1 : S.mode === "after" ? 2 : 0);
export const hasLook = () => !!S.style;

export function exportParams() {       // for the server: the look's EditParams plus the adjustments
  const p = S.params;
  const scene = Object.fromEntries(Object.entries(S.scene).filter(([, v]) => v));
  return { style: p?.style ?? null, renderer: p?.renderer, knots: p?.knots, theta: p?.theta, strength: S.strength,
           overrides: {}, ood_score: p?.ood_score, version: 1, scene, edit: $state.snapshot(S.p) };
}

/** What is on screen: the hovered look (peek), else the current edit; look changes tween. */
function target() {
  const pk = S.peek && S.peek !== "__original" && S.byStyle[S.peek];
  return pk ? { p: layered(pk, pk.strength ?? 1, S.p), look: lookOf(pk, pk.strength ?? 1), K: pk.knots, depth: view.depth } : effective();
}

let raf = 0;
export function render() {
  if (!view.img || !view.main) return;
  let e = target();
  const tw = view.tween;
  if (tw && tw.from.look && e.look && tw.from.look.knots[0].length === e.look.knots[0].length) {
    const t = Math.min(1, (performance.now() - tw.t0) / 320), k = 1 - (1 - t) ** 3;
    e = { ...e, look: mixEffective(tw.from.look, e.look, k) };
    if (t < 1) { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); } else view.tween = null;
  }
  view.shown = e;
  e = { ...e, p: drawnP(e.p) };
  sizeCanvas();
  const full = B.fullSize?.(S.id), c = view.main.canvas;
  const pxScale = full ? pixelScale(c.width, view.main.src, Math.max(...full), drawnP().crop) : 1;
  draw(view.main, { ...e, brushes: brushList() }, { mode: S.peek === "__original" ? 1 : S.peek ? 2 : modeCode(), split: S.split,
                       clipWarn: S.clip ? 1 : 0, pxScale, showMask: S.showMask ? S.maskSel : -1, showDepth: S.depth.show && !!view.depth });
  untrack(() => S.frame++);          // no dependency: effects that call render() must not loop
}
function sizeCanvas() {
  const c = view.main.canvas, crop = S.tool === "crop" ? null : S.p.crop;     // the crop tool shows the whole frame
  const [w, h] = outputSize(view.img.naturalWidth, view.img.naturalHeight, crop ? { ...crop, angle: 0 } : null);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}
/** The adjustments as drawn (the crop tool shows the uncropped, straightened frame). */
export const drawnP = (p = S.p) => (S.tool === "crop" && p.crop ? { ...p, crop: { angle: p.crop.angle ?? 0 } } : p);
function startTween() { if (view.shown) view.tween = { from: view.shown, t0: performance.now() }; }
export function setPeek(name) {
  if (S.peek === name) return;
  startTween(); S.peek = name;
}

// ------------------------------------------------------------------ adjustments
export function setSlider(name, v) {
  if (v === (DEFAULTS[name] ?? 0)) { const { [name]: _, ...rest } = S.p; S.p = rest; }
  else S.p = { ...S.p, [name]: v };
}
export const slider = (name) => S.p[name] ?? DEFAULTS[name] ?? 0;
export function setCurve(key, pts) { S.p = { ...S.p, curves: { ...(S.p.curves || {}), [key]: pts } }; }
export function setCrop(crop) { S.p = { ...S.p, crop }; }
export function resetGroup(names) {
  const p = { ...S.p };
  for (const n of names) delete p[n];
  S.p = p; commit();
}

/** White-balance eyedropper: the temperature and tint that make the picked colour neutral.
 * Two-dimensional Newton iteration on the pipeline's own gains (wbGains), in linear Rec. 2020. */
export function neutralise(rgbDisplay) {
  const lin = rgbDisplay.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  const m = SPEC.m.srgb_to_w, c = [0, 1, 2].map((r) => m[3 * r] * lin[0] + m[3 * r + 1] * lin[1] + m[3 * r + 2] * lin[2]);
  const err = (t, n) => { const g = wbGains(t, n), o = c.map((v, i) => v * g[i]); return [Math.log(o[0] / o[2]), Math.log(o[1] / Math.sqrt(o[0] * o[2]))]; };
  let t = slider("temp"), n = slider("tint");
  for (let it = 0; it < 30; it++) {
    const e0 = err(t, n), h = 0.5, et = err(t + h, n), en = err(t, n + h);
    const J = [[(et[0] - e0[0]) / h, (en[0] - e0[0]) / h], [(et[1] - e0[1]) / h, (en[1] - e0[1]) / h]];
    const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if (Math.abs(det) < 1e-12) break;
    t -= (J[1][1] * e0[0] - J[0][1] * e0[1]) / det; n -= (-J[1][0] * e0[0] + J[0][0] * e0[1]) / det;
    t = Math.max(-100, Math.min(100, t)); n = Math.max(-100, Math.min(100, n));
    if (Math.hypot(...e0) < 1e-5) break;
  }
  setSlider("temp", Math.round(t)); setSlider("tint", Math.round(n)); commit();
}

// ------------------------------------------------------------------ history
const snap = () => JSON.stringify({ p: S.p, strength: S.strength, style: S.style, scene: S.scene, autoOn: S.autoOn });
export function commit() { const s = snap(); if (s === S.last) return; S.undo.push(S.last); S.redo = []; S.last = s; remember(); }
async function restore(s) {
  const o = JSON.parse(s); S.p = o.p; S.strength = o.strength;
  if ((o.autoOn ?? false) !== S.autoOn) { S.autoOn = !!o.autoOn; if (S.autoOn && !S.auto && S.id) loadAuto(S.id); }
  if (o.style !== S.style) { S.style = o.style; await loadLook(); S.strength = o.strength; }
  const sceneChanged = JSON.stringify(o.scene) !== JSON.stringify(S.scene);
  S.scene = o.scene; if (sceneChanged) applyScene();
  S.last = s; remember();
}
export function undo() { if (S.undo.length) { S.redo.push(snap()); restore(S.undo.pop()); } }
export function redo() { if (S.redo.length) { S.undo.push(snap()); restore(S.redo.pop()); } }
export function resetAll() { S.p = {}; S.strength = S.params?.strength ?? 1; commit(); }
let saveTimer = 0;
function remember() {
  if (!S.id) return;
  const id = S.id, e = { p: $state.snapshot(S.p), style: S.style, strength: S.strength, snapshots: $state.snapshot(S.snapshots), autoOn: S.autoOn };
  edits.set(id, e);
  if (S.caps.persist) { clearTimeout(saveTimer); saveTimer = setTimeout(() => store.saveEdit(id, e), 400); }
}

// ------------------------------------------------------------------ data flow
export async function loadStyles() {
  S.styles = await B.styles();
  if (S.style === null && S.styles.length) S.style = S.styles[0].name;
}
export async function loadPhotos() { S.photos = await B.photos(); }
export const photoURL = (id, edge = 1600) => B.photoURL(id, edge);

export async function upload(files) {
  if (!files?.length) return;
  S.busy = true;
  try {
    const res = await B.upload(files);
    await loadPhotos(); if (res.length) await selectPhoto(res[0].id);
  } finally { S.busy = false; }
}
export async function addSample(s) {
  S.busy = true;
  try { const res = await B.addSample(s); await loadPhotos(); if (res.length) await selectPhoto(res[0].id); }
  finally { S.busy = false; }
}

export async function selectPhoto(id) {
  remember();
  S.id = id; view.sceneImg = null; view.tween = null; view.shown = null; view.depth = null;
  S.depth = { loading: false, progress: 0, ready: false, show: false, error: "" };
  S.scene = ZERO_SCENE(); S.byStyle = {}; S.peek = null; S.zoom = 0; S.pan = [0, 0]; S.tool = null;
  const kept = edits.get(id);
  S.p = kept?.p ?? {}; S.snapshots = kept?.snapshots ?? [];
  if (kept) S.style = kept.style;
  const img = await loadImage(photoURL(id, 1600));
  view.img = img;
  setImage(view.main, B.source?.(id) ?? img);
  if (!kept && B.isRaw?.(id)) S.p = { ...RAW_BASE };
  await loadLook();
  if (kept) S.strength = kept.strength;
  S.autoOn = kept?.autoOn ?? (S.caps.auto && autoDefault());
  S.auto = null; if (S.autoOn) loadAuto(id);
  S.undo = []; S.redo = []; S.last = snap();
  S.imgVersion++;
  predictAll(id);
}

/** Params for every style on this photo (style list previews and hover-peek). */
async function predictAll(id) {
  for (const st of S.styles) {
    if (S.id !== id) return;
    if (!S.byStyle[st.name]) S.byStyle[st.name] = await B.predict(id, st.name);
  }
}

async function loadLook() {
  if (!S.id || !S.style) { S.params = null; return; }
  const p = S.byStyle[S.style] || await B.predict(S.id, S.style);
  S.byStyle[S.style] = p;
  S.params = p;
  S.strength = p.strength ?? 1;
}

export async function setStyle(name) {
  startTween(); S.peek = null;
  S.style = name || NO_LOOK; if (S.mode === "before") S.mode = "split";
  await loadLook();
  commit();
}

let sceneTimer = 0, sceneSeq = 0;
/** Debounced: ask the server for the photo with scene tools applied, then use it as the texture. */
export function applyScene() {
  if (!S.caps.scene) return;
  clearTimeout(sceneTimer);
  sceneTimer = setTimeout(async () => {
    if (!S.id) return;
    const seq = ++sceneSeq; S.sceneBusy = true;
    try {
      const r = await api("/api/scene", { id: S.id, scene: S.scene });
      const url = URL.createObjectURL(await r.blob());
      const img = await loadImage(url);
      if (seq !== sceneSeq) return;
      view.sceneImg = img; setImage(view.main, img, false); render();   // "before" stays the original
    } finally { if (seq === sceneSeq) S.sceneBusy = false; }
  }, 250);
}

/** Export with ``opts`` = { fmt, quality (1-100), edge (long side, 0 = full), space ("srgb" | "p3"),
 * metadata (copy EXIF) }. */
export async function exportCurrent(opts) {
  if (!S.id) return;
  S.busy = true;
  try { const note = await B.export(S.id, exportParams(), $state.snapshot(effective()), typeof opts === "string" ? { fmt: opts } : opts); if (note) S.message = note; }
  catch (e) { S.message = String(e); }
  finally { S.busy = false; }
}

export async function rememberEdit() {
  if (!S.params) return;
  const r = await json("/api/remember", { id: S.id, style: S.style, params: exportParams() });
  S.message = `Remembered (${r.stored} correction${r.stored > 1 ? "s" : ""} for ${S.style}).`;
}

export async function personalise() {
  const { job } = await json("/api/personalise", { style: S.style });
  const j = await pollJob(job, (j) => (S.message = `Personalising… ${j.step}/${j.total}`));
  S.message = j.status === "done" ? `Saved style "${S.style.replace(/_personal$/, "")}_personal".` : j.error;
  await loadStyles();
}

/** Render a look on `img` into a small canvas (style list, batch). */
export function renderThumb(img, params, w, target) {
  const h = Math.round((w * img.naturalHeight) / img.naturalWidth);
  view.thumbCanvas.width = w; view.thumbCanvas.height = h; setImage(view.thumb, img);
  draw(view.thumb, { p: layered(params, params.strength ?? 1, {}), look: lookOf(params, params.strength ?? 1), K: params.knots }, { mode: 2 });
  target.width = w; target.height = h; target.getContext("2d").drawImage(view.thumbCanvas, 0, 0);
}

const NAMES = { fivek: "FiveK", fujifilm: "Fujifilm" };
/** Display name of a style: "clean_cool" -> "Clean Cool". */
export const pretty = (n) => n.split("_").map((w) => NAMES[w] || w.charAt(0).toUpperCase() + w.slice(1)).join(" ");

// ------------------------------------------------------------------ masks (local adjustments)
const MASK_DEFAULTS = {
  linear: () => ({ x0: 0.5, y0: 0.15, x1: 0.5, y1: 0.55 }),
  radial: () => ({ cx: 0.5, cy: 0.5, rx: 0.22, ry: 0.22, angle: 0, feather: 50 }),
  brush: () => ({}),
  all: () => ({}),
};
let maskSeq = 0;
export function addMask(type, range = null) {
  const masks = [...(S.p.masks || [])];
  if (masks.length >= SPEC.masks.max) { S.message = `At most ${SPEC.masks.max} masks.`; return; }
  masks.push({ id: `m${Date.now().toString(36)}${maskSeq++}`, type, invert: false, ...MASK_DEFAULTS[type](),
               ...(range ? { range } : {}), sliders: {} });
  S.p = { ...S.p, masks }; S.maskSel = masks.length - 1; S.tool = type === "all" ? null : "mask"; commit();
}
export function updateMask(j, patch) {
  const masks = (S.p.masks || []).map((m, i) => (i === j ? { ...m, ...patch } : m));
  S.p = { ...S.p, masks };
}
export function setMaskSlider(j, k, v) {
  const m = S.p.masks[j], sl = { ...(m.sliders || {}) };
  if (v === 0) delete sl[k]; else sl[k] = v;
  updateMask(j, { sliders: sl });
}
export function removeMask(j) {
  const masks = (S.p.masks || []).filter((_, i) => i !== j);
  const { masks: _, ...rest } = S.p;
  S.p = masks.length ? { ...rest, masks } : rest;
  S.maskSel = Math.min(S.maskSel, masks.length - 1); if (S.maskSel < 0) S.tool = S.tool === "mask" ? null : S.tool;
  commit();
}

/** The painted-mask canvases for the current masks (rebuilt from their stored PNG after undo). */
function brushList() {
  return (S.p.masks || []).map((m) => {
    if (m.type !== "brush") return null;
    let b = view.brushes.get(m.id);
    if (!b) { b = { canvas: newBrushCanvas(), version: 0, data: undefined }; view.brushes.set(m.id, b); }
    if (m.data !== b.data && !b.painting) {                  // restore from the stored PNG (undo, another photo)
      b.data = m.data;
      const ctx = b.canvas.getContext("2d"); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, b.canvas.width, b.canvas.height);
      if (m.data) { const im = new Image(); im.onload = () => { ctx.drawImage(im, 0, 0, b.canvas.width, b.canvas.height); b.version++; render(); }; im.src = "data:image/png;base64," + m.data; }
      b.version++;
    }
    return b;
  });
}
function newBrushCanvas() {
  // source space, 1024 px on the long side (enough for a soft mask; it is sampled with filtering)
  const W = view.img?.naturalWidth || 1024, H = view.img?.naturalHeight || 768, s = 1024 / Math.max(W, H);
  const c = document.createElement("canvas"); c.width = Math.round(W * s); c.height = Math.round(H * s);
  const x = c.getContext("2d"); x.fillStyle = "#000"; x.fillRect(0, 0, c.width, c.height);
  return c;
}
/** Paint a dab at source uv (u, v) into mask j. Soft round brush; hardness sets the solid core. */
export function paint(j, u, v) {
  const m = S.p.masks[j], b = view.brushes.get(m.id); if (!b) return;
  b.painting = true;
  const c = b.canvas, x = c.getContext("2d"), r = (S.brush.size / 100) * Math.max(c.width, c.height);
  const g = x.createRadialGradient(u * c.width, v * c.height, r * (S.brush.hardness / 100) * 0.99, u * c.width, v * c.height, r);
  const col = S.brush.erase ? "0,0,0" : "255,255,255", a = S.brush.flow / 100;
  g.addColorStop(0, `rgba(${col},${a})`); g.addColorStop(1, `rgba(${col},0)`);
  x.fillStyle = g; x.beginPath(); x.arc(u * c.width, v * c.height, r, 0, 2 * Math.PI); x.fill();
  b.version++;
  render();
}
export function endStroke(j) {
  const m = S.p.masks[j], b = view.brushes.get(m.id); if (!b) return;
  b.painting = false;
  b.data = b.canvas.toDataURL("image/png").split(",")[1];
  updateMask(j, { data: b.data }); commit();
}

// ------------------------------------------------------------------ workflow (Phase 25)
const OWN = (p) => { const { crop: _c, masks: _m, ...rest } = p; return rest; };   // settings that transfer between photos

/** Copy the edit (look, strength and global adjustments; not crop or masks, which belong to a photo). */
export function copySettings() {
  S.clipboard = { p: $state.snapshot(OWN(S.p)), style: S.style, strength: S.strength };
  store.saveClip(S.clipboard); S.message = "Settings copied.";
}
export async function pasteSettings() {
  if (!S.clipboard) return;
  S.p = { ...S.clipboard.p, ...(S.p.crop ? { crop: S.p.crop } : {}), ...(S.p.masks ? { masks: S.p.masks } : {}) };
  if (S.clipboard.style !== S.style) { S.style = S.clipboard.style; await loadLook(); }
  S.strength = S.clipboard.strength; commit();
}
/** Apply the copied settings to every photo (each keeps its own crop and masks). */
export function syncAll() {
  if (!S.clipboard) return;
  for (const ph of S.photos) {
    const e = edits.get(ph.id) || { p: {} };
    const p = { ...S.clipboard.p, ...(e.p?.crop ? { crop: e.p.crop } : {}), ...(e.p?.masks ? { masks: e.p.masks } : {}) };
    const ne = { ...e, p, style: S.clipboard.style, strength: S.clipboard.strength };
    edits.set(ph.id, ne); if (S.caps.persist) store.saveEdit(ph.id, ne);
  }
  pasteSettings();
  S.message = `Settings applied to ${S.photos.length} photos.`;
}

export function savePreset(name) {
  if (!name) return;
  const pr = { name, p: $state.snapshot(OWN(S.p)), style: S.style, strength: S.strength };
  S.presets = [...S.presets.filter((x) => x.name !== name), pr];
  store.savePresets($state.snapshot(S.presets));
}
export async function applyPreset(pr) {
  S.p = { ...pr.p, ...(S.p.crop ? { crop: S.p.crop } : {}), ...(S.p.masks ? { masks: S.p.masks } : {}) };
  if (pr.style !== S.style) { S.style = pr.style; await loadLook(); }
  S.strength = pr.strength; commit();
}
export function deletePreset(name) { S.presets = S.presets.filter((x) => x.name !== name); store.savePresets($state.snapshot(S.presets)); }

export function takeSnapshot(name) {
  S.snapshots = [...S.snapshots, { name: name || `Snapshot ${S.snapshots.length + 1}`, p: $state.snapshot(S.p), style: S.style, strength: S.strength }];
  remember();
}
export async function restoreSnapshot(sn) {
  S.p = sn.p; if (sn.style !== S.style) { S.style = sn.style; await loadLook(); }
  S.strength = sn.strength; commit();
}
export function deleteSnapshot(i) { S.snapshots = S.snapshots.filter((_, k) => k !== i); remember(); }

/** The history as labelled steps (what changed between consecutive states). */
export function historySteps() {
  const states = [...S.undo, S.last].filter(Boolean).map((x) => JSON.parse(x));
  const label = (a, b) => {
    if (!a) return "Opened";
    if (a.style !== b.style) return `Look: ${b.style ? pretty(b.style) : "none"}`;
    if (a.strength !== b.strength) return `Strength ${Math.round(b.strength * 100)} %`;
    const keys = new Set([...Object.keys(a.p || {}), ...Object.keys(b.p || {})]);
    const ch = [...keys].filter((k) => JSON.stringify(a.p?.[k]) !== JSON.stringify(b.p?.[k]));
    if (!ch.length) return JSON.stringify(a.scene) !== JSON.stringify(b.scene) ? "Scene" : "Edit";
    const name = (k) => (k === "curves" ? "Curve" : k === "crop" ? "Crop" : k === "masks" ? "Mask" : (SPEC.sliders.find((x) => x.name === k)?.label ?? k));
    return ch.length > 2 ? `${name(ch[0])} and ${ch.length - 1} more` : ch.map(name).join(", ");
  };
  return states.map((st, i) => ({ i, label: label(states[i - 1], st) }));
}
/** Go back (or forward) to history step i. */
export function gotoStep(i) {
  const n = S.undo.length;
  if (i < n) for (let k = n; k > i; k--) undo();
}

/** Remove a photo (and its kept copy). */
export async function removePhoto(id) {
  await B.remove?.(id); edits.delete(id); await loadPhotos();
  if (S.id === id) { if (S.photos.length) await selectPhoto(S.photos[0].id); else { S.id = null; view.img = null; } }
}

// ------------------------------------------------------------------ match a reference photo
/** Statistics of an image in Oklab: mean L, std L, mean a, mean b, mean chroma. */
function stats(px) {
  const m = SPEC.m, n = px.length / 4;
  let sL = 0, sL2 = 0, sa = 0, sb = 0, sc = 0;
  for (let i = 0; i < px.length; i += 4) {
    const rgb = [px[i], px[i + 1], px[i + 2]].map((v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    const w = [0, 1, 2].map((r) => m.srgb_to_w[3 * r] * rgb[0] + m.srgb_to_w[3 * r + 1] * rgb[1] + m.srgb_to_w[3 * r + 2] * rgb[2]);
    const l = [0, 1, 2].map((r) => Math.cbrt(m.w_to_lms[3 * r] * w[0] + m.w_to_lms[3 * r + 1] * w[1] + m.w_to_lms[3 * r + 2] * w[2]));
    const L = m.lms_to_lab[0] * l[0] + m.lms_to_lab[1] * l[1] + m.lms_to_lab[2] * l[2];
    const a = m.lms_to_lab[3] * l[0] + m.lms_to_lab[4] * l[1] + m.lms_to_lab[5] * l[2];
    const b = m.lms_to_lab[6] * l[0] + m.lms_to_lab[7] * l[1] + m.lms_to_lab[8] * l[2];
    sL += L; sL2 += L * L; sa += a; sb += b; sc += Math.hypot(a, b);
  }
  const mL = sL / n;
  return { L: mL, sd: Math.sqrt(Math.max(0, sL2 / n - mL * mL)), a: sa / n, b: sb / n, C: sc / n };
}
function smallPixels(img, p, look) {
  const w = 96, h = Math.max(1, Math.round((96 * img.naturalHeight) / img.naturalWidth));
  view.thumbCanvas.width = w; view.thumbCanvas.height = h; setImage(view.thumb, img);
  draw(view.thumb, { p, look, K: S.params?.knots ?? 9 }, { mode: 2, dither: 0 });
  const px = new Uint8Array(4 * w * h); view.thumb.gl.readPixels(0, 0, w, h, view.thumb.gl.RGBA, view.thumb.gl.UNSIGNED_BYTE, px);
  return px;
}
/** Match the overall brightness, contrast, white balance and saturation of another photo:
 * coordinate search on exposure, contrast, temperature, tint and saturation, measured on 96 px
 * renders through the real pipeline (so the match includes the look and every other setting). */
export async function matchTo(refId) {
  const ref = await loadImage(photoURL(refId, 400)), cur = view.img, look = lookOf(S.params, S.strength);
  const t = stats(smallPixels(ref, edits.get(refId)?.p ?? {}, null));
  const p = { ...S.p }, err = (q) => { const s = stats(smallPixels(cur, q, look)); return [s.L - t.L, s.sd - t.sd, s.a - t.a, s.b - t.b, s.C - t.C]; };
  const solve = (k, idx, lo, hi, sign = 1) => {                 // bisection on one slider for one statistic
    let a = lo, b = hi;
    for (let it = 0; it < 12; it++) { const m = (a + b) / 2; if (sign * err({ ...p, [k]: m })[idx] > 0) b = m; else a = m; }
    p[k] = Math.round(((a + b) / 2) * 100) / 100;
  };
  for (let pass = 0; pass < 2; pass++) {
    solve("exposure", 0, -3, 3); solve("contrast", 1, -100, 100); solve("temp", 3, -100, 100); solve("tint", 2, -100, 100);
    solve("saturation", 4, -100, 100);
  }
  for (const k of ["contrast", "temp", "tint", "saturation"]) p[k] = Math.round(p[k]);
  S.p = p; commit();
  S.message = "Matched to the reference photo.";
}

// ------------------------------------------------------------------ base correction (Phase 21)
const AUTO_KEY = "irodori.auto.v1";
// Off unless the user turned it on: in the owner's blind test it was not preferred to the unedited photo
// (tests/reports/phase20_rating.md). Photos saved with it on keep it.
const autoDefault = () => { try { return localStorage.getItem(AUTO_KEY) === "1"; } catch { return false; } };
async function loadAuto(id) {
  const a = await B.sliders?.(id, "auto_c");
  if (S.id === id) { S.auto = a; }
}
/** Turn the per-photo correction (predicted from MIT-Adobe FiveK expert C) on or off. */
export async function setAuto(on) {
  S.autoOn = on;
  try { localStorage.setItem(AUTO_KEY, on ? "1" : "0"); } catch { /* storage unavailable */ }
  if (on && !S.auto && S.id) await loadAuto(S.id);
  commit();
}

// ------------------------------------------------------------------ lens blur (Phase 24)
/** Estimate depth for this photo (downloads the depth model once), then lens blur can be used. */
export async function estimateDepth() {
  if (!B.depth || !S.id) return;
  const id = S.id;
  S.depth = { ...S.depth, loading: true, error: "" };
  try {
    const d = await B.depth(id, (l, t) => (S.depth.progress = t ? l / t : 0));
    if (S.id !== id) return;
    view.depth = d; S.depth = { ...S.depth, loading: false, ready: true };
    if (!S.p.lens_amount) { setSlider("lens_amount", 40); commit(); }
    render();
  } catch (e) { S.depth = { ...S.depth, loading: false, error: String(e) }; }
}
/** Focus on the depth at output uv (u, v). */
export function focusAt(su, sv) {
  const d = view.depth; if (!d) return;
  const x = Math.min(d.width - 1, Math.max(0, Math.floor(su * d.width))), y = Math.min(d.height - 1, Math.max(0, Math.floor(sv * d.height)));
  setSlider("lens_focus", Math.round(d.data[y * d.width + x] * 100) / 100); commit();
}

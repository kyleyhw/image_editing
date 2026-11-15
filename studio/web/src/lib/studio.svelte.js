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
import { draw, makeGL, outputSize, setImage, wbGains, DEFAULTS, SPEC } from "./pipeline.js";

export const ZERO_SCENE = () => ({ haze: 0, clarity_near: 0, clarity_far: 0, sky_exposure: 0, sky_warmth: 0, sky_saturation: 0 });
const NO_LOOK = "";

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
  tool: null,                // null | "crop" | "wb" (eyedropper)
  clip: 0,                   // clipping warning: 0 off, 1 shadows/highlights
  zoom: 0,                   // 0 = fit, else scale (1 = 100 %)
  pan: [0, 0],
  readout: null,             // {x, y, rgb} under the pointer
  message: "",
  imgVersion: 0,             // bumps when a new photo is on the GPU
  frame: 0,                  // bumps after every draw (histogram)
  undo: [], redo: [], last: "",
});

export const view = { img: null, sceneImg: null, main: null, thumb: null, thumbCanvas: null, tween: null, shown: null };
const edits = new Map();     // photo id -> {p, style, strength}: each photo keeps its own edit

export function initGL(canvas) {
  view.main = makeGL(canvas);
  view.thumbCanvas = document.createElement("canvas");
  view.thumb = makeGL(view.thumbCanvas);
  /** @type {any} */ (window).__studio = { S, effective, exportParams, setSlider, get main() { return view.main; } };   // automated tests
}

export async function boot() {
  const b = await initBackend((m) => Object.assign(S.model, m));
  Object.assign(S, { backend: b.kind, caps: b.caps, formats: b.formats, accept: b.accept });
  if (b.kind === "browser") S.model.ready = false;
  await loadStyles(); await loadPhotos();
  S.ready = true;
  if (S.photos.length) await selectPhoto(S.photos[0].id);
}

const lookOf = (params, strength) => (params ? effectiveFrom(params, strength) : null);
/** Everything the renderer needs for the current edit. */
export const effective = () => ({ p: S.p, look: lookOf(S.params, S.strength), K: S.params?.knots ?? 9 });
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
  return pk ? { p: S.p, look: lookOf(pk, pk.strength ?? 1), K: pk.knots } : effective();
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
  e = { ...e, p: drawnP() };
  sizeCanvas();
  draw(view.main, e, { mode: S.peek === "__original" ? 1 : S.peek ? 2 : modeCode(), split: S.split,
                       clipWarn: S.clip ? 1 : 0 });
  untrack(() => S.frame++);          // no dependency: effects that call render() must not loop
}
function sizeCanvas() {
  const c = view.main.canvas, crop = S.tool === "crop" ? null : S.p.crop;     // the crop tool shows the whole frame
  const [w, h] = outputSize(view.img.naturalWidth, view.img.naturalHeight, crop ? { ...crop, angle: 0 } : null);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}
/** The adjustments as drawn (the crop tool shows the uncropped, straightened frame). */
export const drawnP = () => (S.tool === "crop" && S.p.crop ? { ...S.p, crop: { angle: S.p.crop.angle ?? 0 } } : S.p);
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
const snap = () => JSON.stringify({ p: S.p, strength: S.strength, style: S.style, scene: S.scene });
export function commit() { const s = snap(); if (s === S.last) return; S.undo.push(S.last); S.redo = []; S.last = s; remember(); }
async function restore(s) {
  const o = JSON.parse(s); S.p = o.p; S.strength = o.strength;
  if (o.style !== S.style) { S.style = o.style; await loadLook(); S.strength = o.strength; }
  const sceneChanged = JSON.stringify(o.scene) !== JSON.stringify(S.scene);
  S.scene = o.scene; if (sceneChanged) applyScene();
  S.last = s; remember();
}
export function undo() { if (S.undo.length) { S.redo.push(snap()); restore(S.undo.pop()); } }
export function redo() { if (S.redo.length) { S.undo.push(snap()); restore(S.redo.pop()); } }
export function resetAll() { S.p = {}; S.strength = S.params?.strength ?? 1; commit(); }
function remember() { if (S.id) edits.set(S.id, { p: S.p, style: S.style, strength: S.strength }); }

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
  S.id = id; view.sceneImg = null; view.tween = null; view.shown = null;
  S.scene = ZERO_SCENE(); S.byStyle = {}; S.peek = null; S.zoom = 0; S.pan = [0, 0]; S.tool = null;
  const kept = edits.get(id);
  S.p = kept?.p ?? {};
  if (kept) S.style = kept.style;
  const img = await loadImage(photoURL(id, 1600));
  view.img = img;
  setImage(view.main, img);
  await loadLook();
  if (kept) S.strength = kept.strength;
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

export async function exportCurrent(fmt) {
  if (!S.id) return;
  S.busy = true;
  try { await B.export(S.id, exportParams(), $state.snapshot(effective()), fmt); } finally { S.busy = false; }
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
  draw(view.thumb, { p: {}, look: lookOf(params, params.strength ?? 1), K: params.knots }, { mode: 2 });
  target.width = w; target.height = h; target.getContext("2d").drawImage(view.thumbCanvas, 0, 0);
}

const NAMES = { fivek: "FiveK", fujifilm: "Fujifilm" };
/** Display name of a style: "clean_cool" -> "Clean Cool". */
export const pretty = (n) => n.split("_").map((w) => NAMES[w] || w.charAt(0).toUpperCase() + w.slice(1)).join(" ");

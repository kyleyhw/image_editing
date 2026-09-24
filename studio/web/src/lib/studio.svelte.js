// Studio state and actions (Svelte 5 runes). Heavy objects (images, WebGL) live in `view`,
// outside the reactive proxy. All model calls go through the backend (server or browser).
import { untrack } from "svelte";
import { api, loadImage, pollJob, json } from "./api.js";
import { B, initBackend } from "./backend.js";
import { curveFromKnots, draw, effectiveFrom, makeGL, mixEffective, setImage } from "./render.js";

const ZERO_D = () => ({ warmth: 0, tint: 0, sat: 0, vig: 0 });
export const ZERO_SCENE = () => ({ haze: 0, clarity_near: 0, clarity_far: 0, sky_exposure: 0, sky_warmth: 0, sky_saturation: 0 });

export const S = $state({
  ready: false,
  backend: "server",
  caps: { scene: false, batch: false, create: false, remember: false },
  formats: [["jpg", "JPEG"]],
  accept: "image/*",
  model: { loaded: 0, total: 0, ready: true, error: "" },   // browser backend: backbone download
  photos: [], styles: [], id: null, style: null,
  params: null,              // EditParams payload for the current style
  byStyle: {},               // style name -> params for the current photo (thumbnails, peek)
  peek: null,                // style name previewed while hovering the style list
  strength: 1,
  d: ZERO_D(),               // user deltas (colour, vignette)
  knots: null,               // user curve override: [3][K] or null
  scene: ZERO_SCENE(),       // scene tools (server-rendered, applied before the grade)
  sceneBusy: false, busy: false,
  mode: "split", split: 0.5, holdBefore: false, ch: 0,
  message: "",
  imgVersion: 0,             // bumps when a new photo is on the GPU
  frame: 0,                  // bumps after every draw (ambient light, histogram)
  undo: [], redo: [], last: "",
});

export const view = { img: null, sceneImg: null, main: null, thumb: null, thumbCanvas: null, srcPx: null,
                      tween: null, shown: null };

export function initGL(canvas) {
  view.main = makeGL(canvas);
  view.thumbCanvas = document.createElement("canvas");
  view.thumb = makeGL(view.thumbCanvas);
  /** @type {any} */ (window).__studio = { S, effective, exportParams, draw, get main() { return view.main; } };   // automated tests
}

export async function boot() {
  const b = await initBackend((m) => Object.assign(S.model, m));
  Object.assign(S, { backend: b.kind, caps: b.caps, formats: b.formats, accept: b.accept });
  if (b.kind === "browser") S.model.ready = false;
  await loadStyles(); await loadPhotos();
  S.ready = true;
  if (S.photos.length) await selectPhoto(S.photos[0].id);
}

export const effective = () => effectiveFrom(S.params, S.strength, S.d, S.knots);
export const modeCode = () => (S.holdBefore || S.mode === "before" ? 1 : S.mode === "after" ? 2 : 0);

export function exportParams() {       // EditParams for the server (overrides are absolute)
  const p = S.params, e = effective(), ov = { dM: e.dM, bias: e.bias, vignette: [e.vig] };
  if (S.knots && p.renderer !== "shared") ov.curve = curveFromKnots(p, S.knots);
  const scene = Object.fromEntries(Object.entries(S.scene).filter(([, v]) => v));
  return { style: p.style, renderer: p.renderer, knots: p.knots, theta: p.theta, strength: S.strength,
           overrides: ov, ood_score: p.ood_score, version: 1, scene };
}

/** What is on screen: the hovered style (peek), else the current edit; style changes tween. */
function target() {
  const pk = S.peek && S.byStyle[S.peek];
  return pk ? effectiveFrom(pk, pk.strength ?? 1, ZERO_D(), null) : effective();
}

let raf = 0;
export function render() {
  if (!view.img || !S.params || !view.main) return;
  let e = target();
  const tw = view.tween;
  if (tw && tw.from.knots[0].length === e.knots[0].length) {
    const t = Math.min(1, (performance.now() - tw.t0) / 320), k = 1 - (1 - t) ** 3;
    e = mixEffective(tw.from, e, k);
    if (t < 1) { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); } else view.tween = null;
  }
  view.shown = e;
  draw(view.main, e, S.params.knots, S.peek === "__original" ? 1 : S.peek ? 2 : modeCode(), S.split);
  untrack(() => S.frame++);          // no dependency: effects that call render() must not loop
}
function startTween() { if (view.shown) view.tween = { from: view.shown, t0: performance.now() }; }
export function setPeek(name) {
  if (S.peek === name) return;
  startTween(); S.peek = name;
}

// ------------------------------------------------------------------ history
const snap = () => JSON.stringify({ d: S.d, knots: S.knots, strength: S.strength, scene: S.scene });
export function commit() { S.undo.push(S.last); S.redo = []; S.last = snap(); }
function restore(s) {
  const o = JSON.parse(s); S.d = o.d; S.knots = o.knots; S.strength = o.strength;
  const sceneChanged = JSON.stringify(o.scene) !== JSON.stringify(S.scene);
  S.scene = o.scene; if (sceneChanged) applyScene();
}
export function undo() { if (S.undo.length) { S.redo.push(snap()); restore(S.undo.pop()); } }
export function redo() { if (S.redo.length) { S.undo.push(snap()); restore(S.redo.pop()); } }
export function resetToModel() {
  S.d = ZERO_D(); S.knots = null; S.strength = S.params?.strength ?? 1; commit();
}

// ------------------------------------------------------------------ data flow
export async function loadStyles() {
  S.styles = await B.styles();
  if ((!S.style || !S.styles.some((s) => s.name === S.style)) && S.styles.length) S.style = S.styles[0].name;
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
  S.id = id; view.srcPx = null; view.sceneImg = null; view.tween = null; view.shown = null;
  S.scene = ZERO_SCENE(); S.byStyle = {}; S.peek = null;
  const img = await loadImage(photoURL(id, 1600));
  view.img = img;
  view.main.canvas.width = img.naturalWidth; view.main.canvas.height = img.naturalHeight;
  setImage(view.main, img);
  await predict();
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

export async function predict() {
  if (!S.id || !S.style) return;
  const p = S.byStyle[S.style] || await B.predict(S.id, S.style);
  S.byStyle[S.style] = p;
  S.params = p;
  S.d = ZERO_D(); S.knots = null; S.strength = p.strength ?? 1; S.undo = []; S.redo = []; S.last = snap();
}

export async function setStyle(name) {
  startTween(); S.peek = null;
  S.style = name; if (S.mode === "before") S.mode = "split";
  await predict();
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
  if (!S.params) return;
  S.busy = true;
  try { await B.export(S.id, exportParams(), effective(), fmt); } finally { S.busy = false; }
}

export async function remember() {
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

/** Render `params` on `img` into a small canvas (style list, batch). */
export function renderThumb(img, params, w, target) {
  const h = Math.round((w * img.naturalHeight) / img.naturalWidth);
  view.thumbCanvas.width = w; view.thumbCanvas.height = h; setImage(view.thumb, img);
  const e = effectiveFrom(params, params.strength ?? 1, ZERO_D(), null);
  draw(view.thumb, e, params.knots, 2);
  target.width = w; target.height = h; target.getContext("2d").drawImage(view.thumbCanvas, 0, 0);
}

export const pretty = (n) => n.replace(/_/g, " ");

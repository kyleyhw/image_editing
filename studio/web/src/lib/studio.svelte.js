// Studio state and actions (Svelte 5 runes). Heavy objects (images, WebGL) live in `view`,
// outside the reactive proxy.
import { api, downloadBlob, json, loadImage, pollJob } from "./api.js";
import { curveFromKnots, draw, effectiveFrom, makeGL, setImage } from "./render.js";

const ZERO_D = () => ({ warmth: 0, tint: 0, sat: 0, vig: 0 });
export const ZERO_SCENE = () => ({ haze: 0, clarity_near: 0, clarity_far: 0, sky_exposure: 0, sky_warmth: 0, sky_saturation: 0 });

export const S = $state({
  photos: [], styles: [], id: null, style: null,
  params: null,              // server EditParams payload
  strength: 1,
  d: ZERO_D(),               // user deltas (colour, vignette)
  knots: null,               // user curve override: [3][K] or null
  scene: ZERO_SCENE(),       // scene tools (server-rendered, applied before the grade)
  sceneBusy: false,
  mode: "split", split: 0.5, holdBefore: false, ch: 0,
  message: "",
  imgVersion: 0,             // bumps when a new photo is on the GPU
  undo: [], redo: [], last: "",
});

export const view = { img: null, sceneImg: null, main: null, thumb: null, thumbCanvas: null, srcPx: null };

export function initGL(canvas) {
  view.main = makeGL(canvas);
  view.thumbCanvas = document.createElement("canvas");
  view.thumb = makeGL(view.thumbCanvas);
  window.__studio = { S, effective, exportParams, draw, get main() { return view.main; } };   // automated tests
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

export function render() {
  if (!view.img || !S.params || !view.main) return;
  draw(view.main, effective(), S.params.knots, modeCode(), S.split);
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
  S.styles = await json("/api/styles");
  if ((!S.style || !S.styles.some((s) => s.name === S.style)) && S.styles.length) S.style = S.styles[0].name;
}
export async function loadPhotos() { S.photos = await json("/api/photos"); }

export async function upload(files) {
  const fd = new FormData(); [...files].forEach((f) => fd.append("files", f));
  const res = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  await loadPhotos(); if (res.length) await selectPhoto(res[0].id);
}

export async function selectPhoto(id) {
  S.id = id; view.srcPx = null; view.sceneImg = null; S.scene = ZERO_SCENE();
  const img = await loadImage(`/api/photo/${id}?edge=1600`);
  view.img = img;
  view.main.canvas.width = img.naturalWidth; view.main.canvas.height = img.naturalHeight;
  setImage(view.main, img);
  await predict();
  S.imgVersion++;
}

export async function predict() {
  if (!S.id || !S.style) return;
  S.params = await json("/api/predict", { id: S.id, style: S.style });
  S.d = ZERO_D(); S.knots = null; S.strength = S.params.strength ?? 1; S.undo = []; S.redo = []; S.last = snap();
}

export async function setStyle(name) { S.style = name; S.mode = "split"; await predict(); }

let sceneTimer = 0, sceneSeq = 0;
/** Debounced: ask the server for the photo with scene tools applied, then use it as the texture. */
export function applyScene() {
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
  const r = await api("/api/export", { id: S.id, params: exportParams(), format: fmt });
  await downloadBlob(r, `export.${fmt}`);
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

/** Render `params` on `img` into a small canvas (style strip, batch). */
export function renderThumb(img, params, w, target) {
  const h = Math.round((w * img.naturalHeight) / img.naturalWidth);
  view.thumbCanvas.width = w; view.thumbCanvas.height = h; setImage(view.thumb, img);
  const e = effectiveFrom(params, params.strength ?? 1, ZERO_D(), null);
  draw(view.thumb, e, params.knots, 2);
  target.width = w; target.height = h; target.getContext("2d").drawImage(view.thumbCanvas, 0, 0);
}

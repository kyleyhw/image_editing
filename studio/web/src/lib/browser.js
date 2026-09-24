// Browser-only backend (GitHub Pages build): the photostyle predictor, ported.
//   descriptor = 3 x 256 per-channel CDFs of the 512 px proxy + ResNet-18 pooled features
//                (proxy resized to a 224 px short side), as photostyle.features.FeatureExtractor;
//   head       = the style's MLP (Linear -> LayerNorm -> GELU -> Linear, shrinkage), in JS;
//   ood        = RMS z-score against the style's training features / its 95th percentile.
// Models come from tools/export_web.py. Photos stay in memory on this device.
import * as ort from "onnxruntime-web/wasm";
import { downloadBlob } from "./api.js";
import { bakeCube, draw, makeGL, setImage } from "./render.js";

const BASE = import.meta.env.BASE_URL;
ort.env.wasm.numThreads = 1;              // GitHub Pages is not cross-origin isolated

const PREDICT_EDGE = 512, SHORT_SIDE = 224, PREVIEW_EDGE = 1600, THUMB_EDGE = 320;

const f32 = (b64) => {
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
};

let session = null, modelReady = null, styles = [];
const photos = new Map();
let seq = 0;

async function fetchProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0, reader = r.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length; onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function canvasOf(src, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const x = c.getContext("2d", { willReadFrequently: true });
  x.imageSmoothingEnabled = true; x.imageSmoothingQuality = "high";
  x.drawImage(src, 0, 0, w, h);
  return c;
}
const fit = (w, h, edge) => { const s = Math.min(1, edge / Math.max(w, h)); return [Math.round(w * s), Math.round(h * s)]; };
const toURL = (c, q = 0.92) => new Promise((res) => c.toBlob((b) => res(URL.createObjectURL(b)), "image/jpeg", q));

function features(p) {
  p.featP ??= computeFeatures(p);
  return p.featP;
}

async function computeFeatures(p) {
  await modelReady;
  const [pw, ph] = fit(p.bitmap.width, p.bitmap.height, PREDICT_EDGE);
  const proxy = canvasOf(p.bitmap, pw, ph);
  const px = proxy.getContext("2d").getImageData(0, 0, pw, ph).data;
  const f = new Float32Array(1280), n = pw * ph;
  for (let c = 0; c < 3; c++) {                     // torch.histc(bins=256, 0..1), then cumsum / N
    const h = new Float64Array(256);
    for (let i = c; i < px.length; i += 4) h[Math.min(255, Math.floor((px[i] * 256) / 255))]++;
    let acc = 0;
    for (let b = 0; b < 256; b++) { acc += h[b]; f[c * 256 + b] = acc / n; }
  }
  const s = SHORT_SIDE / Math.min(pw, ph), rw = Math.max(1, Math.round(pw * s)), rh = Math.max(1, Math.round(ph * s));
  const small = canvasOf(proxy, rw, rh).getContext("2d").getImageData(0, 0, rw, rh).data;
  const x = new Float32Array(3 * rw * rh);
  for (let i = 0, j = 0; j < rw * rh; i += 4, j++) {
    x[j] = small[i] / 255; x[rw * rh + j] = small[i + 1] / 255; x[2 * rw * rh + j] = small[i + 2] / 255;
  }
  const out = await session.run({ x: new ort.Tensor("float32", x, [1, 3, rh, rw]) });
  f.set(out.f.data, 768);
  return f;
}

function erf(x) {                                   // Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7)
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

function head(st, f) {
  const H = st.head, D = f.length, z = new Float32Array(D);
  for (let i = 0; i < D; i++) z[i] = (f[i] - H.mu[i]) / H.sigma[i];
  const h = new Float64Array(H.hidden);
  for (let j = 0; j < H.hidden; j++) {
    let a = H.b1[j];
    const row = j * D;
    for (let i = 0; i < D; i++) a += H.w1[row + i] * z[i];
    h[j] = a;
  }
  let m = 0, v = 0;
  for (const a of h) m += a / H.hidden;
  for (const a of h) v += (a - m) ** 2 / H.hidden;
  for (let j = 0; j < H.hidden; j++) {
    const y = ((h[j] - m) / Math.sqrt(v + 1e-5)) * H.ln_w[j] + H.ln_b[j];
    h[j] = 0.5 * y * (1 + erf(y / Math.SQRT2));
  }
  const theta = [];
  for (let k = 0; k < H.out; k++) {
    let a = H.b2[k];
    for (let j = 0; j < H.hidden; j++) a += H.w2[k * H.hidden + j] * h[j];
    theta.push(H.theta_mean[k] + H.alpha * (a - H.theta_mean[k]));
  }
  let zz = 0;
  for (let i = 0; i < D; i++) zz += ((f[i] - st.ood.mean[i]) / st.ood.std[i]) ** 2;
  return { theta, ood: Math.sqrt(zz / D) / st.ood.ref };
}

export const browser = {
  kind: "browser",
  caps: { scene: false, batch: false, create: false, remember: false },
  formats: [["jpg", "JPEG"], ["png", "PNG"], ["cube", ".cube LUT"], ["json", "Params JSON"]],
  accept: "image/*",
  async init(onModel) {
    const data = await (await fetch(`${BASE}models/styles.json`)).json();
    styles = data.styles.map((s) => ({
      ...s,
      head: { ...s.head, ...Object.fromEntries(["mu", "sigma", "w1", "b1", "ln_w", "ln_b", "w2", "b2", "theta_mean"]
        .map((k) => [k, f32(s.head[k])])) },
      ood: { mean: f32(s.ood.mean), std: f32(s.ood.std), ref: s.ood.ref },
    }));
    modelReady = (async () => {
      const bytes = await fetchProgress(`${BASE}models/rn18.onnx`, (l, t) => onModel?.({ loaded: l, total: t, ready: false }));
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      onModel?.({ loaded: bytes.length, total: bytes.length, ready: true });
    })();
    modelReady.catch((e) => onModel?.({ error: String(e) }));
  },
  styles: async () => styles.map((s) => ({ ...s.card, attribution: s.attribution })),
  photos: async () => [...photos.values()].map(({ id, name }) => ({ id, name })),
  async upload(files) {
    const added = [];
    for (const file of files) {
      let bitmap;
      try { bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }); } catch { continue; }
      const id = `p${++seq}`;
      const preview = await toURL(canvasOf(bitmap, ...fit(bitmap.width, bitmap.height, PREVIEW_EDGE)));
      const thumb = await toURL(canvasOf(bitmap, ...fit(bitmap.width, bitmap.height, THUMB_EDGE)), 0.85);
      photos.set(id, { id, name: file.name, bitmap, preview, thumb, featP: null });
      added.push({ id, name: file.name });
    }
    return added;
  },
  async addSample(s) {
    const blob = await (await fetch(`${BASE}samples/${s.file}`)).blob();
    return this.upload([new File([blob], s.file, { type: "image/jpeg" })]);
  },
  photoURL: (id, edge) => (edge <= THUMB_EDGE ? photos.get(id)?.thumb : photos.get(id)?.preview),
  async predict(id, style) {
    const p = photos.get(id), st = styles.find((s) => s.card.name === style);
    const { theta, ood } = head(st, await features(p));
    return { style, renderer: st.card.renderer, knots: st.card.knots, theta,
             strength: st.card.default_strength ?? 1, ood_score: ood, version: 1 };
  },
  async export(id, params, eff, fmt) {
    const p = photos.get(id), stem = `${p.name.replace(/\.[^.]+$/, "")}_${params.style}`;
    if (fmt === "cube") {
      const blob = new Blob([bakeCube(eff, params.knots, 33, `photostyle ${params.style}`)], { type: "text/plain" });
      return save(blob, `${stem}.cube`);
    }
    if (fmt === "json") return save(new Blob([JSON.stringify(params, null, 1)], { type: "application/json" }), `${stem}.json`);
    const c = document.createElement("canvas"), R = makeGL(c);
    const max = Math.min(8192, R.gl.getParameter(R.gl.MAX_TEXTURE_SIZE));
    const [w, h] = fit(p.bitmap.width, p.bitmap.height, max);
    c.width = w; c.height = h;
    setImage(R, w === p.bitmap.width ? p.bitmap : canvasOf(p.bitmap, w, h));
    draw(R, eff, params.knots, 2);
    const type = fmt === "png" ? "image/png" : "image/jpeg";
    const blob = await new Promise((res) => c.toBlob(res, type, 0.95));
    R.gl.getExtension("WEBGL_lose_context")?.loseContext();
    return save(blob, `${stem}.${fmt === "png" ? "png" : "jpg"}`);
  },
};

function save(blob, name) {
  return downloadBlob({ headers: new Headers({ "content-disposition": `attachment; filename="${name}"` }), blob: async () => blob }, name);
}

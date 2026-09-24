// Where the model runs. Two backends with one interface:
//   server  `photostyle serve` (FastAPI): everything, incl. scene tools, batch, style training
//   browser the GitHub Pages build (VITE_STATIC=1): ResNet-18 via ONNX Runtime Web + the style
//           head in JS; photos never leave the device. No scene tools or training.
// The browser backend is imported only in the static build, so the server build stays small.
import { api, downloadBlob, json } from "./api.js";

export const STATIC = import.meta.env.VITE_STATIC === "1";

const server = {
  kind: "server",
  caps: { scene: true, batch: true, create: true, remember: true },
  formats: [["jpg", "JPEG"], ["cube", ".cube LUT"], ["xmp", "Lightroom XMP"], ["json", "Params JSON"]],
  accept: "image/*,.dng,.cr2,.cr3,.nef,.arw",
  async init() {},
  styles: () => json("/api/styles"),
  photos: () => json("/api/photos"),
  async upload(files) {
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    return (await fetch("/api/upload", { method: "POST", body: fd })).json();
  },
  async addSample(s) {
    const blob = await (await fetch(`${import.meta.env.BASE_URL}samples/${s.file}`)).blob();
    return this.upload([new File([blob], s.file, { type: "image/jpeg" })]);
  },
  photoURL: (id, edge) => `/api/photo/${id}?edge=${edge}`,
  predict: (id, style) => json("/api/predict", { id, style }),
  async export(id, params, _eff, fmt) {
    await downloadBlob(await api("/api/export", { id, params, format: fmt }), `export.${fmt}`);
  },
};

/** @type {any} */
export let B = server;

export async function initBackend(onModel) {
  if (STATIC) B = (await import("photostyle:browser-backend")).browser;
  await B.init(onModel);
  return B;
}

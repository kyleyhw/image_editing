// Raw files in the browser (Phase 22): LibRaw compiled to WebAssembly (libraw-wasm, ISC), loaded
// only when a raw file is opened. The decode keeps the data scene-referred:
//   camera white balance, camera colour matrix, output in linear Rec. 2020 (our working space),
//   16 bits per channel, no auto-brightening and no gamma.
// A baseline exposure then places the 99th-percentile luminance at 0.9, which leaves the brightest
// 1 % above 1.0 as recoverable headroom (highlights - brings it back), instead of clipping it as
// LibRaw's auto-brightening would.
import { SPEC } from "./pipeline.js";

export const RAW_EXT = /\.(dng|cr2|cr3|crw|nef|nrw|arw|srf|sr2|raf|orf|rw2|rwl|pef|srw|3fr|iiq|erf|kdc|dcr|mos|mef|mrw|x3f)$/i;
export const isRaw = (f) => RAW_EXT.test(f.name);

const LUMA = [0.2627, 0.678, 0.0593];          // Rec. 2020 luminance weights
const TARGET_P99 = 0.9;                        // see above
const PREVIEW_EDGE = 2048;                     // interactive texture; export decodes at full size

/** Decode a raw file. Returns { width, height, data16 (RGB Uint16), gain, meta }. */
export async function decodeRaw(file) {
  const { default: LibRaw } = await import("libraw-wasm");
  const raw = new LibRaw();
  try {
    await raw.open(new Uint8Array(await file.arrayBuffer()), {
      useCameraWb: true, useCameraMatrix: 1, outputColor: 8, outputBps: 16, gamm: [1, 1],
      noAutoBright: true, highlight: 2, userQual: 3,
    });
    const meta = await raw.metadata();
    const im = await raw.imageData();
    if (im.colors !== 3 || im.bits !== 16) throw new Error(`unexpected raw output: ${im.colors} colours, ${im.bits} bits`);
    const d = /** @type {Uint16Array} */ (im.data), n = im.width * im.height;
    // 99th percentile of luminance from a 1-in-16 sample (64-bin-per-stop histogram is overkill here)
    const ys = [];
    for (let i = 0; i < n; i += 16) ys.push((LUMA[0] * d[3 * i] + LUMA[1] * d[3 * i + 1] + LUMA[2] * d[3 * i + 2]) / 65535);
    ys.sort((a, b) => a - b);
    const p99 = ys[Math.floor(ys.length * 0.99)] || 1;
    const gain = Math.min(16, Math.max(1, TARGET_P99 / p99));
    return { width: im.width, height: im.height, data16: d, gain, meta };
  } finally { raw.dispose(); }
}

/** Linear Rec. 2020 RGBA float data at most ``edge`` px on the long side (box filter). */
export function linearRGBA(r, edge = PREVIEW_EDGE) {
  const s = Math.min(1, edge / Math.max(r.width, r.height)), w = Math.max(1, Math.round(r.width * s)), h = Math.max(1, Math.round(r.height * s));
  const out = new Float32Array(4 * w * h), d = r.data16, k = r.gain / 65535;
  const fx = r.width / w, fy = r.height / h;
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(y * fy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * fy));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(x * fx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * fx));
      let a = 0, b = 0, c = 0, m = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) { const i = 3 * (yy * r.width + xx); a += d[i]; b += d[i + 1]; c += d[i + 2]; m++; }
      const o = 4 * (y * w + x);
      out[o] = (a / m) * k; out[o + 1] = (b / m) * k; out[o + 2] = (c / m) * k; out[o + 3] = 1;
    }
  }
  return { data: out, width: w, height: h, linear: true };
}

/** A display (sRGB, 8-bit) canvas of linear Rec. 2020 data: for thumbnails, features, the picker. */
export function displayCanvas(lin) {
  const m = SPEC.m.w_to_srgb, M = [m.slice(0, 3), m.slice(3, 6), m.slice(6, 9)];   // linear Rec. 2020 -> linear sRGB
  const enc = (v) => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  const c = document.createElement("canvas"); c.width = lin.width; c.height = lin.height;
  const x = c.getContext("2d"), img = x.createImageData(lin.width, lin.height), d = lin.data;
  for (let i = 0; i < d.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      const v = M[k][0] * d[i] + M[k][1] * d[i + 1] + M[k][2] * d[i + 2];
      img.data[i + k] = 255 * enc(Math.min(1, Math.max(0, v))) + 0.5;
    }
    img.data[i + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  return c;
}

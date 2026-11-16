// Export (Phase 22): encode rendered pixels as JPEG / PNG (8-bit, via the browser's encoder, with
// the colour space tagged), 16-bit PNG or 16-bit TIFF (encoded here), and write a Lightroom preset
// (.xmp) of the slider part of an edit.
import { DEFAULTS, BANDS, curvePoints, CANVAS_AREA_LIMIT } from "./pipeline.js";

/** 8-bit JPEG or PNG via a 2D canvas in the output colour space (the browser embeds its profile). */
export async function encode8(px, { type = "image/jpeg", quality = 0.92, space = "srgb", exif = null } = {}) {
  if (px.width * px.height > CANVAS_AREA_LIMIT) return encodeLarge8(px, { type, quality, exif });
  const c = document.createElement("canvas");
  c.width = px.width; c.height = px.height;
  const opts = space === "p3" ? { colorSpace: "display-p3" } : {};
  const x = /** @type {CanvasRenderingContext2D} */ (c.getContext("2d", opts));
  x.putImageData(new ImageData(px.data, px.width, px.height, /** @type {any} */ (opts)), 0, 0);
  let blob = await new Promise((res) => c.toBlob(res, type, quality));
  if (exif && type === "image/jpeg") blob = await withExif(blob, exif);
  return blob;
}

/** 8-bit export above the canvas area limit (iOS Safari refuses larger canvases): PNG through the
 * PNG writer below, JPEG through jpeg-js (BSD-3-Clause, loaded only here). Both are untagged sRGB,
 * so callers render these in sRGB (Display P3 needs the canvas path or a 16-bit PNG). */
async function encodeLarge8(px, { type, quality, exif }) {
  if (type === "image/png") return encodePNG(px, "srgb", 8);
  const { encode } = await import("jpeg-js");
  const g = /** @type {any} */ (globalThis), had = "Buffer" in g;
  if (!had) g.Buffer = { from: (a) => Uint8Array.from(a) };   // jpeg-js returns Buffer.from(bytes) when bundled as CommonJS
  try {
    const out = encode({ data: px.data, width: px.width, height: px.height }, Math.round(quality * 100));
    const blob = new Blob([out.data], { type: "image/jpeg" });
    return exif ? withExif(blob, exif) : blob;
  } finally { if (!had) delete g.Buffer; }
}

/** Insert an EXIF APP1 segment (orientation reset to 1: the pixels are already upright). */
async function withExif(blob, app1) {
  const seg = app1.slice();
  resetOrientation(seg);
  const b = new Uint8Array(await blob.arrayBuffer());
  let i = 2;                                       // after SOI; keep a JFIF APP0 first if present
  if (b[2] === 0xff && b[3] === 0xe0) i = 4 + ((b[4] << 8) | b[5]);
  return new Blob([b.subarray(0, i), seg, b.subarray(i)], { type: "image/jpeg" });
}
function resetOrientation(seg) {
  const t = 10, le = seg[t] === 0x49;               // TIFF header after "Exif\0\0" (APP1 marker 2 + length 2 + 6)
  const u16 = (o) => (le ? seg[o] | (seg[o + 1] << 8) : (seg[o] << 8) | seg[o + 1]);
  const ifd = t + (le ? seg[t + 4] | (seg[t + 5] << 8) | (seg[t + 6] << 16) | (seg[t + 7] << 24)
                      : (seg[t + 4] << 24) | (seg[t + 5] << 16) | (seg[t + 6] << 8) | seg[t + 7]);
  if (ifd + 2 > seg.length) return;
  const n = u16(ifd);
  for (let k = 0; k < n; k++) {
    const e = ifd + 2 + 12 * k;
    if (e + 12 > seg.length) return;
    if (u16(e) === 0x0112) { if (le) { seg[e + 8] = 1; seg[e + 9] = 0; } else { seg[e + 8] = 0; seg[e + 9] = 1; } return; }
  }
}

const to16 = (v) => Math.max(0, Math.min(65535, Math.round(v * 65535)));
/** Channel value reader: pixels are float RGBA (display-encoded, 0-1), 16-bit RGB (Uint16Array,
 * ``channels`` 3, from tiled export) or 8-bit RGBA. Returns a 16-bit or 8-bit code value. */
function reader(px, bits) {
  const { data } = px, ch = px.channels ?? 4;
  if (data instanceof Uint16Array) return bits === 16 ? (i, c) => data[ch * i + c] : (i, c) => data[ch * i + c] >> 8;
  if (data instanceof Float32Array) return bits === 16 ? (i, c) => to16(data[ch * i + c]) : (i, c) => to16(data[ch * i + c]) >> 8;
  return bits === 16 ? (i, c) => data[ch * i + c] * 257 : (i, c) => data[ch * i + c];
}

/** 16-bit RGB PNG (zlib via CompressionStream). sRGB is tagged with an sRGB chunk, Display P3 with
 * a cICP chunk (primaries 12 = P3 D65, transfer 13 = sRGB, full range). */
export const encodePNG16 = (px, space = "srgb") => encodePNG(px, space, 16);

/** RGB PNG of 8 or 16 bits per channel, written here (no canvas, so no size limit but memory). */
export async function encodePNG(px, space = "srgb", bits = 16) {
  const { width: w, height: h } = px, get = reader(px, bits), B = bits / 8;
  const raw = new Uint8Array(h * (1 + 3 * B * w));
  for (let y = 0, o = 0; y < h; y++) {
    raw[o++] = 0;                                   // filter: none
    for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      const v = get(y * w + x, c);
      if (B === 2) { raw[o++] = v >> 8; raw[o++] = v & 255; } else raw[o++] = v;
    }
  }
  const z = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer());
  const ihdr = new Uint8Array(13); const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = bits; ihdr[9] = 2;
  const colour = space === "p3" ? chunk("cICP", new Uint8Array([12, 13, 0, 1])) : chunk("sRGB", new Uint8Array([0]));
  return new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), colour, chunk("IDAT", z), chunk("IEND", new Uint8Array(0))], { type: "image/png" });
}
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function chunk(type, body) {
  const out = new Uint8Array(12 + body.length), dv = new DataView(out.buffer);
  dv.setUint32(0, body.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(body, 8);
  let c = 0xffffffff;
  for (let i = 4; i < 8 + body.length; i++) c = CRC[(c ^ out[i]) & 255] ^ (c >>> 8);
  dv.setUint32(8 + body.length, (c ^ 0xffffffff) >>> 0);
  return out;
}

/** 16-bit RGB TIFF, uncompressed, little-endian (baseline TIFF 6.0; colour space untagged = sRGB). */
export function encodeTIFF16(px) {
  const { width: w, height: h } = px, get = reader(px, 16), nTags = 11, ifdOff = 8, bpsOff = ifdOff + 2 + 12 * nTags + 4, imgOff = bpsOff + 6;
  const buf = new ArrayBuffer(imgOff + 6 * w * h), dv = new DataView(buf);
  dv.setUint16(0, 0x4949, true); dv.setUint16(2, 42, true); dv.setUint32(4, ifdOff, true);
  dv.setUint16(ifdOff, nTags, true);
  const tags = [[256, 4, 1, w], [257, 4, 1, h], [258, 3, 3, bpsOff], [259, 3, 1, 1], [262, 3, 1, 2], [273, 4, 1, imgOff],
                [277, 3, 1, 3], [278, 4, 1, h], [279, 4, 1, 6 * w * h], [284, 3, 1, 1], [339, 3, 1, 1]];
  tags.forEach(([tag, type, count, v], k) => {
    const e = ifdOff + 2 + 12 * k; dv.setUint16(e, tag, true); dv.setUint16(e + 2, type, true); dv.setUint32(e + 4, count, true);
    if (type === 3 && count === 1) dv.setUint16(e + 8, v, true); else dv.setUint32(e + 8, v, true);
  });
  dv.setUint32(ifdOff + 2 + 12 * nTags, 0, true);
  for (let i = 0; i < 3; i++) dv.setUint16(bpsOff + 2 * i, 16, true);
  for (let i = 0, o = imgOff; i < w * h; i++) for (let c = 0; c < 3; c++, o += 2) dv.setUint16(o, get(i, c), true);
  return new Blob([buf], { type: "image/tiff" });
}

// ------------------------------------------------------------------ Lightroom preset
const LR_BANDS = { red: "Red", orange: "Orange", yellow: "Yellow", green: "Green", aqua: "Aqua", blue: "Blue", purple: "Purple", magenta: "Magenta" };
const LR = {
  temp: "IncrementalTemperature", tint: "IncrementalTint", exposure: "Exposure2012", contrast: "Contrast2012",
  highlights: "Highlights2012", shadows: "Shadows2012", whites: "Whites2012", blacks: "Blacks2012",
  vibrance: "Vibrance", saturation: "Saturation",
  grade_shadows_hue: "ColorGradeShadowHue", grade_shadows_sat: "ColorGradeShadowSat", grade_shadows_lum: "ColorGradeShadowLum",
  grade_midtones_hue: "ColorGradeMidtoneHue", grade_midtones_sat: "ColorGradeMidtoneSat", grade_midtones_lum: "ColorGradeMidtoneLum",
  grade_highlights_hue: "ColorGradeHighlightHue", grade_highlights_sat: "ColorGradeHighlightSat", grade_highlights_lum: "ColorGradeHighlightLum",
  grade_global_hue: "ColorGradeGlobalHue", grade_global_sat: "ColorGradeGlobalSat", grade_global_lum: "ColorGradeGlobalLum",
  grade_balance: "SplitToningBalance", grade_blending: "ColorGradeBlending",
  cal_red_hue: "RedHue", cal_red_sat: "RedSaturation", cal_green_hue: "GreenHue", cal_green_sat: "GreenSaturation",
  cal_blue_hue: "BlueHue", cal_blue_sat: "BlueSaturation", cal_shadow_tint: "ShadowTint",
  vig_amount: "PostCropVignetteAmount", vig_midpoint: "PostCropVignetteMidpoint", vig_feather: "PostCropVignetteFeather",
  vig_roundness: "PostCropVignetteRoundness",
};
for (const b of BANDS) {
  LR[`hsl_hue_${b}`] = `HueAdjustment${LR_BANDS[b]}`; LR[`hsl_sat_${b}`] = `SaturationAdjustment${LR_BANDS[b]}`;
  LR[`hsl_lum_${b}`] = `LuminanceAdjustment${LR_BANDS[b]}`; LR[`bw_${b}`] = `GrayMixer${LR_BANDS[b]}`;
}

/** A Lightroom / Camera Raw preset (.xmp, process version 2012+) with the edit's sliders, curves,
 * crop and black-and-white setting. Lightroom applies its own maths to these numbers, so the result
 * is similar, not identical (A6.6: our slider scales are not calibrated against Lightroom). */
export function lightroomXMP(p, name = "Irodori") {
  const attrs = { PresetType: "Normal", Cluster: "", UUID: crypto.randomUUID?.().replace(/-/g, "").toUpperCase() ?? "",
                  SupportsAmount: "False", SupportsColor: "True", SupportsMonochrome: "True", SupportsHighDynamicRange: "True",
                  SupportsNormalDynamicRange: "True", SupportsSceneReferred: "True", SupportsOutputReferred: "True",
                  CameraModelRestriction: "", Copyright: "", ContactInfo: "", Version: "15.0", ProcessVersion: "11.0" };
  const fmt = (k, v) => (k === "exposure" ? (v >= 0 ? "+" : "") + v.toFixed(2) : (v > 0 && !k.endsWith("_hue") && !k.startsWith("vig_m") && !k.startsWith("vig_f") && k !== "grade_blending" ? "+" : "") + Math.round(v));
  for (const [k, lr] of Object.entries(LR)) {
    const v = p[k] ?? DEFAULTS[k];
    if (v !== DEFAULTS[k] || k === "grade_blending") attrs[lr] = fmt(k, v);
  }
  if ((p.bw ?? 0) > 0.5) attrs.ConvertToGrayscale = "True";
  const crop = p.crop;
  if (crop) Object.assign(attrs, { HasCrop: "True", CropLeft: crop.x.toFixed(6), CropTop: crop.y.toFixed(6),
                                   CropRight: (crop.x + crop.w).toFixed(6), CropBottom: (crop.y + crop.h).toFixed(6), CropAngle: (crop.angle ?? 0).toFixed(2) });
  const seq = (pts) => `<rdf:Seq>${pts.map(([x, y]) => `<rdf:li>${Math.round(x * 255)}, ${Math.round(y * 255)}</rdf:li>`).join("")}</rdf:Seq>`;
  const curves = [["rgb", "ToneCurvePV2012"], ["r", "ToneCurvePV2012Red"], ["g", "ToneCurvePV2012Green"], ["b", "ToneCurvePV2012Blue"]]
    .map(([k, tag]) => { const pts = curvePoints(p.curves, k); return pts ? `<crs:${tag}>${seq(pts)}</crs:${tag}>` : ""; }).join("\n    ");
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  return `<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Irodori">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    ${Object.entries(attrs).map(([k, v]) => `crs:${k}="${esc(v)}"`).join("\n    ")}>
    <crs:Name><rdf:Alt><rdf:li xml:lang="x-default">${esc(name)}</rdf:li></rdf:Alt></crs:Name>
    ${curves}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
`;
}

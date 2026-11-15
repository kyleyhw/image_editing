// Irodori's image pipeline in WebGL2: a port of photostyle/pipeline.py (the reference), tested
// against it by tools/check_pipeline.py. Every constant comes from pipeline_spec.json, written by
// `python -m photostyle.pipeline`. One fragment shader does the whole pipeline per pixel in 32-bit
// float; the only 8-bit steps are the input texture and the canvas, and the output is dithered.
//
//   geometry -> [legacy look] -> linear Rec. 2020 -> white balance -> exposure -> tone (luminance,
//   hue-preserving) -> point curves -> colour in Oklab -> vignette -> sRGB, gamut clip, encode.
import SPEC from "./pipeline_spec.json";

export { SPEC };
export const SLIDERS = SPEC.sliders;
export const DEFAULTS = Object.fromEntries(SLIDERS.map((s) => [s.name, s.default]));
export const BANDS = SPEC.bands;
const K = SPEC.k;
const CURVE_PTS = 16;                          // maximum points per point curve (uniform budget)

// ------------------------------------------------------------------ CPU side: sliders -> uniforms
const get = (p, k) => (p?.[k] ?? DEFAULTS[k] ?? 0);

function mat3(a) { return [a.slice(0, 3), a.slice(3, 6), a.slice(6, 9)]; }
const M = Object.fromEntries(Object.entries(SPEC.m).map(([k, v]) => [k, v.length === 9 ? mat3(v) : v]));
const mulv = (m, v) => m.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);

function planckXY(kelvin) {                    // Kim et al. 2002, as pipeline.planck_xy
  const t = Math.min(25000, Math.max(1667, kelvin)), i = 1e3 / t;
  const x = t <= 4000 ? -0.2661239 * i ** 3 - 0.2343589 * i ** 2 + 0.8776956 * i + 0.17991
                      : -3.0258469 * i ** 3 + 2.1070379 * i ** 2 + 0.2226347 * i + 0.24039;
  const y = t <= 2222 ? -1.1063814 * x ** 3 - 1.3481102 * x ** 2 + 2.18555832 * x - 0.20219683
          : t <= 4000 ? -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
                      : 3.081758 * x ** 3 - 5.8733867 * x ** 2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}
const XYZ_TO_W = (() => {                       // inverse of w_to_xyz
  const a = M.w_to_xyz, [[p, q, r], [s, t, u], [v, w, x]] = a;
  const det = p * (t * x - u * w) - q * (s * x - u * v) + r * (s * w - t * v);
  return [[(t * x - u * w) / det, (r * w - q * x) / det, (q * u - r * t) / det],
          [(u * v - s * x) / det, (p * x - r * v) / det, (r * s - p * u) / det],
          [(s * w - t * v) / det, (q * v - p * w) / det, (p * t - q * s) / det]];
})();
function whiteW(kelvin) { const [x, y] = planckXY(kelvin); return mulv(XYZ_TO_W, [x / y, 1, (1 - x - y) / y]); }

/** White-balance gains, as pipeline.wb_gains. */
export function wbGains(temp, tint) {
  const mired = 1e6 / SPEC.wb.ref_k - SPEC.wb.mired_per_unit * temp;
  const ref = whiteW(SPEC.wb.ref_k), src = whiteW(1e6 / mired);
  const g = ref.map((v, i) => v / src[i]);
  g[1] *= 2 ** (-SPEC.wb.tint_stops * tint / 100);
  const y = g[0] * M.luma[0] + g[1] * M.luma[1] + g[2] * M.luma[2];
  return g.map((v) => v / y);
}

/** Fritsch-Carlson slopes, as pipeline.pchip. */
export function pchipSlopes(xs, ys) {
  const n = xs.length, h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h.push(xs[i + 1] - xs[i]); d.push((ys[i + 1] - ys[i]) / h[i]); }
  const m = [d[0]];
  for (let i = 1; i < n - 1; i++) {
    const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1];
    m.push(d[i - 1] * d[i] > 0 ? (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]) : 0);
  }
  m.push(d[n - 2]);
  return m;
}
/** Evaluate PCHIP on the CPU (curve editor drawing); linear beyond the ends, as the reference. */
export function pchipEval(xs, ys, m, t) {
  const n = xs.length;
  let i = 0; while (i < n - 2 && t > xs[i + 1]) i++;
  const h = xs[i + 1] - xs[i], u = (t - xs[i]) / h;
  if (u < 0) return ys[i] + (t - xs[i]) * m[i];
  if (u > 1) return ys[i + 1] + (t - xs[i + 1]) * m[i + 1];
  return (2 * u ** 3 - 3 * u ** 2 + 1) * ys[i] + (u ** 3 - 2 * u ** 2 + u) * h * m[i]
       + (-2 * u ** 3 + 3 * u ** 2) * ys[i + 1] + (u ** 3 - u ** 2) * h * m[i + 1];
}

export function tonePoints(p) {
  const T = SPEC.tone, y = T.x.slice();
  for (const [k, row] of Object.entries(T.rows)) { const v = get(p, k) / 100; row.forEach((r, i) => (y[i] += v * r)); }
  y[y.length - 1] -= T.recovery * Math.max(0, -get(p, "highlights") / 100);
  for (let i = 1; i < y.length; i++) y[i] = Math.max(y[i], y[i - 1] + T.mono_eps);
  return y;
}

/** Control points of a point curve, or null for the identity (as pipeline.curve_points). */
export function curvePoints(curves, key) {
  const pts = curves?.[key];
  if (!pts || pts.length < 2) return null;
  const s = pts.map(([a, b]) => [+a, +b]).sort((u, v) => u[0] - v[0]);
  if (s.length === 2 && s[0][0] === 0 && s[0][1] === 0 && s[1][0] === 1 && s[1][1] === 1) return null;
  return s;
}

function bandGeometry() {
  const c = SPEC.band_oklab.map((d) => (d * Math.PI) / 180), tau = 2 * Math.PI, mod = (v) => ((v % tau) + tau) % tau;
  return c.map((ci, k) => [ci, mod(ci - c[(k + 7) % 8]), mod(c[(k + 1) % 8] - ci)]);
}
const BAND_GEO = bandGeometry();

/** Unit Oklab (a, b) of a colour-wheel hue in HSV degrees, as pipeline._grade_dir. */
export function gradeDir(hueDeg) {
  const anch = [...SPEC.band_hsv, 360], ok = [...SPEC.band_oklab, SPEC.band_oklab[0] + 360].map((d) => (d * Math.PI) / 180);
  const hd = ((hueDeg % 360) + 360) % 360;
  let i = 0; while (i < 7 && hd > anch[i + 1]) i++;
  const u = (hd - anch[i]) / (anch[i + 1] - anch[i]), a = ok[i] + u * (ok[i + 1] - ok[i]);
  return [Math.cos(a), Math.sin(a)];
}

function calMatrix(p) {
  const r = get(p, "cal_red_hue") / 100, g = get(p, "cal_green_hue") / 100, b = get(p, "cal_blue_hue") / 100;
  const sr = 1 + get(p, "cal_red_sat") / 100, sg = 1 + get(p, "cal_green_sat") / 100, sb = 1 + get(p, "cal_blue_sat") / 100;
  const k = K.cal_shift, P = (v) => Math.max(0, v);
  const m = [[1, k * P(-g), k * P(b)], [k * P(r), 1, k * P(-b)], [k * P(-r), k * P(g), 1]];
  const on = [r, g, b, sr - 1, sg - 1, sb - 1].some((v) => v !== 0);
  return { on, m: m.map((row) => [row[0] * sr, row[1] * sg, row[2] * sb]) };
}

/** 3x3 geometry matrix (output uv -> source uv), as pipeline.geometry_matrix. */
export function geometryMatrix(W, H, crop) {
  const c = { x: 0, y: 0, w: 1, h: 1, angle: 0, ...(crop || {}) }, th = (c.angle * Math.PI) / 180;
  const mm = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const toPx = [[c.w * W, 0, c.x * W], [0, c.h * H, c.y * H], [0, 0, 1]];
  const centre = [[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]];
  const rot = [[Math.cos(th), -Math.sin(th), 0], [Math.sin(th), Math.cos(th), 0], [0, 0, 1]];
  const back = [[1 / W, 0, 0.5], [0, 1 / H, 0.5], [0, 0, 1]];
  return mm(back, mm(rot, mm(centre, toPx)));
}
export const outputSize = (W, H, crop) => [Math.max(1, Math.round((crop?.w ?? 1) * W)), Math.max(1, Math.round((crop?.h ?? 1) * H))];

/** Largest centred crop scale that keeps the corners inside a frame rotated by ``angle``. */
export function straightenScale(W, H, angle) {
  const t = Math.abs((angle * Math.PI) / 180), c = Math.cos(t), s = Math.sin(t);
  return Math.min(W / (W * c + H * s), H / (W * s + H * c));
}

// ------------------------------------------------------------------ shader
const VS = `#version 300 es
in vec2 pos; out vec2 uv;
void main() { uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5); gl_Position = vec4(pos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D img; uniform sampler2D orig;
uniform int mode; uniform float split; uniform mat3 geo; uniform int clipWarn; uniform float dither;
uniform int lookOn; uniform vec4 lk[13]; uniform int K; uniform mat3 lM; uniform vec3 lBias; uniform float lVig;
uniform mat3 srgbToW, wToSrgb, wToLms, lmsToW, lmsToLab, labToLms; uniform vec3 luma, srgbY;
uniform vec3 wb; uniform float expo; uniform vec4 tone[8];
uniform vec4 cp[${4 * CURVE_PTS}]; uniform ivec4 cn;
uniform int calOn; uniform mat3 calM; uniform float shadowTint;
uniform float sat, vib; uniform vec4 band[8]; uniform vec4 hsl[8]; uniform int bwOn;
uniform vec4 grade[4]; uniform vec2 gradeBal; uniform int colourOn;
uniform vec4 vig; uniform float vigRound; uniform vec2 frame;
uniform vec4 kc; uniform vec4 kc2; uniform vec4 kc3;
in vec2 uv; out vec4 o;

float knot(int i) { return lk[i / 4][i % 4]; }
float lcurve(int ch, float x) {
  float t = clamp(x, 0.0, 1.0) * float(K - 1); int i = min(int(floor(t)), K - 2); float u = t - float(i);
  float a = knot(ch * K + i), b = knot(ch * K + i + 1); return a + u * (b - a);
}
float dec(float x) { return x <= 0.04045 ? x / 12.92 : pow((max(x, 0.04045) + 0.055) / 1.055, 2.4); }
float enc(float y) { y = max(y, 0.0); return y <= 0.0031308 ? 12.92 * y : 1.055 * pow(max(y, 0.0031308), 1.0 / 2.4) - 0.055; }
vec3 dec3(vec3 x) { return vec3(dec(x.r), dec(x.g), dec(x.b)); }
vec3 enc3(vec3 y) { return vec3(enc(y.r), enc(y.g), enc(y.b)); }
float hermite(vec4 a, vec4 b, float t) {          // a, b = (x, y, slope, -) of a segment's ends
  float h = b.x - a.x, u = (t - a.x) / h;
  if (u < 0.0) return a.y + (t - a.x) * a.z;
  if (u > 1.0) return b.y + (t - b.x) * b.z;
  return (2.0*u*u*u - 3.0*u*u + 1.0) * a.y + (u*u*u - 2.0*u*u + u) * h * a.z
       + (-2.0*u*u*u + 3.0*u*u) * b.y + (u*u*u - u*u) * h * b.z;
}
float toneEval(float t) {
  int i = 0; for (int k = 0; k < 6; k++) { if (t > tone[k + 1].x) i = k + 1; }
  return hermite(tone[i], tone[i + 1], t);
}
float curveEval(int c, int n, float t) {          // flat beyond the end points, + (t - 1) above 1
  int off = c * ${CURVE_PTS};
  float x0 = cp[off].x, xn = cp[off + n - 1].x, tc = clamp(t, x0, xn);
  int i = 0; for (int k = 0; k < ${CURVE_PTS - 2}; k++) { if (k + 1 < n - 1 && tc > cp[off + k + 1].x) i = k + 1; }
  float v = hermite(cp[off + i], cp[off + i + 1], tc);
  return t > 1.0 ? v + (t - 1.0) : v;
}
vec3 cbrt3(vec3 v) { return sign(v) * pow(abs(v) + 1e-30, vec3(1.0 / 3.0)); }
float smooth01(float x) { x = clamp(x, 0.0, 1.0); return x * x * (3.0 - 2.0 * x); }
float bandW(int k, float h) {
  vec4 b = band[k];
  float d = mod(h - b.x + 3.14159265358979, 6.28318530717959) - 3.14159265358979;
  float t = d < 0.0 ? -d / b.y : d / b.z;
  return t < 1.0 ? pow(cos(clamp(t, 0.0, 1.0) * 1.5707963267949), 2.0) : 0.0;
}
float rnd(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec2 suv = (geo * vec3(uv, 1.0)).xy;
  bool inside = all(greaterThanEqual(suv, vec2(0.0))) && all(lessThanEqual(suv, vec2(1.0)));
  vec3 x = inside ? texture(img, suv).rgb : vec3(0.0);
  if (mode == 1 || (mode == 0 && uv.x < split)) { o = vec4(inside ? texture(orig, suv).rgb : vec3(0.0), 1.0); return; }

  if (lookOn == 1) {                                // the Phase-A5 look, in display space
    vec3 e = vec3(lcurve(0, x.r), lcurve(1, x.g), lcurve(2, x.b));
    e = lM * e + lBias;
    vec2 p = suv * 2.0 - 1.0;
    e *= clamp(1.0 - lVig * dot(p, p) / 2.0, 0.0, 1.0);
    x = clamp(e, 0.0, 1.0);
  }
  vec3 c = srgbToW * dec3(clamp(x, 0.0, 1.0));
  c *= wb * expo;

  // tone: curve on perceptual luminance; colour scaled by the luminance ratio
  float Y = dot(luma, c), Lp = toneEval(enc(Y)), Yp = dec(max(Lp, 0.0));
  float kr = Yp / max(Y, kc.x);
  c = kr * c + (Yp - kr * Y);

  if (cn.x + cn.y + cn.z + cn.w > 0) {
    vec3 e = enc3(c);
    if (cn.x > 0) e = vec3(curveEval(0, cn.x, e.r), curveEval(0, cn.x, e.g), curveEval(0, cn.x, e.b));
    if (cn.y > 0) e.r = curveEval(1, cn.y, e.r);
    if (cn.z > 0) e.g = curveEval(2, cn.z, e.g);
    if (cn.w > 0) e.b = curveEval(3, cn.w, e.b);
    c = dec3(e);
  }

  if (colourOn == 1) {
    if (calOn == 1) { float Yc = dot(luma, c); c = Yc + calM * (c - Yc); }
    vec3 lab = lmsToLab * cbrt3(wToLms * c);
    float L = lab.x, C = sqrt(lab.y * lab.y + lab.z * lab.z + 1e-12), h = atan(lab.z, lab.y);
    float cw = smooth01(C / kc.y);
    float dh = 0.0, ds = 0.0, dl = 0.0, mix = 0.0;
    for (int k = 0; k < 8; k++) { float w = bandW(k, h); dh += w * hsl[k].x; ds += w * hsl[k].y; dl += w * hsl[k].z; mix += w * band[k].w; }
    float vw = 1.0 - clamp(C / kc.z, 0.0, 1.0);
    C *= max(1.0 + sat, 0.0) * max(1.0 + vib * vw * vw, 0.0);
    C *= max(1.0 + ds, 0.0);
    h += radians(dh * kc.w) * cw;
    L *= 1.0 + dl * kc2.x * cw;
    if (bwOn == 1) { L *= 1.0 + mix * kc2.y * cw; C = 0.0; }
    float a = C * cos(h), b = C * sin(h);
    float pL = clamp(L + gradeBal.x, 0.0, 1.0), ex = 2.0 / gradeBal.y;
    float ws = pow(1.0 - pL, ex), wh = pow(pL, ex), wm = max(1.0 - ws - wh, 0.0);
    float wr[4] = float[4](ws, wm, wh, 1.0);
    for (int r = 0; r < 4; r++) { a += wr[r] * grade[r].x; b += wr[r] * grade[r].y; L += wr[r] * grade[r].z; }
    a += pow(clamp(1.0 - L, 0.0, 1.0), 2.0) * shadowTint;
    vec3 lms = labToLms * vec3(L, a, b);
    c = lmsToW * (lms * lms * lms);
  }

  if (vig.x != 0.0) {                               // exposure-like vignette, as pipeline.vignette_mask
    vec2 q = uv * 2.0 - 1.0; float asp = frame.x / frame.y, rn = vigRound;
    float sx = 1.0 + max(rn, 0.0) * (asp - 1.0), pw = 2.0 * pow(2.0, -min(rn, 0.0) * 1.5);
    float r = pow((pow(abs(q.x * sx), pw) + pow(abs(q.y), pw)) / (pow(sx, pw) + 1.0), 1.0 / pw);
    float m0 = clamp(vig.y, 0.01, 1.0), f = clamp(vig.z, 0.01, 1.0);
    float lo = m0 * (1.0 - f), hi = m0 + (1.0 - m0) * f;
    c *= pow(2.0, vig.x * kc2.z * smooth01((r - lo) / (hi - lo)));
  }

  vec3 s = wToSrgb * c;
  bool over = any(greaterThan(s, vec3(1.0 + 1e-4))), under = any(lessThan(s, vec3(-1e-4)));
  float Ys = clamp(dot(srgbY, s), 0.0, 1.0); vec3 d = s - Ys; float t = 1.0;
  for (int i = 0; i < 3; i++) {
    if (d[i] > 1e-9) t = min(t, (1.0 - Ys) / d[i]); else if (d[i] < -1e-9) t = min(t, -Ys / d[i]);
  }
  vec3 outc = enc3(clamp(Ys + max(t, 0.0) * d, 0.0, 1.0));
  if (clipWarn == 1) { if (dot(srgbY, s) > 1.0) outc = vec3(1.0, 0.15, 0.15); else if (dot(srgbY, s) <= 0.0) outc = vec3(0.15, 0.35, 1.0); }
  else if (clipWarn == 2 && (over || under)) outc = vec3(1.0, 0.0, 1.0);
  outc += dither * (rnd(gl_FragCoord.xy) - rnd(gl_FragCoord.yx + 17.0)) / 255.0;
  o = vec4(outc, 1.0);
}`;

export function makeGL(canvas) {
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, premultipliedAlpha: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");
  const sh = (t, s) => {
    const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x));
    return x;
  };
  const pr = gl.createProgram();
  gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
  if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
  gl.useProgram(pr);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(pr, "pos"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture(), origTex = gl.createTexture(), cache = {};
  const u = (n) => (n in cache ? cache[n] : (cache[n] = gl.getUniformLocation(pr, n)));
  gl.uniform1i(u("img"), 0); gl.uniform1i(u("orig"), 1);
  const R = { gl, tex, origTex, u, canvas, src: [1, 1] };
  for (const [name, key] of [["srgbToW", "srgb_to_w"], ["wToSrgb", "w_to_srgb"], ["wToLms", "w_to_lms"], ["lmsToW", "lms_to_w"],
                             ["lmsToLab", "lms_to_lab"], ["labToLms", "lab_to_lms"]]) {
    gl.uniformMatrix3fv(u(name), true, new Float32Array(SPEC.m[key]));
  }
  gl.uniform3fv(u("luma"), new Float32Array(SPEC.m.luma)); gl.uniform3fv(u("srgbY"), new Float32Array(SPEC.m.srgb_y));
  gl.uniform4fv(u("band"), new Float32Array(BAND_GEO.flatMap((g) => [...g, 0])));
  return R;
}

/** Upload the image; with ``asOriginal`` (default) also as the "before" texture. ``img`` may be an
 * image, canvas or bitmap (8-bit), or {data: Float32Array RGBA, width, height} (16/32-bit sources). */
export function setImage(R, img, asOriginal = true) {
  upload(R, R.tex, 0, img);
  if (asOriginal) upload(R, R.origTex, 1, img);
  R.src = [img.width ?? img.naturalWidth, img.height ?? img.naturalHeight];
}

function upload(R, tex, unit, img) {
  const { gl } = R;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  if (img.data instanceof Float32Array) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, img.width, img.height, 0, gl.RGBA, gl.FLOAT, img.data);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  }
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) {
    gl.texParameteri(gl.TEXTURE_2D, k, v);
  }
}

/**
 * Draw. ``e`` = { p: slider dict (+ curves, crop), look: legacy look {knots, dM, bias, vig} | null,
 * K: knots of the legacy look }. ``opts``: { mode: 0 split | 1 before | 2 after, split, clipWarn: 0 | 1
 * luminance | 2 gamut, dither (code values, default 1) }. The canvas must already have the output size.
 */
export function draw(R, e, opts = {}) {
  const { gl, u } = R, p = e.p || {};
  const { mode = 2, split = 0.5, clipWarn = 0, dither = 1 } = opts;
  gl.viewport(0, 0, R.canvas.width, R.canvas.height);
  gl.uniform1i(u("mode"), mode); gl.uniform1f(u("split"), split);
  gl.uniform1i(u("clipWarn"), clipWarn); gl.uniform1f(u("dither"), dither);
  gl.uniformMatrix3fv(u("geo"), true, new Float32Array(geometryMatrix(R.src[0], R.src[1], p.crop).flat()));
  gl.uniform2f(u("frame"), R.canvas.width, R.canvas.height);

  const lk = e.look;
  gl.uniform1i(u("lookOn"), lk ? 1 : 0);
  if (lk) {
    const flat = new Float32Array(52); lk.knots.flat().forEach((v, i) => (flat[i] = v));
    gl.uniform4fv(u("lk"), flat); gl.uniform1i(u("K"), e.K);
    const Mm = [1 + lk.dM[0], lk.dM[1], lk.dM[2], lk.dM[3], 1 + lk.dM[4], lk.dM[5], lk.dM[6], lk.dM[7], 1 + lk.dM[8]];
    gl.uniformMatrix3fv(u("lM"), true, new Float32Array(Mm));
    gl.uniform3fv(u("lBias"), new Float32Array(lk.bias)); gl.uniform1f(u("lVig"), lk.vig);
  }

  gl.uniform3fv(u("wb"), new Float32Array(wbGains(get(p, "temp"), get(p, "tint"))));
  gl.uniform1f(u("expo"), 2 ** get(p, "exposure"));
  const tx = SPEC.tone.x, ty = tonePoints(p), tm = pchipSlopes(tx, ty);
  gl.uniform4fv(u("tone"), new Float32Array(tx.flatMap((x, i) => [x, ty[i], tm[i], 0])));

  const cpv = new Float32Array(4 * 4 * CURVE_PTS), cn = [0, 0, 0, 0];
  ["rgb", "r", "g", "b"].forEach((key, c) => {
    const pts = curvePoints(p.curves, key);
    if (!pts) return;
    const xs = pts.map((q) => q[0]).slice(0, CURVE_PTS), ys = pts.map((q) => q[1]).slice(0, CURVE_PTS), m = pchipSlopes(xs, ys);
    xs.forEach((x, i) => cpv.set([x, ys[i], m[i], 0], 4 * (c * CURVE_PTS + i)));
    cn[c] = xs.length;
  });
  gl.uniform4fv(u("cp"), cpv); gl.uniform4i(u("cn"), ...cn);

  const cal = calMatrix(p);
  gl.uniform1i(u("calOn"), cal.on ? 1 : 0); gl.uniformMatrix3fv(u("calM"), true, new Float32Array(cal.m.flat()));
  gl.uniform1f(u("shadowTint"), (get(p, "cal_shadow_tint") / 100) * K.cal_shadow);
  gl.uniform1f(u("sat"), (K.sat_gain * get(p, "saturation")) / 100); gl.uniform1f(u("vib"), get(p, "vibrance") / 100);
  gl.uniform4fv(u("hsl"), new Float32Array(BANDS.flatMap((b) => [get(p, `hsl_hue_${b}`) / 100, get(p, `hsl_sat_${b}`) / 100, get(p, `hsl_lum_${b}`) / 100, 0])));
  const band = new Float32Array(BAND_GEO.flatMap((g, k) => [...g, get(p, `bw_${BANDS[k]}`) / 100]));
  gl.uniform4fv(u("band"), band);
  const bw = get(p, "bw") > 0.5;
  gl.uniform1i(u("bwOn"), bw ? 1 : 0);
  gl.uniform4fv(u("grade"), new Float32Array(SPEC.grade_ranges.flatMap((r) => {
    const s = get(p, `grade_${r}_sat`) / 100, [da, db] = gradeDir(get(p, `grade_${r}_hue`));
    return [s * K.grade_c * da, s * K.grade_c * db, (get(p, `grade_${r}_lum`) / 100) * K.grade_l, 0];
  })));
  gl.uniform2f(u("gradeBal"), get(p, "grade_balance") / 200, 0.5 + get(p, "grade_blending") / 100);
  // the colour stage is skipped when every colour slider is at its default (exact identity, cheaper)
  const colourKeys = SLIDERS.filter((s) => /^(hsl_|grade_|bw|cal_|vibrance|saturation)/.test(s.name));
  gl.uniform1i(u("colourOn"), colourKeys.some((s) => get(p, s.name) !== s.default) ? 1 : 0);
  gl.uniform4f(u("vig"), get(p, "vig_amount") / 100, get(p, "vig_midpoint") / 100, get(p, "vig_feather") / 100, 0);
  gl.uniform1f(u("vigRound"), get(p, "vig_roundness") / 100);
  gl.uniform4f(u("kc"), K.eps_y, K.chroma_full, K.vib_c0, K.hsl_hue_deg);
  gl.uniform4f(u("kc2"), K.hsl_lum, K.bw_mix, K.vig_stops, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

/** Does this slider dict change anything? (The default edit is the identity.) */
export const isDefault = (p) => SLIDERS.every((s) => get(p, s.name) === s.default)
  && !["rgb", "r", "g", "b"].some((k) => curvePoints(p?.curves, k)) && !p?.crop;

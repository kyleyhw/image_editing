// The Phase-A5 look renderer's parameter helpers (photostyle.render.GlobalRenderer): per-channel
// piecewise-linear curves, then a 3x3 matrix and bias, then a vignette, in display space. The
// pixels are drawn by pipeline.js, which applies this look first and the photo's adjustments after it.

export function layout(p) {
  const K = p.knots, curve = p.renderer === "shared" ? K - 2 : 3 * K;
  return { curve: [0, curve], dM: [curve, curve + 9], bias: [curve + 9, curve + 12], vig: [curve + 12, curve + 13] };
}

export function modelGroups(p, strength) {
  const L = layout(p), th = p.theta.map((v) => v * strength), g = {};
  for (const k in L) g[k] = th.slice(L[k][0], L[k][1]);
  return g;
}

export function knotsFromCurve(p, cp) {
  const K = p.knots, out = [];
  if (p.renderer === "shared") {
    const y = [];
    for (let i = 0; i < K; i++) y.push(i / (K - 1) + (i > 0 && i < K - 1 ? cp[i - 1] : 0));
    return [y, y.slice(), y.slice()];
  }
  for (let c = 0; c < 3; c++) {
    const base = c * K, y = [cp[base]];
    for (let j = 1; j < K; j++) y.push(y[j - 1] + Math.exp(cp[base + j]) / (K - 1));
    out.push(y);
  }
  return out;
}

/** The look's renderer inputs at a strength (strength scales the predicted parameters). */
export function effectiveFrom(p, strength) {
  const g = modelGroups(p, strength);
  return { knots: knotsFromCurve(p, g.curve), dM: g.dM.slice(), bias: g.bias.slice(), vig: g.vig[0] };
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
/** The look on one colour (no vignette): curves -> matrix + bias -> clamp. */
export function applyLook(e, K, rgb) {
  const c = [0, 1, 2].map((ch) => {
    const t = clamp01(rgb[ch]) * (K - 1), i = Math.min(Math.floor(t), K - 2), u = t - i, k = e.knots[ch];
    return k[i] + u * (k[i + 1] - k[i]);
  });
  const M = [1 + e.dM[0], e.dM[1], e.dM[2], e.dM[3], 1 + e.dM[4], e.dM[5], e.dM[6], e.dM[7], 1 + e.dM[8]];
  return [0, 1, 2].map((r) => clamp01(M[r * 3] * c[0] + M[r * 3 + 1] * c[1] + M[r * 3 + 2] * c[2] + e.bias[r]));
}

const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
const chroma = (c) => Math.max(...c) - Math.min(...c);
const TEST = [[0.8, 0.3, 0.3], [0.3, 0.7, 0.3], [0.3, 0.4, 0.8], [0.8, 0.7, 0.3], [0.3, 0.7, 0.7], [0.7, 0.3, 0.7]];

/** What a look does, measured through its renderer: [{label, value}] for changes big enough to notice. */
export function describe(e, K) {
  const out = [], pct = (v) => `${v > 0 ? "+" : "−"}${Math.abs(Math.round(v * 100))}`;
  for (const { label, x } of [{ label: "Shadows", x: 0.12 }, { label: "Midtones", x: 0.5 }, { label: "Highlights", x: 0.88 }]) {
    const d = lum(applyLook(e, K, [x, x, x])) - x;
    if (Math.abs(d) > 0.015) out.push({ label, value: pct(d) });
  }
  let warm = 0, tint = 0;
  for (let i = 0; i < 8; i++) {
    const x = 0.15 + (0.7 * i) / 7, o = applyLook(e, K, [x, x, x]);
    warm += (o[0] - o[2]) / 8; tint += (o[1] - (o[0] + o[2]) / 2) / 8;
  }
  if (Math.abs(warm) > 0.01) out.push({ label: warm > 0 ? "Warmer" : "Cooler", value: pct(Math.abs(warm)).slice(1) });
  if (Math.abs(tint) > 0.01) out.push({ label: tint > 0 ? "Green tint" : "Magenta tint", value: pct(Math.abs(tint)).slice(1) });
  const sat = TEST.reduce((s, c) => s + chroma(applyLook(e, K, c)) / chroma(c), 0) / TEST.length - 1;
  if (Math.abs(sat) > 0.03) out.push({ label: "Saturation", value: pct(sat) + "%" });
  if (e.vig > 0.03) out.push({ label: "Vignette", value: String(Math.round(e.vig * 100)) });
  return out;
}

/** Blend two looks (look changes animate instead of jumping). */
export function mixEffective(a, b, t) {
  const l = (x, y) => x + (y - x) * t;
  return { knots: a.knots.map((r, c) => r.map((v, i) => l(v, b.knots[c][i]))), dM: a.dM.map((v, i) => l(v, b.dM[i])),
           bias: a.bias.map((v, i) => l(v, b.bias[i])), vig: l(a.vig, b.vig) };
}

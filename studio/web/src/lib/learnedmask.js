// Learned masks (Phase 23 task 31): a segmentation model's class probability, refined against the
// photo with a guided filter, written as a painted (brush) mask so that the rest of the editor
// (rendering, history, persistence, export, the Python reference) treats it as any other mask.
//
// The model's output is 128 x 128, so its edges are blocky at photo scale. The guided filter (He,
// Sun and Tang 2010) fits, in every window of radius r, a linear map q = a I + b from the photo's
// luminance I to the coarse probability p, with
//     a = cov(I, p) / (var(I) + eps),   b = mean(p) - a mean(I),
// and averages the maps over the windows covering each pixel. Where the photo has an edge the
// mask follows it; where the photo is flat the mask stays smooth. eps sets the edge contrast that
// counts as an edge.

/** Box mean over a (2r+1)^2 window, with the window clipped at the borders (summed-area table). */
function boxMean(src, w, h, r) {
  const S = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) { row += src[y * w + x]; S[(y + 1) * (w + 1) + x + 1] = S[y * (w + 1) + x + 1] + row; }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const s = S[y1 * (w + 1) + x1] - S[y0 * (w + 1) + x1] - S[y1 * (w + 1) + x0] + S[y0 * (w + 1) + x0];
      out[y * w + x] = s / ((x1 - x0) * (y1 - y0));
    }
  }
  return out;
}

/** Guided filter of ``p`` with guide ``I`` (both w x h, values 0-1). */
export function guidedFilter(I, p, w, h, r, eps) {
  const n = w * h, II = new Float32Array(n), Ip = new Float32Array(n);
  for (let i = 0; i < n; i++) { II[i] = I[i] * I[i]; Ip[i] = I[i] * p[i]; }
  const mI = boxMean(I, w, h, r), mp = boxMean(p, w, h, r), mII = boxMean(II, w, h, r), mIp = boxMean(Ip, w, h, r);
  const a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = mII[i] - mI[i] * mI[i], c = mIp[i] - mI[i] * mp[i];
    a[i] = c / (v + eps); b[i] = mp[i] - a[i] * mI[i];
  }
  const ma = boxMean(a, w, h, r), mb = boxMean(b, w, h, r), q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = Math.min(1, Math.max(0, ma[i] * I[i] + mb[i]));
  return q;
}

/** Bilinear resampling of a {data, width, height} map to w x h (pixel centres aligned). */
export function resize(m, w, h) {
  const out = new Float32Array(w * h), sx = m.width / w, sy = m.height / h;
  for (let y = 0; y < h; y++) {
    const fy = Math.min(m.height - 1, Math.max(0, (y + 0.5) * sy - 0.5)), y0 = Math.floor(fy), y1 = Math.min(m.height - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(m.width - 1, Math.max(0, (x + 0.5) * sx - 0.5)), x0 = Math.floor(fx), x1 = Math.min(m.width - 1, x0 + 1), tx = fx - x0;
      const a = m.data[y0 * m.width + x0], b = m.data[y0 * m.width + x1], c = m.data[y1 * m.width + x0], d = m.data[y1 * m.width + x1];
      out[y * w + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    }
  }
  return out;
}

/** Window radius: 1 % of the mask's long edge (10 px at 1024), wide enough to bridge the model's
 * 8 px output cells at that size, narrow enough to keep thin structures such as branches. */
export const GF_RADIUS = 0.01;
/** eps on luminance in 0-1: edges with a luminance step below about sqrt(eps) = 0.03 are smoothed
 * over, larger ones are followed. */
export const GF_EPS = 1e-3;

/** The refined mask (0-1, w x h) of a class probability map, guided by ``img`` drawn at w x h. */
export function refine(prob, img, w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const x = /** @type {CanvasRenderingContext2D} */ (c.getContext("2d", { willReadFrequently: true }));
  x.drawImage(img, 0, 0, w, h);
  const d = x.getImageData(0, 0, w, h).data, I = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) I[i] = (0.2126 * d[4 * i] + 0.7152 * d[4 * i + 1] + 0.0722 * d[4 * i + 2]) / 255;
  return guidedFilter(I, resize(prob, w, h), w, h, Math.max(1, Math.round(GF_RADIUS * Math.max(w, h))), GF_EPS);
}

// Exact port of photostyle.render.GlobalRenderer for the browser:
//   per-channel piecewise-linear curves -> 3x3 matrix + bias -> vignette -> clamp.
// Kept numerically identical to Python; tools/check_studio.py checks it (golden test).

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

export function curveFromKnots(p, kn) {       // inverse of knotsFromCurve (per-channel monotone)
  const K = p.knots, cp = [];
  for (let c = 0; c < 3; c++) {
    cp.push(kn[c][0]);
    for (let j = 1; j < K; j++) cp.push(Math.log(Math.max(1e-4, (kn[c][j] - kn[c][j - 1]) * (K - 1))));
  }
  return cp;
}

/** Effective renderer inputs from model parameters + the user's edits. */
export function effectiveFrom(p, strength, d, knots) {
  const g = modelGroups(p, strength);
  const kn = knots || knotsFromCurve(p, g.curve);
  const dM = g.dM.slice(), bias = g.bias.slice(), s = (d.sat / 100) * 0.5;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) dM[i * 3 + j] += s * ((i === j ? 1 : 0) - 1 / 3);
  const w = d.warmth / 100, t = d.tint / 100;
  bias[0] += 0.05 * w + 0.03 * t; bias[1] += -0.03 * t; bias[2] += -0.05 * w + 0.03 * t;
  return { knots: kn, dM, bias, vig: g.vig[0] + d.vig / 100, g };
}

const VS = `#version 300 es
in vec2 pos; out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }`;
const FS = `#version 300 es
precision highp float;
uniform sampler2D img; uniform sampler2D orig; uniform float knots[51]; uniform int K;
uniform mat3 M; uniform vec3 bias; uniform float vig; uniform float split; uniform int mode;
in vec2 uv; out vec4 o;
float curve(int ch, float x) {
  float t = clamp(x, 0.0, 1.0) * float(K - 1);
  int i = min(int(floor(t)), K - 2); float u = t - float(i);
  float a = knots[ch * K + i], b = knots[ch * K + i + 1]; return a + u * (b - a);
}
void main() {
  vec3 c = texture(img, vec2(uv.x, 1.0 - uv.y)).rgb;
  vec3 e = vec3(curve(0, c.r), curve(1, c.g), curve(2, c.b));
  e = M * e + bias;
  vec2 p = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  e *= clamp(1.0 - vig * dot(p, p) / 2.0, 0.0, 1.0);
  e = clamp(e, 0.0, 1.0);
  if (mode == 1 || (mode == 0 && uv.x < split)) e = texture(orig, vec2(uv.x, 1.0 - uv.y)).rgb;
  o = vec4(e, 1.0);
}`;

export function makeGL(canvas) {
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, premultipliedAlpha: false });
  const sh = (t, s) => {
    const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x));
    return x;
  };
  const pr = gl.createProgram();
  gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
  gl.useProgram(pr);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(pr, "pos"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture(), origTex = gl.createTexture();
  const u = (n) => gl.getUniformLocation(pr, n);
  gl.uniform1i(u("img"), 0); gl.uniform1i(u("orig"), 1);
  return { gl, tex, origTex, u, canvas };
}

/** Upload the image to be graded; with ``asOriginal`` (default) also the "before" texture. */
export function setImage(R, img, asOriginal = true) {
  upload(R, R.tex, 0, img);
  if (asOriginal) upload(R, R.origTex, 1, img);
}

function upload(R, tex, unit, img) {
  const { gl } = R;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) {
    gl.texParameteri(gl.TEXTURE_2D, k, v);
  }
}

/** mode: 0 split, 1 before, 2 after. */
export function draw(R, e, K, mode = 2, split = 0.5) {
  const { gl, u } = R;
  gl.viewport(0, 0, R.canvas.width, R.canvas.height);
  const flat = new Float32Array(51); e.knots.flat().forEach((v, i) => (flat[i] = v));
  gl.uniform1fv(u("knots"), flat); gl.uniform1i(u("K"), K);
  const M = [1 + e.dM[0], e.dM[1], e.dM[2], e.dM[3], 1 + e.dM[4], e.dM[5], e.dM[6], e.dM[7], 1 + e.dM[8]];
  gl.uniformMatrix3fv(u("M"), true, new Float32Array(M));      // row-major -> transpose
  gl.uniform3fv(u("bias"), new Float32Array(e.bias)); gl.uniform1f(u("vig"), e.vig);
  gl.uniform1f(u("split"), split); gl.uniform1i(u("mode"), mode);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

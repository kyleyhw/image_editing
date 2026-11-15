<script>
  // Point curves, as Lightroom's point-curve panel: RGB (the same curve on each channel), then R, G
  // and B. Click to add a point, drag to move it, double-click (or drag it off the side) to remove
  // it. Interpolation is monotone cubic (PCHIP), so the curve never overshoots between points.
  import { S, setCurve, commit } from "../lib/studio.svelte.js";
  import { pchipSlopes, pchipEval } from "../lib/pipeline.js";
  let cv;
  const CH = [["rgb", "RGB", "#e6e6ea"], ["r", "Red", "#ff5d6c"], ["g", "Green", "#4fe39a"], ["b", "Blue", "#5b9dff"]];
  const MAX_PTS = 16;                 // the shader's per-curve budget (pipeline.js CURVE_PTS)
  const ident = () => [[0, 0], [1, 1]];
  const pts = (k) => S.p.curves?.[k] ?? ident();
  const colour = (k) => CH.find((c) => c[0] === k)[2];

  function curveY(p, t) {
    const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]), m = pchipSlopes(xs, ys);
    return pchipEval(xs, ys, m, Math.min(xs[xs.length - 1], Math.max(xs[0], t)));
  }

  $effect(() => {
    S.ch; JSON.stringify(S.p.curves); S.frame;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth * dpr || 560, H = W;
    if (cv.width !== W) { cv.width = W; cv.height = H; }
    const x = cv.getContext("2d");
    x.clearRect(0, 0, W, H);
    x.lineWidth = dpr; x.strokeStyle = "rgba(255,255,255,.06)"; x.beginPath();
    for (let i = 1; i < 4; i++) { x.moveTo((i * W) / 4, 0); x.lineTo((i * W) / 4, H); x.moveTo(0, (i * H) / 4); x.lineTo(W, (i * H) / 4); }
    x.stroke();
    x.setLineDash([2 * dpr, 4 * dpr]); x.strokeStyle = "rgba(255,255,255,.14)"; x.beginPath(); x.moveTo(0, H); x.lineTo(W, 0); x.stroke(); x.setLineDash([]);
    const line = (p, col, w) => {
      x.strokeStyle = col; x.lineWidth = w * dpr; x.beginPath();
      for (let i = 0; i <= 128; i++) { const t = i / 128, px = t * W, py = H - curveY(p, t) * H; i ? x.lineTo(px, py) : x.moveTo(px, py); }
      x.stroke();
    };
    for (const [k] of CH) if (k !== S.ch && S.p.curves?.[k]) line(pts(k), colour(k) + "50", 1.2);
    line(pts(S.ch), colour(S.ch), 2);
    for (const [a, b] of pts(S.ch)) {
      x.beginPath(); x.arc(a * W, H - b * H, 4.5 * dpr, 0, 7);
      x.fillStyle = "#0c0c0e"; x.fill(); x.lineWidth = 1.6 * dpr; x.strokeStyle = colour(S.ch); x.stroke();
    }
  });

  let drag = -1;
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height]; };
  const near = (p, q) => { const [u, v] = q; return p.findIndex(([a, b]) => Math.hypot(a - u, b - v) < 0.045); };
  function down(e) {
    if (!S.id) return;
    const q = pos(e), p = pts(S.ch).map((r) => r.slice());
    let i = near(p, q);
    if (i < 0) {
      if (p.length >= MAX_PTS) return;
      const u = Math.min(1, Math.max(0, q[0]));
      p.push([u, curveY(p, u)]); p.sort((a, b) => a[0] - b[0]); i = p.findIndex((r) => r[0] === u);
    }
    setCurve(S.ch, p); drag = i; cv.setPointerCapture(e.pointerId);
  }
  function move(e) {
    if (drag < 0) return;
    const [u, v] = pos(e), p = pts(S.ch).map((r) => r.slice()), n = p.length;
    const lo = drag > 0 ? p[drag - 1][0] + 0.01 : 0, hi = drag < n - 1 ? p[drag + 1][0] - 0.01 : 1;
    const ends = drag === 0 || drag === n - 1;
    if (!ends && (u < lo - 0.08 || u > hi + 0.08)) { p.splice(drag, 1); setCurve(S.ch, p); drag = -1; return; }   // dragged off: remove
    p[drag] = [ends ? Math.min(hi, Math.max(lo, drag === 0 ? Math.min(u, 0.5) : Math.max(u, 0.5))) : Math.min(hi, Math.max(lo, u)),
               Math.min(1, Math.max(0, v))];
    setCurve(S.ch, p);
  }
  function up() { if (drag >= 0) commit(); drag = -1; }
  function dbl(e) {
    const p = pts(S.ch).map((r) => r.slice()), i = near(p, pos(e));
    if (i > 0 && i < p.length - 1) { p.splice(i, 1); setCurve(S.ch, p); commit(); }
  }
  function resetChannel() { setCurve(S.ch, ident()); commit(); }
</script>

<div class="chan" role="tablist" aria-label="Curve channel">
  {#each CH as [k, n, c]}
    <button role="tab" class="chan-b" class:on={S.ch === k} style={`--c:${c}`} aria-selected={S.ch === k} onclick={() => (S.ch = k)} aria-label={n}>{k === "rgb" ? "RGB" : n[0]}</button>
  {/each}
  {#if S.p.curves?.[S.ch]}<button class="link-btn push" onclick={resetChannel}>Reset</button>{/if}
</div>
<canvas id="curves" bind:this={cv} aria-label="Curve editor: click to add a point, drag to move, double-click to remove"
        onpointerdown={down} onpointermove={move} onpointerup={up} ondblclick={dbl}></canvas>
<p class="hint">Click to add a point, drag to move it, double-click or drag it off the side to remove it.</p>

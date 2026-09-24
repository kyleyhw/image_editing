<script>
  import { S, effective, commit } from "../lib/studio.svelte.js";
  import { knotsFromCurve, modelGroups } from "../lib/render.js";
  let cv;
  const COL = ["#ff5d6c", "#4fe39a", "#5b9dff"], NAMES = ["Red", "Green", "Blue"];

  $effect(() => {
    S.params; S.strength; S.ch; JSON.stringify(S.knots); JSON.stringify(S.d);
    if (!cv || !S.params) return;
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth * dpr || 560, H = W;
    if (cv.width !== W) { cv.width = W; cv.height = H; }
    const x = cv.getContext("2d"), e = effective(), K = S.params.knots;
    x.clearRect(0, 0, W, H);
    x.lineWidth = dpr; x.strokeStyle = "rgba(255,255,255,.06)"; x.beginPath();
    for (let i = 1; i < 4; i++) { x.moveTo((i * W) / 4, 0); x.lineTo((i * W) / 4, H); x.moveTo(0, (i * H) / 4); x.lineTo(W, (i * H) / 4); }
    x.stroke();
    x.setLineDash([2 * dpr, 4 * dpr]); x.strokeStyle = "rgba(255,255,255,.14)"; x.beginPath(); x.moveTo(0, H); x.lineTo(W, 0); x.stroke(); x.setLineDash([]);
    const ghost = knotsFromCurve(S.params, modelGroups(S.params, S.strength).curve);
    const line = (kn) => { x.beginPath(); kn.forEach((v, i) => { const px = (i / (K - 1)) * W, py = H - v * H; i ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke(); };
    for (let c = 0; c < 3; c++) {
      if (c === S.ch) continue;
      x.strokeStyle = COL[c] + "40"; x.lineWidth = 1.2 * dpr; line(e.knots[c]);
    }
    if (S.knots) { x.setLineDash([3 * dpr, 3 * dpr]); x.strokeStyle = "rgba(255,255,255,.35)"; x.lineWidth = dpr; line(ghost[S.ch]); x.setLineDash([]); }
    x.shadowColor = COL[S.ch]; x.shadowBlur = 10 * dpr; x.strokeStyle = COL[S.ch]; x.lineWidth = 2 * dpr; line(e.knots[S.ch]); x.shadowBlur = 0;
    e.knots[S.ch].forEach((v, i) => {
      x.beginPath(); x.arc((i / (K - 1)) * W, H - v * H, 4.5 * dpr, 0, 7);
      x.fillStyle = "#0c0c0e"; x.fill(); x.lineWidth = 1.6 * dpr; x.strokeStyle = COL[S.ch]; x.stroke();
    });
  });

  let drag = -1;
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height]; };
  function down(e) {
    if (!S.params || S.params.renderer === "shared") return;
    const [x] = pos(e), K = S.params.knots; drag = Math.round(x * (K - 1));
    if (!S.knots) S.knots = effective().knots.map((r) => r.slice());
    cv.setPointerCapture(e.pointerId);
  }
  function move(e) {
    if (drag < 0) return;
    const [, y] = pos(e), kn = S.knots[S.ch], K = kn.length;
    const lo = drag > 0 ? kn[drag - 1] + 1e-3 : -0.2, hi = drag < K - 1 ? kn[drag + 1] - 1e-3 : 1.2;   // stay monotone
    kn[drag] = Math.min(hi, Math.max(lo, y));
  }
  function up() { if (drag >= 0) commit(); drag = -1; }
</script>

<div class="chan" role="group" aria-label="Channel">
  {#each NAMES as n, i}
    <button class="chan-b" class:on={S.ch === i} style={`--c:${COL[i]}`} onclick={() => (S.ch = i)} aria-label={n}>{n[0]}</button>
  {/each}
  {#if S.knots}<button class="btn ghost sm push" onclick={() => { S.knots = null; commit(); }}>Model curve</button>{/if}
</div>
<canvas id="curves" bind:this={cv} aria-label="Curve editor: drag the points"
        onpointerdown={down} onpointermove={move} onpointerup={up}></canvas>
<p class="hint">Drag a point. Curves stay monotone, so tones never invert.</p>

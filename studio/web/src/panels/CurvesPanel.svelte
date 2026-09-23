<script>
  import { S, effective, commit } from "../lib/studio.svelte.js";
  import { knotsFromCurve, modelGroups } from "../lib/render.js";
  let cv;
  const COL = ["#ff6b6b", "#6bdc6b", "#6b9bff"];

  $effect(() => {
    S.params; S.strength; S.ch; JSON.stringify(S.knots);
    if (!cv || !S.params) return;
    const x = cv.getContext("2d"), W = cv.width, H = cv.height, e = effective(), K = S.params.knots;
    x.clearRect(0, 0, W, H); x.strokeStyle = "#444"; x.lineWidth = 1; x.beginPath();
    for (let i = 1; i < 4; i++) { x.moveTo((i * W) / 4, 0); x.lineTo((i * W) / 4, H); x.moveTo(0, (i * H) / 4); x.lineTo(W, (i * H) / 4); }
    x.stroke();
    const ghost = knotsFromCurve(S.params, modelGroups(S.params, S.strength).curve);
    const line = (kn) => { x.beginPath(); kn.forEach((v, i) => { const px = (i / (K - 1)) * W, py = H - v * H; i ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke(); };
    for (let c = 0; c < 3; c++) {
      const on = c === S.ch;
      if (S.knots) { x.setLineDash([3, 3]); x.strokeStyle = "#777"; x.lineWidth = 1; line(ghost[c]); x.setLineDash([]); }
      x.strokeStyle = on ? COL[c] : COL[c] + "55"; x.lineWidth = on ? 2 : 1; line(e.knots[c]);
      if (on) e.knots[c].forEach((v, i) => { x.fillStyle = COL[c]; x.fillRect((i / (K - 1)) * W - 3, H - v * H - 3, 6, 6); });
    }
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

<div class="tabs">
  {#each ["R", "G", "B"] as l, i}<button class="tab" class:on={S.ch === i} onclick={() => (S.ch = i)}>{l}</button>{/each}
</div>
<canvas id="curves" bind:this={cv} width="240" height="240" aria-label="Curve editor: drag the knots"
        onpointerdown={down} onpointermove={move} onpointerup={up}></canvas>
<p class="muted small">Curves stay monotone, so tones never invert. Dashed: the model's curve.</p>

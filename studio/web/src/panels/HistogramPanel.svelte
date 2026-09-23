<script>
  import { S, view, effective, modeCode } from "../lib/studio.svelte.js";
  import { draw } from "../lib/render.js";
  let cv;
  $effect(() => {
    S.params; S.strength; S.imgVersion; JSON.stringify(S.d); JSON.stringify(S.knots); JSON.stringify(S.scene);
    if (!cv || !S.params || !view.img || !view.main) return;
    const x = cv.getContext("2d"), W = cv.width, H = cv.height, gl = view.main.gl;
    const w = view.main.canvas.width, h = view.main.canvas.height, px = new Uint8Array(w * h * 4);
    draw(view.main, effective(), S.params.knots, 2); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const after = [0, 1, 2].map(() => new Array(64).fill(0)), lum = new Array(64).fill(0);
    for (let i = 0; i < px.length; i += 4 * 7) for (let c = 0; c < 3; c++) after[c][px[i + c] >> 2]++;
    if (!view.srcPx) {
      const t = document.createElement("canvas"); t.width = w; t.height = h; const tx = t.getContext("2d");
      tx.drawImage(view.img, 0, 0, w, h); view.srcPx = tx.getImageData(0, 0, w, h).data;
    }
    for (let i = 0; i < view.srcPx.length; i += 4 * 7) lum[(0.2126 * view.srcPx[i] + 0.7152 * view.srcPx[i + 1] + 0.0722 * view.srcPx[i + 2]) >> 2]++;
    const mx = Math.max(...lum, ...after.flat());
    x.clearRect(0, 0, W, H); x.fillStyle = "#666";
    lum.forEach((v, i) => x.fillRect((i * W) / 64, H - (v / mx) * H, W / 64, (v / mx) * H));
    ["#ff6b6b", "#6bdc6b", "#6b9bff"].forEach((c, k) => { x.strokeStyle = c; x.beginPath();
      after[k].forEach((v, i) => { const X = ((i + 0.5) * W) / 64, Y = H - (v / mx) * H; i ? x.lineTo(X, Y) : x.moveTo(X, Y); }); x.stroke(); });
    draw(view.main, effective(), S.params.knots, modeCode(), S.split);
  });
</script>

<canvas id="hist" bind:this={cv} width="240" height="90" aria-label="Before (grey) and after (colour) histograms"></canvas>

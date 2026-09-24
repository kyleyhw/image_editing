<script>
  import { S, view } from "../lib/studio.svelte.js";
  let cv;
  let last = 0, timer = 0;
  // Histogram of what is on screen (the graded image), channels blended additively.
  $effect(() => {
    S.frame; S.imgVersion;
    const now = performance.now();
    clearTimeout(timer);
    if (now - last < 120) { timer = setTimeout(paint, 130); return; }
    paint();
  });
  function paint() {
    last = performance.now();
    if (!cv || !view.main || !S.params) return;
    const W = (cv.width = cv.clientWidth * (window.devicePixelRatio || 1) || 560), H = (cv.height = Math.round(W * 0.3));
    const src = view.main.canvas, t = document.createElement("canvas"), tw = 256, th = Math.max(1, Math.round((256 * src.height) / src.width));
    t.width = tw; t.height = th;
    const tx = t.getContext("2d", { willReadFrequently: true }); tx.drawImage(src, 0, 0, tw, th);
    const px = tx.getImageData(0, 0, tw, th).data, bins = [0, 1, 2].map(() => new Float32Array(64));
    for (let i = 0; i < px.length; i += 4) for (let c = 0; c < 3; c++) bins[c][px[i + c] >> 2]++;
    const mx = Math.max(...bins.map((b) => Math.max(...b.subarray(1, 63)))) || 1;
    const x = cv.getContext("2d"); x.clearRect(0, 0, W, H); x.globalCompositeOperation = "lighter";
    ["rgba(255,80,100,.55)", "rgba(70,230,150,.5)", "rgba(80,140,255,.6)"].forEach((col, c) => {
      x.fillStyle = col; x.beginPath(); x.moveTo(0, H);
      bins[c].forEach((v, i) => x.lineTo(((i + 0.5) * W) / 64, H - Math.min(1, v / mx) * H * 0.95));
      x.lineTo(W, H); x.closePath(); x.fill();
    });
    x.globalCompositeOperation = "source-over";
  }
</script>

<canvas id="hist" class="hist" bind:this={cv} aria-label="Histogram of the edited photo"></canvas>

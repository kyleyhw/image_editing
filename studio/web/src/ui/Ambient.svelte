<script>
  // The room is lit by the photo: a tiny copy of what is on screen, blurred behind the UI,
  // and the accent colour taken from its most vivid pixels.
  import { S, view } from "../lib/studio.svelte.js";
  let cv, last = 0, timer = 0;
  $effect(() => {
    S.frame; S.id;
    clearTimeout(timer);
    const wait = 90 - (performance.now() - last);
    timer = setTimeout(paint, Math.max(0, wait));
  });
  function paint() {
    last = performance.now();
    if (!cv || !view.main || !S.id) return;
    const x = cv.getContext("2d", { willReadFrequently: true });
    x.drawImage(view.main.canvas, 0, 0, cv.width, cv.height);
    const d = x.getImageData(0, 0, cv.width, cv.height).data;
    let sx = 0, sy = 0, w = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255, mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sat = mx - mn;
      if (sat < 0.12 || mx < 0.15) continue;
      let h = mx === r ? (g - b) / sat : mx === g ? 2 + (b - r) / sat : 4 + (r - g) / sat;
      h = (h * Math.PI) / 3;
      const wt = sat * sat * mx;
      sx += Math.cos(h) * wt; sy += Math.sin(h) * wt; w += wt;
    }
    const root = document.documentElement.style;
    if (w > 0.4) {
      const hue = ((Math.atan2(sy, sx) * 180) / Math.PI + 360) % 360;
      root.setProperty("--accent", `hsl(${hue.toFixed(0)} 92% 66%)`);
      root.setProperty("--accent-2", `hsl(${((hue + 40) % 360).toFixed(0)} 90% 62%)`);
    } else {
      root.removeProperty("--accent"); root.removeProperty("--accent-2");
    }
  }
</script>

<div class="ambient" aria-hidden="true" class:on={!!S.id}>
  <canvas bind:this={cv} width="48" height="32"></canvas>
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
</div>
<div class="grain" aria-hidden="true"></div>

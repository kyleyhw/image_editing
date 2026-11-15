<script>
  // Crop and straighten. The crop rectangle is stored as fractions of the photo (x, y, w, h) with the
  // straighten angle; the stage draws handles while the crop tool is on.
  import { S, view, setCrop, commit } from "../lib/studio.svelte.js";
  import { straightenScale } from "../lib/pipeline.js";
  import Slider from "../ui/Slider.svelte";
  import { I } from "../lib/icons.js";
  const ASPECTS = [["free", "Free"], ["orig", "Original"], ["1", "1 : 1"], ["0.8", "4 : 5"], ["1.25", "5 : 4"],
                   ["1.5", "3 : 2"], ["0.6667", "2 : 3"], ["1.7778", "16 : 9"], ["0.5625", "9 : 16"]];
  const crop = () => ({ x: 0, y: 0, w: 1, h: 1, angle: 0, aspect: "free", ...(S.p.crop || {}) });
  const dims = () => [view.img?.naturalWidth || 1, view.img?.naturalHeight || 1];

  /** Fit a rectangle of pixel aspect `r` (w/h), as large as possible inside the straightened frame. */
  export function fitAspect(c, aspect) {
    const [W, H] = dims(), s = straightenScale(W, H, c.angle);
    const r = aspect === "free" ? (c.w * W) / (c.h * H) : aspect === "orig" ? W / H : +aspect;
    let w = s, h = (s * W) / (H * r);
    if (h > s) { h = s; w = (s * H * r) / W; }
    return { ...c, w, h, x: (1 - w) / 2, y: (1 - h) / 2, aspect };
  }
  function setAngle(a) {
    const c = crop(), [W, H] = dims(), s = straightenScale(W, H, a), s0 = straightenScale(W, H, c.angle);
    const full = Math.abs(c.w - s0) < 1e-3 && Math.abs(c.h - s0) < 1e-3;
    setCrop(full || c.aspect !== "free" ? fitAspect({ ...c, angle: a }, c.aspect === "free" && full ? "orig" : c.aspect) : { ...c, angle: a });
  }
  function reset() { const { crop: _, ...p } = S.p; S.p = p; commit(); }
</script>

<div class="row-between">
  <button class="btn" class:primary={S.tool === "crop"} disabled={!S.id} onclick={() => { S.tool = S.tool === "crop" ? null : "crop"; if (!S.tool) commit(); }}>
    {@html I.crop}{S.tool === "crop" ? "Done" : "Crop"}</button>
  <select aria-label="Aspect ratio" value={crop().aspect} disabled={!S.id}
          onchange={(e) => { setCrop(fitAspect(crop(), e.currentTarget.value)); commit(); }}>
    {#each ASPECTS as [v, l]}<option value={v}>{l}</option>{/each}
  </select>
  <button class="link-btn" onclick={reset} disabled={!S.p.crop}>Reset</button>
</div>
<Slider label="Straighten" min={-45} max={45} step={0.1} value={crop().angle} disabled={!S.id}
        oninput={setAngle} onchange={commit} />
<p class="hint">Drag the frame or its corners in the photo. Straightening keeps the largest crop that fits.</p>

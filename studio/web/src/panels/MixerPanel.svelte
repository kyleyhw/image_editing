<script>
  import { BANDS, SPEC } from "../lib/pipeline.js";
  import Sliders from "./Sliders.svelte";
  let mode = $state("hue");
  const hsv = (h, s = 85, l = 55) => `hsl(${(h + 360) % 360} ${s}% ${l}%)`;
  // tracks show what each slider does to its band: hue to its neighbours, saturation from grey,
  // luminance from dark to light
  const tracks = $derived(Object.fromEntries(BANDS.map((b, k) => {
    const h = SPEC.band_hsv[k];
    const t = mode === "hue" ? `linear-gradient(90deg,${hsv(h - 30)},${hsv(h)},${hsv(h + 30)})`
            : mode === "sat" ? `linear-gradient(90deg,hsl(${h} 0% 55%),${hsv(h)})`
            : `linear-gradient(90deg,${hsv(h, 70, 18)},${hsv(h)},${hsv(h, 70, 85)})`;
    return [`hsl_${mode}_${b}`, t];
  })));
</script>

<div class="tabs" role="tablist" aria-label="Colour mixer">
  {#each [["hue", "Hue"], ["sat", "Saturation"], ["lum", "Luminance"]] as [m, label]}
    <button role="tab" class="tab" class:on={mode === m} aria-selected={mode === m} onclick={() => (mode = m)}>{label}</button>
  {/each}
</div>
<Sliders names={BANDS.map((b) => `hsl_${mode}_${b}`)} {tracks} />

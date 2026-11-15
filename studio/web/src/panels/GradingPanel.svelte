<script>
  import Sliders from "./Sliders.svelte";
  import { S, slider } from "../lib/studio.svelte.js";
  let range = $state("shadows");
  const RANGES = [["shadows", "Shadows"], ["midtones", "Mids"], ["highlights", "Highs"], ["global", "Global"]];
  const rainbow = "linear-gradient(90deg,hsl(0 80% 55%),hsl(60 80% 55%),hsl(120 80% 50%),hsl(180 80% 50%),hsl(240 80% 60%),hsl(300 80% 55%),hsl(360 80% 55%))";
  const swatch = (r) => `hsl(${slider(`grade_${r}_hue`)} ${Math.min(100, slider(`grade_${r}_sat`) * 1.6)}% 55%)`;
</script>

<div class="tabs" role="tablist" aria-label="Tonal range">
  {#each RANGES as [r, label]}
    <button role="tab" class="tab" class:on={range === r} aria-selected={range === r} onclick={() => (range = r)}>
      <span class="dot" style={`background:${swatch(r)}`}></span>{label}</button>
  {/each}
</div>
{#key range}
  <Sliders names={[`grade_${range}_hue`, `grade_${range}_sat`, `grade_${range}_lum`]}
           tracks={{ [`grade_${range}_hue`]: rainbow, [`grade_${range}_sat`]: `linear-gradient(90deg,#777,${swatch(range)})` }} />
{/key}
<div class="sub-head"><span>Range</span></div>
<Sliders names={["grade_balance", "grade_blending"]} />
{#if !S.id}<p class="hint">Open a photo to grade it.</p>{/if}

<script>
  import { S, commit } from "../lib/studio.svelte.js";
  import Slider from "../ui/Slider.svelte";
  const sign = (v) => (v > 0 ? `+${v}` : String(v));
</script>

<div class="stack">
  {#each [["warmth", "Warmth", "Cooler ← → warmer"], ["tint", "Tint", "Magenta ← → green"], ["sat", "Saturation", ""]] as [k, label, hint]}
    <Slider id={k} {label} title={hint} value={S.d[k]} fmt={sign} disabled={!S.params}
            oninput={(v) => (S.d[k] = v)} onchange={commit} />
  {/each}
</div>

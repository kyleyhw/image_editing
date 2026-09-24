<script>
  import { S, commit } from "../lib/studio.svelte.js";
  import Slider from "../ui/Slider.svelte";
</script>

<div class="stack">
  {#each [["warmth", "Warmth", "Cooler ← → warmer"], ["tint", "Tint", "Magenta ← → green"], ["sat", "Saturation", ""]] as [k, label, hint]}
    <Slider id={k} {label} title={hint} value={S.d[k]} disabled={!S.params}
            oninput={(v) => (S.d[k] = v)} onchange={commit} />
  {/each}
</div>

<script>
  import { S, commit } from "../lib/studio.svelte.js";
</script>

{#each [["warmth", "Warmth"], ["tint", "Tint"], ["sat", "Saturation"]] as [k, label]}
  <label class="row">{label}
    <input id={k} type="range" min="-100" max="100" value={S.d[k]} oninput={(e) => (S.d[k] = +e.currentTarget.value)} onchange={commit} />
    <output>{S.d[k]}</output>
  </label>
{/each}

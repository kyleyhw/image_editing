<script>
  import { S, commit } from "./lib/studio.svelte.js";
  let { onAdjust, sheetOpen } = $props();
  const pct = $derived(Math.round(S.strength * 100));
</script>

<div id="dock" aria-label="Quick controls">
  <label class="dock-strength">
    <span>Strength</span>
    <input id="dockStrength" type="range" min="0" max="150" value={pct}
           oninput={(e) => (S.strength = Number(e.currentTarget.value) / 100)} onchange={commit} disabled={!S.params} />
    <output>{pct}%</output>
  </label>
  <div class="dock-modes" role="group" aria-label="Compare">
    {#each ["split", "after", "before"] as m}
      <button class="seg" class:on={S.mode === m} onclick={() => (S.mode = m)}>{m[0].toUpperCase() + m.slice(1)}</button>
    {/each}
  </div>
  <button class="btn primary adjust" aria-expanded={sheetOpen} onclick={onAdjust}>{sheetOpen ? "Done" : "Adjust"}</button>
</div>

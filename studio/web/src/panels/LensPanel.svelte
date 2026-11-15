<script>
  // Lens blur (Phase 24): depth of field from an estimated depth map, like a wide aperture. The depth
  // model (27 MB) is downloaded only when asked for.
  import { S, estimateDepth } from "../lib/studio.svelte.js";
  import Sliders from "./Sliders.svelte";
</script>

{#if !S.caps.depth}
  <p class="hint">Lens blur needs the depth model, which runs in the hosted (browser) version.</p>
{:else if !S.depth.ready}
  <p class="hint">Blurs the background (or foreground) by its estimated distance, as a wide-aperture lens would. The depth
    model (Depth Anything V2, 27 MB) is downloaded once and runs on your device.</p>
  <button class="btn" disabled={!S.id || S.depth.loading} onclick={estimateDepth}>
    {S.depth.loading ? `Estimating depth… ${Math.round(S.depth.progress * 100)} %` : "Estimate depth"}</button>
  {#if S.depth.error}<p class="note">{S.depth.error}</p>{/if}
{:else}
  <Sliders names={["lens_amount", "lens_focus", "lens_range"]} />
  <div class="row-between">
    <button class="link-btn" class:on={S.tool === "focus"} onclick={() => (S.tool = S.tool === "focus" ? null : "focus")}>Pick focus on the photo</button>
    <label class="toggle small"><input type="checkbox" bind:checked={S.depth.show} /> Show depth</label>
  </div>
  <p class="hint">Depth view: brighter is nearer; green is in focus.</p>
{/if}

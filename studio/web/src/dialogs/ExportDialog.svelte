<script>
  // Export settings: size, quality, colour space, metadata. The format is chosen next to the
  // Export button; these settings are remembered.
  import Modal from "./Modal.svelte";
  import { S } from "../lib/studio.svelte.js";
  /** @type {{ open?: boolean, opts: any }} */
  let { open = $bindable(false), opts = $bindable() } = $props();
  const SIZES = [[0, "Full size"], [4096, "4096 px"], [3000, "3000 px"], [2048, "2048 px"], [1080, "1080 px"]];
  const p3 = typeof window !== "undefined" && window.matchMedia?.("(color-gamut: p3)").matches;
  const canShare = typeof navigator !== "undefined" && !!navigator.canShare;
</script>

<Modal bind:open title="Export settings">
  <div class="form">
    <label>Long edge
      <select bind:value={opts.edge}>{#each SIZES as [v, l]}<option value={v}>{l}</option>{/each}</select></label>
    <label>JPEG quality <input type="number" min="40" max="100" bind:value={opts.quality} /></label>
    <label>Colour space
      <select bind:value={opts.space}>
        <option value="srgb">sRGB (any screen, the web)</option>
        <option value="p3">Display P3 (wide colour{p3 ? ", this screen shows it" : ""})</option>
      </select></label>
    <label class="toggle"><input type="checkbox" bind:checked={opts.metadata} /> Keep the photo's metadata (EXIF)</label>
    {#if canShare}<label class="toggle"><input type="checkbox" bind:checked={opts.share} /> Share instead of download (save to Photos on a phone)</label>{/if}
    <p class="hint">16-bit PNG and TIFF keep the full precision of the pipeline for further editing. The Lightroom preset
      carries the sliders and curves; Lightroom's own maths makes it similar, not identical.
      {#if S.backend === "server"}On the local server, size and colour space options do not apply yet.{/if}</p>
  </div>
</Modal>

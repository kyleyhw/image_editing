<script>
  import { S, slider, setSlider, commit } from "../lib/studio.svelte.js";
  import { BANDS, SPEC } from "../lib/pipeline.js";
  import Sliders from "./Sliders.svelte";
  const on = $derived(slider("bw") > 0.5);
  const tracks = Object.fromEntries(BANDS.map((b, k) => [`bw_${b}`, `linear-gradient(90deg,#222,hsl(${SPEC.band_hsv[k]} 60% 50%),#eee)`]));
</script>

<label class="toggle"><input type="checkbox" checked={on} disabled={!S.id}
       onchange={(e) => { setSlider("bw", e.currentTarget.checked ? 1 : 0); commit(); }} /> Convert to black and white</label>
{#if on}<p class="hint">Each slider brightens or darkens the greys that colour becomes.</p><Sliders names={BANDS.map((b) => `bw_${b}`)} {tracks} />{/if}

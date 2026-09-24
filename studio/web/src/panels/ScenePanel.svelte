<script>
  import { S, applyScene, commit, ZERO_SCENE } from "../lib/studio.svelte.js";
  import Slider from "../ui/Slider.svelte";
  // Depth-aware (Depth Anything V2 Small) and sky-aware (learned segmenter) tools, rendered by the
  // server on the original before the grade. Values are -100..100 (%), mapped to SceneParams.
  /** @type {Array<[string, string, string, number]>} */
  const CONTROLS = [
    ["haze", "Haze", "+ adds distance haze, − removes it", 1],
    ["clarity_near", "Clarity near", "local contrast in the foreground", 1],
    ["clarity_far", "Clarity far", "local contrast in the distance", 1],
    ["sky_exposure", "Sky exposure", "brighten or darken the sky only (stops ×2)", 2],
    ["sky_warmth", "Sky warmth", "", 1],
    ["sky_saturation", "Sky saturation", "", 1],
  ];
  function set(k, v, scale) { S.scene[k] = (v / 100) * scale; applyScene(); }
  function reset() { S.scene = ZERO_SCENE(); applyScene(); commit(); }
</script>

<div class="stack">
  {#each CONTROLS as [k, label, hint, scale]}
    <Slider {label} title={hint} value={Math.round((S.scene[k] / scale) * 100)} disabled={!S.id}
            oninput={(v) => set(k, v, scale)} onchange={commit} />
  {/each}
  <div class="row-between">
    <span class="hint">Depth and sky are found by two small models; the first change takes ~2 s.</span>
    <button class="link-btn" onclick={reset} disabled={!Object.values(S.scene).some(Boolean)}>Reset</button>
  </div>
</div>

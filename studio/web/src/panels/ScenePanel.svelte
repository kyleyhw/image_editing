<script>
  import { S, applyScene, commit, ZERO_SCENE } from "../lib/studio.svelte.js";
  // Depth-aware (Depth Anything V2 Small) and sky-aware (learned segmenter) tools, rendered by the
  // server on the original before the grade. Values are -100..100 (%), mapped to SceneParams.
  /** @type {Array<[string, string, string, number]>} */
  const CONTROLS = [
    ["haze", "Haze", "+ adds distance haze, − removes it", 1],
    ["clarity_near", "Clarity (near)", "local contrast in the foreground", 1],
    ["clarity_far", "Clarity (far)", "local contrast in the distance", 1],
    ["sky_exposure", "Sky exposure", "brighten or darken the sky only (stops ×2)", 2],
    ["sky_warmth", "Sky warmth", "", 1],
    ["sky_saturation", "Sky saturation", "", 1],
  ];
  function set(k, v, scale) { S.scene[k] = (v / 100) * scale; applyScene(); }
  function reset() { S.scene = ZERO_SCENE(); applyScene(); commit(); }
</script>

{#each CONTROLS as [k, label, hint, scale]}
  <label class="row" title={hint}>{label}
    <input type="range" min="-100" max="100" value={Math.round((S.scene[k] / scale) * 100)}
           oninput={(e) => set(k, +e.currentTarget.value, scale)} onchange={commit} disabled={!S.id} />
    <output>{Math.round((S.scene[k] / scale) * 100)}</output>
  </label>
{/each}
<div class="row"><button class="btn small" onclick={reset} disabled={!Object.values(S.scene).some(Boolean)}>Reset scene</button>
  <span class="muted small">First use on a photo takes ~2 s (depth + sky).</span></div>

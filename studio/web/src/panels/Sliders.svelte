<script>
  // A column of pipeline sliders bound to the photo's adjustments (S.p), by name. `tracks` gives
  // optional gradients by name; `labels` overrides the schema's labels.
  import { S, slider, setSlider, commit } from "../lib/studio.svelte.js";
  import { SLIDERS } from "../lib/pipeline.js";
  import Slider from "../ui/Slider.svelte";
  /** @type {{ names: string[], tracks?: Record<string, string>, labels?: Record<string, string> }} */
  let { names, tracks = {}, labels = {} } = $props();
  const spec = (n) => SLIDERS.find((s) => s.name === n);
</script>

<div class="stack">
  {#each names as n (n)}
    {@const s = spec(n)}
    <Slider id={`s_${n}`} label={labels[n] ?? s.label} min={s.min} max={s.max} step={s.step} reset={s.default}
            value={slider(n)} disabled={!S.id} track={tracks[n]}
            oninput={(v) => setSlider(n, v)} onchange={commit} />
  {/each}
</div>

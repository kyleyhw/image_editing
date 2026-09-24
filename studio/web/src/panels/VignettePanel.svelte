<script>
  import { S, commit } from "../lib/studio.svelte.js";
  import Slider from "../ui/Slider.svelte";
</script>

<Slider id="vig" label="Vignette" min={0} max={60} value={S.d.vig} disabled={!S.params}
        oninput={(v) => (S.d.vig = v)} onchange={commit} />

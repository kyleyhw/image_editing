<script>
  import Sliders from "./Sliders.svelte";
</script>

<div class="sub-head"><span>Grain</span></div>
<Sliders names={["grain", "grain_size", "grain_roughness"]} />
<div class="sub-head"><span>Film</span></div>
<Sliders names={["halation"]} />
<p class="hint">Halation is the red-orange glow film shows around bright lights.</p>

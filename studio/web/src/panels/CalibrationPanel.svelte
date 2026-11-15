<script>
  import Sliders from "./Sliders.svelte";
</script>

<p class="hint">Moves the red, green and blue primaries, as a camera profile does. Neutrals are not affected.</p>
<Sliders names={["cal_red_hue", "cal_red_sat", "cal_green_hue", "cal_green_sat", "cal_blue_hue", "cal_blue_sat", "cal_shadow_tint"]} />

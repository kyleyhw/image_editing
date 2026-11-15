<script>
  import Sliders from "./Sliders.svelte";
</script>

<div class="sub-head"><span>Presence</span></div>
<Sliders names={["texture", "clarity", "dehaze"]} />
<div class="sub-head"><span>Sharpening</span></div>
<Sliders names={["sharpen", "sharpen_masking"]} />
<div class="sub-head"><span>Noise reduction</span></div>
<Sliders names={["nr_luma", "nr_colour"]} labels={{ nr_luma: "Luminance", nr_colour: "Colour" }} />
<p class="hint">Sharpening and noise act at the pixel level: judge them at 100 % (Z).</p>

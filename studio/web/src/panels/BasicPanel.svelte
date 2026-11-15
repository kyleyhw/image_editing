<script>
  import { S } from "../lib/studio.svelte.js";
  import Sliders from "./Sliders.svelte";
  import { I } from "../lib/icons.js";
  const T = { temp: "linear-gradient(90deg,#3d78d8,#c9c9c9 50%,#e0b53e)", tint: "linear-gradient(90deg,#46b04a,#c9c9c9 50%,#c24fc4)" };
</script>

<div class="sub-head"><span>White balance</span>
  <button class="icon-btn small" class:on={S.tool === "wb"} title="Pick something that should be neutral grey (W)"
          aria-label="White balance picker" onclick={() => (S.tool = S.tool === "wb" ? null : "wb")}>{@html I.dropper}</button>
</div>
<Sliders names={["temp", "tint"]} tracks={T} />
<div class="sub-head"><span>Tone</span></div>
<Sliders names={["exposure", "contrast", "highlights", "shadows", "whites", "blacks"]} />
<div class="sub-head"><span>Presence</span></div>
<Sliders names={["vibrance", "saturation"]} />

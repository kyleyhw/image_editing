<script>
  import { S, commit } from "../lib/studio.svelte.js";
</script>

<label class="row">Amount
  <input id="vig" type="range" min="0" max="60" value={S.d.vig} oninput={(e) => (S.d.vig = +e.currentTarget.value)} onchange={commit} />
  <output>{S.d.vig}</output>
</label>

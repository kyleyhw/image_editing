<script>
  import { S, resetToModel, remember, personalise } from "../lib/studio.svelte.js";
</script>

<div class="actions">
  <button id="btnReset" class="btn" onclick={resetToModel} disabled={!S.params}>Reset to model</button>
  <button id="btnRemember" class="btn" title="Use this edit to personalise the style" onclick={remember} disabled={!S.params}>Remember this edit</button>
  <button id="btnPersonalise" class="btn" onclick={personalise} disabled={!S.style}>Personalise style</button>
</div>

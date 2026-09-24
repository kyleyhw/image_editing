<script>
  // A blurred copy of the *original* photo behind the translucent panels. It changes only when
  // another photo is opened, never while editing.
  import { S, photoURL } from "../lib/studio.svelte.js";
</script>

<div class="backdrop" aria-hidden="true">
  {#key S.id}{#if S.id}<img src={photoURL(S.id, 240)} alt="" />{/if}{/key}
</div>

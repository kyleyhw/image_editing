<script>
  /** @type {{ open?: boolean, title: string, children?: import('svelte').Snippet, footer?: import('svelte').Snippet }} */
  let { open = $bindable(false), title, children = undefined, footer = undefined } = $props();
  let dlg;
  $effect(() => { if (!dlg) return; if (open && !dlg.open) dlg.showModal(); else if (!open && dlg.open) dlg.close(); });
</script>

<dialog bind:this={dlg} onclose={() => (open = false)} aria-label={title}>
  <h2>{title}</h2>
  {@render children?.()}
  <div class="row end">{@render footer?.()}<button class="btn" onclick={() => (open = false)}>Close</button></div>
</dialog>

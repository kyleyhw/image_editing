<script>
  /** @type {{ title: string, open?: boolean, edited?: boolean, onreset?: () => void, children: import('svelte').Snippet }} */
  let { title, open = true, edited = false, onreset = undefined, children } = $props();
</script>

<details class="sec" {open}>
  <summary>
    <span class="sec-title">{title}{#if edited}<span class="edited-dot" title="Edited"></span>{/if}</span>
    {#if edited && onreset}<button class="link-btn sec-reset" onclick={(e) => { e.preventDefault(); onreset(); }}>Reset</button>{/if}
    <span class="sec-chev" aria-hidden="true"></span>
  </summary>
  <div class="sec-body">{@render children()}</div>
</details>

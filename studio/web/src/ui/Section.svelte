<script>
  /** @type {{ title: string, open?: boolean, children: import('svelte').Snippet, aside?: import('svelte').Snippet }} */
  let { title, open = true, children, aside = undefined } = $props();
</script>

<details class="sec" {open}>
  <summary><span class="sec-title">{title}</span>{@render aside?.()}<span class="sec-chev" aria-hidden="true"></span></summary>
  <div class="sec-body">{@render children()}</div>
</details>

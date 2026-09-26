<script>
  // A look's example: an openly licensed photo of the subject its tutorial was written for,
  // before/after with the trained look, plus the tutorials the recipe comes from.
  let { look } = $props();
  const base = import.meta.env.BASE_URL;
  let x = $state(50);
  const ex = $derived(look.example);
  const GRADES = { A: "Tutorial gives the numbers", B: "Tutorial gives some numbers; the rest are directions",
                   C: "Tutorial gives mostly directions; amounts were chosen" };
  function move(e) { const r = e.currentTarget.getBoundingClientRect(); x = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)); }
  const lic = (p) => `${(p.license || "").toUpperCase()}${p.license_version && !["cc0", "pdm"].includes(p.license) ? " " + p.license_version : ""}`;
</script>

<div class="look-ex">
  {#if ex}
    <div class="ex-frame" role="img" aria-label={`${look.title || look.name}: before and after`} onpointermove={move}>
      <img src={base + ex.before} alt="" loading="lazy" />
      <img class="after" src={base + ex.after} alt="" loading="lazy" style={`clip-path: inset(0 0 0 ${x}%)`} />
      <span class="sc-line" style={`left:${x}%`}></span>
      <span class="tag tag-r">Before</span><span class="tag tag-l">After</span>
    </div>
  {/if}
  {#if look.subject}<p class="ex-line"><b>Best for</b> {look.subject}</p>{/if}
  {#if look.tutorials?.length}
    <p class="ex-line"><b>Recipe from</b></p>
    <ul class="ex-links">
      {#each look.tutorials as t}<li><a href={t.url} target="_blank" rel="noopener">{t.title}</a></li>{/each}
    </ul>
    <p class="ex-note">The authors' own before/after photos are on their pages.</p>
  {/if}
  {#if look.grade}<p class="ex-note">{GRADES[look.grade]} ({look.grade}).</p>{/if}
  {#if ex?.photo}
    <p class="ex-note">Example photo:
      <a href={ex.photo.url} target="_blank" rel="noopener">{ex.photo.title || "untitled"}</a>
      by {ex.photo.creator || "unknown"}, {lic(ex.photo)}.</p>
  {/if}
</div>

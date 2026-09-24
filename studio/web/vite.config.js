import { defineConfig, loadEnv } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { fileURLToPath } from "node:url";
import { cpSync, existsSync } from "node:fs";

// Two builds from one source:
//   npm run build        -> ../dist, served by studio/server.py at "/" (model on the server)
//   npm run build:pages  -> ../../site, the GitHub Pages build: VITE_STATIC=1, the model runs in the
//                           browser (ONNX Runtime Web); relative base so it works under /<repo>/.
// The browser backend is swapped for a stub in the server build so ONNX Runtime isn't bundled.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const pages = mode === "pages" || env.VITE_STATIC === "1";
  const lib = (f) => fileURLToPath(new URL(`./src/lib/${f}`, import.meta.url));
  // Pages build: also ship ./pages (models/styles.json, committed; models/rn18.onnx, made in CI).
  const copyPages = { name: "copy-pages", closeBundle() { if (existsSync("pages")) cpSync("pages", "../../site", { recursive: true }); } };
  return {
    plugins: pages ? [svelte(), copyPages] : [svelte()],
    base: pages ? "./" : "/",
    define: { "import.meta.env.VITE_STATIC": JSON.stringify(pages ? "1" : "0") },
    resolve: { alias: { "photostyle:browser-backend": lib(pages ? "browser.js" : "browser-stub.js") } },
    build: pages ? { outDir: "../../site", emptyOutDir: true, assetsDir: "assets" }
                 : { outDir: "../dist", emptyOutDir: true, assetsDir: "assets" },
    server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  };
});

import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Built into ../dist and served by studio/server.py at "/". During development,
// `npm run dev` proxies the API to a running `photostyle serve`.
export default defineConfig({
  plugins: [svelte()],
  base: "/",
  build: { outDir: "../dist", emptyOutDir: true, assetsDir: "assets" },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});

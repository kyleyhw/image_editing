import { mount } from "svelte";
import "@fontsource-variable/inter";
import App from "./App.svelte";
import "./app.css";

mount(App, { target: document.getElementById("app") });

// Hosted build: work offline and install to the home screen (public/sw.js, manifest.webmanifest).
if (import.meta.env.VITE_STATIC === "1" && "serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js").catch(() => {}));
}

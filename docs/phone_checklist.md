# Phone checklist

The automated tests drive the editor in headless Chromium at a 390×844 phone viewport, with a
software GPU. They cannot test a real phone's GPU, memory, touch input, photo library or
installation. This checklist covers those. It takes about ten minutes. Note the phone, the
browser and anything that fails, and send the result back.

Open <https://kyleyhw.github.io/image_editing/> in the phone's browser.

| # | Step | Expected |
|---|---|---|
| 1 | Wait for the model to load (first visit downloads about 27 MB) | The model bar fills; the sample photos appear |
| 2 | Tap a sample photo | It opens; the look strip fills with previews |
| 3 | Drag the before/after divider | It follows your finger without lag |
| 4 | Pinch to zoom on the photo, then double-tap | It zooms in; double-tap returns to fit |
| 5 | Tap **Adjust**, open **Basic**, drag Exposure | The photo updates while dragging (smooth, not stepwise) |
| 6 | Open **Detail**, set Clarity to +60 | The photo updates within about a second |
| 7 | Open **Masks**, tap **+ Radial**, drag its centre handle | The handle follows your finger; the effect moves |
| 8 | Tap **+ Brush**, paint a stroke with one finger | The stroke appears where you painted |
| 9 | Open **Crop & straighten**, choose 4 : 5, tap **Crop**, drag a corner | The frame resizes; tap **Done** |
| 10 | Tap the photos button, add a photo from your camera roll | It opens; for an iPhone ProRAW or other raw file, wait up to a minute |
| 11 | Export settings: tick **Share instead of download**, then **Export** as JPEG | The share sheet opens; **Save Image** puts it in Photos |
| 12 | Close the browser tab, reopen the site | Your photos and edits are still there |
| 13 | Add to Home Screen (Safari: Share → Add to Home Screen; Chrome: menu → Install app), open it from the icon | It opens full-screen with the Irodori icon |
| 14 | Turn on aeroplane mode, open the installed app | It opens and edits the photos kept from before |

If a step fails, note:

- the step number;
- the phone model and OS version;
- the browser and its version;
- what happened instead;
- a screenshot if possible.

Known limits:

- **Raw files.** Decoding happens on the phone, in WebAssembly. A 24-megapixel raw file can take
  30 s or more and needs several hundred megabytes of memory.
- **Wide-colour export.** Display P3 export needs a browser that tags canvases with their colour
  space (current Safari and Chrome do).

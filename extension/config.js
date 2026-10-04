// Settings shared by the background worker, content script and popup.
// Site-specific selectors and headers live here so a markup change is a one-file fix.
var ATX = globalThis.ATX || (globalThis.ATX = {});

ATX.config = {
  server: "http://127.0.0.1:8765",

  // mycomic.com: pages are <img class="page">; the first few have src, the rest
  // data-src, which lozad moves into src (and marks data-loaded) near the viewport.
  pageSelector: "img.page",
  // The image host answers 403 without this Referer and sends no CORS headers,
  // so the background worker fetches images, with a session rule adding it.
  imageHost: "biccam.com",
  referer: "https://mycomic.com/",

  tiers: [
    { id: "very_quick", label: "Very quick" },
    { id: "quick", label: "Quick" },
    { id: "accurate", label: "Accurate" },
  ],
  defaults: { enabled: true, tier: "quick", highlight: true },

  maxInFlight: 2, // pages being translated at once
  prefetchAhead: 2, // pages past the one on screen
  keepTranslated: 6, // pages further than this from the screen go back to the original

  // Drawing
  font: '"Comic Sans MS", "Comic Neue", "Segoe UI", sans-serif',
  fontWeight: 700,
  maxFontPx: 28,
  minFontPx: 10,
  comfortableFontPx: 14, // narrow boxes are widened until text fits at least this size
  lineHeight: 1.15,
  padPx: 4, // box padding around the OCR'd text
};

// Talking to the local server. Runs in the background worker; the dev test page
// (dev/harness.html) loads it directly.
var ATX = globalThis.ATX || (globalThis.ATX = {});

ATX.api = {
  // Fetch a page image, send it to the server, and return the translation with
  // the image bytes (base64), which the content script needs to draw on.
  async translate({ url, prevUrl, tier }) {
    const img = await fetch(url, { credentials: "omit" });
    if (!img.ok) throw new Error(`image ${img.status} for ${url}`);
    const type = img.headers.get("content-type") || "image/jpeg";
    const bytes = await img.arrayBuffer();

    const form = new FormData();
    form.append("image", new Blob([bytes], { type }), "page");
    form.append("tier", tier);
    form.append("page_url", url);
    if (prevUrl) form.append("prev_url", prevUrl);
    const res = await fetch(`${ATX.config.server}/translate`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`server ${res.status}: ${await res.text()}`);
    return { result: await res.json(), image: toBase64(bytes), type };
  },

  async health() {
    const res = await fetch(`${ATX.config.server}/health`, { signal: AbortSignal.timeout(2000) });
    return res.json();
  },

  async warm(tier) {
    const form = new FormData();
    form.append("tier", tier);
    const res = await fetch(`${ATX.config.server}/warm`, { method: "POST", body: form });
    return res.json();
  },
};

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

// Popup: on/off switch, tier picker, server status, and this tab's progress.
const C = ATX.config;
const $ = (id) => document.getElementById(id);

const TIER_HINTS = {
  very_quick: "Fastest. Fine for short dialogue; drops detail in long narration.",
  quick: "Hy-MT2, one bubble at a time. Complete but literal.",
  accurate: "Uses surrounding text for names and pronouns. Slower.",
};

let settings = { ...C.defaults };

async function init() {
  settings = await chrome.storage.local.get(C.defaults);
  $("enabled").checked = settings.enabled;
  document.body.classList.toggle("off", !settings.enabled);
  renderTiers();

  $("enabled").addEventListener("change", async (e) => {
    settings.enabled = e.target.checked;
    document.body.classList.toggle("off", !settings.enabled);
    await chrome.storage.local.set({ enabled: settings.enabled });
  });
  $("original").addEventListener("click", async () => showPage(await toTab({ type: "toggleOriginal" })));
  $("copy-urls").addEventListener("click", copyUrls);

  refresh();
  setInterval(refresh, 1500);
}

function renderTiers() {
  const box = $("tiers");
  box.replaceChildren(...C.tiers.map((t) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = t.label;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(t.id === settings.tier));
    b.addEventListener("click", () => setTier(t.id));
    return b;
  }));
  $("tier-hint").textContent = TIER_HINTS[settings.tier] || "";
}

async function setTier(tier) {
  if (tier === settings.tier) return;
  settings.tier = tier;
  renderTiers();
  await chrome.storage.local.set({ tier });
  chrome.runtime.sendMessage({ type: "warm", tier }); // load the model now; don't wait
  refresh();
}

async function refresh() {
  const health = await chrome.runtime.sendMessage({ type: "health" }).catch(() => null);
  showServer(health);
  showPage(await toTab({ type: "status" }));
}

function showServer(h) {
  const dot = $("dot");
  if (!h || h.ok === undefined) { // the background worker couldn't reach the server
    dot.className = "dot down";
    $("server").textContent = "Server not running";
    $("device").textContent = "Start it with: python -m atx.app (in server/)";
    $("note").textContent = "";
    return;
  }
  if (!h.ok || h.loading) {
    dot.className = "dot loading";
    $("server").textContent = h.error ? `Server failed: ${h.error}` : `Loading ${h.loading || "models"}…`;
  } else {
    dot.className = "dot ok";
    const tier = C.tiers.find((t) => t.id === h.tier)?.label || h.tier || "none";
    $("server").textContent = `Ready · ${tier}${h.resolved && h.resolved !== h.tier ? ` (${h.resolved})` : ""}`;
  }
  $("device").textContent = h.device || "";
  const notes = [h.note, ...Object.entries(h.fallbacks || {}).map(([k, v]) => `${k}: ${v}`)].filter(Boolean);
  $("note").textContent = notes.join(" · ");
}

function showPage(s) {
  $("page-section").hidden = !s;
  if (!s) return;
  const parts = [`${s.done} / ${s.total} pages translated`];
  if (s.busy) parts.push(`${s.busy} in progress`);
  if (s.avgMs) parts.push(`~${(s.avgMs / 1000).toFixed(1)} s each`);
  $("progress").textContent = parts.join(" · ");
  $("original").setAttribute("aria-pressed", String(s.showOriginal));
  $("original").textContent = s.showOriginal ? "Show translation" : "Show original";
  $("page-error").textContent = s.errors ? `${s.errors} failed: ${s.lastError || ""}` : "";
}

// Message the content script in the active tab; null when it isn't a chapter page.
async function toTab(msg) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return null;
  return chrome.tabs.sendMessage(tab.id, msg).catch(() => null);
}

async function copyUrls() {
  const urls = await toTab({ type: "pageUrls" });
  if (!urls) {
    $("copy-result").textContent = "Open a chapter page first.";
    return;
  }
  await navigator.clipboard.writeText(JSON.stringify(urls, null, 1));
  const n = Object.values(Object.values(urls)[0])[0].length;
  $("copy-result").textContent = `Copied ${n} URLs (samples/urls.json format).`;
}

init();

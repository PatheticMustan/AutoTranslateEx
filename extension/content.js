// Content script for mycomic.com chapter pages.
//
// Every page's URL is in the HTML from the start (data-src or src), so the whole
// chapter is known upfront. Pages are translated in order from the one on
// screen, at most maxInFlight at a time and prefetchAhead pages ahead. A
// translated page replaces the <img> src with a blob URL; pages far off screen
// go back to the original to keep memory down, and come back from the server's
// cache when scrolled to again.
(() => {
  const C = ATX.config;
  const pages = [...document.querySelectorAll(C.pageSelector)].map((img, i) => ({
    i,
    img,
    url: img.getAttribute("data-src") || img.getAttribute("src"),
    status: "idle", // idle | busy | done | error
    blobUrl: null,
    retryAt: 0,
  }));
  if (!pages.length) return;

  const state = {
    enabled: C.defaults.enabled,
    tier: C.defaults.tier,
    showOriginal: false,
    current: 0,
    inFlight: 0,
    generation: 0, // bumped on tier change / disable, so stale results are dropped
    lastError: null,
    ms: [], // server time per translated page, for the popup
  };

  // ---- which page is on screen ----------------------------------------------
  const visibleHeight = new Map();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) visibleHeight.set(Number(e.target.dataset.atxPage), e.intersectionRect.height);
    let best = state.current, bestH = -1;
    for (const [i, h] of visibleHeight) if (h > bestH) [best, bestH] = [i, h];
    if (best !== state.current) {
      state.current = best;
      evictFarPages();
    }
    pump();
  }, { threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] });

  // lozad (or anything else) setting src on a translated page: put ours back.
  const mo = new MutationObserver((records) => {
    for (const r of records) {
      const p = pages[Number(r.target.dataset.atxPage)];
      if (p && shouldShow(p) && r.target.src !== p.blobUrl) apply(p);
    }
  });

  for (const p of pages) {
    p.img.dataset.atxPage = String(p.i);
    io.observe(p.img);
    mo.observe(p.img, { attributes: true, attributeFilter: ["src"] });
  }

  // ---- queue ----------------------------------------------------------------
  function pump() {
    if (!state.enabled) return;
    const last = Math.min(pages.length - 1, state.current + C.prefetchAhead);
    for (let i = state.current; i <= last && state.inFlight < C.maxInFlight; i++) {
      const p = pages[i];
      if (p.status === "idle" || (p.status === "error" && Date.now() >= p.retryAt)) start(p);
    }
  }

  async function start(p) {
    p.status = "busy";
    state.inFlight++;
    const gen = state.generation;
    try {
      const resp = await chrome.runtime.sendMessage({
        type: "translate", url: p.url, prevUrl: pages[p.i - 1]?.url, tier: state.tier,
      });
      if (!resp || resp.error) throw new Error(resp?.error || "no response from the extension");
      const blob = await ATX.render(resp.image, resp.type, resp.result);
      if (gen !== state.generation) return;
      p.blobUrl = URL.createObjectURL(blob);
      p.status = "done";
      state.lastError = null;
      state.ms.push(resp.result.ms?.total ?? 0);
      apply(p);
    } catch (e) {
      if (gen !== state.generation) return;
      p.status = "error";
      p.retryAt = Date.now() + 10000;
      state.lastError = String(e?.message || e);
      console.warn("[AutoTranslateEx]", p.url, state.lastError);
      setTimeout(pump, 10000);
    } finally {
      state.inFlight--;
      pump();
    }
  }

  // ---- showing pages --------------------------------------------------------
  function shouldShow(p) {
    return state.enabled && !state.showOriginal && p.status === "done" && p.blobUrl;
  }

  function apply(p) {
    if (!shouldShow(p) || p.img.src === p.blobUrl) return;
    p.img.setAttribute("data-loaded", "true"); // lozad skips loaded images
    p.img.src = p.blobUrl;
  }

  function restore(p) {
    if (p.img.src.startsWith("blob:")) p.img.src = p.url;
  }

  function forget(p) {
    restore(p);
    if (p.blobUrl) URL.revokeObjectURL(p.blobUrl);
    p.blobUrl = null;
    p.status = "idle";
  }

  function evictFarPages() {
    for (const p of pages) {
      if (p.status === "done" && Math.abs(p.i - state.current) > C.keepTranslated) forget(p);
    }
  }

  function resetAll() {
    state.generation++;
    for (const p of pages) forget(p);
  }

  // ---- settings and popup ---------------------------------------------------
  chrome.storage.local.get(C.defaults).then((s) => {
    state.enabled = s.enabled;
    state.tier = s.tier;
    pump();
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if (changes.tier && changes.tier.newValue !== state.tier) {
      state.tier = changes.tier.newValue;
      resetAll();
    }
    if (changes.enabled) {
      state.enabled = changes.enabled.newValue;
      if (!state.enabled) resetAll();
    }
    pump();
  });

  const messages = {
    status: () => ({
      total: pages.length,
      done: pages.filter((p) => p.status === "done").length,
      busy: pages.filter((p) => p.status === "busy").length,
      errors: pages.filter((p) => p.status === "error").length,
      current: state.current + 1,
      showOriginal: state.showOriginal,
      lastError: state.lastError,
      avgMs: state.ms.length ? Math.round(state.ms.reduce((a, b) => a + b) / state.ms.length) : null,
    }),
    toggleOriginal: () => {
      state.showOriginal = !state.showOriginal;
      for (const p of pages) (state.showOriginal ? restore(p) : apply(p));
      return messages.status();
    },
    // Dev: this chapter's page URLs in server/samples/urls.json format.
    pageUrls: () => {
      const back = [...document.querySelectorAll('a[href*="/comics/"]')]
        .find((a) => a.textContent.trim() === "返回目錄") || document.querySelector('a[href*="/comics/"]');
      const series = back?.href.match(/\/comics\/(\d+)/)?.[1] || "unknown";
      const chapter = location.pathname.match(/\/chapters\/(\d+)/)?.[1] || "unknown";
      return { [series]: { [chapter]: pages.map((p) => p.url) } };
    },
  };

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    const handler = messages[msg?.type];
    if (!handler) return false;
    sendResponse(handler());
    return false;
  });
})();

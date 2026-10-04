// Stand-in for the chrome.* APIs the content script uses, so dev/harness.html can
// run the real content.js and render.js on a mock chapter without installing
// the extension. Messages go straight to ATX.api (what background.js does).
(() => {
  const tabListeners = [];
  const storageListeners = [];
  const store = { ...ATX.config.defaults };

  globalThis.chrome = {
    runtime: {
      async sendMessage(msg) {
        try {
          if (msg.type === "translate") return await ATX.api.translate(msg);
          if (msg.type === "health") return await ATX.api.health();
          if (msg.type === "warm") return await ATX.api.warm(msg.tier);
        } catch (e) {
          return { error: String(e?.message || e) };
        }
        return undefined;
      },
      onMessage: { addListener: (fn) => tabListeners.push(fn) },
    },
    storage: {
      local: {
        async get(defaults) { return { ...defaults, ...store }; },
        async set(values) {
          const changes = {};
          for (const [k, v] of Object.entries(values)) {
            changes[k] = { oldValue: store[k], newValue: v };
            store[k] = v;
          }
          storageListeners.forEach((fn) => fn(changes, "local"));
        },
      },
      onChanged: { addListener: (fn) => storageListeners.push(fn) },
    },
  };

  // What the popup does: message the content script.
  ATX.devTab = (msg) => {
    let reply;
    tabListeners.forEach((fn) => fn(msg, {}, (r) => { reply = r; }));
    return reply;
  };
})();

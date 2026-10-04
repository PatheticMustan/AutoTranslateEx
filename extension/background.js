// Service worker: fetches page images (adding the Referer the image host needs)
// and relays them to the local server for the content script and popup.
importScripts("config.js", "api.js");

const REFERER_RULE = 1;

// Session rules survive service-worker restarts but not a browser restart, so
// (re)install on every start. Only this extension's own requests (tabId -1) to
// the image host are changed; the site's requests are left alone.
const ruleReady = chrome.declarativeNetRequest.updateSessionRules({
  removeRuleIds: [REFERER_RULE],
  addRules: [{
    id: REFERER_RULE,
    priority: 1,
    action: {
      type: "modifyHeaders",
      requestHeaders: [{ header: "referer", operation: "set", value: ATX.config.referer }],
    },
    condition: {
      requestDomains: [ATX.config.imageHost],
      tabIds: [chrome.tabs.TAB_ID_NONE],
      resourceTypes: ["xmlhttprequest"],
    },
  }],
});

const handlers = {
  async translate(msg) {
    await ruleReady;
    return ATX.api.translate(msg);
  },
  health: () => ATX.api.health(),
  warm: (msg) => ATX.api.warm(msg.tier),
};

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const handler = handlers[msg?.type];
  if (!handler) return false;
  handler(msg)
    .then(sendResponse)
    .catch((e) => sendResponse({ error: String(e?.message || e) }));
  return true; // respond asynchronously
});

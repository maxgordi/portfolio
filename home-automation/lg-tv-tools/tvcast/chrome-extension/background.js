// Фоновый скрипт: мост между страницей YouTube / окошком расширения и помощником на Python
// (Native Messaging). Вся работа с телевизором — в помощнике: расширение не умеет UDP (WOL, SSDP)
// и не может открыть WebSocket к телевизору без проверки сертификата/ключа пары.

const HOST = "com.tvcast.helper";
const HELPER_MISSING =
  "Не установлен помощник. Дважды кликни «Установить помощник.command» в папке ~/Public/tvcast и перезапусти Chrome.";
const YT_LINKS = [
  "https://www.youtube.com/watch*", "https://www.youtube.com/shorts/*", "https://youtu.be/*",
  "https://m.youtube.com/watch*", "https://www.youtube.com/live/*",
];

// ---------- помощник ----------

function runHost(msg, onProgress) {
  return new Promise((resolve) => {
    let port;
    try {
      port = chrome.runtime.connectNative(HOST);
    } catch (e) {
      resolve({ ok: false, code: "no_helper", error: HELPER_MISSING });
      return;
    }
    let done = false;
    port.onMessage.addListener((m) => {
      if (m.type === "progress") {
        if (onProgress) onProgress(m.text);
        return;
      }
      done = true;
      resolve(m);
      port.disconnect();
    });
    port.onDisconnect.addListener(() => {
      if (done) return;
      const err = (chrome.runtime.lastError && chrome.runtime.lastError.message) || "";
      let text;
      if (/not found/i.test(err)) text = HELPER_MISSING;
      else if (/forbidden/i.test(err))
        text = "Помощник установлен для другого расширения. Запусти «Установить помощник.command» ещё раз.";
      else text = "Помощник неожиданно завершился" + (err ? ` (${err})` : "") +
        ". Подробности: ~/Library/Logs/tvcast-helper.log";
      resolve({ ok: false, code: "no_helper", error: text });
    });
    port.postMessage(msg);
  });
}

// ---------- значок: статус телевизора ----------

async function setBadge(power) {
  const map = {
    on: { text: "●", color: "#1e8e3e", title: "Телевизор в сети" },
    off: { text: "●", color: "#9aa0a6", title: "Телевизор выключен" },
    error: { text: "!", color: "#d93025", title: "Есть проблема — открой окошко расширения" },
  };
  const b = map[power];
  if (!b) {
    await chrome.action.setBadgeText({ text: "" });
    return;
  }
  await chrome.action.setBadgeText({ text: b.text });
  await chrome.action.setBadgeBackgroundColor({ color: b.color });
  if (chrome.action.setBadgeTextColor) await chrome.action.setBadgeTextColor({ color: "#ffffff" });
  await chrome.action.setTitle({ title: "На телевизор — " + b.title });
}

function badgeFromResult(r) {
  if (r.power === "on" || r.power === "off") return setBadge(r.power);
  if (r.ok && r.state) return setBadge("on");
  if (r.code === "off" || r.code === "off_nomac") return setBadge("off");
  if (r.code === "no_helper" || r.code === "lan") return setBadge("error");
}

async function refreshBadge() {
  const r = await runHost({ cmd: "ping" });
  badgeFromResult(r);
  return r;
}

// ---------- отправка видео ----------

async function toast(tabId, text, kind) {
  if (tabId == null) return;
  try {
    await chrome.tabs.sendMessage(tabId, { cmd: "toast", text, kind });
  } catch (e) { /* на вкладке нет нашего скрипта */ }
}

async function cast(info, tabId) {
  const r = await runHost({ cmd: "cast", videoId: info.videoId, time: info.time || 0 },
    (text) => toast(tabId, text, "info"));
  badgeFromResult(r);
  return r;
}

// Кнопка на странице, Alt+T и пункт меню идут одним путём: скрипт страницы сам берёт видео
// и секунду, ставит паузу и присылает {cmd: "cast"}.
async function triggerInTab(tab) {
  if (!tab || !/^https:\/\/www\.youtube\.com\//.test(tab.url || "")) return false;
  try {
    await chrome.tabs.sendMessage(tab.id, { cmd: "trigger" });
  } catch (e) {
    // вкладка открыта до установки расширения — подключаем скрипт сейчас
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["content.css"] });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    await chrome.tabs.sendMessage(tab.id, { cmd: "trigger" });
  }
  return true;
}

function videoFromUrl(u) {
  try {
    const url = new URL(u);
    let id = null;
    if (url.hostname === "youtu.be") id = url.pathname.slice(1, 12);
    else if (/^\/(shorts|live|embed)\//.test(url.pathname)) id = url.pathname.split("/")[2];
    else id = url.searchParams.get("v");
    if (!id || !/^[A-Za-z0-9_-]{11}$/.test(id)) return null;
    const t = url.searchParams.get("t") || url.searchParams.get("start") || "0";
    let sec = 0;
    const m = String(t).match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$/);
    if (m) sec = (+m[1] || 0) * 3600 + (+m[2] || 0) * 60 + (+m[3] || 0);
    return { videoId: id, time: sec };
  } catch (e) {
    return null;
  }
}

// ---------- события ----------

function setupMenus() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "cast-page",
      title: "На телевизор",
      contexts: ["page", "video", "frame"],
      documentUrlPatterns: ["https://www.youtube.com/watch*", "https://www.youtube.com/shorts/*",
        "https://www.youtube.com/live/*"],
    });
    chrome.contextMenus.create({
      id: "cast-link",
      title: "Открыть ссылку на телевизоре",
      contexts: ["link"],
      targetUrlPatterns: YT_LINKS,
    });
  });
}

chrome.runtime.onInstalled.addListener(() => {
  setupMenus();
  chrome.alarms.create("status", { periodInMinutes: 1 });
  refreshBadge();
});

chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create("status", { periodInMinutes: 1 });
  refreshBadge();
});

chrome.alarms.onAlarm.addListener((a) => {
  if (a.name === "status") refreshBadge();
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "cast-page") {
    await triggerInTab(tab);
  } else if (info.menuItemId === "cast-link") {
    const v = videoFromUrl(info.linkUrl);
    const onYT = tab && /^https:\/\/www\.youtube\.com\//.test(tab.url || "");
    const tabId = onYT ? tab.id : null;
    if (!v) {
      toast(tabId, "Это не ссылка на видео YouTube.", "err");
      return;
    }
    toast(tabId, "Отправляю на телевизор…", "info");
    const r = await cast(v, tabId);
    if (onYT) toast(tabId, r.ok ? r.text : r.error, r.ok ? "ok" : "err");
    else if (!r.ok) setBadge("error");
  }
});

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command !== "cast-to-tv") return;
  if (!tab) [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  await triggerInTab(tab);
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.cmd === "cast") {
    cast(msg, sender.tab ? sender.tab.id : msg.tabId).then(sendResponse);
    return true;
  }
  if (msg.cmd === "host") {
    runHost(msg.payload).then((r) => {
      badgeFromResult(r);
      sendResponse(r);
    });
    return true;
  }
  if (msg.cmd === "triggerActive") {
    chrome.tabs.query({ active: true, lastFocusedWindow: true }).then(async ([tab]) => {
      sendResponse({ ok: await triggerInTab(tab).catch(() => false) });
    });
    return true;
  }
});

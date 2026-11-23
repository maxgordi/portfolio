// Окошко расширения: статус телевизора, громкость, пауза, включить/выключить.
const $ = (id) => document.getElementById(id);
let power = "unknown";
let confirmOff = false;

const host = (payload) => chrome.runtime.sendMessage({ cmd: "host", payload });

function msg(text, kind = "info") {
  const m = $("msg");
  if (!text) {
    m.hidden = true;
    return;
  }
  m.hidden = false;
  m.textContent = text;
  m.className = "msg " + kind;
}

function render(r) {
  if (r.power) power = r.power;
  const dot = $("dot");
  dot.className = "dot" + (power === "on" ? " on" : r.ok === false && power !== "off" ? " err" : "");
  if (power === "on") {
    $("title").textContent = "Телевизор LG — в сети";
    const parts = [];
    if (r.appName) parts.push("Сейчас: " + r.appName);
    if (r.muted) parts.push("звук выключен");
    $("sub").textContent = parts.join(" · ") || "Включён";
  } else if (power === "off") {
    $("title").textContent = "Телевизор LG — выключен";
    $("sub").textContent = r.canWake === false ? "Включи его пультом один раз" : "Можно включить отсюда";
  } else {
    $("title").textContent = "Телевизор LG";
    $("sub").textContent = "Статус неизвестен";
  }
  if (typeof r.volume === "number") $("vol").textContent = r.volume;
  document.querySelectorAll(".tv-on").forEach((b) => (b.disabled = power !== "on"));
  const p = $("power");
  confirmOff = false;
  if (power === "on") {
    p.textContent = "⏻ Выключить телевизор";
    p.className = "danger";
  } else {
    p.textContent = "⏻ Включить телевизор";
    p.className = "";
  }
  if (r.ok === false && r.error) msg(r.error, "err");
}

async function busy(btn, fn) {
  const all = document.querySelectorAll("button");
  const was = [...all].map((b) => b.disabled);
  all.forEach((b) => (b.disabled = true));
  try {
    return await fn();
  } finally {
    all.forEach((b, i) => (b.disabled = was[i]));
    if (btn === $("power") || btn === null) render({});
  }
}

async function refresh() {
  const r = await host({ cmd: "status" });
  render(r);
}

$("volup").onclick = () => busy($("volup"), async () => {
  const r = await host({ cmd: "volume", dir: "up" });
  r.ok ? ($("vol").textContent = r.volume, msg("")) : msg(r.error, "err");
});
$("voldown").onclick = () => busy($("voldown"), async () => {
  const r = await host({ cmd: "volume", dir: "down" });
  r.ok ? ($("vol").textContent = r.volume, msg("")) : msg(r.error, "err");
});
$("playpause").onclick = () => busy($("playpause"), async () => {
  msg("…", "info");
  const r = await host({ cmd: "playpause" });
  msg(r.ok ? r.text : r.error, r.ok ? "ok" : "err");
});
$("power").onclick = async () => {
  if (power === "on") {
    if (!confirmOff) {
      confirmOff = true;
      $("power").textContent = "Нажми ещё раз — выключить";
      return;
    }
    await busy(null, async () => {
      const r = await host({ cmd: "off" });
      msg(r.ok ? r.text : r.error, r.ok ? "ok" : "err");
      if (r.ok) power = "off";
    });
    render({ power });
  } else {
    await busy(null, async () => {
      msg("Включаю телевизор… (до 40 секунд)", "info");
      const r = await host({ cmd: "wake" });
      msg(r.ok ? r.text : r.error, r.ok ? "ok" : "err");
      if (r.ok) power = "on";
    });
    await refresh();
  }
};
$("cast").onclick = async () => {
  msg("Отправляю — ход видно на странице YouTube.", "info");
  const r = await chrome.runtime.sendMessage({ cmd: "triggerActive" });
  if (r && r.ok) setTimeout(() => window.close(), 600);
  else msg("Открой вкладку с видео YouTube.", "err");
};

(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = (tab && tab.url) || "";
  $("cast").disabled = !/^https:\/\/www\.youtube\.com\/(watch|shorts\/|live\/)/.test(url);
  const cmds = await chrome.commands.getAll();
  const c = cmds.find((x) => x.name === "cast-to-tv");
  $("key").textContent = (c && c.shortcut) || "не назначена (chrome://extensions/shortcuts)";
  render({});
  await refresh();
})();

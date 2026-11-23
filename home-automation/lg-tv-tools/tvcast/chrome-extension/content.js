// Скрипт страницы youtube.com: кнопка «На телевизор» в плеере (и плавающая на Shorts),
// берёт videoId и текущую секунду, ставит видео на Маке на паузу, показывает ход отправки.
// YouTube требует Trusted Types, поэтому разметку строим через createElement, без innerHTML.
(() => {
  const RUN_ID = String(Math.random());
  document.documentElement.dataset.tvcast = RUN_ID; // новая копия скрипта вытесняет старую
  const alive = () => document.documentElement.dataset.tvcast === RUN_ID;
  document.querySelectorAll(".tvcast-btn, .tvcast-float, #tvcast-toast").forEach((e) => e.remove());

  const SVG = "http://www.w3.org/2000/svg";
  const lastGood = new Map(); // videoId → последняя секунда настоящего видео (не рекламы)
  let busy = false;
  let lastFinal = 0;

  // ---------- что сейчас открыто ----------

  const isShorts = () => location.pathname.startsWith("/shorts/");

  function currentVideoId() {
    const p = location.pathname;
    let id = null;
    if (p.startsWith("/shorts/") || p.startsWith("/live/")) id = p.split("/")[2];
    else if (p === "/watch") id = new URLSearchParams(location.search).get("v");
    return id && /^[A-Za-z0-9_-]{11}$/.test(id) ? id : null;
  }

  function visibleArea(el) {
    const r = el.getBoundingClientRect();
    const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
    const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
    return w * h;
  }

  function activeVideo() {
    if (!isShorts()) {
      const main = document.querySelector("#movie_player video.html5-main-video");
      if (main) return main;
    }
    // Shorts: на странице несколько роликов — берём самый видимый (активный)
    let best = null, bestScore = -1;
    for (const v of document.querySelectorAll("video")) {
      const score = visibleArea(v) + (v.paused ? 0 : 1e7);
      if (v.readyState > 0 && score > bestScore) { best = v; bestScore = score; }
    }
    return best;
  }

  const isAd = (v) => !!(v && v.closest(".html5-video-player.ad-showing"));
  const isLive = (v) => !!(v && (v.duration === Infinity ||
    v.closest(".html5-video-player.ytp-live")));

  // Запоминаем секунду, пока идёт само видео: во время рекламы currentTime у <video> — рекламный
  document.addEventListener("timeupdate", (e) => {
    const v = e.target;
    if (!(v instanceof HTMLVideoElement) || isAd(v)) return;
    const id = currentVideoId();
    if (id && v === activeVideo() && v.currentTime > 0) lastGood.set(id, v.currentTime);
  }, true);

  function startTime(id, v) {
    if (isLive(v)) return 0; // прямой эфир: телевизор сам встанет на «сейчас»
    if (v && !isAd(v) && v.currentTime > 0) return v.currentTime;
    if (lastGood.has(id)) return lastGood.get(id);
    const t = new URLSearchParams(location.search).get("t");
    return t ? parseInt(t, 10) || 0 : 0;
  }

  // ---------- отправка ----------

  async function castNow() {
    if (!alive()) return;
    if (busy) {
      toast("Уже отправляю на телевизор…", "info", 3000);
      return;
    }
    const id = currentVideoId();
    if (!id) {
      toast("Открой видео или Shorts — тогда его можно отправить на телевизор.", "err", 6000);
      return;
    }
    const v = activeVideo();
    const time = Math.floor(startTime(id, v));
    const wasPlaying = !!(v && !v.paused);
    if (v) v.pause();
    busy = true;
    setBusy(true);
    toast("Отправляю на телевизор…", "info", 0);
    let r;
    try {
      r = await chrome.runtime.sendMessage({ cmd: "cast", videoId: id, time });
    } catch (e) {
      r = { ok: false, error: "Расширение обновилось — обнови страницу (⌘R) и нажми ещё раз." };
    }
    busy = false;
    setBusy(false);
    if (!r) r = { ok: false, error: "Нет ответа от расширения." };
    lastFinal = Date.now();
    if (r.ok) {
      toast(r.text, "ok", 6000);
    } else {
      toast(r.error, "err", 15000);
      if (wasPlaying && v) v.play().catch(() => {});
    }
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!alive()) return;
    if (msg.cmd === "trigger") castNow();
    else if (msg.cmd === "toast") {
      // промежуточные сообщения не должны перекрыть уже показанный итог
      if (msg.kind === "info" && Date.now() - lastFinal < 1500) return;
      if (msg.kind !== "info") lastFinal = Date.now();
      toast(msg.text, msg.kind, msg.kind === "info" ? 0 : msg.kind === "err" ? 15000 : 6000);
    }
  });

  // ---------- интерфейс ----------

  function tvIcon(size) {
    const s = document.createElementNS(SVG, "svg");
    s.setAttribute("viewBox", "0 0 24 24");
    s.setAttribute("width", size);
    s.setAttribute("height", size);
    s.setAttribute("aria-hidden", "true");
    const add = (tag, attrs) => {
      const e = document.createElementNS(SVG, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      s.appendChild(e);
    };
    add("rect", { x: "2", y: "4", width: "20", height: "13", rx: "2", fill: "none",
      stroke: "currentColor", "stroke-width": "2" });
    add("path", { d: "M8 21h8", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round" });
    add("path", { d: "M10 7.8v5.4l4.6-2.7z", fill: "currentColor" });
    return s;
  }

  function setBusy(on) {
    document.querySelectorAll(".tvcast-btn, .tvcast-float").forEach((b) => b.classList.toggle("tvcast-busy", on));
  }

  function makePlayerButton() {
    const b = document.createElement("button");
    b.className = "ytp-button tvcast-btn";
    b.title = "На телевизор (⌥T)";
    b.setAttribute("aria-label", "На телевизор");
    b.appendChild(tvIcon(24));
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      castNow();
    });
    return b;
  }

  function makeFloatButton() {
    const b = document.createElement("button");
    b.className = "tvcast-float";
    b.title = "На телевизор (⌥T)";
    b.appendChild(tvIcon(20));
    const t = document.createElement("span");
    t.textContent = "На телевизор";
    b.appendChild(t);
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      castNow();
    });
    return b;
  }

  function ensureUI() {
    if (!alive()) {
      clearInterval(timer);
      return;
    }
    // обычное видео: кнопка в правой части панели плеера
    const controls = document.querySelector("#movie_player .ytp-right-controls");
    if (controls && !controls.querySelector(".tvcast-btn")) controls.prepend(makePlayerButton());
    // Shorts: у плеера нет панели — плавающая кнопка. Она же запасная, если YouTube поменял
    // разметку и кнопку в панель вставить не удалось.
    let fl = document.querySelector(".tvcast-float");
    const noPlayerBtn = !document.querySelector("#movie_player .tvcast-btn");
    if (isShorts() || (currentVideoId() && noPlayerBtn)) {
      if (!fl) document.body.appendChild((fl = makeFloatButton()));
      fl.style.display = "";
    } else if (fl) {
      fl.style.display = "none";
    }
  }

  let toastTimer = null;
  function toast(text, kind = "info", ms = 5000) {
    let el = document.getElementById("tvcast-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "tvcast-toast";
      el.addEventListener("click", () => el.classList.remove("tvcast-show"));
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.className = "tvcast-show tvcast-" + kind;
    clearTimeout(toastTimer);
    if (ms > 0) toastTimer = setTimeout(() => el.classList.remove("tvcast-show"), ms);
  }

  const timer = setInterval(ensureUI, 1000);
  document.addEventListener("yt-navigate-finish", ensureUI);
  ensureUI();
})();

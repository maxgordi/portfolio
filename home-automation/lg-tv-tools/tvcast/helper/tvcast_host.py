#!/usr/bin/python3
"""Помощник расширения «На телевизор» (Chrome Native Messaging).

Chrome запускает этот файл на каждую команду расширения и общается с ним через stdin/stdout:
4 байта длины + JSON. Промежуточные сообщения {"type":"progress"} показываются на странице,
последнее {"type":"result"} — итог.

Можно запускать и из Терминала для проверки:
  ./tvcast_host.py cast <ссылка или videoId> [секунда]
  ./tvcast_host.py status | ping | playpause | volup | voldown | off | wake | enter
Лог: ~/Library/Logs/tvcast-helper.log
"""
import errno, json, os, re, struct, sys, time, traceback, uuid

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from lgtv import (TV, TVError, YT_APP, DEFAULT_IP, log, load_state, save_state, tcp_errno, port_open,  # noqa: E402
                  lan_blocked, arp_mac, send_wol, ssdp_search, dial_youtube, dial_launch_youtube)
from lounge import LoungeError, Session, lounge_token  # noqa: E402

LAN_MSG = ("macOS не пускает Chrome в домашнюю сеть. Открой «Системные настройки → Конфиденциальность "
           "и безопасность → Локальная сеть», включи Google Chrome и перезапусти Chrome.")
OFF_MSG = ("Телевизор не включился по сети. Включи его пультом и нажми кнопку ещё раз.\n"
           "Если телевизор выдёргивали из розетки, по сети он может не проснуться, пока его один раз "
           "не включат пультом. Ещё проверь на ТВ: Настройки → Общие → «Mobile TV On» "
           "(включение по Wi-Fi) — должно быть «Вкл».")
NOMAC_MSG = ("Телевизор не отвечает, а его сетевой адрес (MAC) для включения ещё не известен. "
             "Включи телевизор пультом и нажми кнопку ещё раз — дальше он будет включаться сам.")
APP_NAMES = {YT_APP: "YouTube", "com.webos.app.livetv": "ТВ-каналы", "com.webos.app.home": "Главный экран",
             "com.webos.app.browser": "Браузер", "netflix": "Netflix"}


def fmt_time(sec):
    sec = int(sec)
    h, m, s = sec // 3600, sec // 60 % 60, sec % 60
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def video_id_from(text):
    text = (text or "").strip()
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})", text)
    if m:
        return m.group(1)
    return text if re.fullmatch(r"[A-Za-z0-9_-]{11}", text) else None


# ---------- телевизор ----------

def ensure_on(st, progress, wake=True):
    """IP включённого телевизора. Выключен — будим по Wake-on-LAN и ждём порт 3000."""
    ip = st.get("ip") or DEFAULT_IP
    e = tcp_errno(ip, 3000, 1.2)
    if e == 0:
        return ip
    log("tv port 3000 closed:", ip, errno.errorcode.get(e, e))
    if e in (errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPERM, errno.EACCES) and lan_blocked():
        raise TVError("lan", LAN_MSG)
    # IP мог смениться (роутер раздал другой) — ищем телевизор по SSDP
    for fip, _, srv in ssdp_search(timeout=1.5):
        if fip != ip and port_open(fip):
            log("tv found at new ip", fip, srv)
            st["ip"] = fip
            save_state(st)
            return fip
    if not wake:
        return None
    macs = st.get("macs") or []
    if not macs:
        raise TVError("off_nomac", NOMAC_MSG)
    progress("Телевизор выключен — включаю по сети…")
    end = time.time() + 35
    while time.time() < end:
        send_wol(macs, ip)
        for _ in range(6):
            if port_open(ip, 3000, 0.6):
                log("tv woke up")
                progress("Телевизор включился, жду запуска…")
                return ip
            time.sleep(0.4)
    raise TVError("off", OFF_MSG)


def connect_tv(ip, progress, wait=20):
    """SSAP с повторами: сразу после включения webOS ещё поднимает службы."""
    end = time.time() + wait
    last = None
    while True:
        try:
            return TV(ip).connect(on_prompt=lambda: progress(
                "На телевизоре появился запрос — нажми пультом «Да» (один раз)."))
        except TVError:
            raise
        except (OSError, ConnectionError, ValueError) as e:
            last = e
            if time.time() > end:
                break
            time.sleep(1.5)
    log("ssap connect failed", last)
    raise TVError("no_tv", "Телевизор в сети, но не отвечает на команды. Подожди несколько секунд и "
                  "нажми ещё раз (после включения ему нужно время).")


def learn(st, tv, ip):
    """Запоминаем IP и MAC-адреса (Wi-Fi из ARP, проводной из информации о прошивке) для WOL."""
    macs = list(st.get("macs") or [])
    mac = arp_mac(ip)
    if mac and mac not in macs:
        macs.insert(0, mac)
    if not st.get("wired_mac_checked"):
        try:
            dev = tv.req("ssap://com.webos.service.update/getCurrentSWInformation").get("device_id", "")
            if re.fullmatch(r"[0-9a-fA-F:]{17}", dev or "") and dev.lower() not in macs:
                macs.append(dev.lower())
            st["wired_mac_checked"] = True
        except TVError:
            pass
    st["ip"], st["macs"] = ip, macs
    save_state(st)


def find_screen(ip, st):
    """(состояние приложения по DIAL, screenId). screenId переживает перезагрузку ТВ, храним его."""
    state, sid = dial_youtube(ip, st)
    if sid and sid != st.get("screen_id"):
        log("screenId", sid)
        st["screen_id"] = sid
    save_state(st)
    return state, sid or st.get("screen_id")


def get_token(screen):
    try:
        return lounge_token(screen)
    except LoungeError as e:
        raise TVError("yt", f"YouTube не ответил ({e}). Проверь интернет и нажми ещё раз.")


# ---------- команды ----------

def cmd_cast(msg, progress):
    vid = video_id_from(msg.get("videoId") or msg.get("url"))
    if not vid:
        raise TVError("bad_video", "Не понял, какое это видео. Открой страницу видео на YouTube.")
    start = float(msg.get("time") or 0)
    st = load_state()
    ip = ensure_on(st, progress)
    tv = connect_tv(ip, progress)
    try:
        learn(st, tv, ip)
        dial_state, screen = find_screen(ip, st)
        if not screen:
            raise TVError("no_screen", "Телевизор не сообщил данные приложения YouTube (DIAL). "
                          "Проверь, что YouTube установлен на ТВ, и нажми ещё раз.")
        token = get_token(screen)
        remote_id = st.get("remote_id") or str(uuid.uuid4())
        if st.get("remote_id") != remote_id:
            st["remote_id"] = remote_id
            save_state(st)
        sess = Session(token, remote_id).connect()
        try:
            return cast_in_session(tv, ip, st, sess, vid, start, progress, dial_state)
        finally:
            sess.close()
    finally:
        tv.close()


def connect_screen(tv, ip, st, sess, progress, dial_state=None, force=False):
    """Приложение YouTube на ТВ должно быть запущено и сидеть в сессии Lounge.
    После окончания видео оно иногда выпадает из сессии (сервер при этом пишет «online»), и
    команды уходят в пустоту. Возвращаем его DIAL-запуском с pairingCode, как телефон."""
    fg = tv.foreground()
    if fg == YT_APP and sess.screen_seen and not force:
        return
    progress("Запускаю YouTube на телевизоре…" if fg != YT_APP else "Подключаюсь к YouTube на телевизоре…")
    if not dial_launch_youtube(ip, st, f"pairingCode={uuid.uuid4()}&theme=cl"):
        tv.launch(YT_APP)
    if dial_state == "running" and sess.screen_seen and not force:
        # приложение жило в фоне и не выходило из сессии — переподключаться ему незачем
        time.sleep(2)
        return
    if sess.wait_screen(30):
        return
    # не подключился: возможно, висит экран «Кто смотрит?»
    if tv.foreground() == YT_APP:
        progress("Выбираю профиль на телевизоре…")
        tv.button("ENTER")
    else:
        tv.launch(YT_APP)
    if not sess.wait_screen(20):
        raise TVError("yt_noscreen", "YouTube на телевизоре не подключился. Посмотри на экран ТВ: если там "
                      "окно или вопрос — закрой его пультом и нажми кнопку ещё раз.")


def cast_in_session(tv, ip, st, sess, vid, start, progress, dial_state):
    t0 = time.time()
    connect_screen(tv, ip, st, sess, progress, dial_state)
    log("screen ready in", round(time.time() - t0, 1))
    progress("Отправляю видео на телевизор…")
    sess.play_video(vid, start)
    res = sess.wait_video(vid, 10)
    log("cast", vid, start, "first try:", res, sess.state.get("state"), round(time.time() - t0, 1))
    if res is None:
        # приложение в сессии, но видео не взяло — скорее всего экран выбора профиля
        if tv.foreground() == YT_APP:
            progress("Выбираю профиль на телевизоре…")
            tv.button("ENTER")
            time.sleep(2)
        connect_screen(tv, ip, st, sess, progress, force=True)
        sess.play_video(vid, start)
        res = sess.wait_video(vid, 12)
        log("cast", vid, "second try:", res, sess.state.get("state"), round(time.time() - t0, 1))
    if res is None:
        raise TVError("yt_noresp", "YouTube на телевизоре не ответил. Посмотри на экран ТВ: если там "
                      "окно или вопрос — закрой его пультом и нажми кнопку ещё раз.")
    if res == "loading":
        # телевизор взял видео и буферизует; не держим пользователя долго
        progress("Телевизор загружает видео…")
        res = sess.wait_video(vid, 6) or "loading"
    log("cast done", res, round(time.time() - t0, 1))
    at = f" с {fmt_time(start)}" if start >= 1 else ""
    if res == "playing":
        return {"ok": True, "state": "playing", "text": f"▶ Играет на телевизоре{at}"}
    return {"ok": True, "state": "loading", "text": f"▶ Видео запущено на телевизоре{at} (ещё загружается)"}


def cmd_status(msg, progress, quick=False):
    st = load_state()
    ip = st.get("ip") or DEFAULT_IP
    e = tcp_errno(ip, 3000, 0.8)
    if e != 0:
        if e in (errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPERM, errno.EACCES) and lan_blocked():
            return {"ok": False, "code": "lan", "error": LAN_MSG, "power": "unknown"}
        if not quick:
            ip2 = ensure_on(st, progress, wake=False)
            if ip2:
                ip = ip2
                e = 0
        if e != 0:
            return {"ok": True, "power": "off", "canWake": bool(st.get("macs"))}
    if quick:
        return {"ok": True, "power": "on"}
    try:
        tv = TV(ip).connect(prompt_timeout=4)
    except TVError as ex:
        return {"ok": False, "power": "on", "code": ex.code, "error": ex.msg}
    except (OSError, ConnectionError, ValueError):
        return {"ok": True, "power": "on", "app": "", "appName": "запускается…"}
    try:
        learn(st, tv, ip)
        fg = tv.foreground()
        vol = {}
        try:
            vol = tv.req("ssap://audio/getVolume")
        except TVError:
            pass
        vs = vol.get("volumeStatus") or vol
        return {"ok": True, "power": "on", "app": fg, "appName": APP_NAMES.get(fg, fg),
                "volume": vs.get("volume"), "muted": vs.get("muted", vs.get("muteStatus"))}
    finally:
        tv.close()


def need_tv(progress):
    st = load_state()
    ip = ensure_on(st, progress, wake=False)
    if not ip:
        raise TVError("off", "Телевизор выключен.")
    return st, connect_tv(ip, progress, wait=5)


def cmd_volume(msg, progress):
    st, tv = need_tv(progress)
    try:
        tv.req("ssap://audio/volumeUp" if msg.get("dir") == "up" else "ssap://audio/volumeDown")
        vol = tv.req("ssap://audio/getVolume")
        vs = vol.get("volumeStatus") or vol
        return {"ok": True, "volume": vs.get("volume"), "muted": vs.get("muted"),
                "text": f"Громкость {vs.get('volume')}"}
    finally:
        tv.close()


def cmd_playpause(msg, progress):
    st, tv = need_tv(progress)
    try:
        if tv.foreground() != YT_APP:
            # не YouTube — обычная кнопка паузы/воспроизведения webOS
            tv.req("ssap://media.controls/pause" if msg.get("want") != "play" else "ssap://media.controls/play")
            return {"ok": True, "text": "Команда отправлена"}
    finally:
        tv.close()
    screen = st.get("screen_id")
    if not screen:
        raise TVError("no_screen", "Сначала отправь на телевизор любое видео кнопкой.")
    sess = Session(get_token(screen), st.get("remote_id")).connect()
    try:
        vid, state = sess.current(3.0)
        if state == "1":
            sess.command("pause")
            return {"ok": True, "playing": False, "text": "⏸ Пауза на телевизоре"}
        if vid:
            sess.command("play")
            return {"ok": True, "playing": True, "text": "▶ Продолжаю на телевизоре"}
    finally:
        sess.close()
    raise TVError("nothing", "В YouTube на телевизоре сейчас ничего не открыто.")


def cmd_off(msg, progress):
    st, tv = need_tv(progress)
    try:
        tv.req("ssap://system/turnOff")
        return {"ok": True, "power": "off", "text": "Телевизор выключается"}
    finally:
        tv.close()


def cmd_wake(msg, progress):
    st = load_state()
    ip = ensure_on(st, progress)
    tv = connect_tv(ip, progress)
    try:
        learn(st, tv, ip)
    finally:
        tv.close()
    return {"ok": True, "power": "on", "text": "Телевизор включён"}


def cmd_enter(msg, progress):
    st, tv = need_tv(progress)
    try:
        tv.button(msg.get("button") or "ENTER")
        return {"ok": True, "text": "Нажато"}
    finally:
        tv.close()


COMMANDS = {
    "cast": cmd_cast,
    "status": cmd_status,
    "ping": lambda m, p: cmd_status(m, p, quick=True),
    "volume": cmd_volume,
    "playpause": cmd_playpause,
    "off": cmd_off,
    "wake": cmd_wake,
    "enter": cmd_enter,
}


def handle(msg, progress):
    cmd = msg.get("cmd")
    t0 = time.time()
    log(">>", json.dumps(msg, ensure_ascii=False))
    try:
        fn = COMMANDS.get(cmd)
        if not fn:
            res = {"ok": False, "code": "bad_cmd", "error": f"Неизвестная команда {cmd!r}"}
        else:
            res = fn(msg, progress)
    except TVError as e:
        res = {"ok": False, "code": e.code, "error": e.msg}
    except LoungeError as e:
        res = {"ok": False, "code": "yt", "error": f"YouTube не ответил ({e}). Нажми ещё раз."}
    except Exception as e:  # чтобы расширение всегда получило понятный ответ
        log(traceback.format_exc())
        res = {"ok": False, "code": "internal",
               "error": f"Ошибка помощника: {e}. Подробности в ~/Library/Logs/tvcast-helper.log"}
    res["type"] = "result"
    log("<<", f"{time.time() - t0:.1f}s", json.dumps(res, ensure_ascii=False))
    return res


# ---------- Native Messaging ----------

def nm_read():
    raw = sys.stdin.buffer.read(4)
    if len(raw) < 4:
        return None
    n = struct.unpack("=I", raw)[0]
    return json.loads(sys.stdin.buffer.read(n).decode("utf-8"))


def nm_write(obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(data)) + data)
    sys.stdout.buffer.flush()


def main():
    args = sys.argv[1:]
    if args and not args[0].startswith("chrome-extension://"):
        # режим Терминала
        cmd = args[0]
        msg = {"cmd": cmd}
        if cmd == "cast":
            msg["videoId"] = args[1] if len(args) > 1 else ""
            msg["time"] = float(args[2]) if len(args) > 2 else 0
        elif cmd in ("volup", "voldown"):
            msg = {"cmd": "volume", "dir": "up" if cmd == "volup" else "down"}
        elif cmd == "enter" and len(args) > 1:
            msg["button"] = args[1]
        res = handle(msg, lambda t: print("…", t, file=sys.stderr, flush=True))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res.get("ok") else 1
    while True:
        msg = nm_read()
        if msg is None:
            return 0
        nm_write(handle(msg, lambda t: nm_write({"type": "progress", "text": t})))


if __name__ == "__main__":
    sys.exit(main())

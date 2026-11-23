#!/usr/bin/env python3
"""tvremote — пульт для LG webOS TV (42LF652V) с Mac по локальной сети (порт 3000).
Только стандартная библиотека. Ключ пары берётся из ~/.lgtv_client_key (общий с tvcast)."""
import base64, json, os, re, socket, struct, subprocess, sys, time

TV_IP = os.environ.get("TV_IP", "192.168.1.50")
KEY_FILE = os.path.expanduser("~/.lgtv_client_key")

SIGNATURE = ("eyJhbGdvcml0aG0iOiJSU0EtU0hBMjU2Iiwia2V5SWQiOiJ0ZXN0LXNpZ25pbm"
             "ctY2VydCIsInNpZ25hdHVyZVZlcnNpb24iOjF9.hrVRgjCwXVvE2OOSpDZ58hR"
             "+59aFNwYDyjQgKk3auukd7pcegmE2CzPCa0bJ0ZsRAcKkCTJrWo5iDzNhMBWRy"
             "aMOv5zWSrthlf7G128qvIlpMT0YNY+n/FaOHE73uLrS/g7swl3/qH/BGFG2Hu4"
             "RlL48eb3lLKqTt2xKHdCs6Cd4RMfJPYnzgvI4BNrFUKsjkcu+WD4OO2A27Pq1n"
             "50cMchmcaXadJhGrOqH5YmHdOCj5NSHzJYrsW0HPlpuAx/ECMeIZYDh6RMqaFM"
             "2DXzdKX9NmmyqzJ3o/0lkk/N97gfVRLW5hA29yeAwaCViZNCP8iC9aO0q9fQoj"
             "oa7NQnAtw==")
REG = {"forcePairing": False, "pairingType": "PROMPT", "manifest": {
    "appVersion": "1.1", "manifestVersion": 1,
    "permissions": ["LAUNCH","LAUNCH_WEBAPP","APP_TO_APP","CLOSE","TEST_OPEN","TEST_PROTECTED",
        "CONTROL_AUDIO","CONTROL_DISPLAY","CONTROL_INPUT_JOYSTICK","CONTROL_INPUT_MEDIA_RECORDING",
        "CONTROL_INPUT_MEDIA_PLAYBACK","CONTROL_INPUT_TV","CONTROL_POWER","READ_APP_STATUS",
        "READ_CURRENT_CHANNEL","READ_INPUT_DEVICE_LIST","READ_NETWORK_STATE","READ_RUNNING_APPS",
        "READ_TV_CHANNEL_LIST","WRITE_NOTIFICATION_TOAST","READ_POWER_STATE","READ_COUNTRY_INFO",
        "READ_SETTINGS","CONTROL_TV_SCREEN","CONTROL_TV_STANBY","CONTROL_FAVORITE_GROUP",
        "CONTROL_USER_INFO","CHECK_BLUETOOTH_DEVICE","CONTROL_BLUETOOTH","CONTROL_TIMER_INFO",
        "STB_INTERNAL_CONNECTION","CONTROL_RECORDING","READ_RECORDING_STATE","WRITE_RECORDING_LIST",
        "READ_RECORDING_LIST","READ_RECORDING_SCHEDULE","WRITE_RECORDING_SCHEDULE",
        "READ_STORAGE_DEVICE_LIST","READ_TV_PROGRAM_INFO","CONTROL_BOX_CHANNEL","READ_TV_ACR_AUTH_TOKEN",
        "READ_TV_CONTENT_STATE","READ_TV_CURRENT_TIME","ADD_LAUNCHER_CHANNEL","SET_CHANNEL_SKIP",
        "RELEASE_CHANNEL_SKIP","CONTROL_CHANNEL_BLOCK","DELETE_SELECT_CHANNEL","CONTROL_CHANNEL_GROUP",
        "SCAN_TV_CHANNELS","CONTROL_TV_POWER","CONTROL_WOL"],
    "signatures": [{"signature": SIGNATURE, "signatureVersion": 1}],
    "signed": {"appId": "com.lge.test","created": "20140509",
        "localizedAppNames": {"": "LG Remote App","ko-KR": "리모컨 앱","zxx-XX": "ЛГ Rэмotэ AПП"},
        "localizedVendorNames": {"": "LG Electronics"},
        "permissions": ["TEST_SECURE","CONTROL_INPUT_TEXT","CONTROL_MOUSE_AND_KEYBOARD",
            "READ_INSTALLED_APPS","READ_LGE_SDX","READ_NOTIFICATIONS","SEARCH","WRITE_SETTINGS",
            "WRITE_NOTIFICATION_ALERT","CONTROL_POWER","READ_CURRENT_CHANNEL","READ_RUNNING_APPS",
            "READ_UPDATE_INFO","UPDATE_FROM_REMOTE_APP","READ_LGE_TV_INPUT_EVENTS","READ_TV_CURRENT_TIME"],
        "serial": "2f930e2d2cfe083771f68e4fe7bb07","vendorId": "com.lge"}}}

class WS:
    def __init__(self, host, port=3000):
        self.s = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            c = self.s.recv(4096)
            if not c: raise ConnectionError("телевизор закрыл соединение")
            resp += c
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        if b" 101" not in head.split(b"\r\n")[0]:
            raise ConnectionError("не открылся WebSocket")
    def _exact(self, n):
        while len(self.buf) < n:
            c = self.s.recv(65536)
            if not c: raise ConnectionError("телевизор закрыл соединение")
            self.buf += c
        d, self.buf = self.buf[:n], self.buf[n:]; return d
    def send(self, obj):
        data = json.dumps(obj).encode(); n = len(data)
        hdr = bytes([0x81])
        if n < 126: hdr += bytes([0x80 | n])
        elif n < 65536: hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else: hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        m = os.urandom(4)
        self.s.sendall(hdr + m + bytes(b ^ m[i % 4] for i, b in enumerate(data)))
    def recv(self):
        msg = b""
        while True:
            b1, b2 = self._exact(2); op, n = b1 & 0x0F, b2 & 0x7F
            if n == 126: n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._exact(8))[0]
            mask = self._exact(4) if b2 & 0x80 else None
            data = self._exact(n)
            if mask: data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 9: continue
            if op == 10: continue
            if op == 8: raise ConnectionError("телевизор закрыл WebSocket")
            msg += data
            if b1 & 0x80: return json.loads(msg.decode("utf-8", "ignore"))

def connect():
    ws = WS(TV_IP)
    payload = json.loads(json.dumps(REG))
    try: payload["client-key"] = open(KEY_FILE).read().strip()
    except FileNotFoundError: pass
    ws.send({"type": "register", "id": "reg0", "payload": payload})
    ws.s.settimeout(90)
    while True:
        m = ws.recv()
        if m.get("type") == "response" and m.get("payload", {}).get("pairingType") == "PROMPT":
            print("📺 На телевизоре — запрос на подключение, нажми пультом «Да» (один раз).")
        elif m.get("type") == "registered":
            open(KEY_FILE, "w").write(m["payload"]["client-key"]); os.chmod(KEY_FILE, 0o600)
            ws.s.settimeout(15); return ws
        elif m.get("type") == "error":
            raise RuntimeError(f"телевизор отказал: {m.get('error')}")

def req(ws, uri, payload=None):
    rid = f"r{time.time_ns()}"
    ws.send({"type": "request", "id": rid, "uri": uri, "payload": payload or {}})
    while True:
        m = ws.recv()
        if m.get("id") == rid: return m.get("payload", {})

def launch_points(ws):
    p = req(ws, "ssap://com.webos.applicationManager/listLaunchPoints")
    return p.get("launchPoints", [])

def video_id(url):
    url = url.strip()
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else (url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url) else None)

def open_youtube(ws, url):
    vid = video_id(url)
    yt = None
    for lp in launch_points(ws):
        blob = (lp.get("id","") + " " + lp.get("title","")).lower()
        if "youtube" in blob: yt = lp.get("id"); break
    yt = yt or "youtube.leanback.v4"
    params = {}
    if vid: params["contentTarget"] = f"https://www.youtube.com/watch?v={vid}"
    r = req(ws, "ssap://system.launcher/launch", {"id": yt, "params": params})
    ok = r.get("returnValue", False)
    if ok and vid: print(f"✅ YouTube открыт на видео {vid}. Телек играет его сам, Мак свободен.")
    elif ok: print("✅ YouTube открыт (конкретное видео не распозналось — включи нужное на телеке).")
    else: print("⚠ Телевизор ответил:", r)

def clipboard():
    try: return subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    except Exception: return ""

def show_apps(ws):
    lps = sorted(launch_points(ws), key=lambda x: x.get("title","").lower())
    print("\n— Приложения на телевизоре —")
    for i, lp in enumerate(lps, 1):
        rem = "" if lp.get("removable") else "  (системное, не удаляется)"
        print(f"{i:2}) {lp.get('title','?'):<28} [{lp.get('id','')}]{rem}")
    print("Удаление: на телеке выдели плитку приложения, зажми OK/колесо и выбери «Удалить». "
          "Системные (ютуб к ним не относится) убрать нельзя — только спрятать.")
    return lps

def main():
    print(f"Подключаюсь к телевизору {TV_IP}…")
    try: ws = connect()
    except Exception as e:
        sys.exit(f"✖ Не вышло: {e}\n  Телек включён? Мак в той же Wi-Fi?")
    req(ws, "ssap://system.notifications/createToast", {"message": "Пульт Mac подключён"})
    lps = []
    while True:
        print("""
  1) Показать приложения
  2) Запустить приложение (по номеру)
  3) ▶ YouTube-видео на телек (вставь ссылку)
  4) Громче   5) Тише   6) Звук вкл/выкл
  7) Выключить телевизор
  0) Выход""")
        try: c = input("> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if c == "1": lps = show_apps(ws)
        elif c == "2":
            if not lps: lps = show_apps(ws)
            n = input("Номер приложения: ").strip()
            if n.isdigit() and 1 <= int(n) <= len(lps):
                req(ws, "ssap://system.launcher/launch", {"id": lps[int(n)-1]["id"]})
                print("✅ Запущено.")
        elif c == "3":
            u = input("Ссылка на YouTube (Enter — взять из буфера обмена): ").strip() or clipboard()
            if u: open_youtube(ws, u)
            else: print("Пусто.")
        elif c == "4": req(ws, "ssap://audio/volumeUp")
        elif c == "5": req(ws, "ssap://audio/volumeDown")
        elif c == "6":
            st = req(ws, "ssap://audio/getStatus"); req(ws, "ssap://audio/setMute", {"mute": not st.get("mute", False)})
        elif c == "7":
            if input("Точно выключить? (y/n): ").strip().lower() == "y":
                req(ws, "ssap://system/turnOff"); print("📴 Выключаю."); break
        elif c == "0": break

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass

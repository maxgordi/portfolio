#!/usr/bin/env python3
"""tvcast — трансляция экрана Mac на LG webOS TV (42LF652V) одной командой.

Как работает: Mac захватывает экран -> отдаёт его как MJPEG-поток по локальной сети ->
по webOS API (порт 3000) говорит телевизору открыть браузер на этой странице.
Только стандартная библиотека Python. ffmpeg желателен (плавно), без него — запасной режим ~2 к/с.
Остановка: Ctrl+C или закрыть окно Терминала.
"""
import base64, json, os, re, shutil, signal, socket, struct, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

TV_IP   = os.environ.get("TV_IP", "192.168.1.50")
PORT    = int(os.environ.get("TVCAST_PORT", "8099"))
WIDTH   = int(os.environ.get("TVCAST_WIDTH", "1280"))   # 1920 = чётче, но тяжелее для ТВ
FPS     = int(os.environ.get("TVCAST_FPS", "12"))
QUALITY = os.environ.get("TVCAST_Q", "6")                 # 2 = лучше ... 31 = хуже
KEY_FILE = os.path.expanduser("~/.lgtv_client_key")

# Подписанный манифест «LG Remote App» (тот же, что в pywebostv / lgtv2)
SIGNATURE = ("eyJhbGdvcml0aG0iOiJSU0EtU0hBMjU2Iiwia2V5SWQiOiJ0ZXN0LXNpZ25pbm"
             "ctY2VydCIsInNpZ25hdHVyZVZlcnNpb24iOjF9.hrVRgjCwXVvE2OOSpDZ58hR"
             "+59aFNwYDyjQgKk3auukd7pcegmE2CzPCa0bJ0ZsRAcKkCTJrWo5iDzNhMBWRy"
             "aMOv5zWSrthlf7G128qvIlpMT0YNY+n/FaOHE73uLrS/g7swl3/qH/BGFG2Hu4"
             "RlL48eb3lLKqTt2xKHdCs6Cd4RMfJPYnzgvI4BNrFUKsjkcu+WD4OO2A27Pq1n"
             "50cMchmcaXadJhGrOqH5YmHdOCj5NSHzJYrsW0HPlpuAx/ECMeIZYDh6RMqaFM"
             "2DXzdKX9NmmyqzJ3o/0lkk/N97gfVRLW5hA29yeAwaCViZNCP8iC9aO0q9fQoj"
             "oa7NQnAtw==")
REG = {
    "forcePairing": False, "pairingType": "PROMPT",
    "manifest": {
        "appVersion": "1.1", "manifestVersion": 1,
        "permissions": ["LAUNCH", "LAUNCH_WEBAPP", "APP_TO_APP", "CLOSE", "TEST_OPEN", "TEST_PROTECTED",
            "CONTROL_AUDIO", "CONTROL_DISPLAY", "CONTROL_INPUT_JOYSTICK", "CONTROL_INPUT_MEDIA_RECORDING",
            "CONTROL_INPUT_MEDIA_PLAYBACK", "CONTROL_INPUT_TV", "CONTROL_POWER", "READ_APP_STATUS",
            "READ_CURRENT_CHANNEL", "READ_INPUT_DEVICE_LIST", "READ_NETWORK_STATE", "READ_RUNNING_APPS",
            "READ_TV_CHANNEL_LIST", "WRITE_NOTIFICATION_TOAST", "READ_POWER_STATE", "READ_COUNTRY_INFO",
            "READ_SETTINGS", "CONTROL_TV_SCREEN", "CONTROL_TV_STANBY", "CONTROL_FAVORITE_GROUP",
            "CONTROL_USER_INFO", "CHECK_BLUETOOTH_DEVICE", "CONTROL_BLUETOOTH", "CONTROL_TIMER_INFO",
            "STB_INTERNAL_CONNECTION", "CONTROL_RECORDING", "READ_RECORDING_STATE", "WRITE_RECORDING_LIST",
            "READ_RECORDING_LIST", "READ_RECORDING_SCHEDULE", "WRITE_RECORDING_SCHEDULE",
            "READ_STORAGE_DEVICE_LIST", "READ_TV_PROGRAM_INFO", "CONTROL_BOX_CHANNEL",
            "READ_TV_ACR_AUTH_TOKEN", "READ_TV_CONTENT_STATE", "READ_TV_CURRENT_TIME",
            "ADD_LAUNCHER_CHANNEL", "SET_CHANNEL_SKIP", "RELEASE_CHANNEL_SKIP", "CONTROL_CHANNEL_BLOCK",
            "DELETE_SELECT_CHANNEL", "CONTROL_CHANNEL_GROUP", "SCAN_TV_CHANNELS", "CONTROL_TV_POWER",
            "CONTROL_WOL"],
        "signatures": [{"signature": SIGNATURE, "signatureVersion": 1}],
        "signed": {
            "appId": "com.lge.test", "created": "20140509",
            "localizedAppNames": {"": "LG Remote App", "ko-KR": "리모컨 앱", "zxx-XX": "ЛГ Rэмotэ AПП"},
            "localizedVendorNames": {"": "LG Electronics"},
            "permissions": ["TEST_SECURE", "CONTROL_INPUT_TEXT", "CONTROL_MOUSE_AND_KEYBOARD",
                "READ_INSTALLED_APPS", "READ_LGE_SDX", "READ_NOTIFICATIONS", "SEARCH", "WRITE_SETTINGS",
                "WRITE_NOTIFICATION_ALERT", "CONTROL_POWER", "READ_CURRENT_CHANNEL", "READ_RUNNING_APPS",
                "READ_UPDATE_INFO", "UPDATE_FROM_REMOTE_APP", "READ_LGE_TV_INPUT_EVENTS", "READ_TV_CURRENT_TIME"],
            "serial": "2f930e2d2cfe083771f68e4fe7bb07", "vendorId": "com.lge"}}}

# ---------------- минимальный WebSocket-клиент ----------------
class WS:
    def __init__(self, host, port=3000):
        self.s = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                        "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            c = self.s.recv(4096)
            if not c: raise ConnectionError("телевизор закрыл соединение")
            resp += c
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        if b" 101" not in head.split(b"\r\n")[0]:
            raise ConnectionError("не удалось открыть WebSocket: " + head.split(b"\r\n")[0].decode(errors="ignore"))

    def _exact(self, n):
        while len(self.buf) < n:
            c = self.s.recv(65536)
            if not c: raise ConnectionError("телевизор закрыл соединение")
            self.buf += c
        d, self.buf = self.buf[:n], self.buf[n:]
        return d

    def send(self, obj, opcode=1):
        data = json.dumps(obj).encode() if opcode == 1 else obj
        n = len(data)
        hdr = bytes([0x80 | opcode])
        if n < 126:     hdr += bytes([0x80 | n])
        elif n < 65536: hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:           hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        m = os.urandom(4)
        self.s.sendall(hdr + m + bytes(b ^ m[i % 4] for i, b in enumerate(data)))

    def recv(self):
        msg = b""
        while True:
            b1, b2 = self._exact(2)
            op, n = b1 & 0x0F, b2 & 0x7F
            if n == 126:   n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._exact(8))[0]
            mask = self._exact(4) if b2 & 0x80 else None
            data = self._exact(n)
            if mask: data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 9:  self.send(data, opcode=10); continue      # ping -> pong
            if op == 10: continue
            if op == 8:  raise ConnectionError("телевизор закрыл WebSocket")
            msg += data
            if b1 & 0x80: return json.loads(msg.decode("utf-8", "ignore"))

def connect_tv():
    ws = WS(TV_IP)
    payload = json.loads(json.dumps(REG))
    try:
        payload["client-key"] = open(KEY_FILE).read().strip()
    except FileNotFoundError:
        pass
    ws.send({"type": "register", "id": "reg0", "payload": payload})
    ws.s.settimeout(90)
    while True:
        m = ws.recv()
        if m.get("type") == "response" and m.get("payload", {}).get("pairingType") == "PROMPT":
            print("📺 На телевизоре запрос на подключение — нажми пультом «Да» (это нужно один раз).")
        elif m.get("type") == "registered":
            with open(KEY_FILE, "w") as f:
                f.write(m["payload"]["client-key"])
            os.chmod(KEY_FILE, 0o600)
            ws.s.settimeout(15)
            return ws
        elif m.get("type") == "error":
            raise RuntimeError(f"телевизор отказал: {m.get('error')}")

def tv_request(ws, uri, payload=None):
    rid = f"r{time.time_ns()}"
    ws.send({"type": "request", "id": rid, "uri": uri, "payload": payload or {}})
    while True:
        m = ws.recv()
        if m.get("id") == rid:
            return m

# ---------------- кадры ----------------
class Frames:
    def __init__(self):
        self.cond = threading.Condition(); self.frame = None; self.seq = 0
    def put(self, f):
        with self.cond:
            self.frame = f; self.seq += 1; self.cond.notify_all()
    def wait(self, last, timeout=5):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last, timeout=timeout)
            return self.seq, self.frame

FR = Frames()
PROCS = []

def find_ffmpeg():
    return shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)), None)

def screen_index(ff):
    r = subprocess.run([ff, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                       capture_output=True, text=True)
    m = re.search(r"\[(\d+)\] Capture screen 0", r.stderr)
    return m.group(1) if m else None

def capture_ffmpeg(ff, idx):
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-capture_cursor", "1",
           "-framerate", "30", "-i", f"{idx}:none", "-vf", f"fps={FPS},scale={WIDTH}:-2",
           "-q:v", QUALITY, "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE); PROCS.append(p)
    buf = b""
    while True:
        c = p.stdout.read1(1 << 16)
        if not c: return
        buf += c
        while True:
            s = buf.find(b"\xff\xd8")
            if s < 0: buf = b""; break
            e = buf.find(b"\xff\xd9", s + 2)
            if e < 0: buf = buf[s:]; break
            FR.put(buf[s:e + 2]); buf = buf[e + 2:]

def capture_screencapture():
    tmp = "/tmp/tvcast_frame.jpg"
    while True:
        subprocess.run(["screencapture", "-x", "-C", "-t", "jpg", tmp])
        subprocess.run(["sips", "-Z", str(WIDTH), tmp], capture_output=True)
        try:
            FR.put(open(tmp, "rb").read())
        except FileNotFoundError:
            time.sleep(0.5)

# ---------------- HTTP ----------------
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Mac</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden;cursor:none}
img{width:100%;height:100%;object-fit:contain;display:block}</style></head>
<body><img src="/stream" onerror="setTimeout(function(){location.reload()},1000)"></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = 0
            try:
                while True:
                    seq, f = FR.wait(seq)
                    if f is None: continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f))
                    self.wfile.write(f + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)

def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((TV_IP, 9)); ip = s.getsockname()[0]; s.close()
    return ip

def stop_on_tv():
    try:
        ws = connect_tv()
        tv_request(ws, "ssap://system.launcher/close", {"id": "com.webos.app.browser"})
    except Exception:
        pass

def main():
    def _sig(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, _sig); signal.signal(signal.SIGTERM, _sig)

    ff = find_ffmpeg()
    idx = screen_index(ff) if ff else None
    if ff and idx is not None:
        print(f"▶ Захват экрана через ffmpeg ({WIDTH}px, {FPS} к/с)")
        threading.Thread(target=capture_ffmpeg, args=(ff, idx), daemon=True).start()
    else:
        print("⚠ ffmpeg не найден — запасной режим (~2 к/с). Для плавной картинки: brew install ffmpeg")
        threading.Thread(target=capture_screencapture, daemon=True).start()

    _, first = FR.wait(0, timeout=8)
    if first is None:
        sys.exit("✖ Нет картинки с экрана. Дай Терминалу доступ: Системные настройки → Конфиденциальность "
                 "и безопасность → Запись экрана (и звука) → включи Терминал, потом перезапусти Терминал.")

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://{local_ip()}:{PORT}/"
    print(f"▶ Поток: {url}")

    try:
        ws = connect_tv()
    except (OSError, ConnectionError) as e:
        sys.exit(f"✖ Не достучался до телевизора {TV_IP}: {e}\n  Телек включён? Мак в той же Wi-Fi сети?")
    try:
        tv_request(ws, "ssap://system.notifications/createToast", {"message": "Подключаю экран Mac…"})
    except Exception:
        pass
    r = tv_request(ws, "ssap://system.launcher/open", {"target": url})
    if not r.get("payload", {}).get("returnValue", True):
        print("⚠ Телевизор ответил:", r)
    print("✅ Экран на телевизоре. Остановить: Ctrl+C или закрой это окно.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n⏹ Останавливаю…")
        stop_on_tv()
        for p in PROCS:
            p.terminate()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

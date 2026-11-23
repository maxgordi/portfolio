"""Телевизор LG webOS: SSAP (порт 3000), кнопки пульта, Wake-on-LAN, DIAL, поиск по SSDP.

Основано на ../tvremote.py: тот же подписанный манифест «LG Remote App» и тот же ключ пары
~/.lgtv_client_key, поэтому телевизор не спрашивает подтверждение заново.
Только стандартная библиотека (Python 3.9 из macOS).
"""
import base64, errno, json, os, re, socket, struct, subprocess, time, urllib.parse, urllib.request

KEY_FILE = os.path.expanduser("~/.lgtv_client_key")
STATE_FILE = os.path.expanduser("~/.config/tvcast/state.json")
LOG_FILE = os.path.expanduser("~/Library/Logs/tvcast-helper.log")
DEFAULT_IP = "192.168.1.50"
YT_APP = "youtube.leanback.v4"


class TVError(Exception):
    """Ошибка, текст которой показывается пользователю как есть (по-русски)."""
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


def log(*a):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    except OSError:
        pass


# ---------- состояние: IP, MAC-адреса, screenId, адрес DIAL ----------

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_FILE)


# ---------- подписанная регистрация (из tvremote.py) ----------

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
    "permissions": ["LAUNCH", "LAUNCH_WEBAPP", "APP_TO_APP", "CLOSE", "TEST_OPEN", "TEST_PROTECTED",
        "CONTROL_AUDIO", "CONTROL_DISPLAY", "CONTROL_INPUT_JOYSTICK", "CONTROL_INPUT_MEDIA_RECORDING",
        "CONTROL_INPUT_MEDIA_PLAYBACK", "CONTROL_INPUT_TV", "CONTROL_POWER", "READ_APP_STATUS",
        "READ_CURRENT_CHANNEL", "READ_INPUT_DEVICE_LIST", "READ_NETWORK_STATE", "READ_RUNNING_APPS",
        "READ_TV_CHANNEL_LIST", "WRITE_NOTIFICATION_TOAST", "READ_POWER_STATE", "READ_COUNTRY_INFO",
        "READ_SETTINGS", "CONTROL_TV_SCREEN", "CONTROL_TV_STANBY", "CONTROL_FAVORITE_GROUP",
        "CONTROL_USER_INFO", "CHECK_BLUETOOTH_DEVICE", "CONTROL_BLUETOOTH", "CONTROL_TIMER_INFO",
        "STB_INTERNAL_CONNECTION", "CONTROL_RECORDING", "READ_RECORDING_STATE", "WRITE_RECORDING_LIST",
        "READ_RECORDING_LIST", "READ_RECORDING_SCHEDULE", "WRITE_RECORDING_SCHEDULE",
        "READ_STORAGE_DEVICE_LIST", "READ_TV_PROGRAM_INFO", "CONTROL_BOX_CHANNEL", "READ_TV_ACR_AUTH_TOKEN",
        "READ_TV_CONTENT_STATE", "READ_TV_CURRENT_TIME", "ADD_LAUNCHER_CHANNEL", "SET_CHANNEL_SKIP",
        "RELEASE_CHANNEL_SKIP", "CONTROL_CHANNEL_BLOCK", "DELETE_SELECT_CHANNEL", "CONTROL_CHANNEL_GROUP",
        "SCAN_TV_CHANNELS", "CONTROL_TV_POWER", "CONTROL_WOL"],
    "signatures": [{"signature": SIGNATURE, "signatureVersion": 1}],
    "signed": {"appId": "com.lge.test", "created": "20140509",
        "localizedAppNames": {"": "LG Remote App", "ko-KR": "리모컨 앱", "zxx-XX": "ЛГ Rэмotэ AПП"},
        "localizedVendorNames": {"": "LG Electronics"},
        "permissions": ["TEST_SECURE", "CONTROL_INPUT_TEXT", "CONTROL_MOUSE_AND_KEYBOARD",
            "READ_INSTALLED_APPS", "READ_LGE_SDX", "READ_NOTIFICATIONS", "SEARCH", "WRITE_SETTINGS",
            "WRITE_NOTIFICATION_ALERT", "CONTROL_POWER", "READ_CURRENT_CHANNEL", "READ_RUNNING_APPS",
            "READ_UPDATE_INFO", "UPDATE_FROM_REMOTE_APP", "READ_LGE_TV_INPUT_EVENTS", "READ_TV_CURRENT_TIME"],
        "serial": "2f930e2d2cfe083771f68e4fe7bb07", "vendorId": "com.lge"}}}


# ---------- минимальный WebSocket-клиент (как в tvremote.py + путь и текстовые кадры) ----------

class WS:
    def __init__(self, host, port=3000, path="/", timeout=10):
        self.s = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            c = self.s.recv(4096)
            if not c:
                raise ConnectionError("телевизор закрыл соединение")
            resp += c
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        if b" 101" not in head.split(b"\r\n")[0]:
            raise ConnectionError("не открылся WebSocket")

    def _exact(self, n):
        while len(self.buf) < n:
            c = self.s.recv(65536)
            if not c:
                raise ConnectionError("телевизор закрыл соединение")
            self.buf += c
        d, self.buf = self.buf[:n], self.buf[n:]
        return d

    def send_text(self, text):
        data = text.encode()
        n = len(data)
        hdr = bytes([0x81])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        m = os.urandom(4)
        self.s.sendall(hdr + m + bytes(b ^ m[i % 4] for i, b in enumerate(data)))

    def send(self, obj):
        self.send_text(json.dumps(obj))

    def recv(self):
        msg = b""
        while True:
            b1, b2 = self._exact(2)
            op, n = b1 & 0x0F, b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._exact(8))[0]
            mask = self._exact(4) if b2 & 0x80 else None
            data = self._exact(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op in (9, 10):
                continue
            if op == 8:
                raise ConnectionError("телевизор закрыл WebSocket")
            msg += data
            if b1 & 0x80:
                return json.loads(msg.decode("utf-8", "ignore"))

    def close(self):
        try:
            self.s.sendall(bytes([0x88, 0x80]) + os.urandom(4))
        except OSError:
            pass
        try:
            self.s.close()
        except OSError:
            pass


class TV:
    """Подключение к телевизору по SSAP с сохранённым ключом пары."""

    def __init__(self, ip):
        self.ip = ip
        self.ws = None
        self._n = 0

    def connect(self, on_prompt=None, prompt_timeout=60):
        self.ws = WS(self.ip)
        payload = json.loads(json.dumps(REG))
        try:
            with open(KEY_FILE) as f:
                payload["client-key"] = f.read().strip()
        except FileNotFoundError:
            pass
        self.ws.send({"type": "register", "id": "reg0", "payload": payload})
        self.ws.s.settimeout(prompt_timeout)
        try:
            while True:
                m = self.ws.recv()
                if m.get("type") == "response" and m.get("payload", {}).get("pairingType") == "PROMPT":
                    log("pairing prompt shown on TV")
                    if on_prompt:
                        on_prompt()
                elif m.get("type") == "registered":
                    with open(KEY_FILE, "w") as f:
                        f.write(m["payload"]["client-key"])
                    os.chmod(KEY_FILE, 0o600)
                    self.ws.s.settimeout(15)
                    return self
                elif m.get("type") == "error":
                    raise TVError("pair_denied", "Телевизор отклонил подключение. Нажми кнопку ещё раз "
                                  "и на телевизоре ответь пультом «Да».")
        except socket.timeout:
            raise TVError("pair", "Нужно подтвердить подключение на телевизоре: на экране ТВ нажми пультом "
                          "«Да», затем нажми кнопку ещё раз.")

    def req(self, uri, payload=None):
        self._n += 1
        rid = f"r{self._n}_{time.time_ns()}"
        self.ws.send({"type": "request", "id": rid, "uri": uri, "payload": payload or {}})
        while True:
            m = self.ws.recv()
            if m.get("id") != rid:
                continue
            if m.get("type") == "error":
                raise TVError("ssap", f"Телевизор отказал в команде {uri.split('/')[-1]}: {m.get('error')}")
            return m.get("payload", {})

    def foreground(self):
        try:
            return self.req("ssap://com.webos.applicationManager/getForegroundAppInfo").get("appId", "")
        except TVError:
            return ""

    def launch(self, app_id, params=None):
        p = {"id": app_id}
        if params:
            p["params"] = params
        return self.req("ssap://system.launcher/launch", p)

    def button(self, name):
        """Кнопка пульта (ENTER, BACK, HOME, UP…) через сокет указки webOS."""
        path = self.req("ssap://com.webos.service.networkinput/getPointerInputSocket").get("socketPath")
        if not path:
            raise TVError("ssap", "Телевизор не дал доступ к кнопкам пульта.")
        u = urllib.parse.urlparse(path)
        p = WS(u.hostname, u.port or 3000, u.path or "/", timeout=5)
        try:
            p.send_text(f"type:button\nname:{name}\n\n")
            time.sleep(0.15)
        finally:
            p.close()

    def close(self):
        if self.ws:
            self.ws.close()
            self.ws = None


# ---------- сеть: доступность, MAC, WOL, SSDP, DIAL ----------

def tcp_errno(ip, port=3000, timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port))
    except socket.timeout:
        return errno.ETIMEDOUT
    except OSError as e:
        return e.errno or -1
    finally:
        s.close()


def port_open(ip, port=3000, timeout=1.0):
    return tcp_errno(ip, port, timeout) == 0


def gateway_ip():
    try:
        out = subprocess.run(["/sbin/route", "-n", "get", "default"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r"gateway:\s*([\d.]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def lan_blocked():
    """macOS 15: без разрешения «Локальная сеть» Chrome (и запущенный им помощник) не видит
    устройства в домашней сети: connect сразу даёт «No route to host». Проверяем на роутере:
    если и он «недоступен», виновато разрешение, а не выключенный телевизор."""
    gw = gateway_ip()
    if not gw:
        return False
    for port in (80, 53, 443):
        e = tcp_errno(gw, port, 1.0)
        if e in (0, errno.ECONNREFUSED):
            return False
    return tcp_errno(gw, 80, 1.0) in (errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPERM, errno.EACCES)


def arp_mac(ip):
    try:
        out = subprocess.run(["/usr/sbin/arp", "-n", ip], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    m = re.search(r" at ([0-9a-fA-F:]{11,17}) ", out)
    if not m:
        return None
    # arp на macOS печатает без ведущих нулей (0:1a:…) — нормализуем
    return ":".join(p.zfill(2) for p in m.group(1).lower().split(":"))


def send_wol(macs, ip):
    """Волшебный пакет на все MAC-адреса телевизора (Wi-Fi и проводной), на широковещательные
    адреса сети и прямо на IP (по ARP-кэшу роутера иногда доходит только так)."""
    targets = {"255.255.255.255", ".".join(ip.split(".")[:3] + ["255"]), ip}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sent = 0
    for mac in macs:
        raw = bytes.fromhex(mac.replace(":", "").replace("-", ""))
        if len(raw) != 6:
            continue
        pkt = b"\xff" * 6 + raw * 16
        for t in targets:
            for port in (9, 7):
                try:
                    s.sendto(pkt, (t, port))
                    sent += 1
                except OSError as e:
                    log("wol sendto", t, port, e)
    s.close()
    return sent


def ssdp_search(st_value="urn:dial-multiscreen-org:service:dial:1", timeout=2.0):
    """Возвращает [(ip, location, server)] устройств webOS в сети."""
    msg = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\n"
           f"MX: 1\r\nST: {st_value}\r\n\r\n").encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    s.settimeout(0.3)
    found = {}
    try:
        s.sendto(msg, ("239.255.255.250", 1900))
        s.sendto(msg, ("239.255.255.250", 1900))
        end = time.time() + timeout
        while time.time() < end:
            try:
                d, a = s.recvfrom(65535)
            except socket.timeout:
                continue
            txt = d.decode("utf-8", "ignore")
            server = (re.search(r"(?im)^server:\s*(.*)$", txt) or [None, ""])[1].strip()
            loc = (re.search(r"(?im)^location:\s*(\S+)", txt) or [None, ""])[1]
            if "webos" in server.lower():
                found[a[0]] = (a[0], loc, server)
    except OSError as e:
        log("ssdp", e)
    finally:
        s.close()
    return list(found.values())


def _http(url, method="GET", data=None, timeout=5):
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "text/plain; charset=utf-8"} if data is not None else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.headers, resp.read().decode("utf-8", "ignore")


def dial_app_url(ip, st):
    """URL приложений DIAL (заголовок Application-URL). Порт у LG меняется после перезагрузки,
    поэтому сначала пробуем сохранённый, потом ищем заново через SSDP."""
    locs = []
    if st.get("dial_location"):
        locs.append(st["dial_location"])
    locs.append(f"http://{ip}:1562/")
    tried = set()
    for attempt in range(2):
        for loc in locs:
            if loc in tried:
                continue
            tried.add(loc)
            try:
                _, headers, _ = _http(loc, timeout=3)
                app_url = headers.get("Application-URL")
                if app_url:
                    st["dial_location"] = loc
                    return app_url.rstrip("/")
            except Exception as e:
                log("dial desc", loc, e)
        if attempt == 0:
            locs = [loc for (i, loc, _) in ssdp_search(timeout=2.0) if i == ip]
    return None


def dial_youtube(ip, st):
    """(state, screenId) приложения YouTube по DIAL, либо (None, None)."""
    app_url = dial_app_url(ip, st)
    if not app_url:
        return None, None
    try:
        _, _, body = _http(app_url + "/YouTube", timeout=4)
    except Exception as e:
        log("dial youtube", e)
        return None, None
    state = (re.search(r"<state>(.*?)</state>", body) or [None, None])[1]
    sid = (re.search(r"<screenId>(.*?)</screenId>", body) or [None, None])[1]
    return state, sid


def dial_launch_youtube(ip, st, body):
    """Запуск/«подключение» YouTube по DIAL (как делает телефон перед трансляцией).
    Пустое тело этот телевизор не принимает (висит), поэтому шлём pairingCode."""
    app_url = dial_app_url(ip, st)
    if not app_url:
        return False
    try:
        status, _, _ = _http(app_url + "/YouTube", method="POST", data=body.encode(), timeout=8)
        return status in (200, 201)
    except Exception as e:
        log("dial launch", e)
        return False

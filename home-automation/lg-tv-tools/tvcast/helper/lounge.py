"""YouTube Lounge API: тот же канал, которым пользуется кнопка «Трансляция» в приложении YouTube.

Телевизор (screenId из DIAL) → lounge token → сессия BrowserChannel (/bc/bind) → команды
setPlaylist / play / pause. Проверено на LG 42LF652V (webOS 2.0, YouTube TVHTML5 7.2026):
setPlaylist с videoId и currentTime сам запускает видео с нужной секунды.

Формат команды (как в casttube): count=1, req0__sc=<команда>, req0_<параметр>=<значение>.
"""
import json, re, socket, time, urllib.error, urllib.parse, urllib.request, uuid

BASE = "https://www.youtube.com/api/lounge"
BIND = BASE + "/bc/bind"
REMOTE = {"device": "REMOTE_CONTROL", "name": "Mac Chrome", "mdx-version": 3,
          "pairing_type": "cast", "app": "android-phone-13.14.55"}


class LoungeError(Exception):
    pass


def _post(url, data, headers=None, timeout=15):
    r = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(),
                               headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        raise LoungeError(f"HTTP {e.code}") from e
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        raise LoungeError(f"нет связи с youtube.com ({e})") from e


def lounge_token(screen_id):
    body = _post(BASE + "/pairing/get_lounge_token_batch", {"screen_ids": screen_id})
    try:
        return json.loads(body)["screens"][0]["loungeToken"]
    except (ValueError, KeyError, IndexError):
        raise LoungeError("YouTube не выдал токен для телевизора")


def screen_status(token):
    """'online' — приложение YouTube на ТВ подключено к серверу, иначе 'offline'."""
    try:
        body = _post(BASE + "/pairing/get_screen_availability", {"lounge_token": token}, timeout=8)
        return (json.loads(body).get("screens") or [{}])[0].get("status", "offline")
    except (LoungeError, ValueError):
        return "offline"


class _Parser:
    """Поток BrowserChannel: «<длина>\\n<JSON-массив>» подряд. Длину не считаем (она в символах
    UTF-16), а берём JSON через raw_decode — так не ломается на не-ASCII."""

    def __init__(self):
        self.buf = ""
        self.dec = json.JSONDecoder()

    def feed(self, text):
        self.buf += text
        out = []
        while True:
            s = self.buf.lstrip()
            m = re.match(r"(\d+)\s*\n", s)
            if not m:
                self.buf = s
                return out
            try:
                obj, end = self.dec.raw_decode(s, m.end())
            except ValueError:
                self.buf = s
                return out
            self.buf = s[end:]
            for item in obj if isinstance(obj, list) else []:
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list) and item[1]:
                    out.append((item[0], item[1][0], item[1][1:]))


class Session:
    def __init__(self, token, device_id=None):
        self.token = token
        self.headers = {"X-YouTube-LoungeId-Token": token}
        self.id = device_id or str(uuid.uuid4())
        self.rid = 0
        self.aid = 0
        self.sid = self.gsid = None
        self.video = None      # videoId из последнего nowPlaying
        self.state = {}        # последнее состояние плеера (state, currentTime, loadedTime…)
        self.screen_seen = False

    # --- соединение ---
    def connect(self):
        self.rid += 1
        body = _post(BIND + "?" + urllib.parse.urlencode({"RID": self.rid, "VER": 8, "CVER": 1}),
                     dict(REMOTE, id=self.id), self.headers)
        self._apply(_Parser().feed(body))
        if not (self.sid and self.gsid):
            raise LoungeError("YouTube не открыл сессию с телевизором")
        return self

    def _apply(self, events):
        for eid, name, args in events:
            if isinstance(eid, int):
                self.aid = max(self.aid, eid)
            a0 = args[0] if args else None
            if name == "c":
                self.sid = a0
            elif name == "S":
                self.gsid = a0
            elif name == "loungeStatus" and isinstance(a0, dict):
                self.screen_seen = "LOUNGE_SCREEN" in a0.get("devices", "")
            elif name == "loungeScreenDisconnected":
                self.screen_seen = False
            elif name == "nowPlaying" and isinstance(a0, dict):
                self.video = a0.get("videoId")
                self.state = dict(a0)
            elif name == "onStateChange" and isinstance(a0, dict):
                self.state.update(a0)
        return events

    def command(self, sc, **params):
        data = {"count": 1, "req0__sc": sc}
        data.update({f"req0_{k}": v for k, v in params.items()})
        for attempt in range(2):
            self.rid += 1
            q = {"SID": self.sid, "gsessionid": self.gsid, "RID": self.rid, "VER": 8, "CVER": 1}
            try:
                return _post(BIND + "?" + urllib.parse.urlencode(q), data, self.headers)
            except LoungeError as e:
                # 400/404/410 — сессия протухла: открываем новую и повторяем
                if attempt == 0 and str(e).startswith("HTTP 4"):
                    self.connect()
                    continue
                raise

    def close(self):
        """Закрыть сессию, чтобы на ТВ не копились «подключённые пульты»."""
        if not self.sid:
            return
        self.rid += 1
        q = {"SID": self.sid, "gsessionid": self.gsid, "RID": self.rid, "VER": 8, "CVER": 1, "TYPE": "terminate"}
        try:
            _post(BIND + "?" + urllib.parse.urlencode(q), {}, self.headers, timeout=4)
        except LoungeError:
            pass
        self.sid = None

    # --- события ---
    def poll(self, seconds, stop=None):
        """Читает события ~seconds секунд (или пока stop(self, event) не вернёт True)."""
        end = time.time() + seconds
        while time.time() < end:
            q = {"SID": self.sid, "gsessionid": self.gsid, "RID": "rpc", "VER": 8, "CVER": 1,
                 "CI": 0, "TYPE": "xmlhttp", "AID": self.aid}
            req = urllib.request.Request(BIND + "?" + urllib.parse.urlencode(q), headers=self.headers)
            parser = _Parser()
            try:
                with urllib.request.urlopen(req, timeout=max(0.5, end - time.time())) as resp:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        for ev in self._apply(parser.feed(line.decode("utf-8", "ignore"))):
                            if stop and stop(self, ev):
                                return True
                        if time.time() >= end:
                            return False
            except (socket.timeout, TimeoutError):
                return False
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 410):
                    self.connect()
                    continue
                raise LoungeError(f"HTTP {e.code}")
            except (urllib.error.URLError, OSError) as e:
                if "timed out" in str(e):
                    return False
                raise LoungeError(f"нет связи с youtube.com ({e})")
        return False

    # --- удобные команды ---
    def play_video(self, video_id, start=0.0):
        self.video, self.state = None, {}
        self.command("setPlaylist", videoId=video_id, currentTime=str(int(max(0, start))),
                     currentIndex=-1, audioOnly="false")

    def wait_video(self, video_id, seconds):
        """Ждёт, пока телевизор возьмёт наше видео. Возвращает 'playing', 'loading' или None."""
        def stop(s, ev):
            return s.video == video_id and s.state.get("state") == "1"
        if self.poll(seconds, stop):
            return "playing"
        return "loading" if self.video == video_id else None

    def wait_screen(self, seconds):
        """После запуска приложения ждём СВЕЖИЙ loungeStatus с телевизором. Состояние из bind
        не годится: сервер ещё ~10 с помнит прежнее подключение (замер: loungeScreenDisconnected
        через 10,6 с после запуска, новый loungeStatus с ТВ через 14,1 с)."""
        self.screen_seen = False
        return self.poll(seconds, lambda s, ev: s.screen_seen)

    def current(self, seconds=2.5):
        """Текущее состояние плеера на ТВ (телевизор присылает nowPlaying новому пульту)."""
        self.poll(seconds, lambda s, ev: ev[1] in ("nowPlaying", "onStateChange") and "state" in s.state)
        return self.video, self.state.get("state")

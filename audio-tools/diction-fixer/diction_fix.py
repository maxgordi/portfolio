#!/usr/bin/env python3
"""
diction_fix.py — автокорректор дикции для вокальных стемов (Suno → Kits).

Что делает:
  1. Раскладывает ваш текст песни по вокальной дорожке с точностью до буквы
     (украинская нейросеть, распознающая буквы, + принудительное выравнивание).
  2. Оценивает, насколько чётко слышна каждая согласная на своём месте.
  3. Для «проглоченных» букв находит в этой же песне чёткий вариант той же буквы
     (или берёт вашу запись из папки доноров) и подмешивает его ровно в нужное место:
     с подгонкой громкости, высоты (для звонких) и плавными краями.
  4. Сохраняет исправленный WAV (та же длина, тот же формат), таблицу всех букв
     с таймингом, список исправлений и маркеры для DAW.

Пример:
  python3 diction_fix.py vocal.wav lyrics.txt
  python3 diction_fix.py vocal.wav lyrics.txt --letters рдт --threshold 0.4
  python3 diction_fix.py vocal.wav lyrics.txt --report-only
"""
import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac M1/M2: незнакомые операции — на процессор

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "robinhad/wav2vec2-xls-r-300m-uk"
CONSONANTS = set("бвгґджзклмнпрстфхцчшщ")
VOICED = set("бвгґджзлмнр")
VOWELS = set("аеєиіїоуюя")
SOFT = set("іїєюяь")  # после этих букв согласная звучит мягко  # для них подгоняем высоту тона донора
_PAIRS = ["бп", "дт", "зс", "жш", "гх", "ґк", "зц", "дц", "тц", "сш", "зж", "тч", "дч"]
ASSIMILATES = {}  # конечная согласная → начальные согласные следующего слова, с которыми она сливается
for _a, _b in _PAIRS:
    ASSIMILATES.setdefault(_a, {_a}).add(_b)
    ASSIMILATES.setdefault(_b, {_b}).add(_a)
APOSTROPHES ="’ʼ`´‘′"
# Если в украинском тексте попались русские буквы — переводим в ближайшие украинские.
FALLBACK_MAP = {"ы": "и", "э": "е", "ъ": "'", "ё": "ьо"}
# Транслит для имён файлов-доноров (р.wav или r.wav — оба подходят).
TRANSLIT = {
    "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "ж": "zh", "з": "z", "к": "k",
    "л": "l", "м": "m", "н": "n", "п": "p", "р": "r", "с": "s", "т": "t", "ф": "f",
    "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- текст

def parse_lyrics(text, vocab_letters, drop_parens=False):
    """Возвращает список слов: [(номер_строки, слово)]. Буквы вне словаря модели отбрасываются."""
    words = []
    dropped = set()
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\[[^\]]*\]", " ", line)  # теги Suno: [Chorus], [Verse 1]
        if drop_parens:
            line = re.sub(r"\([^)]*\)", " ", line)
        line = line.lower()
        for a in APOSTROPHES:
            line = line.replace(a, "'")
        for src, dst in FALLBACK_MAP.items():
            line = line.replace(src, dst)
        for token in re.split(r"[^\w']+", line):
            word = "".join(ch for ch in token if ch in vocab_letters)
            dropped.update(ch for ch in token if ch not in vocab_letters and ch.isalpha())
            word = word.strip("'")
            if word:
                words.append((line_no, word))
    if dropped:
        log(f"  ! Пропущены символы, которых нет в алфавите модели: {''.join(sorted(dropped))}")
    return words


# ---------------------------------------------------------------- модель

class LetterModel:
    def __init__(self, model_id, device_pref="auto"):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).eval()
        self.device = self._pick_device(device_pref)
        try:
            self.model.to(self.device)
        except Exception as e:  # noqa: BLE001
            log(f"  ! Не удалось перейти на {self.device} ({e}); работаю на процессоре")
            self.device = "cpu"
            self.model.to("cpu")
        tok = self.processor.tokenizer
        self.vocab = tok.get_vocab()
        self.blank = tok.pad_token_id
        self.word_sep = self.vocab.get("|")
        self.letters = {ch for ch in self.vocab if len(ch) == 1 and (ch.isalpha() or ch == "'")}
        self.hop = self._measure_hop()

    def _pick_device(self, pref):
        torch = self.torch
        if pref != "auto":
            return pref
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _logprobs(self, audio16):
        torch = self.torch
        inputs = self.processor(audio16, sampling_rate=16000, return_tensors="pt")
        key = "input_features" if "input_features" in inputs else "input_values"
        x = inputs[key].to(self.device)
        with torch.inference_mode():
            try:
                logits = self.model(**{key: x}).logits[0]
            except Exception as e:  # noqa: BLE001
                if self.device == "cpu":
                    raise
                log(f"  ! Ошибка на {self.device} ({e}); переключаюсь на процессор")
                self.device = "cpu"
                self.model.to("cpu")
                logits = self.model(**{key: x.cpu()}).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def _measure_hop(self):
        n = 16000 * 4
        frames = self._logprobs(np.zeros(n, dtype=np.float32)).shape[0]
        hop = int(round(n / frames / 160.0)) * 160  # шаг кадра в отсчётах 16 кГц
        return max(hop, 160)

    def emissions(self, audio16, chunk_s=20.0, overlap_s=2.0):
        """Лог-вероятности букв по кадрам для всей дорожки (длинные треки — окнами с перекрытием)."""
        hop = self.hop
        n = len(audio16)
        chunk = int(chunk_s * 16000) // hop * hop
        ov = int(overlap_s * 16000) // hop * hop
        step = chunk - ov
        total = int(math.ceil(n / hop))
        out = None
        start = 0
        while True:
            seg = audio16[start:start + chunk]
            if len(seg) < hop * 4:
                seg = np.pad(seg, (0, hop * 4 - len(seg)))
            lp = self._logprobs(seg)
            if out is None:
                out = np.full((total, lp.shape[1]), -1e4, dtype=np.float32)
                out[:, self.blank] = 0.0
            g0 = start // hop
            first = start + chunk >= n
            lo = 0 if start == 0 else ov // hop // 2
            hi = lp.shape[0] if first else (chunk - ov // 2) // hop
            for i in range(lo, min(hi, lp.shape[0])):
                if g0 + i < total:
                    out[g0 + i] = lp[i]
            if first:
                break
            start += step
        return out

    def emissions_hires(self, audio16, **kw):
        """Двойной прогон со сдвигом на полкадра → вдвое точнее по времени."""
        half = self.hop // 2
        a = self.emissions(audio16, **kw)
        b = self.emissions(np.concatenate([np.zeros(half, dtype=audio16.dtype), audio16]), **kw)
        m = min(len(a), len(b) - 1)
        out = np.empty((2 * m, a.shape[1]), dtype=np.float32)
        out[0::2] = a[:m]
        out[1::2] = b[1:m + 1]
        return out, half


# ---------------------------------------------------------------- выравнивание

def ctc_force_align(logp, targets, blank):
    """Витерби-выравнивание CTC: для каждого кадра — номер буквы текста или -1 (пауза)."""
    T = logp.shape[0]
    L = len(targets)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    skip = np.zeros(S, dtype=bool)
    if L > 1:
        skip[3::2] = ext[3::2] != ext[1:-2:2]
    NEG = np.float32(-1e30)
    emit = logp[:, ext]
    dp = np.full(S, NEG, dtype=np.float32)
    dp[0] = emit[0, 0]
    if S > 1:
        dp[1] = emit[0, 1]
    bp = np.zeros((T, S), dtype=np.int8)
    neg1 = np.array([NEG], dtype=np.float32)
    neg2 = np.array([NEG, NEG], dtype=np.float32)
    for t in range(1, T):
        p1 = np.concatenate([neg1, dp[:-1]])
        p2 = np.where(skip, np.concatenate([neg2, dp[:-2]]), NEG)
        stacked = np.stack([dp, p1, p2])
        choice = stacked.argmax(axis=0)
        dp = stacked[choice, np.arange(S)] + emit[t]
        bp[t] = choice
    s = S - 1 if S == 1 or dp[S - 1] >= dp[S - 2] else S - 2
    if dp[s] < -1e29:
        raise RuntimeError("Текст не помещается в дорожку: проверьте, что это тот вокал и тот текст.")
    states = np.empty(T, dtype=np.int64)
    s = int(s)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(bp[t, s])
    return np.where(states % 2 == 1, states // 2, -1)


# ---------------------------------------------------------------- звук

def load_audio(path):
    """Читает WAV/FLAC/AIFF/MP3; что не открывает soundfile (m4a, aac…) — через ffmpeg.
    Возвращает (звук [сэмплы × каналы], частота, формат исходника, subtype)."""
    try:
        info = sf.info(str(path))
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return audio, sr, info.format, info.subtype
    except Exception:  # noqa: BLE001
        pass
    import json
    import subprocess

    try:
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
             "-of", "json", str(path)], capture_output=True, check=True).stdout)["streams"][0]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, IndexError):
        sys.exit(f"Не получается прочитать {path}. Сохраните вокал в WAV или MP3 (или установите ffmpeg).")
    ch = int(probe["channels"])
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).copy(), int(probe["sample_rate"]), "OTHER", None


def to_mono16k(audio, sr):
    import librosa

    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    if sr != 16000:
        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=16000)
    return mono.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def median_f0(mono, sr, center, half_win=0.09):
    import librosa

    a = max(0, int((center - half_win) * sr))
    b = min(len(mono), int((center + half_win) * sr))
    seg = mono[a:b]
    if len(seg) < int(0.06 * sr):
        return None
    frame = 2048 if sr > 32000 else 1024
    if len(seg) < frame:
        seg = np.pad(seg, (0, frame - len(seg)))
    f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=1100, sr=sr, frame_length=frame, hop_length=frame // 8)
    f0 = f0[voiced] if voiced is not None else f0
    f0 = f0[~np.isnan(f0)] if f0 is not None else []
    return float(np.median(f0)) if len(f0) else None


def pitch_shift_segment(src, sr, a, b, n_steps, ctx=0.12):
    """Сдвигает высоту куска src[a:b] (каналы по столбцам), обрабатывая его с запасом по краям."""
    import librosa

    ca, cb = max(0, a - int(ctx * sr)), min(len(src), b + int(ctx * sr))
    block = src[ca:cb]
    n_fft = 2048 if sr > 32000 else 1024
    if len(block) < n_fft:
        block = np.pad(block, ((0, n_fft - len(block)), (0, 0)))
    shifted = np.stack([librosa.effects.pitch_shift(block[:, c], sr=sr, n_steps=n_steps, n_fft=n_fft)
                        for c in range(block.shape[1])], axis=1)
    shifted = np.nan_to_num(shifted)
    return shifted[a - ca:a - ca + (b - a)]


def _xfade(a, b):
    """Плавный кроссфейд двух кусков одинаковой длины (каналы по столбцам), как X в Cubase.
    S-кривая (приподнятый косинус): начинается и заканчивается без излома, поэтому стык не щёлкает;
    в середине разные звуки чуть притихают — вставка меньше выпирает."""
    n = len(a)
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)[:, None]
    return a * np.cos(t) ** 2 + b * np.sin(t) ** 2


def find_cut(mono, sr, zones, lo, hi, r0, max_dev, protected, silence, backward=False, W=None):
    """Ищет, где незаметно убрать ~r0 сэмплов на отрезке [lo, hi]. Разрешено только внутри «зон»:
    середина гласной (длинная и ровная в пении) или пауза между словами (весь кусок тихий).
    backward=True — точка вставки справа от отрезка (ближе к hi лучше), иначе слева.
    Возвращает (начало, точная_длина, «цена») или None."""
    from scipy.signal import correlate

    W = W or int(0.02 * sr)  # окно сравнения = длина кроссфейда в месте выреза
    step = max(1, int(0.002 * sr))
    need = r0 + max_dev + W
    best = None
    span = max(1, hi - lo)
    if hi - lo < need:
        return None
    for zs, ze, kind in zones:
        share = need / max(1, ze - zs)  # какую долю гласной съест вырез: из короткой резать нельзя
        zs, ze = max(zs, lo), min(ze, hi)
        if ze - zs < need:
            continue
        for c in range(zs, ze - need + 1, step):
            if any(c < e and c + need > s for s, e in protected):
                continue
            # чем ближе к вставке, тем лучше: меньше звука сдвигается
            dist = 0.15 * ((hi - c - need) if backward else (c - lo)) / span
            if kind == "pause":
                if np.abs(mono[c:c + need]).max() >= silence * 3:
                    continue
                cost, r = 0.02 + dist, r0
            else:
                a = mono[c:c + W]
                b = mono[c + r0 - max_dev:c + r0 + max_dev + W]
                if rms(a) < silence or rms(b) < silence:
                    continue
                xc = correlate(b, a, mode="valid", method="fft")
                cs = np.concatenate([[0.0], np.cumsum(b.astype(np.float64) ** 2)])
                eb = np.sqrt(np.maximum(cs[W:] - cs[:-W], 1e-12))[:len(xc)]
                ncc = xc / (np.sqrt(np.sum(a.astype(np.float64) ** 2)) * eb + 1e-9)
                k = int(np.argmax(ncc))
                if ncc[k] < 0.8:  # волна на стыке совпадает плохо — срезка будет слышна
                    continue
                cost, r = 1.0 - float(ncc[k]) + dist + 0.3 * share, r0 - max_dev + k
            if best is None or cost < best[2]:
                best = (c, r, cost)
    return best


def insert_and_compensate(buf, sr, p, seg, zones, protected, silence, floor=0, xf_s=0.02):
    """Вставляет согласную в точку p, раздвигая звук, и забирает столько же времени
    из ближайшей паузы или середины гласной рядом (после или до вставки) — общий тайминг не меняется.
    seg — донор с запасом xf_s по краям: на запасе идут кроссфейды, сама буква остаётся целой."""
    xf = int(xf_s * sr)
    xc = max(xf, int(0.04 * sr))  # кроссфейд в месте, где забираем время
    if len(seg) < 2 * xf + int(0.01 * sr) or p - xf < 0 or p + xf > len(buf):
        return None
    r0 = len(seg) - 2 * xf
    max_dev = int(0.008 * sr)
    mono = buf.mean(axis=1)
    gap = int(0.03 * sr)
    cut = None
    for reach in (0.8, 2.0):
        reach_n = int(reach * sr)
        after = find_cut(mono, sr, zones, p + gap, min(len(buf) - xc, p + reach_n), r0, max_dev, protected, silence, W=xc)
        # до вставки: ищем в обратную сторону, «цена» растёт с удалением от p
        before = find_cut(mono, sr, zones, max(floor, p - reach_n), p - gap, r0, max_dev, protected, silence,
                          backward=True, W=xc)
        cands = [(cv, d) for cv, d in ((after, 1), (before, -1)) if cv and cv[2] < 0.4]
        if cands:
            cut, direction = min(cands, key=lambda x: x[0][2])
            break
    if cut is None:
        return None
    c, r, _ = cut
    if direction < 0:
        return _insert_cut_before(buf, sr, p, seg, c, r, xf, xc, protected)
    # подгоняем длину донора под точную длину выреза (r), чтобы итоговая длина совпала до сэмпла
    seg = _fit_len(seg, r + 2 * xf)
    # 1) убираем r сэмплов в точке c (с кроссфейдом)
    out = np.concatenate([buf[:c], _xfade(buf[c:c + xc], buf[c + r:c + r + xc]), buf[c + r + xc:]])
    # 2) вставляем донора в точку p (с кроссфейдами на стыках)
    out = np.concatenate([out[:p - xf], _xfade(out[p - xf:p], seg[:xf]), seg[xf:-xf],
                          _xfade(seg[-xf:], out[p:p + xf]), out[p + xf:]])
    assert len(out) == len(buf)
    # то, что лежало между вставкой и вырезом, сдвинулось вправо на r
    for i, (s, e) in enumerate(protected):
        if s >= p and e <= c:
            protected[i] = (s + r, e + r)
    protected.append((p - xf, p + r + xf))
    protected.append((c + r - xc, c + r + xc))
    return out.astype(np.float32), c, r


def _fit_len(seg, need):
    if len(seg) > need:
        trim = len(seg) - need
        return seg[trim // 2:len(seg) - (trim - trim // 2)]
    if len(seg) < need:
        extra = need - len(seg)
        return np.pad(seg, ((extra // 2, extra - extra // 2), (0, 0)))
    return seg


def _insert_cut_before(buf, sr, p, seg, c, r, xf, xc, protected):
    """Как insert_and_compensate, но время забирается в точке c ДО вставки:
    звук между c и p сдвигается влево на r, донор встаёт в p - r."""
    seg = _fit_len(seg, r + 2 * xf)
    out = np.concatenate([buf[:c], _xfade(buf[c:c + xc], buf[c + r:c + r + xc]), buf[c + r + xc:]])
    q = p - r  # точка вставки после выреза
    out = np.concatenate([out[:q - xf], _xfade(out[q - xf:q], seg[:xf]), seg[xf:-xf],
                          _xfade(seg[-xf:], out[q:q + xf]), out[q + xf:]])
    assert len(out) == len(buf)
    for i, (s, e) in enumerate(protected):
        if s >= c + r and e <= p:
            protected[i] = (s - r, e - r)
    protected.append((q - xf, p + xf))
    protected.append((c - xc, c + xc))
    return out.astype(np.float32), c, -r


def shift_zones(zones, p, c, r):
    """Пересчитывает зоны после вставки в p и выреза в c (длина r; r < 0 — вырез был до вставки).
    Задетая зона не выбрасывается, а режется: остаётся всё, кроме вырезанного куска;
    стыки вставки и выреза закрыты списком protected."""
    if r >= 0:  # звук [p, c) уехал вправо на r, кусок [c, c+r) вырезан
        maps = [(-10**12, p, 0), (p, c, r), (c + r, 10**12, 0)]
    else:  # кусок [c, c+r) вырезан, звук [c+r, p) уехал влево на r
        r = -r
        maps = [(-10**12, c, 0), (c + r, p, -r), (p, 10**12, 0)]
    out = []
    for s, e, kind in zones:
        for a, b, d in maps:
            lo, hi = max(s, a), min(e, b)
            if hi > lo:
                out.append((lo + d, hi + d, kind))
    return out


def edge_window(n, fade):
    w = np.ones(n, dtype=np.float32)
    fade = min(fade, n // 2)
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))
        w[:fade] = ramp
        w[-fade:] = ramp[::-1]
    return w


# ---------------------------------------------------------------- правки (общие для командной строки и окна)

def read_own_donor(path, sr, channels):
    """Ваша запись буквы: под частоту и число каналов вокала."""
    seg, own_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if own_sr != sr:
        import librosa
        seg = np.stack([librosa.resample(seg[:, c], orig_sr=own_sr, target_sr=sr)
                        for c in range(seg.shape[1])], axis=1)
    if seg.shape[1] != channels:
        seg = np.repeat(seg.mean(axis=1, keepdims=True), channels, axis=1)
    return seg


def cut_donor(audio, sr, D, xf_s, pad=0.008, max_len_ms=140.0, width_s=0.0):
    """Вырезает букву-донора D (lo/start/end/hi в секундах) с запасом xf_s по краям под кроссфейды.
    width_s расширяет (>0) или сужает (<0) саму букву с каждой стороны."""
    core_lo = (D["lo"] + D["start"]) / 2 - width_s  # без лишних кусков соседних звуков
    core_hi = (D["end"] + D["hi"]) / 2 + width_s
    if core_hi - core_lo < 0.015:
        mid = (core_lo + core_hi) / 2
        core_lo, core_hi = mid - 0.0075, mid + 0.0075
    a = max(0, int((core_lo - pad - xf_s) * sr))
    b = min(len(audio), int((core_hi + pad + xf_s) * sr))
    b = min(b, a + int((max_len_ms / 1000 + 2 * xf_s + 2 * max(width_s, 0)) * sr))
    return audio[a:b].copy()


def apply_fix(buf, sr, zones, protected, lo, hi, seg, mode, pos, floor, silence, xf_s, duck_db=-20.0):
    """Применяет одну правку к копии дорожки: буква на месте [lo, hi] (секунды), донор seg (уже с громкостью).
    Возвращает (дорожка, зоны, защита, заметка) или None, если не нашлось места."""
    zones, protected = list(zones), list(protected)
    if mode == "insert":
        p = int((lo + pos * (hi - lo)) * sr)
        res = insert_and_compensate(buf, sr, p, seg, zones, protected, silence, floor, xf_s)
        if res is None:
            return None
        nb, cut_at, cut_len = res
        return nb, shift_zones(zones, p, cut_at, cut_len), protected, \
            f"раздвинуто на {abs(cut_len) / sr * 1000:.0f} мс, время возвращено на {cut_at / sr:.2f} с"
    n = len(seg)
    fade = int(xf_s * sr)
    # поверх: буква плавно «выплывает» и уходит (S-кривая на треть длины), оригинал звучит под ней как был —
    # две дорожки сводятся в одну, без вклеенного куска
    win = edge_window(n, max(fade, int(0.35 * n)) if mode == "overlay" else fade)[:, None]
    start = int((lo + pos * (hi - lo)) * sr) - n // 2
    s0, s1 = max(0, start), min(len(buf), start + n)
    if s1 <= s0:
        return None
    nb = buf.copy()
    w = win[s0 - start:s1 - start]
    ins = seg[s0 - start:s1 - start] * w
    if mode == "replace":
        keep = 10 ** (duck_db / 20.0)  # сколько оригинала остаётся под вставкой
        nb[s0:s1] = nb[s0:s1] * (1 - w * (1 - keep)) + ins
    else:
        nb[s0:s1] += ins
    protected.append((s0, s1))
    return nb, zones, protected, "заменено поверх" if mode == "replace" else "подмешано поверх"


def listen_words(lm, buf, sr, words, ws, t_lo, t_hi, hires=True):
    """Прослушивает кусок дорожки моделью: {(номер слова, номер буквы): уверенность} для слов ws."""
    seg16 = to_mono16k(buf[int(t_lo * sr):int(t_hi * sr)], sr)
    tg, mt = [], []
    for j, wi in enumerate(ws):
        if j and lm.word_sep is not None:
            tg.append(lm.word_sep)
            mt.append(None)
        for ci, ch in enumerate(words[wi][1]):
            tg.append(lm.vocab[ch])
            mt.append((wi, ci))
    lp = lm.emissions_hires(seg16)[0] if hires else lm.emissions(seg16)
    try:
        ft = ctc_force_align(lp, np.array(tg), lm.blank)
    except RuntimeError:
        return {}
    pr = np.exp(lp)
    out = {}
    for k, m in enumerate(mt):
        if m is None:
            continue
        fr = np.nonzero(ft == k)[0]
        out[m] = float(pr[max(0, fr[0] - 1):fr[-1] + 2, tg[k]].max())
    return out


def render_project(audio, proj, only=None, on_progress=None):
    """Собирает исправленную дорожку из оригинала по файлу проекта (с ручными правками из окна).
    only — набор индексов правок, которые применять (по умолчанию все включённые).
    Возвращает (дорожка, {индекс правки: заметка или None, если не нашлось места})."""
    sr = proj["sr"]
    letters = proj["letters"]
    buf, zones, protected = audio.copy(), [tuple(z) for z in proj["zones"]], []
    own_cache = {}
    notes = {}
    order = sorted((i for i, F in enumerate(proj["fixes"]) if (F["enabled"] if only is None else i in only)),
                   key=lambda i: -letters[proj["fixes"][i]["letter"]]["start"])
    for n, i in enumerate(order, 1):
        F = proj["fixes"][i]
        L = letters[F["letter"]]
        D = F["donors"][F["donor"]]
        xf_s = (F.get("fade_ms") or proj["xfade_ms"]) / 1000.0
        if D["kind"] == "own":
            if D["path"] not in own_cache:
                own_cache[D["path"]] = read_own_donor(D["path"], sr, audio.shape[1])
            seg = own_cache[D["path"]]
        else:
            seg = cut_donor(audio, sr, D, xf_s, proj["pad"], proj["max_len_ms"], F.get("width_ms", 0) / 1000.0)
        gain = 10 ** ((D["match_gain_db"] + F.get("gain_offset_db", 0.0)) / 20.0)
        shift = F.get("shift_ms", 0) / 1000.0
        res = apply_fix(buf, sr, zones, protected, L["lo"] + shift, L["hi"] + shift, seg * gain, F["mode"],
                        F["pos"], F["floor"], proj["silence"], xf_s, proj["duck_db"])
        if res is None:
            notes[i] = None
        else:
            buf, zones, protected, notes[i] = res
        if on_progress:
            on_progress(n, len(order))
    return buf, notes


# ---------------------------------------------------------------- основная логика

def main():
    ap = argparse.ArgumentParser(description="Автокорректор дикции: находит проглоченные буквы и вставляет их.")
    ap.add_argument("vocal", help="вокальный стем (WAV/FLAC/AIFF)")
    ap.add_argument("lyrics", help="текст песни (.txt), строки в том порядке, как поются, с повторами припевов")
    ap.add_argument("--out-dir", default=None, help="куда сохранить результаты (по умолчанию рядом с вокалом)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"модель распознавания букв (по умолчанию {DEFAULT_MODEL})")
    ap.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    ap.add_argument("--letters", default="", help="какие буквы проверять, например 'рдт' (по умолчанию все согласные)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="порог: буква ниже этой уверенности (0..1) считается проглоченной (по умолчанию 0.5; "
                         "в пении буква на 0.4 часто уже не слышна)")
    ap.add_argument("--force", default="",
                    help="исправить эти буквы в этих словах, даже если модель их слышит: 'град:р,шлях:х'")
    ap.add_argument("--xfade-ms", type=float, default=30.0,
                    help="длина плавных переходов на стыках вставки, мс (по умолчанию 30; больше — мягче)")
    ap.add_argument("--mp3", action="store_true", help="сохранить результат ещё и в MP3 (для MP3 на входе — всегда)")
    ap.add_argument("--donor-min", type=float, default=0.8, help="минимальная уверенность для буквы-донора")
    ap.add_argument("--gain", type=float, default=1.0, help="общая громкость вставок (1.0 = как в доноре)")
    ap.add_argument("--mode", choices=["insert", "replace", "overlay"], default="insert",
                    help="insert — раздвинуть слово и вставить букву, забрав время из паузы/гласной (по умолчанию); "
                         "replace — заменить кусок; overlay — подмешать поверх")
    ap.add_argument("--overlay-letters", default="",
                    help="эти буквы не вставлять, а плавно накладывать поверх оригинала, например 'т' или 'тдк'")
    ap.add_argument("--tries", type=int, default=5, help="сколько разных доноров пробовать для каждой буквы")
    ap.add_argument("--accept", type=float, default=0.5,
                    help="вставка принимается, если после неё буква слышна с уверенностью не ниже этой (0..1)")
    ap.add_argument("--duck-db", type=float, default=-20.0,
                    help="в режиме replace: насколько приглушить оригинал под вставкой, дБ (по умолчанию -20)")
    ap.add_argument("--max-len-ms", type=float, default=140.0, help="максимальная длина вставки, мс")
    ap.add_argument("--own-donors", default=None,
                    help="папка с вашими записями букв: р.wav, д.wav… (или r.wav, d.wav)")
    ap.add_argument("--prefer-own", action="store_true", help="сначала брать ваши записи, а не буквы из песни")
    ap.add_argument("--pitch", action="store_true",
                    help="подгонять высоту тона звонких букв (по умолчанию выключено: размывает короткие звуки, "
                         "а Kits всё равно выравнивает высоту сам)")
    ap.add_argument("--no-hires", action="store_true", help="один прогон (быстрее, шаг 40 мс вместо 20 мс)")
    ap.add_argument("--drop-parens", action="store_true", help="не учитывать текст в (скобках) — подпевки")
    ap.add_argument("--report-only", action="store_true", help="только отчёт и маркеры, без нового WAV")
    ap.add_argument("--verbose", action="store_true", help="показывать каждый опробованный вариант вставки")
    args = ap.parse_args()

    t0 = time.time()
    vocal_path = Path(args.vocal)
    out_dir = Path(args.out_dir) if args.out_dir else vocal_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = vocal_path.stem

    log("1/5 Загружаю вокал и модель…")
    audio, sr, in_format, in_subtype = load_audio(vocal_path)
    mono_full = audio.mean(axis=1)
    mono16 = to_mono16k(audio, sr)
    lm = LetterModel(args.model, args.device)
    log(f"  вокал: {len(mono_full)/sr:.1f} с, {sr} Гц, {audio.shape[1]} кан.; устройство: {lm.device}")

    words = parse_lyrics(Path(args.lyrics).read_text(encoding="utf-8"), lm.letters, args.drop_parens)
    if not words:
        sys.exit("Текст пустой или не на украинском/русском — нечего выравнивать.")
    check = set(args.letters.lower()) if args.letters else CONSONANTS
    forced = set()
    for item in filter(None, (x.strip().lower() for x in args.force.split(","))):
        w_, _, ch_ = item.partition(":")
        forced.update((w_, c) for c in (ch_ or w_) if c in CONSONANTS)

    # цели для выравнивания
    targets, meta = [], []  # meta: (word_idx, char_idx, char, line_no) или None для «|»
    for wi, (line_no, word) in enumerate(words):
        if wi > 0 and lm.word_sep is not None:
            targets.append(lm.word_sep)
            meta.append(None)
        for ci, ch in enumerate(word):
            targets.append(lm.vocab[ch])
            meta.append((wi, ci, ch, line_no))
    log(f"  текст: {len(words)} слов, {sum(1 for m in meta if m)} букв")

    log("2/5 Слушаю дорожку по буквам" + (" (двойной прогон, шаг 20 мс)…" if not args.no_hires else "…"))
    if args.no_hires:
        logp = lm.emissions(mono16)
        dt, offset = lm.hop / 16000.0, 0.0
    else:
        logp, half = lm.emissions_hires(mono16)
        dt, offset = (lm.hop / 2) / 16000.0, 0.0

    log("3/5 Раскладываю текст по времени…")
    frame_tok = ctc_force_align(logp, np.array(targets), lm.blank)
    probs = np.exp(logp)
    letters = []
    for k, m in enumerate(meta):
        if m is None:
            continue
        frames = np.nonzero(frame_tok == k)[0]
        f0, f1 = int(frames[0]), int(frames[-1])
        lo, hi = max(0, f0 - 1), min(len(probs), f1 + 2)
        conf = float(probs[lo:hi, targets[k]].max())
        wi, ci, ch, line_no = m
        letters.append(dict(k=k, idx=len(letters), word_idx=wi, char_idx=ci, char=ch, line=line_no, word=words[wi][1],
                            start=offset + f0 * dt, end=offset + (f1 + 1) * dt, conf=conf))

    mean_conf = float(np.mean([L["conf"] for L in letters]))
    if mean_conf < 0.4:
        log(f"  ! Средняя уверенность низкая ({mean_conf:.2f}). Проверьте, что текст совпадает с тем, что спето "
            "(повторы припевов, подпевки), и что это сухой вокал без музыки.")

    for i, L in enumerate(letters):
        P = letters[i - 1] if i > 0 else None
        N = letters[i + 1] if i + 1 < len(letters) else None
        L["prev"] = P["char"] if P and P["word_idx"] == L["word_idx"] else " "
        L["next"] = N["char"] if N and N["word_idx"] == L["word_idx"] else " "
        # реальный звук буквы шире «пика» модели: от конца соседа слева до начала соседа справа
        left = P["end"] if P and L["start"] - P["end"] < 0.10 else L["start"] - 0.03
        right = N["start"] if N and N["start"] - L["end"] < 0.08 else L["end"] + 0.02
        L["lo"], L["hi"] = min(left, L["start"]), max(right, L["end"])
        # украинская «в» перед согласной и в конце слова звучит как короткое «ў» — это норма, не ошибка
        soft_v = L["char"] == "в" and (L["next"] == " " or L["next"] in CONSONANTS) and not args.letters
        # «над темною», «без суму»: конечная согласная сливается с такой же или парной в начале следующего слова
        if L["next"] == " " and N and N["word_idx"] == L["word_idx"] + 1 and not args.letters:
            soft_v = soft_v or N["char"] in ASSIMILATES.get(L["char"], ())
        edge = L["start"] < 0.06 or L["end"] > len(mono_full) / sr - 0.06  # край файла — оценка ненадёжна
        L["exempt"] = soft_v or edge
        L["weak"] = L["char"] in check and L["conf"] < args.threshold and not soft_v and not edge
        L["forced"] = (L["word"], L["char"]) in forced and not edge
        L["weak"] = L["weak"] or L["forced"]

    weak = [L for L in letters if L["weak"]]
    exempt = {(L["word_idx"], L["char_idx"]) for L in letters if L["exempt"]}
    log(f"  найдено проглоченных букв: {len(weak)} (порог {args.threshold})")

    log("4/5 Подбираю доноров и вставляю буквы…")
    own = {}
    if args.own_donors:
        d = Path(args.own_donors)
        for ch in CONSONANTS:
            for name in (ch, TRANSLIT.get(ch, "")):
                for ext in (".wav", ".flac", ".aif", ".aiff"):
                    p = d / f"{name}{ext}"
                    if name and p.exists() and ch not in own:
                        own[ch] = p
        log(f"  ваши записи букв: {''.join(sorted(own)) or 'не найдены'}")

    fixed = audio.copy()
    fixes = []
    xf_s = args.xfade_ms / 1000.0
    pad = 0.008
    protected = []  # участки, которые нельзя резать (уже вставленные буквы)
    silence = 10 ** (-45 / 20) * max(float(np.abs(audio).max()), 1e-6)
    # где можно «забрать» время для вставки: середины гласных и паузы между словами
    zones = []
    for i, L in enumerate(letters):
        if L["char"] in VOWELS:
            # модель отмечает гласную коротким пиком, а спетая нота тянется до следующей буквы
            a_, b_ = L["lo"], L["hi"]
            if i + 1 < len(letters) and letters[i + 1]["start"] - L["end"] < 1.5:
                b_ = max(b_, letters[i + 1]["start"])
            m_ = max(0.2 * (b_ - a_), 0.03)
            m_end = max(0.25 * (b_ - a_), 0.06)
            zones.append((int((a_ + m_) * sr), int((b_ - m_end) * sr), "vowel"))
        if i + 1 < len(letters) and letters[i + 1]["start"] - L["end"] > 0.15:
            zones.append((int((L["end"] + 0.04) * sr), int((letters[i + 1]["start"] - 0.04) * sr), "pause"))
    # тишина до первой и после последней буквы
    if letters[0]["start"] > 0.15:
        zones.append((int(0.02 * sr), int((letters[0]["start"] - 0.04) * sr), "pause"))
    if len(audio) / sr - letters[-1]["end"] > 0.15:
        zones.append((int((letters[-1]["end"] + 0.04) * sr), len(audio) - int(0.02 * sr), "pause"))

    by_word = {}
    for L in letters:
        by_word.setdefault(L["word_idx"], []).append(L)

    def word_span(wi):
        ws = by_word[wi]
        return ws[0]["lo"], ws[-1]["hi"]

    def eval_window(L):
        """Окно для проверки на слух: слово + до двух соседних слов с каждой стороны, если они рядом.
        Короткое окно обманывает модель (вставка «р» в нём слышна как «р», а в целой песне — как «д»),
        поэтому контекста даём побольше."""
        wi = L["word_idx"]
        ws = [wi]
        for _ in range(2):
            if ws[0] - 1 in by_word and word_span(ws[0])[0] - word_span(ws[0] - 1)[1] < 1.0:
                ws.insert(0, ws[0] - 1)
            if ws[-1] + 1 in by_word and word_span(ws[-1] + 1)[0] - word_span(ws[-1])[1] < 1.0:
                ws.append(ws[-1] + 1)
        t_lo = word_span(ws[0])[0] - 0.3
        t_hi = word_span(ws[-1])[1] + 0.3
        if ws[0] - 1 in by_word:
            t_lo = max(t_lo, word_span(ws[0] - 1)[1] + 0.02)
        if ws[-1] + 1 in by_word:
            t_hi = min(t_hi, word_span(ws[-1] + 1)[0] - 0.02)
        return ws, max(0.0, t_lo), min(len(audio) / sr, t_hi)

    def listen(buf, ws, t_lo, t_hi):
        return listen_words(lm, buf, sr, words, ws, t_lo, t_hi, hires=not args.no_hires)

    def donor_segment(L, D):
        if D == "own":
            seg = read_own_donor(own[L["char"]], sr, audio.shape[1])
            return seg, rms(seg), None, f"ваша запись {own[L['char']].name}"
        # запас xfade по краям: на нём пойдут кроссфейды, сама буква не съедается
        seg = cut_donor(audio, sr, D, xf_s, pad, args.max_len_ms)
        dc = (D["lo"] + D["hi"]) / 2
        if L["char"] in VOICED and args.pitch:
            core_lo, core_hi = (D["lo"] + D["start"]) / 2, (D["end"] + D["hi"]) / 2
            a_, b_ = max(0, int((core_lo - pad - xf_s) * sr)), int((core_hi + pad + xf_s) * sr)
            f_t = median_f0(mono_full, sr, (L["lo"] + L["hi"]) / 2)
            f_d = median_f0(mono_full, sr, dc)
            if f_t and f_d and 0.3 < abs(12 * math.log2(f_t / f_d)) <= 7:
                seg = pitch_shift_segment(audio, sr, a_, a_ + len(seg), 12 * math.log2(f_t / f_d))
        ca, cb = max(0, int((dc - 0.12) * sr)), min(len(audio), int((dc + 0.12) * sr))
        return seg, rms(mono_full[ca:cb]), D["conf"], f"«{D['word']}» на {D['start']:.2f} с"

    def apply(buf, zones_, protected_, L, seg, mode, pos, floor=0):
        """Применяет один вариант вставки к копии дорожки. Возвращает (дорожка, зоны, защита, заметка)."""
        return apply_fix(buf, sr, zones_, protected_, L["lo"], L["hi"], seg, mode, pos, floor, silence, xf_s,
                         args.duck_db)

    def letter_confs(buf):
        """Уверенность каждой буквы текста при прослушивании всей дорожки целиком: {k: уверенность}."""
        m16 = to_mono16k(buf, sr)
        lp = lm.emissions(m16) if args.no_hires else lm.emissions_hires(m16)[0]
        ft = ctc_force_align(lp, np.array(targets), lm.blank)
        pr = np.exp(lp)
        out = {}
        for L in letters:
            fr = np.nonzero(ft == L["k"])[0]
            out[L["k"]] = float(pr[max(0, fr[0] - 1):fr[-1] + 2, targets[L["k"]]].max())
        return out

    zones0 = list(zones)

    def replay(keep):
        """Собирает дорожку заново из оригинала, применяя только вставки из keep (в том же порядке)."""
        buf, z, pr = audio.copy(), list(zones0), []
        for F in sorted(keep, key=lambda F: -F["L"]["start"]):
            res = apply(buf, z, pr, *F["args"])
            if res is None:
                F["status"] = "не исправлено (не нашлось места при пересборке) — проверьте вручную"
                F["after"] = None
                continue
            buf, z, pr, F["note"] = res
        return buf

    todo = sorted(weak if not args.report_only else [], key=lambda x: -x["start"])  # с конца: позиции раньше не сдвигаются
    for n_done, L in enumerate(todo, 1):
        cands = [D for D in letters if D["char"] == L["char"] and not D["weak"] and D["conf"] >= args.donor_min
                 and abs(D["start"] - L["start"]) > 0.3]
        for D in cands:
            same_softness = (D["next"] in SOFT) == (L["next"] in SOFT)
            D["_rank"] = D["conf"] + 0.15 * (D["next"] == L["next"]) + 0.2 * same_softness \
                + 0.05 * (D["prev"] == L["prev"]) + 0.05 * min(1.0, (D["end"] - D["start"]) / 0.06)
        cands.sort(key=lambda D: -D["_rank"])
        donors = cands[:args.tries]
        if L["char"] in own:
            donors = (["own"] + donors) if args.prefer_own else (donors + ["own"])
        if not donors:
            fixes.append(dict(L=L, status=f"нет донора — запишите «{L['char']}» в папку --own-donors "
                                          f"({L['char']}.wav) или проверьте вручную"))
            continue

        ws, t_lo, t_hi = eval_window(L)
        base = listen(fixed, ws, t_lo, t_hi)
        key = (L["word_idx"], L["char_idx"])
        letter_mode = "overlay" if L["char"] in args.overlay_letters.lower() else args.mode

        def rate(heard):
            """(уверенность целевой буквы, испорченные соседние, средняя просадка соседних)."""
            damaged = [m for m, c0 in base.items() if m != key and m not in exempt and c0 >= args.threshold
                       and heard.get(m, 0.0) < args.threshold]
            drop = np.mean([max(0.0, c0 - heard.get(m, 0.0)) for m, c0 in base.items() if m != key] or [0.0])
            return heard.get(key, 0.0), damaged, drop
        # время до вставки можно забирать только правее следующих (более ранних) проглоченных букв
        pending = [P["hi"] for P in todo[n_done:]]
        floor = int((max(pending) + 0.03) * sr) if pending else 0
        best = None
        tried = 0
        donor_info = []  # для файла проекта: окно сможет переключать доноров вручную
        for D in donors:
            seg, ctx_rms, donor_conf, source = donor_segment(L, D)
            center = (L["lo"] + L["hi"]) / 2
            ta, tb = max(0, int((center - 0.12) * sr)), min(len(audio), int((center + 0.12) * sr))
            # не громче окружения +2 дБ: согласные в пении тише гласных, громкая вставка выпирает
            gain = float(np.clip(rms(mono_full[ta:tb]) / max(ctx_rms, 1e-6), 0.25, 1.26)) * args.gain
            if D == "own":
                donor_info.append(dict(kind="own", label=source, path=str(own[L["char"]].resolve()),
                                       match_gain_db=20 * math.log10(gain)))
            else:
                donor_info.append(dict(kind="song", label=source, word=D["word"], conf=D["conf"],
                                       lo=D["lo"], start=D["start"], end=D["end"], hi=D["hi"],
                                       match_gain_db=20 * math.log10(gain)))
            variants = [(letter_mode, pos) for pos in (0.5, 0.8, 1.0)]
            if letter_mode == "insert":
                variants += [("replace", 0.8)]
            for mode, pos in variants:
                res = apply(fixed, zones, protected, L, seg * gain, mode, pos, floor)
                if res is None:
                    if args.verbose:
                        log(f"      {mode:7s} {pos:.1f} {source}: нет места")
                    continue
                tried += 1
                heard = listen(res[0], ws, t_lo, t_hi)
                if not heard:
                    continue
                tconf, damaged, drop = rate(heard)
                damage = len(damaged)
                score = tconf - 0.6 * damage - 0.5 * drop
                if mode != letter_mode:
                    score -= 0.05  # запасной режим выигрывает, только если он заметно лучше
                if args.verbose:
                    hurt = ", ".join(f"{words[m[0]][1]}:{words[m[0]][1][m[1]]} {base[m]:.2f}→{heard.get(m, 0.0):.2f}"
                                     for m in damaged)
                    log(f"      {mode:7s} {pos:.1f} {source}: буква {tconf:.2f}, испорчено {damage}"
                        + (f" ({hurt})" if hurt else "") + f", просадка {drop:.2f} → {score:.2f}")
                if best is None or score > best["score"]:
                    best = dict(score=score, tconf=tconf, damage=damage, res=res, source=source,
                                donor_conf=donor_conf, gain=gain, mode=mode, pos=pos, seg=seg,
                                donor_idx=len(donor_info) - 1,
                                args=(L, seg * gain, mode, pos, floor))
        log(f"  [{n_done}/{len(todo)}] {L['word']} — «{L['char']}»: проверено вариантов: {tried}"
            + (f", лучший: {best['tconf']:.2f}" if best else ""))
        # буква должна стать слышна заметно лучше, чем была: иначе это не исправление, а лишний звук
        # (сравниваем с оценкой по всей дорожке: в коротком окне модель иногда слышит букву лучше)
        need_conf = max(args.accept, min(L["conf"] + 0.2, 0.9)) if not L["forced"] else args.accept
        # делаем вставку тише на 3 дБ, если буква всё ещё чётко слышна (сильнее не убавляем: модель слышит
        # тихую букву лучше человека и Kits)
        if best and best["damage"] == 0 and best["tconf"] >= need_conf:
            for quieter in (0.7,):
                g = best["gain"] * quieter
                res = apply(fixed, zones, protected, L, best["seg"] * g, best["mode"], best["pos"], floor)
                heard = listen(res[0], ws, t_lo, t_hi) if res else {}
                if not heard:
                    break
                tconf, damaged, _ = rate(heard)
                if damaged or tconf < need_conf:
                    break
                best.update(res=res, gain=g, tconf=tconf, args=(L, best["seg"] * g, best["mode"], best["pos"], floor))
        pj = dict(letter=L["idx"], donors=donor_info, floor=floor, ws=ws, t_lo=t_lo, t_hi=t_hi,
                  donor=best["donor_idx"] if best else 0, mode=best["mode"] if best else letter_mode,
                  pos=best["pos"] if best else 0.8, gain_offset_db=(20 * math.log10(best["gain"])
                  - donor_info[best["donor_idx"]]["match_gain_db"]) if best else 0.0)
        if best and best["damage"] == 0 and best["tconf"] >= need_conf:
            fixed, zones, protected, note = best["res"]
            fixes.append(dict(L=L, status="вставлено", source=best["source"], donor_conf=best["donor_conf"],
                              note=note, gain_db=20 * math.log10(best["gain"]), after=best["tconf"],
                              args=best["args"], proj=pj))
        else:
            why = "вставка не прозвучала чётко" if best else "не нашлось места для вставки"
            if L["char"] not in own:
                why += f"; можно записать «{L['char']}» в папку --own-donors"
            fixes.append(dict(L=L, status=f"не исправлено ({why}) — проверьте вручную",
                              after=best["tconf"] if best else None, proj=pj))
    # Итоговая проверка: окно вокруг слова модель слышит иначе, чем всю дорожку. Слушаем дорожку целиком;
    # вставки, после которых ослабла буква в том же или соседнем слове, отменяем и собираем дорожку заново.
    applied = [F for F in fixes if F["status"] == "вставлено"]
    for check_round in range(5):
        if not applied:
            break
        log("  итоговая проверка всей дорожки…")
        conf2 = letter_confs(fixed)
        bad = {}
        fixed_k = {F["L"]["k"] for F in applied}
        for F in applied:
            F["after"] = conf2[F["L"]["k"]]
            if F["after"] < args.accept:
                bad[id(F)] = f"в контексте всей дорожки буква всё равно не слышна ({F['after']:.2f})"
        # ослабшая соседняя буква — виновата ближайшая к ней вставка (в том же или соседнем слове)
        for N in letters:
            if N["k"] in fixed_k or N["exempt"] or N["conf"] < args.threshold or conf2[N["k"]] >= args.threshold \
                    or N["conf"] - conf2[N["k"]] < 0.2:
                continue
            near = [F for F in applied if abs(F["L"]["word_idx"] - N["word_idx"]) <= 1 and id(F) not in bad]
            if near:
                F = min(near, key=lambda F: abs(F["L"]["start"] - N["start"]))
                bad[id(F)] = f"ослабла «{N['char']}» в «{N['word']}» {N['conf']:.2f}→{conf2[N['k']]:.2f}"
        if not bad:
            break
        if check_round == 4:  # больше не откатываем, только предупреждаем
            for F in applied:
                if id(F) in bad:
                    F["status"] = f"вставлено, но {bad[id(F)]} — проверьте на слух"
            break
        for F in applied:
            if id(F) in bad:
                log(f"  отменяю «{F['L']['char']}» в «{F['L']['word']}» ({F['L']['start']:.2f} с): "
                    f"{bad[id(F)]}")
                F["status"] = f"не исправлено ({bad[id(F)]}) — проверьте вручную"
                F["after"] = None
                F["note"] = ""
        applied = [F for F in applied if id(F) not in bad]
        fixed = replay(applied)
    if args.report_only:
        fixes = [dict(L=L, status="найдена") for L in weak]
    fixes.sort(key=lambda F: F["L"]["start"])

    log("5/5 Сохраняю результаты…")
    peak = float(np.abs(fixed).max())
    if peak > 0.999:
        fixed *= 0.999 / peak
        log(f"  ! Пик превысил 0 дБ — вся дорожка опущена на {20*math.log10(peak/0.999):.1f} дБ")

    letters_csv = out_dir / f"{stem}_letters.csv"
    with open(letters_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["строка", "слово", "буква", "№ в слове", "начало, с", "конец, с", "уверенность", "статус"])
        for L in letters:
            w.writerow([L["line"], L["word"], L["char"], L["char_idx"] + 1, f"{L['start']:.3f}", f"{L['end']:.3f}",
                        f"{L['conf']:.2f}", "проглочена" if L["weak"] else "ок"])

    fixes_csv = out_dir / f"{stem}_fixes.csv"
    with open(fixes_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["время, с", "строка", "слово", "буква", "уверенность до", "уверенность после", "результат",
                    "донор", "уверенность донора", "усиление, дБ", "как вставлено"])
        for F in fixes:
            L = F["L"]
            after = F.get("after")
            w.writerow([f"{L['start']:.3f}", L["line"], L["word"], L["char"], f"{L['conf']:.2f}",
                        f"{after:.2f}" if after is not None else "", F["status"],
                        F.get("source", ""), f"{F['donor_conf']:.2f}" if F.get("donor_conf") else "",
                        f"{F['gain_db']:.1f}" if "gain_db" in F else "", F.get("note", "")])

    def mark_word(L):
        w = L["word"]
        i = L["char_idx"]
        return w[:i] + w[i].upper() + w[i + 1:]

    aud = out_dir / f"{stem}_markers_audacity.txt"
    with open(aud, "w", encoding="utf-8") as f:
        for F in fixes:
            L = F["L"]
            f.write(f"{L['start']:.3f}\t{L['end']:.3f}\t{mark_word(L)} — {L['char']} ({F['status']})\n")
    rpr = out_dir / f"{stem}_markers_reaper.csv"
    with open(rpr, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["#", "Name", "Start", "End", "Length"])
        for i, F in enumerate(fixes, 1):
            L = F["L"]
            w.writerow([f"M{i}", f"{mark_word(L)} - {L['char']} ({F['status']})", f"{L['start']:.3f}", "", ""])

    outputs = [letters_csv, fixes_csv, aud, rpr]
    if not args.report_only:
        # проект для окна ручной правки (bukvi_app.py): всё, чтобы пересобрать дорожку с другими настройками
        import json
        keys = ("k", "idx", "word_idx", "char_idx", "char", "line", "word", "start", "end", "lo", "hi", "conf", "weak")
        proj_fixes = []
        for F in fixes:
            if "proj" not in F:
                continue
            pj = dict(F["proj"], status=F["status"], enabled=F["status"].startswith("вставлено"),
                      before=F["L"]["conf"], after=F.get("after"), note=F.get("note", ""))
            proj_fixes.append(pj)
        proj = dict(version=1, vocal=str(vocal_path.resolve()), lyrics=str(Path(args.lyrics).resolve()),
                    model=args.model, sr=sr, xfade_ms=args.xfade_ms, duck_db=args.duck_db, pad=pad,
                    max_len_ms=args.max_len_ms, silence=silence, threshold=args.threshold,
                    zones=[list(z) for z in zones0], words=words,
                    letters=[{k_: L[k_] for k_ in keys} for L in letters], fixes=proj_fixes)
        proj_path = out_dir / f"{stem}_project.json"
        proj_path.write_text(json.dumps(proj, ensure_ascii=False, indent=1), encoding="utf-8")
        outputs.insert(0, proj_path)
    if not args.report_only:
        out_wav = out_dir / f"{stem}_fixed.wav"
        # из MP3 и других сжатых форматов — в WAV 24 бит: повторное сжатие перед Kits только навредит
        subtype = in_subtype if in_format == "WAV" else "PCM_24"
        sf.write(str(out_wav), fixed, sr, subtype=subtype)
        outputs.insert(0, out_wav)
        if args.mp3 or in_format == "MP3":
            out_mp3 = out_dir / f"{stem}_fixed.mp3"
            try:
                sf.write(str(out_mp3), fixed, sr, format="MP3", subtype="MPEG_LAYER_III", compression_level=0.0)
                outputs.insert(1, out_mp3)
            except Exception as e:  # noqa: BLE001
                log(f"  ! MP3 не сохранён ({e}); используйте WAV")

    done = sum(1 for F in fixes if F["status"].startswith("вставлено"))
    log("")
    log(f"Готово за {time.time()-t0:.0f} с. Букв в тексте: {len(letters)}, проглочено: {len(weak)}, вставлено: {done}.")
    for F in fixes:
        L = F["L"]
        extra = f" ← {F['source']}, {F['gain_db']:+.1f} дБ" if F["status"].startswith("вставлено") else ""
        if F.get("note"):
            extra += f"; {F['note']}"
        after = f" → {F['after']:.2f}" if F.get("after") is not None else ""
        log(f"  {L['start']:7.2f} с  {mark_word(L):<14} {L['char']}  ({L['conf']:.2f}{after})  {F['status']}{extra}")
    log("")
    for p in outputs:
        log(f"  → {p}")


if __name__ == "__main__":
    main()

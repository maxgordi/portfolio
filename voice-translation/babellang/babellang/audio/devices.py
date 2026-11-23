from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class DeviceResolution:
    """Result of resolving saved devices against current hardware."""
    changed: bool = False           # indices were updated — caller should save settings
    errors: list[str] = None        # devices that could not be found (blocks start)
    warnings: list[str] = None      # suspicious routing that likely causes echo

    def __post_init__(self) -> None:
        self.errors = self.errors or []
        self.warnings = self.warnings or []

    @property
    def ok(self) -> bool:
        return not self.errors


def rescan() -> None:
    """Force PortAudio to re-enumerate devices (it caches the list at init time,
    so devices plugged in after launch — Bluetooth headphones, iPhone mic — are invisible
    without this)."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:
        logger.warning("PortAudio rescan failed: %s", exc)


def _find_by_name(name: str, need_input: bool, need_output: bool) -> Optional[int]:
    """Exact name match first, then substring match."""
    devices = sd.query_devices()

    def _fits(d: dict) -> bool:
        if need_input and d["max_input_channels"] == 0:
            return False
        if need_output and d["max_output_channels"] == 0:
            return False
        return True

    for i, d in enumerate(devices):
        if d["name"] == name and _fits(d):
            return i
    low = name.lower()
    for i, d in enumerate(devices):
        if low in d["name"].lower() and _fits(d):
            return i
    return None


def resolve_devices(settings) -> DeviceResolution:
    """
    Re-resolve every saved device by NAME against the current device list.

    macOS renumbers audio devices whenever Bluetooth headphones / iPhone mics
    appear or disappear, so saved indices go stale. The saved name is the source
    of truth; the index is just a cache. Mutates settings in place.
    """
    rescan()
    res = DeviceResolution()

    specs = (
        ("input_device",    "input_device_name",    True,  False, "Микрофон",                 True),
        ("output_device",   "output_device_name",   False, True,  "Выход собеседнику",        True),
        ("incoming_device", "incoming_device_name", True,  False, "Захват собеседника",       False),
        ("speaker_device",  "speaker_device_name",  False, True,  "Динамики (ты слышишь)",    False),
    )

    for attr, name_attr, need_in, need_out, label, required in specs:
        idx: Optional[int] = getattr(settings, attr)
        name: str = getattr(settings, name_attr) or ""

        if idx is None and not name:
            if required:
                res.errors.append(f"{label}: устройство не выбрано")
            continue

        # 1) Saved name is the source of truth
        if name:
            found = _find_by_name(name, need_in, need_out)
            if found is not None:
                if found != idx:
                    logger.info("Device '%s' (%s) moved: index %s → %s", name, attr, idx, found)
                    setattr(settings, attr, found)
                    res.changed = True
                continue

        # 2) Fall back to the saved index if it still points at a valid device
        if idx is not None:
            try:
                d = sd.query_devices(idx)
                dir_ok = (not need_in or d["max_input_channels"] > 0) and \
                         (not need_out or d["max_output_channels"] > 0)
                if dir_ok:
                    if name and name.lower() not in d["name"].lower():
                        res.warnings.append(
                            f"{label}: '{name}' не найдено, использую '{d['name']}' (индекс {idx})"
                        )
                    setattr(settings, name_attr, d["name"])
                    res.changed = True
                    continue
            except Exception:
                pass

        msg = f"{label}: устройство '{name or idx}' не найдено"
        if required:
            res.errors.append(msg)
        else:
            res.warnings.append(msg + " — канал отключён")
            setattr(settings, attr, None)
            res.changed = True

    _check_routing(settings, res)
    return res


def _check_routing(settings, res: DeviceResolution) -> None:
    """Detect device combinations that create feedback loops."""
    out_idx = settings.output_device
    spk_idx = settings.speaker_device
    inc_idx = settings.incoming_device

    if out_idx is not None and spk_idx is not None and out_idx == spk_idx:
        res.warnings.append(
            "⚠ Выход собеседнику и твои динамики — одно устройство: "
            "ты будешь слышать свой переведённый голос"
        )
    if out_idx is not None and inc_idx is not None:
        try:
            out_name = sd.query_devices(out_idx)["name"].lower()
            inc_name = sd.query_devices(inc_idx)["name"].lower()
            if out_name == inc_name:
                res.warnings.append(
                    "⚠ Выход собеседнику и захват собеседника — одно устройство: "
                    "программа будет переводить собственный перевод (петля)"
                )
        except Exception:
            pass
    if spk_idx is not None and settings.input_device is not None:
        try:
            spk_name = sd.query_devices(spk_idx)["name"].lower()
            mic_name = sd.query_devices(settings.input_device)["name"].lower()
            if "динамики" in spk_name and "macbook" in spk_name and "macbook" in mic_name:
                res.warnings.append(
                    "⚠ Перевод играет через колонки MacBook, а микрофон MacBook его слышит — "
                    "выбери наушники в «Динамики (ты слышишь)»"
                )
        except Exception:
            pass

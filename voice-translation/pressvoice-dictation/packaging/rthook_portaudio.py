# PyInstaller runtime hook (Linux): sounddevice locates PortAudio with
# ctypes.util.find_library, which only searches system paths. Point it at
# the copy bundled into the app, so the AppImage works without
# libportaudio2 installed.
import ctypes.util
import glob
import os
import sys

_find_library = ctypes.util.find_library


def _find_bundled(name):
    if name == "portaudio":
        hits = glob.glob(os.path.join(getattr(sys, "_MEIPASS", ""), "libportaudio.so*"))
        if hits:
            return hits[0]
    return _find_library(name)


ctypes.util.find_library = _find_bundled

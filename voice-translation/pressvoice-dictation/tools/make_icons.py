"""Render assets/pressvoice.png, .ico (Windows) and .icns (macOS) from the same
drawing the tray uses (qtapp/icons.py). Run after changing the icon:
    python tools/make_icons.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from qtapp.icons import make_icon  # noqa: E402

SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
ICNS_SIZE = 1024  # Pillow derives every macOS icon size from this one


def main():
    app = QGuiApplication([])  # noqa: F841
    out = os.path.join(ROOT, "assets")
    os.makedirs(out, exist_ok=True)
    icon = make_icon("idle")
    tmp = []
    for s in SIZES:
        path = os.path.join(out, f".icon-{s}.png")
        icon.pixmap(QSize(s, s)).toImage().save(path)
        tmp.append(path)
    images = [Image.open(p).convert("RGBA") for p in tmp]
    images[-1].save(os.path.join(out, "pressvoice.png"))
    images[-1].save(os.path.join(out, "pressvoice.ico"),
                    sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    big = os.path.join(out, ".icon-big.png")
    icon.pixmap(QSize(ICNS_SIZE, ICNS_SIZE)).toImage().save(big)
    Image.open(big).convert("RGBA").save(os.path.join(out, "pressvoice.icns"))
    for p in tmp + [big]:
        os.remove(p)
    print("wrote assets/pressvoice.png, .ico and .icns")


if __name__ == "__main__":
    main()

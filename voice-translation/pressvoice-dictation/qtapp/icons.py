"""Tray and window icons, drawn at runtime so every state stays crisp at
any DPI without shipping image files."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

COLORS = {
    "idle": "#2563EB",
    "recording": "#DC2626",
    "busy": "#D97706",
    "style": "#7C3AED",
}


def _draw_mic(p: QPainter, s: float) -> None:
    white = QColor("white")
    # Capsule
    body = QRectF(s * 0.39, s * 0.20, s * 0.22, s * 0.38)
    p.setPen(Qt.NoPen)
    p.setBrush(white)
    p.drawRoundedRect(body, s * 0.11, s * 0.11)
    # Holder arc, stem and base
    pen = QPen(white, s * 0.055, Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    arc = QPainterPath()
    arc.moveTo(s * 0.30, s * 0.46)
    arc.cubicTo(s * 0.30, s * 0.72, s * 0.70, s * 0.72, s * 0.70, s * 0.46)
    p.drawPath(arc)
    p.drawLine(QPointF(s * 0.50, s * 0.66), QPointF(s * 0.50, s * 0.78))
    p.drawLine(QPointF(s * 0.40, s * 0.78), QPointF(s * 0.60, s * 0.78))


def _draw_dots(p: QPainter, s: float) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("white"))
    r = s * 0.065
    for x in (0.32, 0.50, 0.68):
        p.drawEllipse(QPointF(s * x, s * 0.5), r, r)


def make_icon(state: str = "idle") -> QIcon:
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256, 1024):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(COLORS.get(state, COLORS["idle"])))
        p.drawEllipse(QRectF(0.5, 0.5, size - 1, size - 1))
        if state in ("busy", "style"):
            _draw_dots(p, size)
        else:
            _draw_mic(p, size)
        p.end()
        icon.addPixmap(pm)
    return icon

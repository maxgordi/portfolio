"""PressVoice for Windows and Linux. (The macOS app starts from main.py.)"""
import argparse
import getpass
import os
import sys

from PySide6.QtCore import QLockFile
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

import config as cfg_module
from qtapp import platform_utils
from qtapp.icons import make_icon
from qtapp.logs import install_crash_logging, log
from version import APP_NAME, __version__

_SERVER_NAME = f"PressVoice-{getpass.getuser()}"


def _parse_args(argv):
    p = argparse.ArgumentParser(prog=APP_NAME)
    p.add_argument("--selftest", metavar="REPORT_JSON",
                   help="check the installation, write a JSON report, exit")
    p.add_argument("--restarted", action="store_true",
                   help="relaunched by the app itself after an audio hang")
    p.add_argument("--version", action="version", version=__version__)
    return p.parse_args(argv)


def _signal_running_instance() -> bool:
    """Ask an already running PressVoice to open its Settings window."""
    sock = QLocalSocket()
    sock.connectToServer(_SERVER_NAME)
    if not sock.waitForConnected(1000):
        return False
    sock.write(b"show-settings\n")
    sock.waitForBytesWritten(1000)
    sock.disconnectFromServer()
    return True


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    install_crash_logging()
    if platform_utils.IS_WINDOWS:
        from qtapp import winapi
        winapi.set_app_user_model_id(platform_utils.AUMID)

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)  # lives in the tray
    app.setWindowIcon(make_icon("idle"))

    if args.selftest:
        from qtapp.selftest import run_selftest
        return run_selftest(app, args.selftest)

    os.makedirs(cfg_module.DATA_DIR, exist_ok=True)
    lock = QLockFile(os.path.join(cfg_module.DATA_DIR, "pressvoice.lock"))
    lock.setStaleLockTime(0)  # held for the app's whole life; staleness = dead PID
    if not lock.tryLock(5000 if args.restarted else 200):
        if _signal_running_instance():
            log("second launch: asked the running instance to show Settings")
        return 0

    from qtapp.app import PressVoiceQt
    controller = PressVoiceQt()

    QLocalServer.removeServer(_SERVER_NAME)  # left over after a crash
    server = QLocalServer()

    def on_connection():
        sock = server.nextPendingConnection()
        if sock is None:
            return
        sock.waitForReadyRead(500)
        if b"show-settings" in bytes(sock.readAll()):
            controller.open_settings()
        sock.disconnectFromServer()

    server.newConnection.connect(on_connection)
    if not server.listen(_SERVER_NAME):
        log(f"instance server: {server.errorString()}")

    controller.start()
    log(f"PressVoice {__version__} running (config: {cfg_module.CONFIG_DIR})")
    code = app.exec()
    lock.unlock()
    return code


if __name__ == "__main__":
    sys.exit(main())

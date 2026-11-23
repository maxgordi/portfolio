import sys


def main() -> int:
    if "--selftest" in sys.argv:
        from mac_selftest import run_selftest
        i = sys.argv.index("--selftest")
        report = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        return run_selftest(report)

    import mac_support
    if not mac_support.acquire_single_instance():
        return 0  # already running; a clean exit keeps launchd from retrying

    from app import PressVoiceApp
    PressVoiceApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

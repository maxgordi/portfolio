# babellang

Real-time two-way voice translation for video calls on macOS (Python MVP).

* **Outgoing:** microphone → VAD → speech-to-text → GPT-4o-mini translation → ElevenLabs TTS → virtual mic (BlackHole 2ch). The other side hears you in their language.
* **Incoming:** call audio (BlackHole 16ch) → the same pipeline → your headphones.
* Web UI on `localhost:8765` with live subtitles, volume sliders and auto-ducking of the original voice while the translation plays.
* API keys live in the macOS Keychain (`babellang --setup`), never in files.

```bash
pip install -e ".[dev]"
babellang --setup   # devices + keys
babellang           # opens the web UI
pytest
```

Requires macOS, [BlackHole](https://github.com/ExistentialAudio/BlackHole) 2ch/16ch, and OpenAI, ElevenLabs (and optionally Deepgram) API keys.

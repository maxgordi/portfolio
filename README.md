# Projects

AI assistants, real-time voice translation, audio tools and automation.
Each folder is self-contained, with its own README and a `.env.example` where configuration is needed.
No credentials or personal data are stored in this repository: secrets are read from environment
variables or the OS keychain, and sample data is fictional.

## AI assistants

| Project | Stack | What it does |
|---|---|---|
| [rag-knowledge-assistant](ai-assistants/rag-knowledge-assistant) | Python, Ollama, Qdrant, FastAPI, Docker | Self-hosted RAG over a private knowledge base: chunking, hash-based re-indexing, similarity threshold, strict JSON answers, fallback model, request queue. *Demo / WIP* |
| [job-agent](ai-assistants/job-agent) | Python, Playwright, Telegram, LLM | Collects AI vacancies from Ukrainian job boards, scores them against a CV with an LLM, writes cover letters, applies automatically, reports to Telegram. Ships with a fictional CV |

## Voice translation

| Project | Stack | What it does |
|---|---|---|
| [realtime-translator](voice-translation/realtime-translator) | Rust, egui, Core Audio, Soniox | Push-to-talk interpreter for video calls on macOS: you speak Ukrainian/Russian, the other side hears English through a virtual mic; their speech is translated back into your headphones |
| [realtime-translator-gemini](voice-translation/realtime-translator-gemini) | Rust, Gemini Live | The same interpreter on the Gemini Live API (earlier version) |
| [babellang](voice-translation/babellang) | Python, OpenAI, ElevenLabs, BlackHole | Two-way voice translation for calls with a web UI and live subtitles |
| [pressvoice-dictation](voice-translation/pressvoice-dictation) | Python, Qt, OpenAI | Hold a key, speak, release: the text is typed where the cursor is. Windows, Linux, macOS installers |
| [offline-translator](voice-translation/offline-translator) | Python, Whisper, Vosk, Piper | Speech translation fully offline on a Mac |

## Audio tools

| Project | Stack | What it does |
|---|---|---|
| [voxmatch](audio-tools/voxmatch) | Python, WORLD vocoder, DTW, Qt | Matches vocal doubles to a guide take in timing and pitch (like VocAlign / Melodyne) |
| [voxmatch-lite](audio-tools/voxmatch-lite) | Python, Qt | Lighter version without resynthesis: picks the best phrases and builds a comp |
| [vocal-sync](audio-tools/vocal-sync) | Python, DTW | Timing-only alignment of a vocal track to a guide, with a GUI and quality benchmarks |
| [diction-fixer](audio-tools/diction-fixer) | Python, PyTorch, transformers | Finds and repairs swallowed consonants in AI vocal stems |
| [voice-dataset-builder](audio-tools/voice-dataset-builder) | Python | Builds a balanced voice dataset for voice-model training from raw singing recordings |
| [mix-assistant](audio-tools/mix-assistant) | Python, VST3, Claude skill | Measurement-driven vocal + acoustic guitar mixing for Cubase |

## Home automation

| Project | Stack | What it does |
|---|---|---|
| [lg-tv-tools](home-automation/lg-tv-tools) | Python, Chrome extension, Bash | Send YouTube to an LG webOS TV, terminal remote, app cleanup, lossless Mac audio to the TV |

## Automation

| Project | Stack | What it does |
|---|---|---|
| [tender-monitoring-system](automation/tender-monitoring-system) | Python | Monitors tender platforms for keywords, detects duplicates, sends Telegram notifications |
| [wordpress-telegram-bot](automation/wordpress-telegram-bot) | Python, Telegram Bot API, WordPress REST API | Creates WordPress posts with images from a conversational Telegram interface |
| [crypto-paradex-bot](automation/crypto-paradex-bot) | Node.js | StarkNet-based authentication and JWT handling for the Paradex API (testnet) |
| [web-scrapers](automation/web-scrapers) | Python, Selenium | Documentation scraper that watches pages and saves their main content |
| [content-extractor](automation/content-extractor) | Python, Selenium | Standalone script that extracts and logs the main content of documentation pages |
| [powershell-tools](automation/powershell-tools) | PowerShell | SQL Server backup automation with compression, retention, logging and email alerts |

## Running a project

1. Open the project folder and read its README.
2. Copy `.env.example` to `.env` (if present) and fill in your own values.
3. Install dependencies and run as described in that README.

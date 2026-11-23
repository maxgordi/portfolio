# LG TV tools (webOS)

Small utilities for an older LG webOS TV on the home network.

| Folder | What it does |
|---|---|
| `tvcast/` | Chrome extension + local helper: a button on YouTube sends the current video to the TV app at the same second and pauses it on the Mac. Also a terminal remote and screen mirroring launcher. Wake-on-LAN if the TV is off. |
| `lg-remote/` | Python CLI over the SSAP websocket API: find the TV, pair, list/close/remove apps, read settings. |
| `mac-tv-audio/` | Streams all Mac audio to the TV losslessly: AirPlay → AirConnect → DLNA (FLAC). |

Set your TV address with the `TV_IP` environment variable (the code default `192.168.1.50` is a placeholder). The pairing key is stored locally in `~/.lgtv_client_key` / `lgtv_config.json` and is never committed.

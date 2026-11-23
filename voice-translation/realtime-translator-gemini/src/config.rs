//! Where PressVoice keeps its files, the Gemini key and the window's choices.
//!
//! Shared with the Python PressVoice: the key is read from the same
//! `config.json`, and the window's choices live in the same
//! `live_window.json`, so either version opens with the last settings.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::journal;

pub fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("PRESSVOICE_CONFIG_DIR") {
        return PathBuf::from(dir); // tests and portable setups
    }
    home().join(".config/pressvoice")
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"))
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

/// The Google AI Studio key (`gemini_api_key` in config.json), if set.
pub fn gemini_api_key() -> Option<String> {
    let text = fs::read_to_string(config_path()).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let key = value.get("gemini_api_key")?.as_str()?.trim();
    (!key.is_empty()).then(|| key.to_owned())
}

/// The window's choices, restored at the next start. Devices are kept by
/// name: their numbers change when one is plugged in or out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowSettings {
    /// The other side's language: what my voice is translated into.
    pub direction: String,
    /// My language: what the other side is translated into.
    pub my_lang: String,
    /// Microphone name; `None` = the system default.
    pub mic: Option<String>,
    /// Headphones name; `None` = the system default.
    pub out: Option<String>,
    pub key: String,
    /// Semitones the voice is lowered by before the model hears it.
    pub deeper: i32,
    pub echo: bool,
    /// My own voice in the call, % (0 = translation only).
    pub voice_level: i32,
    /// Turn my voice down while the translation sounds.
    pub auto_duck: bool,
    /// …to this % of its volume.
    pub duck_to: i32,
    /// How soon it comes back: "fast", "medium" or "smooth".
    pub duck_speed: String,

    /// Translate the other side of the call (off = only my voice uses the model).
    pub his_on: bool,
    /// The app the call is in, by bundle ID; `None` = all apps.
    pub his_source: Option<String>,
    /// Where I hear the other side; `None` = the system default.
    pub headphones: Option<String>,
    /// The other side's original voice in my headphones, %.
    pub his_level: i32,
    pub his_auto_duck: bool,
    pub his_duck_to: i32,
    pub his_duck_speed: String,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            direction: "en".into(),
            my_lang: "ru".into(),
            mic: None,
            out: None,
            key: "alt_r".into(),
            deeper: 3,
            echo: false,
            voice_level: 0,
            auto_duck: true,
            duck_to: 30,
            duck_speed: "medium".into(),
            his_on: false,
            his_source: None,
            headphones: None,
            his_level: 20,
            his_auto_duck: true,
            his_duck_to: 30,
            his_duck_speed: "medium".into(),
        }
    }
}

fn settings_path() -> PathBuf {
    config_dir().join("live_window.json")
}

impl WindowSettings {
    /// Never fails: a missing or damaged file gives the defaults.
    pub fn load() -> Self {
        fs::read_to_string(settings_path())
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// Written to a temporary file first, so an interrupted save can't
    /// leave a half-written file behind.
    pub fn save(&self) {
        let path = settings_path();
        let result = (|| -> std::io::Result<()> {
            fs::create_dir_all(config_dir())?;
            let tmp = path.with_extension("json.tmp");
            fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
            fs::rename(&tmp, &path)
        })();
        if let Err(e) = result {
            journal::log(&format!("window: settings not saved: {e}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_python_versions_file() {
        let text = r#"{"direction": "nl", "mic": "Scarlett 2i2 USB", "out": null,
                       "key": "alt_r", "deeper": 3, "echo": false}"#;
        let s: WindowSettings = serde_json::from_str(text).unwrap();
        assert_eq!(s.direction, "nl");
        assert_eq!(s.mic.as_deref(), Some("Scarlett 2i2 USB"));
        assert_eq!(s.out, None);
    }

    #[test]
    fn missing_fields_take_defaults() {
        let s: WindowSettings = serde_json::from_str(r#"{"direction": "uk"}"#).unwrap();
        assert_eq!(s.direction, "uk");
        assert_eq!(s.key, "alt_r");
        assert_eq!(s.deeper, 3);
    }
}

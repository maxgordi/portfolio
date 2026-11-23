//! Soniox's voices: the built-in ones (tts-rt-v2, as listed by
//! api.soniox.com/v1/tts-models on 2026-10-10, kept in
//! assets/soniox_voices.json) and the account's own cloned voices, read from
//! api.soniox.com/v1/voices at start. Every voice — a clone too — speaks every
//! language. Clones come first, then a few voices that suit Russian and
//! Ukrainian speakers.

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::Deserialize;

const CLONES_URL: &str = "https://api.soniox.com/v1/voices";

#[derive(Clone, Debug, Deserialize)]
pub struct Voice {
    pub id: String,
    #[serde(default)]
    pub gender: String,
    #[serde(default)]
    pub description: String,
    /// The name it was given (cloned voices).
    #[serde(default)]
    pub name: Option<String>,
    /// One of the account's own cloned voices.
    #[serde(default)]
    pub cloned: bool,
}

/// Offered first after the clones: the default, then native Russian and Ukrainian voices.
const FIRST: [&str; 7] = ["Adrian", "Andrei", "Kirill", "Vadim", "Taras", "Bohdan", "Daniel"];

static BUILT_IN: LazyLock<Vec<Voice>> = LazyLock::new(|| {
    let mut voices: Vec<Voice> = serde_json::from_str(include_str!("../assets/soniox_voices.json")).expect("valid voices list");
    let rank = |v: &Voice| FIRST.iter().position(|f| *f == v.id).unwrap_or(FIRST.len());
    voices.sort_by(|a, b| {
        (rank(a), a.gender != "male", a.id.as_str()).cmp(&(rank(b), b.gender != "male", b.id.as_str()))
    });
    voices
});

/// Clones, then the built-in voices (rebuilt when the clones are read).
static ALL: LazyLock<RwLock<Arc<Vec<Voice>>>> = LazyLock::new(|| RwLock::new(Arc::new(BUILT_IN.clone())));

/// Clones first, then recommended voices, men, women, each by name.
pub fn all() -> Arc<Vec<Voice>> {
    ALL.read().clone()
}

pub fn exists(id: &str) -> bool {
    ALL.read().iter().any(|v| v.id == id)
}

/// A cloned voice's ID is a UUID (built-in voices have names): kept even
/// when the list of clones couldn't be read this time.
pub fn looks_cloned(id: &str) -> bool {
    id.len() == 36 && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

#[derive(Deserialize)]
struct ClonedList {
    voices: Vec<ClonedVoice>,
}

#[derive(Deserialize)]
struct ClonedVoice {
    id: String,
    name: Option<String>,
    #[serde(default)]
    models: Vec<ClonedModel>,
}

#[derive(Deserialize)]
struct ClonedModel {
    model: String,
    status: String,
}

/// Reads the account's cloned voices (those ready for the voice model) and
/// puts them at the top of the list. Blocks up to a few seconds.
pub fn load_cloned(api_key: &str, model: &str) -> Result<usize> {
    let mut response = ureq::get(CLONES_URL)
        .header("Authorization", format!("Bearer {api_key}"))
        .config()
        .timeout_global(Some(Duration::from_secs(4)))
        .build()
        .call()
        .context("список своих голосов Soniox не получен")?;
    let text = response.body_mut().read_to_string()?;
    let list: ClonedList = serde_json::from_str(&text).context("непонятный ответ Soniox о голосах")?;
    let clones: Vec<Voice> = list
        .voices
        .into_iter()
        .filter(|v| v.models.iter().any(|m| m.model == model && m.status == "ready"))
        .map(|v| Voice { id: v.id, gender: String::new(), description: String::new(), name: v.name, cloned: true })
        .collect();
    let count = clones.len();
    let mut all = clones;
    all.extend(BUILT_IN.iter().cloned());
    *ALL.write() = Arc::new(all);
    Ok(count)
}

/// "★ Alex — ваш клон" / "Adrian — муж., A deep, focused male voice…"
pub fn label(voice: &Voice) -> String {
    if voice.cloned {
        return format!("★ {} — ваш клон", voice.name.as_deref().unwrap_or(&voice.id));
    }
    let gender = if voice.gender == "male" { "муж." } else { "жен." };
    let mut description: String = voice.description.chars().take(60).collect();
    if voice.description.chars().count() > 60 {
        description.push('…');
    }
    format!("{} — {gender}, {description}", voice.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_starts_with_the_default() {
        assert!(all().len() >= 100);
        assert_eq!(BUILT_IN[0].id, crate::soniox::DEFAULT_VOICE);
        assert!(exists("Andrei") && !exists("Nobody"));
    }

    #[test]
    fn clone_ids_are_recognised() {
        assert!(looks_cloned("2ff3699c-c143-4418-be00-407fa3c4ad56"));
        assert!(!looks_cloned("Adrian"));
    }

    /// Needs the network and the key: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn the_accounts_clones_are_found() {
        let key = crate::config::soniox_api_key().expect("soniox_api_key");
        let n = load_cloned(&key, crate::soniox::TTS_MODEL).expect("list of clones");
        let list = all();
        for v in list.iter().take(n) {
            println!("{} ({})", label(v), v.id);
        }
        assert!(n >= 1 && list[0].cloned);
    }

    #[test]
    fn clones_are_labelled_as_such() {
        let v = Voice { id: "x".into(), gender: String::new(), description: String::new(), name: Some("Alex".into()), cloned: true };
        assert_eq!(label(&v), "★ Alex — ваш клон");
    }
}

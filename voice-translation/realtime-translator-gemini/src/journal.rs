//! A few lines per key press in ~/Library/Logs/PressVoice/live.log — press,
//! release, errors, what was said and translated — so a session can be
//! checked afterwards. Written on key events, never on the audio path.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use parking_lot::Mutex;

const MAX_BYTES: u64 = 1_000_000;

static LOCK: Mutex<()> = Mutex::new(());

pub fn log_path() -> PathBuf {
    crate::config::home().join("Library/Logs/PressVoice/live.log")
}

pub fn log(message: &str) {
    let line = format!("[{}] {message}\n", chrono::Local::now().format("%H:%M:%S%.3f"));
    let path = log_path();
    let _guard = LOCK.lock();
    let _ = (|| -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        if fs::metadata(&path).map(|m| m.len() > MAX_BYTES).unwrap_or(false) {
            fs::rename(&path, path.with_extension("log.1"))?;
        }
        OpenOptions::new().create(true).append(true).open(&path)?.write_all(line.as_bytes())
    })();
}

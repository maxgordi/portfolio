//! Only one PressVoice Live at a time: every copy listens to the same key,
//! and two of them would each translate every press.
//!
//! Each running copy records its PID in a file (the same one the Python
//! version uses). A new copy stops exactly that process, and only after
//! checking it really is one of ours — never by matching command lines,
//! which would also hit an editor or a shell that merely mentions the name.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use crate::config;

/// The Python version's scripts (run by a `python…` interpreter).
const PYTHON_SCRIPTS: [&str; 3] = ["live_console.py", "interpreter_console.py", "live_gui.py"];
/// The Rust windows: this one and its Soniox twin.
const OUR_PROGRAMS: [&str; 2] = ["pressvoice-live", "pressvoice-soniox"];

fn alive(pid: i32) -> bool {
    // Signal 0 checks the process exists without touching it.
    unsafe { libc::kill(pid, 0) == 0 }
}

fn file_name(path: &str) -> &str {
    Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("")
}

/// The process is this program, or Python running one of our scripts.
fn is_ours(pid: i32) -> bool {
    let Ok(out) = Command::new("ps").args(["-p", &pid.to_string(), "-o", "args="]).output() else {
        return false;
    };
    let args = String::from_utf8_lossy(&out.stdout);
    let mut words = args.split_whitespace();
    let program = file_name(words.next().unwrap_or("")).to_lowercase();
    let script = file_name(words.next().unwrap_or(""));
    let me = std::env::current_exe().ok();
    let my_name = me.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("");
    // Either PressVoice window (Gemini or Soniox) — they listen to the same key.
    program == my_name
        || OUR_PROGRAMS.contains(&program.as_str())
        || (program.starts_with("python") && PYTHON_SCRIPTS.contains(&script))
}

/// Stop the copy recorded in the PID file (if it is still ours and alive),
/// then record this process. True if one was stopped.
pub fn take_over() -> bool {
    let pid_file = config::config_dir().join("pressvoice-live.pid");
    let old: i32 = fs::read_to_string(&pid_file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let mut stopped = false;
    if old > 0 && old != std::process::id() as i32 && alive(old) && is_ours(old) {
        unsafe { libc::kill(old, libc::SIGTERM) };
        let deadline = Instant::now() + Duration::from_secs(3);
        while alive(old) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(100));
        }
        if alive(old) {
            unsafe { libc::kill(old, libc::SIGKILL) };
        }
        stopped = true;
    }
    let _ = fs::create_dir_all(config::config_dir());
    let _ = fs::write(&pid_file, std::process::id().to_string());
    stopped
}

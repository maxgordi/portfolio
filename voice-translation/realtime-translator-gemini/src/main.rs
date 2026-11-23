//! PressVoice Live — the interpreter in a small window (macOS).
//!
//! Pick the language to translate into (the spoken one is detected), the
//! microphone and the headphones; hold Right ⌥ and speak. The translation
//! plays in the headphones while you talk.
//!
//! `--simulate in.wav [--to xx] [--save out.wav]` translates a recording
//! instead (no window, microphone or key) — for checking the Gemini side.

mod app_audio;
mod audio;
mod config;
mod core_audio;
mod dsp;
mod echo;
mod fonts;
mod gemini;
mod hotkey;
mod incoming;
mod instances;
mod journal;
mod languages;
mod simulate;
mod translator;
mod ui;
mod usage;
mod virtual_mic;
mod voice_mix;

use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(api_key) = config::gemini_api_key() else {
        eprintln!("Нет ключа Google AI Studio в {} (поле gemini_api_key).", config::config_path().display());
        return ExitCode::FAILURE;
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--simulate") {
        return simulate::run(&api_key, &args[1..]);
    }

    if instances::take_over() {
        println!("Остановил предыдущий PressVoice Live.");
    }
    if let Err(e) = ui::run(api_key) {
        eprintln!("Окно не открылось: {e}");
        return ExitCode::FAILURE;
    }
    // Background threads (a translator still closing) must not keep the
    // process alive once the window is gone.
    std::process::exit(0)
}

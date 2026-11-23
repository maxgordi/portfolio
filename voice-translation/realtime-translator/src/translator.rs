//! One running interpreter: microphone → echo canceller (optional) → Soniox
//! (recognition with translation) → its voice, or text only → headphones or
//! the call, push-to-talk.
//!
//! The microphone, the output and the Soniox sessions are ready before the
//! key is pressed, so audio flows the moment it goes down.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::thread;
use std::time::Instant;

use anyhow::Result;
use parking_lot::Mutex;

use crate::audio::{ChunkHandler, Microphone, Output, Player, RawHandler, VOICE_RATE};
use crate::dsp::{self};
use crate::echo::EchoCanceller;
use crate::soniox::{self, Interpreter, Listener, Setup, TurnReport};
use crate::journal;
use crate::usage::{Channel, Meter};
use crate::voice_mix::{MixSettings, VoiceMix, VoiceTap, live_voice};

/// The choices a translator is built with (from the window).
#[derive(Clone)]
pub struct Options {
    /// Language to translate into (the other side's).
    pub target: String,
    /// The language I speak (a hint for the recognition).
    pub source: String,
    /// The voice the other side hears my translation in; `None` = text only.
    pub voice: Option<String>,
    /// …and how fast it speaks.
    pub speed: f32,
    pub mic: Option<String>,
    pub out: Option<String>,
    pub echo: bool,
    /// The speaker's own voice in the call (used when the output is the
    /// virtual microphone).
    pub mix: MixSettings,
    /// Where the spending is counted.
    pub meter: Arc<Meter>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    /// What I say.
    Said,
    /// What the other side hears (my translation).
    Heard,
    /// What the other side says.
    TheySaid,
    /// What I hear (their translation).
    IHear,
}

/// Which way a channel translates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// My voice → the call.
    Mine,
    /// The call → my headphones.
    Theirs,
}

impl Side {
    fn panes(self) -> (Pane, Pane) {
        match self {
            Side::Mine => (Pane::Said, Pane::Heard),
            Side::Theirs => (Pane::TheySaid, Pane::IHear),
        }
    }

    /// Marks the log lines of the other side's channel.
    pub fn tag(self) -> &'static str {
        match self {
            Side::Mine => "",
            Side::Theirs => "[собеседник] ",
        }
    }
}

/// Where the live texts go (any thread).
pub type TextSink = Arc<dyn Fn(Pane, &str) + Send + Sync>;

struct PressState {
    on: bool,
    /// The release tail in progress (its number); a new press cancels it.
    tail: Option<u64>,
    tails: u64,
    pressed_at: Instant,
}

pub struct Translator {
    interp: Arc<Interpreter>,
    player: Arc<Player>,
    mic: Mutex<Option<Microphone>>,
    press: Arc<Mutex<PressState>>,
    mic_level_bits: Arc<AtomicU32>,
    /// Samples of translation received since the last press (for the log).
    received: Arc<AtomicUsize>,
    /// Only when the translation goes into the call.
    voice_mix: Option<Arc<VoiceMix>>,
    /// The conversation's stopwatch runs from key down to the end of the tail.
    meter: Arc<Meter>,
}

/// Passes the interpreter's output on: the voice to the headphones, the
/// texts to the window, the reports and errors to the log.
pub(crate) struct Bridge {
    pub player: Arc<Player>,
    pub text: TextSink,
    pub received: Arc<AtomicUsize>,
    pub side: Side,
}

impl Listener for Bridge {
    fn voice(&self, pcm: &[i16]) {
        self.received.fetch_add(pcm.len(), Ordering::Relaxed);
        self.player.play(pcm);
    }

    fn said(&self, text: &str) {
        (self.text)(self.side.panes().0, text);
    }

    fn heard(&self, text: &str) {
        (self.text)(self.side.panes().1, text);
    }

    fn turn_done(&self, r: &TurnReport) {
        let tag = self.side.tag();
        let last = r.last_speech_after_end.map_or("—".to_owned(), |s| format!("+{s:.1}s"));
        journal::log(&format!(
            "  {tag}итог: речь перевода {:.1}s, после отпускания+хвоста {:.1}s (последняя через {last}), итог через {:.1}s (pause)",
            r.speech, r.speech_after_end, r.closed_after
        ));
        if r.said.is_empty() && r.heard.is_empty() {
            return;
        }
        journal::log(&format!("  {tag}сказал:    {}", if r.said.is_empty() { "—" } else { &r.said }));
        journal::log(&format!("  {tag}перевод:   {}", if r.heard.is_empty() { "—" } else { &r.heard }));
        let (twice_in, twice_out) = (repeats(&r.said, 3), repeats(&r.heard, 3));
        if !twice_in.is_empty() {
            journal::log(&format!("  {tag}! повтор уже во входе (модель так услышала микрофон): {}", twice_in.join(", ")));
        } else if !twice_out.is_empty() {
            journal::log(&format!("  {tag}! повтор только в переводе (модель повторила сама): {}", twice_out.join(", ")));
        }
    }

    fn error(&self, message: &str) {
        eprintln!("{message}");
        journal::log(&format!("error: {}{message}", self.side.tag()));
    }
}

/// Word groups of `n` that occur more than once (a doubled phrase).
pub fn repeats(text: &str, n: usize) -> Vec<String> {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    let mut first_seen = std::collections::HashMap::new();
    let mut found: Vec<String> = Vec::new();
    for i in 0..words.len().saturating_sub(n - 1) {
        let gram = words[i..i + n].join(" ");
        if let Some(&first) = first_seen.get(&gram)
            && i - first >= n
            && !found.contains(&gram)
        {
            found.push(gram.clone());
        }
        first_seen.entry(gram).or_insert(i);
    }
    found
}

impl Translator {
    /// Opens the output, connects to Soniox (recognition, and the voice), then
    /// opens the microphone. Takes a few seconds; call it off the UI thread.
    pub fn start(api_key: &str, options: &Options, text: TextSink) -> Result<Arc<Self>> {
        let output = Output::find(options.out.as_deref())?;
        let (echo, far) = if options.echo {
            let (canceller, tap) = EchoCanceller::new(output.rate());
            (Some(canceller), Some(tap))
        } else {
            (None, None)
        };
        // Into the call, the speaker's own voice can go under the translation.
        let into_call = matches!(output, Output::VirtualMic { .. });
        let voice_mix = into_call.then(|| VoiceMix::new(options.mix));
        let (tap, mixer) = match &voice_mix {
            Some(mix) => {
                let (tap, mixer) = live_voice(output.rate(), mix.clone());
                (Some(tap), Some(mixer))
            }
            None => (None, None),
        };
        let player = Arc::new(Player::start(output, far, mixer, true, |seconds| {
            journal::log(&format!("! playback fell {seconds:.1}s too far behind — oldest audio cut"))
        })?);
        let received = Arc::new(AtomicUsize::new(0));
        let bridge = Arc::new(Bridge { player: player.clone(), text, received: received.clone(), side: Side::Mine });
        let setup = Setup {
            target: options.target.clone(),
            source: options.source.clone(),
            voice: options.voice.clone(),
            speed: options.speed,
        };
        let interp = Arc::new(Interpreter::connect(api_key, setup, bridge, options.meter.clone())?);

        let press = Arc::new(Mutex::new(PressState { on: false, tail: None, tails: 0, pressed_at: Instant::now() }));
        let mic_level_bits = Arc::new(AtomicU32::new((-120f32).to_bits()));
        let handler = mic_handler(interp.clone(), press.clone(), mic_level_bits.clone(), echo);
        let raw = tap.map(|tap| voice_tap(tap, press.clone()));
        let mic = Microphone::open(options.mic.as_deref(), handler, raw)?;

        journal::log(&format!(
            "=== window (Soniox): {} {} → {}, перевод {}, mic={}, out={}, echo={}{}",
            soniox::MODEL, options.source, options.target,
            options.voice.as_deref().map_or("только текстом".to_owned(), |v| format!("голосом {v}")),
            mic.name, player.name, options.echo,
            if into_call { mix_summary(&options.mix) } else { String::new() }
        ));
        Ok(Arc::new(Self {
            interp,
            player,
            mic: Mutex::new(Some(mic)),
            press,
            mic_level_bits,
            received,
            voice_mix,
            meter: options.meter.clone(),
        }))
    }

    /// Key down: the press starts (or goes on, if within the release tail).
    pub fn press(&self) {
        let mut state = self.press.lock();
        if state.tail.take().is_some() {
            drop(state);
            journal::log("▶ key down (continues)");
            return;
        }
        if state.on {
            return;
        }
        self.interp.start_turn(); // queued before any audio of this press
        self.meter.set_active(Channel::Mine, true);
        state.on = true;
        state.pressed_at = Instant::now();
        drop(state);
        self.received.store(0, Ordering::Relaxed);
        let queued = self.player.queued_seconds();
        let playing = if queued > 0.05 { format!(" (ещё играет {queued:.1}s прошлого перевода)") } else { String::new() };
        journal::log(&format!("▶ key down{playing}"));
    }

    /// Key up: the microphone keeps going for the release tail, then the
    /// press ends and the model finishes the phrase.
    pub fn release(self: &Arc<Self>) {
        let (tail, held) = {
            let mut state = self.press.lock();
            if !state.on || state.tail.is_some() {
                return;
            }
            state.tails += 1;
            state.tail = Some(state.tails);
            (state.tails, state.pressed_at.elapsed())
        };
        let me: Weak<Self> = Arc::downgrade(self);
        thread::spawn(move || {
            thread::sleep(soniox::RELEASE_TAIL);
            if let Some(me) = me.upgrade() {
                me.end_press(tail);
            }
        });
        let translated = self.received.load(Ordering::Relaxed) as f64 / VOICE_RATE as f64;
        journal::log(&format!("■ key up after {:.1}s · translation so far {translated:.1}s", held.as_secs_f64()));
    }

    fn end_press(&self, tail: u64) {
        {
            let mut state = self.press.lock();
            if state.tail != Some(tail) {
                return; // pressed again meanwhile
            }
            state.tail = None;
            state.on = false; // from now on nothing is sent
            self.interp.end_turn();
            self.meter.set_active(Channel::Mine, false);
        }
        journal::log(&format!(
            "  хвост {:.1}s закончился → модель договаривает",
            soniox::RELEASE_TAIL.as_secs_f64()
        ));
    }

    /// The voice's speed, from the next piece of translation on.
    pub fn set_speed(&self, speed: f32) {
        self.interp.set_speed(speed);
    }

    /// Takes effect at once; the session is kept.
    pub fn set_voice_mix(&self, settings: MixSettings) {
        if let Some(mix) = &self.voice_mix {
            mix.set(settings);
        }
    }

    pub fn mic_level_db(&self) -> f32 {
        f32::from_bits(self.mic_level_bits.load(Ordering::Relaxed))
    }

    pub fn out_level_db(&self) -> f32 {
        self.player.level_db()
    }

    /// The microphone, the Soniox sessions; the output stops when the
    /// translator is dropped. Blocks up to ~2 s. Safe to call twice.
    pub fn close(&self) {
        self.meter.set_active(Channel::Mine, false);
        drop(self.mic.lock().take());
        self.interp.close();
    }
}

/// The speaker's voice into the call, only while the key is held (and the
/// release tail): the same moments the model hears it.
fn voice_tap(mut tap: VoiceTap, press: Arc<Mutex<PressState>>) -> RawHandler {
    Box::new(move |samples: &[f32], rate: u32| {
        let open = press.lock().on;
        tap.push(samples, rate, open);
    })
}

pub fn mix_summary(mix: &MixSettings) -> String {
    format!(
        ", мой голос в звонке {:.0}%{}",
        mix.level * 100.0,
        if mix.auto_duck {
            format!(" (под переводом {:.0}%, возврат {:.1} с)", mix.duck_to * 100.0, mix.hold.as_secs_f32())
        } else {
            String::new()
        }
    )
}

/// Runs on the microphone's worker for every 20 ms chunk, pressed or not
/// (so a press starts with no stale audio inside the echo canceller).
fn mic_handler(
    interp: Arc<Interpreter>,
    press: Arc<Mutex<PressState>>,
    level_bits: Arc<AtomicU32>,
    mut echo: Option<EchoCanceller>,
) -> ChunkHandler {
    Box::new(move |chunk: &mut [f32]| {
        level_bits.store(dsp::dbfs(chunk).to_bits(), Ordering::Relaxed);
        // The canceller first: it needs the microphone exactly as captured.
        if let Some(echo) = echo.as_mut() {
            echo.process(chunk);
        }
        let voice = dsp::to_i16(chunk);
        // Fed while holding the press lock, so no audio slips in after the
        // press has ended (end_press takes the same lock).
        let state = press.lock();
        if state.on {
            interp.feed(&voice);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::repeats;

    #[test]
    fn finds_a_doubled_phrase() {
        let text = "I checked the logs today. I checked the logs today.";
        assert_eq!(repeats(text, 3), vec!["i checked the", "checked the logs", "the logs today"]);
    }

    #[test]
    fn overlapping_words_are_not_a_repeat() {
        assert!(repeats("да да да да", 3).is_empty());
        assert!(repeats("Привет, как дела у тебя", 3).is_empty());
    }
}

//! The other side of the call → my headphones.
//!
//! The call app's sound (see `app_audio`) goes two ways: the original, at the
//! chosen volume, to my headphones (lower still under the translation — the
//! same mixer as my own voice into the call), and to a second Gemini session
//! that translates it into my language. No key: a turn starts when sound
//! appears and ends after a pause. Nothing is sent while the call is quiet (no
//! cost), nor while my own translation is sounding in the call — the other
//! side's microphone can echo it back, and it must not be translated again.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use parking_lot::Mutex;

use crate::app_audio::{self, AppAudio, Source};
use crate::audio::{self, MIC_CHUNK, MIC_RATE, Output, Player, Worker};
use crate::dsp::{self, resample::StreamResampler};
use crate::soniox::{self, Interpreter, Setup};
use crate::journal;
use crate::usage::{Channel, Meter};
use crate::translator::{Bridge, Side, TextSink};
use crate::voice_mix::{MixSettings, VoiceMix, VoiceTap, live_voice};

/// A 20 ms chunk louder than this is someone speaking (apps send true
/// silence between words; a call's comfort noise is far below this).
const SPEECH_DB: f32 = -50.0;
/// This much quiet ends the turn (the model then finishes the phrase).
const PAUSE: Duration = Duration::from_millis(1200);
/// Sent along when the sound begins, so the first syllable isn't lost.
const PREROLL: Duration = Duration::from_millis(300);
/// After my translation stops sounding in the call, the other side's
/// microphone may still return its echo this long.
const ECHO_GUARD: Duration = Duration::from_millis(750);
/// How often a chosen app is checked for new sound processes (a call often
/// starts in a helper process that wasn't there before).
const WATCH_EVERY: Duration = Duration::from_secs(2);

/// The choices the channel is built with (from the window).
#[derive(Clone)]
pub struct IncomingOptions {
    /// My language: what the other side is translated into.
    pub target: String,
    /// The other side's language (a hint for the recognition).
    pub language: String,
    /// The voice I hear their translation in; `None` = text only (nothing
    /// is synthesised or paid for; the original plays as set).
    pub voice: Option<String>,
    /// …and how fast it speaks.
    pub speed: f32,
    pub source: Source,
    /// My headphones (`None` = the system default output).
    pub headphones: Option<String>,
    /// The other side's original voice in my headphones.
    pub mix: MixSettings,
    /// Where the spending is counted.
    pub meter: Arc<Meter>,
}

/// What the channel is doing (for the window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// The chosen app has no sound yet (no call started).
    Waiting,
    Listening,
    /// The other side is speaking; the translation is on its way.
    Speaking,
}

pub struct Incoming {
    interp: Arc<Interpreter>,
    player: Arc<Player>,
    voice_mix: Arc<VoiceMix>,
    shared: Arc<Shared>,
    watcher: Mutex<Option<Worker>>,
}

/// Shared by the capture handler, the watcher and the window.
struct Shared {
    capture: Mutex<Option<AppAudio>>,
    pipeline: Arc<Mutex<Pipeline>>,
    level_bits: AtomicU32,
    speaking: AtomicBool,
}

impl Incoming {
    /// Opens my headphones, connects to Gemini, then starts listening to the
    /// call. Takes a few seconds; call it off the UI thread.
    pub fn start(api_key: &str, options: &IncomingOptions, text: TextSink) -> Result<Arc<Self>> {
        let output = Output::find(options.headphones.as_deref())?;
        let voice_mix = VoiceMix::new(options.mix);
        let (tap, mixer) = live_voice(output.rate(), voice_mix.clone());
        let player = Arc::new(Player::start(output, None, Some(mixer), false, |seconds| {
            journal::log(&format!("! [собеседник] playback fell {seconds:.1}s too far behind — oldest audio cut"))
        })?);
        let bridge = Arc::new(Bridge { player: player.clone(), text, received: Arc::new(AtomicUsize::new(0)), side: Side::Theirs });
        let setup = Setup {
            target: options.target.clone(),
            source: options.language.clone(),
            voice: options.voice.clone(),
            speed: options.speed,
        };
        let interp = Arc::new(Interpreter::connect(api_key, setup, bridge, options.meter.clone())?);

        let shared = Arc::new(Shared {
            capture: Mutex::new(None),
            pipeline: Arc::new(Mutex::new(Pipeline::new(tap, interp.clone(), options.meter.clone()))),
            level_bits: AtomicU32::new((-120f32).to_bits()),
            speaking: AtomicBool::new(false),
        });
        let incoming = Arc::new(Self { interp, player, voice_mix, shared, watcher: Mutex::new(None) });
        let source_name = match &options.source {
            Source::AllApps => "все программы".to_owned(),
            Source::App(bundle) => app_audio::app_name(bundle).unwrap_or_else(|| bundle.clone()),
        };
        match &options.source {
            Source::AllApps => incoming.listen(&Source::AllApps)?,
            Source::App(bundle) => incoming.watch(bundle.clone()),
        }
        journal::log(&format!(
            "=== [собеседник] {} {} → {}, перевод {}, звук из: {source_name}, наушники={}, оригинал {:.0}%{}",
            soniox::MODEL,
            options.language,
            options.target,
            options.voice.as_deref().map_or("только текстом".to_owned(), |v| format!("голосом {v}")),
            incoming.player.name,
            options.mix.level * 100.0,
            if options.mix.auto_duck { format!(" (под переводом {:.0}%)", options.mix.duck_to * 100.0) } else { String::new() }
        ));
        Ok(incoming)
    }

    /// (Re)starts the capture of `source`.
    fn listen(&self, source: &Source) -> Result<()> {
        let shared = self.shared.clone();
        let capture = AppAudio::open(
            source,
            Box::new(move |samples, rate| {
                shared.level_bits.store(dsp::dbfs(samples).to_bits(), Ordering::Relaxed);
                let speaking = shared.pipeline.lock().process(samples, rate);
                shared.speaking.store(speaking, Ordering::Relaxed);
            }),
        )?;
        *self.shared.capture.lock() = Some(capture);
        Ok(())
    }

    /// Follows one app: (re)taps it whenever its set of sound processes
    /// changes — a call usually starts in a helper process.
    fn watch(self: &Arc<Self>, bundle: String) {
        let me = Arc::downgrade(self);
        let worker = Worker::spawn("call watcher", move |stop| {
            let mut tapped: Vec<u32> = Vec::new();
            while !stop.load(Ordering::Acquire) {
                let Some(incoming) = me.upgrade() else { break };
                let objects = app_audio::process_objects(&bundle);
                if objects != tapped {
                    drop(incoming.shared.capture.lock().take());
                    tapped = objects;
                    if !tapped.is_empty() {
                        match incoming.listen(&Source::App(bundle.clone())) {
                            Ok(()) => journal::log(&format!("[собеседник] слушаю {bundle} ({} процесс.)", tapped.len())),
                            Err(e) => {
                                journal::log(&format!("error: [собеседник] {e:#}"));
                                tapped.clear(); // try again next time
                            }
                        }
                    }
                }
                drop(incoming);
                thread::park_timeout(WATCH_EVERY);
            }
        });
        match worker {
            Ok(worker) => *self.watcher.lock() = Some(worker),
            Err(e) => journal::log(&format!("error: [собеседник] {e:#}")),
        }
    }

    pub fn state(&self) -> State {
        if self.shared.capture.lock().is_none() {
            State::Waiting
        } else if self.shared.speaking.load(Ordering::Relaxed) {
            State::Speaking
        } else {
            State::Listening
        }
    }

    /// The voice's speed, from the next piece of translation on.
    pub fn set_speed(&self, speed: f32) {
        self.interp.set_speed(speed);
    }

    /// Takes effect at once.
    pub fn set_voice_mix(&self, settings: MixSettings) {
        self.voice_mix.set(settings);
    }

    /// Loudness of the call (the other side, as captured).
    pub fn level_db(&self) -> f32 {
        f32::from_bits(self.shared.level_bits.load(Ordering::Relaxed))
    }

    /// Loudness of what plays in my headphones.
    pub fn out_level_db(&self) -> f32 {
        self.player.level_db()
    }

    /// Gives the app its sound back and closes every session. Blocks up to
    /// ~3 s. Safe to call twice.
    pub fn close(&self) {
        drop(self.watcher.lock().take());
        drop(self.shared.capture.lock().take());
        self.shared.pipeline.lock().meter.set_active(Channel::Theirs, false);
        self.interp.close();
    }
}

/// The call's sound → the original to my ears, speech to the model.
struct Pipeline {
    tap: VoiceTap,
    interp: Arc<Interpreter>,
    /// The conversation's stopwatch runs while the other side's speech is sent.
    meter: Arc<Meter>,
    /// The call at its own rate → 16 kHz for the model.
    resampler: Option<(u32, StreamResampler)>,
    ready: Vec<f32>,
    preroll: VecDeque<Vec<i16>>,
    in_turn: bool,
    quiet: Duration,
    turn_started: Instant,
}

impl Pipeline {
    fn new(tap: VoiceTap, interp: Arc<Interpreter>, meter: Arc<Meter>) -> Self {
        Self {
            tap,
            interp,
            meter,
            resampler: None,
            ready: Vec::new(),
            preroll: VecDeque::new(),
            in_turn: false,
            quiet: Duration::ZERO,
            turn_started: Instant::now(),
        }
    }

    /// One block of the call's sound; true while the other side is speaking.
    fn process(&mut self, samples: &[f32], rate: u32) -> bool {
        self.tap.push(samples, rate, true);
        if self.resampler.as_ref().is_none_or(|(r, _)| *r != rate) {
            self.resampler = Some((rate, StreamResampler::new(rate, MIC_RATE)));
        }
        let (_, resampler) = self.resampler.as_mut().expect("just set");
        resampler.process(samples, &mut self.ready);
        let whole = self.ready.len() / MIC_CHUNK * MIC_CHUNK;
        let chunks: Vec<Vec<f32>> = self.ready[..whole].chunks(MIC_CHUNK).map(<[f32]>::to_vec).collect();
        self.ready.drain(..whole);
        for chunk in chunks {
            self.chunk(&chunk);
        }
        self.in_turn
    }

    fn chunk(&mut self, chunk: &[f32]) {
        const STEP: Duration = Duration::from_millis(20);
        let echo = audio::call_voice_within(ECHO_GUARD);
        let speech = !echo && dsp::dbfs(chunk) > SPEECH_DB;
        // Echo of my own translation is replaced by silence, never sent.
        let pcm = if echo { vec![0i16; chunk.len()] } else { dsp::to_i16(chunk) };
        if !self.in_turn {
            self.preroll.push_back(pcm);
            while self.preroll.len() > (PREROLL.as_millis() / STEP.as_millis()) as usize {
                self.preroll.pop_front();
            }
            if speech {
                self.interp.start_turn();
                self.meter.set_active(Channel::Theirs, true);
                for piece in self.preroll.drain(..) {
                    self.interp.feed(&piece);
                }
                self.in_turn = true;
                self.quiet = Duration::ZERO;
                self.turn_started = Instant::now();
                journal::log("▶ [собеседник] говорит");
            }
            return;
        }
        self.interp.feed(&pcm);
        self.quiet = if speech { Duration::ZERO } else { self.quiet + STEP };
        if self.quiet >= PAUSE {
            self.interp.end_turn();
            self.meter.set_active(Channel::Theirs, false);
            self.in_turn = false;
            journal::log(&format!("■ [собеседник] пауза после {:.1}s", self.turn_started.elapsed().as_secs_f64()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::Pane;

    /// End to end, with the network and the "System Audio Recording"
    /// permission (run from Terminal): an English phrase played by another app
    /// (`afplay`) comes out translated into Russian. The translation goes into
    /// the virtual microphone, so nothing is heard.
    #[test]
    #[ignore]
    fn translates_another_app() {
        let key = crate::config::soniox_api_key().expect("soniox_api_key");
        let dir = std::env::temp_dir();
        let speech = dir.join("pressvoice_incoming_test.aiff");
        let made = std::process::Command::new("say")
            .args(["-v", "Samantha", "-o"]).arg(&speech)
            .arg("Good morning. Let us meet tomorrow at ten and discuss the new project.")
            .status().expect("say");
        assert!(made.success());

        let texts: Arc<Mutex<Vec<(Pane, String)>>> = Arc::default();
        let sink = texts.clone();
        let options = IncomingOptions {
            target: "ru".into(),
            language: "en".into(),
            voice: Some(crate::soniox::DEFAULT_VOICE.into()),
            speed: 1.0,
            source: Source::AllApps,
            headphones: Some(crate::virtual_mic::NAME.into()),
            mix: MixSettings { level: 0.2, auto_duck: true, duck_to: 0.3, hold: Duration::from_millis(700) },
            meter: Meter::detached(),
        };
        let started = Instant::now();
        let incoming = Incoming::start(&key, &options, Arc::new(move |pane: Pane, t: &str| sink.lock().push((pane, t.to_owned()))))
            .expect("incoming channel");
        println!("ready in {:.1}s, state {:?}", started.elapsed().as_secs_f64(), incoming.state());
        let played = Instant::now();
        assert!(std::process::Command::new("afplay").arg(&speech).status().expect("afplay").success());
        println!("phrase played in {:.1}s", played.elapsed().as_secs_f64());
        let mut first_translation = None;
        for _ in 0..150 {
            thread::sleep(Duration::from_millis(100));
            if first_translation.is_none() && texts.lock().iter().any(|(p, _)| *p == Pane::IHear) {
                first_translation = Some(played.elapsed());
            }
        }
        incoming.close();
        let texts = texts.lock();
        let join = |pane| texts.iter().filter(|(p, _)| *p == pane).map(|(_, t)| t.as_str()).collect::<String>();
        println!("они сказали: {}", join(Pane::TheySaid));
        println!("я слышу:     {}", join(Pane::IHear));
        println!("первый перевод через {:.1}s после начала фразы", first_translation.map_or(f64::NAN, |d| d.as_secs_f64()));
        assert!(!join(Pane::IHear).trim().is_empty(), "no translation");
    }
}

//! The live interpreter on Soniox: speech-to-text with translation
//! (stt-rt-v5) and, when the translation is to be heard, text-to-speech
//! (tts-rt-v2). Push-to-talk, like the Gemini version it replaces.
//!
//! One recognition session serves every press. Audio is sent only during a
//! press; between presses a keep-alive holds the connection (and the
//! session's context). At the end of a press a `finalize` makes Soniox
//! return the rest of the phrase at once (measured: 0.15 s).
//!
//! Soniox translates a phrase in pieces as it is spoken (measured: each
//! clause ~0.7–1.5 s after it ends). Each finished piece of translation goes
//! to the voice in order, on one warm connection (first sound 0.3–0.4 s).
//! Without a voice ("text only") nothing is synthesised and nothing is paid
//! for it.
//!
//! The controller task owns the recognition socket and the turn; a voice
//! task owns the speech socket. They talk through channels — no locks around
//! the session logic.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::{interval, sleep, timeout};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::audio::{MIC_RATE, VOICE_RATE};
use crate::usage::{self, Meter};

const STT_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";
const TTS_URL: &str = "wss://tts-rt.soniox.com/tts-websocket";
pub const STT_MODEL: &str = "stt-rt-v5";
pub const TTS_MODEL: &str = "tts-rt-v2";
/// The models, for the window and the log.
pub const MODEL: &str = "Soniox stt-rt-v5 + tts-rt-v2";
/// The voice offered first (a deep, clear male voice).
pub const DEFAULT_VOICE: &str = "Adrian";

/// Microphone kept after key-up: people let go on the last syllable.
pub const RELEASE_TAIL: Duration = Duration::from_millis(500);
/// Audio goes out in pieces of this many samples (60 ms).
const PIECE: usize = (MIC_RATE / 1000 * 60) as usize;
/// Between presses the connections are kept alive this often (Soniox closes
/// idle ones after ~40 s).
const KEEPALIVE_EVERY: Duration = Duration::from_secs(15);
const OPEN_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(2);
/// A press is reported once its translation (and voice) has been quiet this long.
const REPORT_AFTER: Duration = Duration::from_millis(1500);
/// The end-of-phrase markers Soniox puts among the tokens.
const MARKERS: [&str; 2] = ["<fin>", "<end>"];

/// What a press produced, once its translation has gone quiet.
#[derive(Clone, Debug, Default)]
pub struct TurnReport {
    /// What Soniox heard.
    pub said: String,
    /// Its translation.
    pub heard: String,
    /// Seconds of translated voice (0 in text-only mode).
    pub speech: f64,
    /// …of which after the press ended (release + tail).
    pub speech_after_end: f64,
    /// When the last piece of voice came, counted from the end of the press.
    pub last_speech_after_end: Option<f64>,
    /// When this report was made, counted from the end of the press.
    pub closed_after: f64,
}

/// Where the interpreter's output goes. Called from its network thread.
pub trait Listener: Send + Sync + 'static {
    /// A chunk of the translated voice, 24 kHz mono.
    fn voice(&self, pcm: &[i16]);
    /// A piece of the transcription of what the speaker says.
    fn said(&self, text: &str);
    /// A piece of the translation.
    fn heard(&self, text: &str);
    fn turn_done(&self, report: &TurnReport);
    fn error(&self, message: &str);
}

/// What one interpreter translates, and how it is heard.
#[derive(Clone, Debug)]
pub struct Setup {
    /// Language to translate into.
    pub target: String,
    /// The language the speaker uses (a hint; others are still recognised).
    pub source: String,
    /// The voice to speak the translation in; `None` = text only.
    pub voice: Option<String>,
    /// How fast the voice speaks (Soniox: 0.7–1.3, 1 = normal).
    pub speed: f32,
}

/// The speeds Soniox's voice can speak at.
pub const SPEED_RANGE: (f32, f32) = (0.7, 1.3);

// ── The handle the rest of the program uses ──────────────────────────────────

enum Command {
    Begin,
    Audio(Vec<i16>),
    End,
    Close,
}

pub struct Interpreter {
    shared: Arc<Shared>,
    commands: mpsc::UnboundedSender<Command>,
    /// Microphone audio not yet sent (< one piece).
    pending: Mutex<Vec<i16>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Interpreter {
    /// Opens the recognition session (and the voice, if any); returns once ready.
    pub fn connect(api_key: &str, setup: Setup, listener: Arc<dyn Listener>, meter: Arc<Meter>) -> Result<Self> {
        let speed = AtomicU32::new(setup.speed.to_bits());
        let shared = Arc::new(Shared { key: api_key.to_owned(), setup, listener, meter, speed, closed: AtomicBool::new(false) });
        let handle = shared.clone();
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = thread::Builder::new().name("soniox".into()).spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("a tokio runtime");
            runtime.block_on(Controller::new(shared, command_rx).run(ready_tx));
        })?;
        let this = Self { shared: handle, commands, pending: Mutex::new(Vec::new()), thread: Mutex::new(Some(thread)) };
        let result = ready_rx
            .recv_timeout(CONNECT_TIMEOUT)
            .unwrap_or_else(|_| Err(anyhow!("Soniox не ответил за {} с", CONNECT_TIMEOUT.as_secs())));
        match result {
            Ok(()) => Ok(this),
            Err(e) => {
                this.close();
                Err(e)
            }
        }
    }

    /// The voice's speed from the next piece of translation on (no reconnect).
    pub fn set_speed(&self, speed: f32) {
        let speed = speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1);
        self.shared.speed.store(speed.to_bits(), Ordering::Relaxed);
    }

    pub fn start_turn(&self) {
        self.pending.lock().clear();
        let _ = self.commands.send(Command::Begin);
    }

    /// 16 kHz audio of the current press.
    pub fn feed(&self, pcm: &[i16]) {
        let mut pending = self.pending.lock();
        pending.extend_from_slice(pcm);
        while pending.len() >= PIECE {
            let piece: Vec<i16> = pending.drain(..PIECE).collect();
            let _ = self.commands.send(Command::Audio(piece));
        }
    }

    pub fn end_turn(&self) {
        let rest = std::mem::take(&mut *self.pending.lock());
        if !rest.is_empty() {
            let _ = self.commands.send(Command::Audio(rest));
        }
        let _ = self.commands.send(Command::End);
    }

    /// Closes the sessions and waits (≤ ~2 s) for the network thread. Safe to call twice.
    pub fn close(&self) {
        let _ = self.commands.send(Command::Close);
        if let Some(thread) = self.thread.lock().take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        self.close();
    }
}

struct Shared {
    key: String,
    setup: Setup,
    listener: Arc<dyn Listener>,
    meter: Arc<Meter>,
    /// The voice's speed now (f32 bits); changeable while running.
    speed: AtomicU32,
    closed: AtomicBool,
}

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type Sink = SplitSink<Socket, Message>;

async fn open(url: &str, key: &str) -> Result<Socket> {
    let mut request = url.into_client_request()?;
    request.headers_mut().insert("Authorization", HeaderValue::from_str(&format!("Bearer {key}"))?);
    match timeout(OPEN_TIMEOUT, tokio_tungstenite::connect_async_with_config(request, None, true)).await {
        Err(_) => bail!("Soniox не отвечает"),
        Ok(Err(tungstenite::Error::Http(response))) => {
            let body = response.body().as_deref().map(String::from_utf8_lossy).unwrap_or_default();
            Err(refusal(response.status().as_u16(), &body))
        }
        Ok(Err(e)) => bail!("нет связи с Soniox: {e}"),
        Ok(Ok((socket, _))) => Ok(socket),
    }
}

/// A refusal from Soniox, in plain words when it is about the key or money.
fn refusal(code: u16, message: &str) -> anyhow::Error {
    match code {
        401 => anyhow!("Soniox: неверный ключ API (проверьте soniox_api_key)"),
        402 => anyhow!("Баланс Soniox закончился — пополните его в консоли Soniox и введите новую сумму в окне"),
        429 => anyhow!("Soniox: превышен лимит запросов (429) — {message}"),
        _ => anyhow!("Soniox: ошибка {code}: {message}"),
    }
}

// ── Recognition ──────────────────────────────────────────────────────────────

/// What the recognition reader tells the controller.
enum Event {
    Said(String),
    Heard(String),
    /// The pending phrase was finalized (`<fin>`).
    Finalized,
    /// Audio processed so far in this session, ms (to charge the difference).
    Processed(u64),
    /// Seconds of voice played for the current turn.
    Voice(f64),
    Ended(Option<String>),
}

fn stt_config(setup: &Setup) -> Message {
    let config = json!({
        "model": STT_MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": MIC_RATE,
        "num_channels": 1,
        "language_hints": [setup.source],
        "enable_endpoint_detection": true,
        "translation": {"type": "one_way", "target_language": setup.target},
    });
    Message::text(config.to_string())
}

async fn open_stt(shared: &Shared, events: mpsc::UnboundedSender<Event>) -> Result<Sink> {
    let socket = open(STT_URL, &shared.key).await?;
    let (mut sink, stream) = socket.split();
    sink.send(stt_config(&shared.setup)).await?;
    tokio::spawn(stt_reader(stream, events));
    Ok(sink)
}

async fn stt_reader(mut stream: SplitStream<Socket>, events: mpsc::UnboundedSender<Event>) {
    let mut ended = None;
    while let Some(message) = stream.next().await {
        let text = match message {
            Ok(Message::Text(text)) => text.to_string(),
            Ok(Message::Close(frame)) => {
                ended = frame.map(|f| f.reason.to_string()).filter(|r| !r.is_empty());
                break;
            }
            Ok(_) => continue,
            Err(e) => {
                ended = Some(e.to_string());
                break;
            }
        };
        let Ok(message) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if let Some(code) = message.get("error_code").and_then(Value::as_u64) {
            let what = message.get("error_message").and_then(Value::as_str).unwrap_or("");
            ended = Some(format!("{:#}", refusal(code as u16, what)));
            break;
        }
        let tokens = message.get("tokens").and_then(Value::as_array);
        let (mut said, mut heard, mut finalized) = (String::new(), String::new(), false);
        for token in tokens.into_iter().flatten().filter(|t| t.get("is_final").and_then(Value::as_bool) == Some(true)) {
            let text = token.get("text").and_then(Value::as_str).unwrap_or("");
            if MARKERS.contains(&text) {
                finalized |= text == "<fin>";
                continue;
            }
            match token.get("translation_status").and_then(Value::as_str) {
                Some("translation") => heard.push_str(text),
                _ => said.push_str(text),
            }
        }
        if !said.is_empty() {
            let _ = events.send(Event::Said(said));
        }
        if !heard.is_empty() {
            let _ = events.send(Event::Heard(heard));
        }
        if finalized {
            let _ = events.send(Event::Finalized);
        }
        if let Some(ms) = message.get("total_audio_proc_ms").and_then(Value::as_u64) {
            let _ = events.send(Event::Processed(ms));
        }
        if message.get("finished").and_then(Value::as_bool) == Some(true) {
            break;
        }
    }
    let _ = events.send(Event::Ended(ended));
}

// ── Voice ────────────────────────────────────────────────────────────────────

/// Speaks pieces of translation in order, on one warm connection (opened
/// again if Soniox closed it). Each piece is a stream of its own, ended at once.
async fn voice_task(shared: Arc<Shared>, voice: String, mut pieces: mpsc::UnboundedReceiver<String>, events: mpsc::UnboundedSender<Event>) {
    let mut socket: Option<Socket> = open(TTS_URL, &shared.key).await.ok();
    let mut keepalive = interval(KEEPALIVE_EVERY);
    let mut streams = 0u64;
    loop {
        tokio::select! {
            piece = pieces.recv() => {
                let Some(text) = piece else { break };
                if socket.is_none() {
                    socket = open(TTS_URL, &shared.key).await.ok();
                }
                let Some(ws) = socket.as_mut() else {
                    shared.listener.error("Soniox: голос недоступен (нет связи)");
                    continue;
                };
                streams += 1;
                match speak(ws, &shared, &voice, &format!("s{streams}"), &text, &events).await {
                    Ok(()) => {}
                    Err(e) => {
                        shared.listener.error(&format!("{e:#}"));
                        socket = None; // opened again for the next piece
                    }
                }
            }
            _ = keepalive.tick() => {
                if let Some(ws) = socket.as_mut()
                    && ws.send(Message::text(json!({"keep_alive": true}).to_string())).await.is_err()
                {
                    socket = None;
                }
            }
        }
    }
    if let Some(mut ws) = socket {
        let _ = ws.close(None).await;
    }
}

async fn speak(ws: &mut Socket, shared: &Shared, voice: &str, id: &str, text: &str, events: &mpsc::UnboundedSender<Event>) -> Result<()> {
    let mut config = json!({
        "stream_id": id, "model": TTS_MODEL, "language": shared.setup.target, "voice": voice,
        "audio_format": "pcm_s16le", "sample_rate": VOICE_RATE,
    });
    // Spoken faster by Soniox itself: no distortion, no added delay (and
    // shorter audio costs less).
    let speed = f32::from_bits(shared.speed.load(Ordering::Relaxed));
    if (speed - 1.0).abs() > 0.001 {
        config["speed"] = json!((speed * 100.0).round() / 100.0);
    }
    ws.send(Message::text(config.to_string())).await?;
    ws.send(Message::text(json!({"stream_id": id, "text": text, "text_end": true}).to_string())).await?;
    shared.meter.charge(usage::tts_text_cost(text.chars().count()));
    loop {
        let message = timeout(Duration::from_secs(15), ws.next()).await.map_err(|_| anyhow!("Soniox: голос не пришёл"))?;
        let Some(message) = message else { bail!("Soniox закрыл соединение голоса") };
        let Message::Text(raw) = message? else { continue };
        let Ok(message) = serde_json::from_str::<Value>(&raw) else { continue };
        if message.get("stream_id").and_then(Value::as_str).is_some_and(|s| s != id && !s.is_empty()) {
            continue;
        }
        if let Some(code) = message.get("error_code").and_then(Value::as_u64) {
            let what = message.get("error_message").and_then(Value::as_str).unwrap_or("");
            return Err(refusal(code as u16, what));
        }
        if let Some(audio) = message.get("audio").and_then(Value::as_str) {
            let bytes = BASE64.decode(audio).unwrap_or_default();
            let pcm: Vec<i16> = bytes.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect();
            if !pcm.is_empty() {
                let seconds = pcm.len() as f64 / VOICE_RATE as f64;
                shared.meter.charge(usage::tts_audio_cost(seconds));
                let _ = events.send(Event::Voice(seconds));
                shared.listener.voice(&pcm);
            }
        }
        if message.get("terminated").and_then(Value::as_bool) == Some(true) {
            return Ok(());
        }
    }
}

// ── The controller ───────────────────────────────────────────────────────────

struct Turn {
    said: String,
    heard: String,
    speech: f64,
    after_end: f64,
    ended: Option<Instant>,
    last_speech: Option<f64>,
    last_activity: Instant,
    finalized: bool,
}

impl Turn {
    fn new() -> Self {
        Self {
            said: String::new(),
            heard: String::new(),
            speech: 0.0,
            after_end: 0.0,
            ended: None,
            last_speech: None,
            last_activity: Instant::now(),
            finalized: false,
        }
    }
}

fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct Controller {
    shared: Arc<Shared>,
    commands: mpsc::UnboundedReceiver<Command>,
    events: mpsc::UnboundedSender<Event>,
    event_rx: mpsc::UnboundedReceiver<Event>,
    stt: Option<Sink>,
    voice: Option<mpsc::UnboundedSender<String>>,
    turn: Option<Turn>,
    /// Audio processed by the current recognition session, ms (already charged).
    processed_ms: u64,
    last_sent: Instant,
}

impl Controller {
    fn new(shared: Arc<Shared>, commands: mpsc::UnboundedReceiver<Command>) -> Self {
        let (events, event_rx) = mpsc::unbounded_channel();
        Self {
            shared,
            commands,
            events,
            event_rx,
            stt: None,
            voice: None,
            turn: None,
            processed_ms: 0,
            last_sent: Instant::now(),
        }
    }

    async fn run(mut self, ready: std::sync::mpsc::Sender<Result<()>>) {
        match open_stt(&self.shared, self.events.clone()).await {
            Ok(sink) => self.stt = Some(sink),
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        }
        if let Some(voice) = self.shared.setup.voice.clone() {
            let (tx, rx) = mpsc::unbounded_channel();
            tokio::spawn(voice_task(self.shared.clone(), voice, rx, self.events.clone()));
            self.voice = Some(tx);
        }
        let _ = ready.send(Ok(()));

        let mut tick = interval(Duration::from_millis(200));
        loop {
            tokio::select! {
                command = self.commands.recv() => match command {
                    Some(Command::Begin) => self.begin().await,
                    Some(Command::Audio(pcm)) => self.send_audio(pcm).await,
                    Some(Command::End) => self.end().await,
                    Some(Command::Close) | None => break,
                },
                Some(event) = self.event_rx.recv() => self.event(event),
                _ = tick.tick() => self.tick().await,
            }
        }
        self.shutdown().await;
    }

    async fn ensure_stt(&mut self) {
        if self.stt.is_none() && !self.shared.closed.load(Ordering::Acquire) {
            match open_stt(&self.shared, self.events.clone()).await {
                Ok(sink) => {
                    self.stt = Some(sink);
                    self.processed_ms = 0;
                }
                Err(e) => self.shared.listener.error(&format!("{e:#}")),
            }
        }
    }

    async fn send(&mut self, message: Message) {
        if let Some(sink) = self.stt.as_mut()
            && sink.send(message).await.is_err()
        {
            self.stt = None; // opened again at the next press
        }
        self.last_sent = Instant::now();
    }

    async fn begin(&mut self) {
        self.report_turn(); // the previous press, if not reported yet
        self.ensure_stt().await;
        self.turn = Some(Turn::new());
    }

    async fn send_audio(&mut self, pcm: Vec<i16>) {
        if self.turn.is_none() {
            return; // between presses nothing is sent
        }
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.send(Message::binary(bytes)).await;
    }

    async fn end(&mut self) {
        if let Some(turn) = self.turn.as_mut() {
            turn.ended = Some(Instant::now());
            turn.last_activity = Instant::now();
        }
        // The rest of the phrase, now rather than after the end-of-speech wait.
        self.send(Message::text(json!({"type": "finalize"}).to_string())).await;
    }

    fn event(&mut self, event: Event) {
        let now = Instant::now();
        match event {
            Event::Said(text) => {
                self.shared.listener.said(&text);
                if let Some(turn) = self.turn.as_mut() {
                    turn.said.push_str(&text);
                }
            }
            Event::Heard(text) => {
                self.shared.listener.heard(&text);
                if let Some(turn) = self.turn.as_mut() {
                    turn.heard.push_str(&text);
                    turn.last_activity = now;
                }
                if let Some(voice) = &self.voice {
                    let piece = text.trim();
                    if !piece.is_empty() {
                        let _ = voice.send(piece.to_owned());
                    }
                }
            }
            Event::Finalized => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.finalized = true;
                    turn.last_activity = now;
                }
            }
            Event::Processed(ms) => {
                // Soniox reports the session's total; charge what is new.
                let new = ms.saturating_sub(self.processed_ms);
                self.processed_ms = self.processed_ms.max(ms);
                if new > 0 {
                    self.shared.meter.charge(usage::stt_cost(new as f64 / 1000.0));
                }
            }
            Event::Voice(seconds) => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.speech += seconds;
                    turn.last_activity = now;
                    if let Some(ended) = turn.ended {
                        turn.after_end += seconds;
                        turn.last_speech = Some((now - ended).as_secs_f64());
                    }
                }
            }
            Event::Ended(reason) => {
                self.stt = None;
                if let Some(reason) = reason.filter(|_| !self.shared.closed.load(Ordering::Acquire)) {
                    if reason.contains("Баланс") {
                        self.shared.meter.mark_exhausted();
                    }
                    self.shared.listener.error(&reason);
                }
            }
        }
    }

    async fn tick(&mut self) {
        if let Some(turn) = &self.turn
            && turn.ended.is_some()
            && (turn.finalized || turn.ended.is_some_and(|t| t.elapsed() > Duration::from_secs(5)))
            && turn.last_activity.elapsed() > REPORT_AFTER
        {
            self.report_turn();
        }
        // Between presses: keep the recognition connection alive.
        if self.turn.as_ref().is_none_or(|t| t.ended.is_some()) && self.last_sent.elapsed() > KEEPALIVE_EVERY {
            self.send(Message::text(json!({"type": "keepalive"}).to_string())).await;
        }
    }

    fn report_turn(&mut self) {
        let Some(turn) = self.turn.take() else { return };
        let Some(ended) = turn.ended else { return };
        self.shared.listener.turn_done(&TurnReport {
            said: clean(&turn.said),
            heard: clean(&turn.heard),
            speech: turn.speech,
            speech_after_end: turn.after_end,
            last_speech_after_end: turn.last_speech,
            closed_after: ended.elapsed().as_secs_f64(),
        });
    }

    async fn shutdown(&mut self) {
        self.shared.closed.store(true, Ordering::Release);
        self.report_turn();
        self.voice = None; // the voice task ends when its queue closes
        if let Some(mut sink) = self.stt.take() {
            let _ = sink.send(Message::binary(Vec::<u8>::new())).await; // end of audio
            let _ = timeout(CLOSE_TIMEOUT, sink.close()).await;
        }
        sleep(Duration::from_millis(50)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recognition_is_asked_for_translation() {
        let setup = Setup { target: "en".into(), source: "ru".into(), voice: None, speed: 1.0 };
        let Message::Text(text) = stt_config(&setup) else { panic!("text expected") };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["model"], STT_MODEL);
        assert_eq!(v["sample_rate"], 16000);
        assert_eq!(v["translation"]["type"], "one_way");
        assert_eq!(v["translation"]["target_language"], "en");
        assert_eq!(v["language_hints"][0], "ru");
    }

    #[test]
    fn refusals_are_explained() {
        assert!(format!("{}", refusal(402, "balance exhausted")).contains("Баланс Soniox"));
        assert!(format!("{}", refusal(401, "")).contains("ключ"));
    }

    #[test]
    fn spaces_are_tidied() {
        assert_eq!(clean("  We'll   meet \n tomorrow "), "We'll meet tomorrow");
    }
}

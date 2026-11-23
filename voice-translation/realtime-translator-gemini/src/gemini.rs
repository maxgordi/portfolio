//! The live interpreter on Google's Gemini Live API
//! (gemini-3.5-live-translate-preview), push-to-talk.
//!
//! One session serves many presses. A fresh session per press made the model
//! re-learn the speaker's voice and context every time: the first words came
//! out garbled and the voice changed between presses. Between presses
//! nothing is sent; the session simply waits.
//!
//! On release the caller keeps the microphone for `RELEASE_TAIL` (the last
//! syllable), then `FINISH_SILENCE` of near-silence is sent: the model voices
//! a phrase to the end only after it hears a pause. If the key is pressed
//! again during that pause, the pause stops and speech goes on.
//!
//! Every new session is first "warmed up" with a recording of a deep male
//! voice («Да, слушаю.») so the model starts out copying a male voice (it
//! copies the voice it hears, and left to itself sometimes picked a female
//! one). What the model says back to the warm-up is dropped, not played.
//!
//! A session is kept as long as Google keeps it (~10 min); when it closes or
//! announces it will (`goAway`), the next press goes to a warm spare.
//!
//! Inside, a single task (`Controller`) owns all the state and is the only
//! one to change it; each session has a reader and a writer task that talk
//! to it through messages — no locks around the session logic.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, interval_at, sleep, timeout, timeout_at};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::audio::{MIC_RATE, VOICE_RATE};
use crate::dsp::{self, Hush};
use crate::usage::{self, Kind, Meter};

const URL: &str = "wss://generativelanguage.googleapis.com/ws/\
                   google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
pub const MODEL: &str = "gemini-3.5-live-translate-preview";

/// Microphone kept after key-up: people let go on the last syllable.
pub const RELEASE_TAIL: Duration = Duration::from_millis(500);
/// Then this much near-silence, so the last phrase is voiced in full
/// (`audioStreamEnd` instead cut the last word out of the voice).
const FINISH_SILENCE: Duration = Duration::from_secs(2);
/// The API wants ~100 ms pieces of audio.
const PIECE: usize = (MIC_RATE / 10) as usize;
const PIECE_TIME: Duration = Duration::from_millis(100);
/// A translated chunk quieter than this throughout is silence, not speech.
const SPEECH_PEAK: i32 = 300;
/// A press is reported once its translation has been quiet this long.
const REPORT_AFTER: Duration = Duration::from_secs(4);
/// A replaced session is closed this much later: its last words still play.
const RETIRE_AFTER: Duration = Duration::from_secs(6);
const OPEN_TIMEOUT: Duration = Duration::from_secs(15);
/// Opening includes the warm-up (~5 s, at most ~15 s).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(35);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(3);
const PING_EVERY: Duration = Duration::from_secs(20);
/// The warm-up's own answer is waited for at most this long.
const WARM_UP_LIMIT: Duration = Duration::from_secs(10);

/// 1.5 s of a deep male «Да, слушаю.» (16 kHz mono).
const PRIME_WAV: &[u8] = include_bytes!("../assets/prime_male.wav");

/// What a press produced, once its translation has gone quiet.
#[derive(Clone, Debug, Default)]
pub struct TurnReport {
    /// What the model heard (its transcription of the microphone).
    pub said: String,
    /// What it said in the headphones.
    pub heard: String,
    /// Seconds of translated speech (not silence).
    pub speech: f64,
    /// …of which after the press ended (release + tail).
    pub speech_after_end: f64,
    /// When the last piece of speech came, counted from the end of the press.
    pub last_speech_after_end: Option<f64>,
    /// When this report was made, counted from the end of the press.
    pub closed_after: f64,
}

/// Where the interpreter's output goes. Called from its network thread.
pub trait Listener: Send + Sync + 'static {
    /// A chunk of the translated voice, 24 kHz mono (often silence).
    fn voice(&self, pcm: &[i16]);
    /// A piece of the transcription of what the speaker says.
    fn said(&self, text: &str);
    /// A piece of the transcription of the translation.
    fn heard(&self, text: &str);
    fn turn_done(&self, report: &TurnReport);
    fn error(&self, message: &str);
}

// ── The handle the rest of the program uses ──────────────────────────────────

enum Command {
    Begin,
    Audio(Vec<i16>),
    End,
    Close,
}

pub struct GeminiInterpreter {
    commands: mpsc::UnboundedSender<Command>,
    /// Microphone audio not yet sent (< 100 ms).
    pending: Mutex<Vec<i16>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl GeminiInterpreter {
    /// Opens the first session; returns once it is ready. `warm_up`: start
    /// every session with the male warm-up voice (for the speaker's own voice;
    /// the other side's voice is copied as it is).
    pub fn connect(api_key: &str, target: &str, listener: Arc<dyn Listener>, warm_up: bool, meter: Arc<Meter>) -> Result<Self> {
        let shared = Arc::new(Shared {
            meter,
            key: api_key.to_owned(),
            target: target.to_owned(),
            prime: if warm_up { load_prime() } else { Vec::new() },
            closed: AtomicBool::new(false),
            listener,
        });
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = thread::Builder::new().name("gemini".into()).spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a tokio runtime");
            runtime.block_on(Controller::new(shared, command_rx).run(ready_tx));
        })?;
        let this = Self { commands, pending: Mutex::new(Vec::new()), thread: Mutex::new(Some(thread)) };
        let result = match ready_rx.recv_timeout(CONNECT_TIMEOUT) {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Gemini не ответил за {} с", CONNECT_TIMEOUT.as_secs())),
        };
        match result {
            Ok(()) => Ok(this),
            Err(e) => {
                this.close();
                Err(e)
            }
        }
    }

    pub fn start_turn(&self) {
        self.pending.lock().clear();
        let _ = self.commands.send(Command::Begin);
    }

    /// 16 kHz microphone audio of the current press, sent in 100 ms pieces.
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

    /// Closes every session, including a spare still warming up, and waits
    /// (≤ ~3 s) for the network thread to finish. Safe to call twice.
    pub fn close(&self) {
        let _ = self.commands.send(Command::Close);
        if let Some(thread) = self.thread.lock().take() {
            let _ = thread.join();
        }
    }
}

impl Drop for GeminiInterpreter {
    fn drop(&mut self) {
        self.close();
    }
}

fn load_prime() -> Vec<i16> {
    let Ok(reader) = hound::WavReader::new(Cursor::new(PRIME_WAV)) else {
        return Vec::new();
    };
    let spec = reader.spec();
    if spec.sample_rate != MIC_RATE || spec.channels != 1 || spec.bits_per_sample != 16 {
        return Vec::new();
    }
    reader.into_samples::<i16>().filter_map(|s| s.ok()).collect()
}

// ── Shared by every task ─────────────────────────────────────────────────────

struct Shared {
    key: String,
    target: String,
    prime: Vec<i16>,
    /// After close() no session may be (re)opened.
    closed: AtomicBool,
    listener: Arc<dyn Listener>,
    /// Where every session's spending goes.
    meter: Arc<Meter>,
}

impl Shared {
    fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

// ── A session ────────────────────────────────────────────────────────────────

enum Outgoing {
    Audio(Vec<i16>),
    EndOfPress,
    Close,
}

#[derive(Default)]
struct SessionState {
    /// Output is passed on only once a press has used the session; before
    /// that it is the answer to the warm-up.
    used: AtomicBool,
    open: AtomicBool,
    /// Google announced it will close the session (goAway).
    closing: AtomicBool,
    /// When the model last said something (the warm-up waits for silence).
    last_output: Mutex<Option<Instant>>,
}

#[derive(Clone)]
struct Session {
    id: u64,
    outgoing: mpsc::UnboundedSender<Outgoing>,
    state: Arc<SessionState>,
}

impl Session {
    fn usable(&self) -> bool {
        self.state.open.load(Ordering::Acquire) && !self.state.closing.load(Ordering::Acquire)
    }

    fn send(&self, message: Outgoing) {
        let _ = self.outgoing.send(message);
    }
}

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type Sink = SplitSink<Socket, Message>;

/// What the session tasks tell the controller.
enum Event {
    /// The socket is up (still warming up): from now on close() must reach it.
    Started(Session),
    Opened(u64, Result<Session, String>),
    Ended(u64),
    SendFailed(u64, String),
    /// Seconds of translated speech (not silence) from a used session.
    Speech(f64),
    Heard(String),
    Said(String),
    ReopenSpare,
}

fn audio_message(pcm: &[i16]) -> Message {
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    let message = json!({"realtimeInput": {"audio": {
        "data": BASE64.encode(bytes), "mimeType": format!("audio/pcm;rate={MIC_RATE}")}}});
    Message::text(message.to_string())
}

fn setup_message(target: &str) -> Message {
    let setup = json!({"setup": {
        "model": format!("models/{MODEL}"),
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "translationConfig": {"targetLanguageCode": target}},
        // The window shows both texts (measured: no effect on the delay).
        "inputAudioTranscription": {},
        "outputAudioTranscription": {}}});
    Message::text(setup.to_string())
}

fn message_text(message: Message) -> Option<String> {
    match message {
        Message::Text(text) => Some(text.to_string()),
        Message::Binary(bytes) => String::from_utf8(bytes.to_vec()).ok(),
        _ => None,
    }
}

/// Connects, sets the session up and starts its reader and writer; returns
/// once the warm-up is over.
async fn open_session(id: u64, shared: Arc<Shared>, events: mpsc::UnboundedSender<Event>) -> Result<Session> {
    if shared.closed() {
        bail!("closed");
    }
    let mut request = URL.into_client_request()?;
    request.headers_mut().insert("x-goog-api-key", HeaderValue::from_str(&shared.key)?);
    let connecting = tokio_tungstenite::connect_async_with_config(request, None, true);
    let mut socket = match timeout(OPEN_TIMEOUT, connecting).await {
        Err(_) => bail!("Gemini не отвечает"),
        Ok(Err(tungstenite::Error::Http(response))) => {
            let body = response.body().as_deref().map(String::from_utf8_lossy).unwrap_or_default();
            return Err(refusal(&shared, &format!("Gemini refused: HTTP {} {body}", response.status())));
        }
        Ok(Err(e)) => bail!("нет связи с Gemini: {e}"),
        Ok(Ok((socket, _))) => socket,
    };

    socket.send(setup_message(&shared.target)).await?;
    let reply = timeout(OPEN_TIMEOUT, async {
        while let Some(message) = socket.next().await {
            match message? {
                Message::Close(frame) => {
                    let reason = frame.map(|f| f.reason.to_string()).unwrap_or_default();
                    return Err(refusal(&shared, &format!("Gemini rejected the session: {reason}")));
                }
                other => {
                    if let Some(text) = message_text(other) {
                        return Ok(text);
                    }
                }
            }
        }
        bail!("Gemini closed the connection")
    })
    .await
    .map_err(|_| anyhow!("Gemini не ответил на настройку сессии"))??;
    if !serde_json::from_str::<Value>(&reply).is_ok_and(|v| v.get("setupComplete").is_some()) {
        let reply: String = reply.chars().take(200).collect();
        return Err(refusal(&shared, &format!("Gemini rejected the session: {reply}")));
    }
    if shared.closed() {
        let _ = socket.close(None).await;
        bail!("closed");
    }

    let (sink, stream) = socket.split();
    let (outgoing, outgoing_rx) = mpsc::unbounded_channel();
    let state = Arc::new(SessionState::default());
    state.open.store(true, Ordering::Release);
    let session = Session { id, outgoing, state };
    tokio::spawn(reader(session.clone(), stream, shared.clone(), events.clone()));
    let _ = events.send(Event::Started(session.clone()));
    let (ready_tx, ready_rx) = oneshot::channel();
    tokio::spawn(writer(session.clone(), sink, outgoing_rx, shared, events, ready_tx));
    ready_rx.await.unwrap_or_else(|_| Err(anyhow!("closed")))?;
    Ok(session)
}

/// A refusal from Google, in plain words when it is about money.
fn refusal(shared: &Shared, error: &str) -> anyhow::Error {
    if usage::is_out_of_money(error) {
        shared.meter.mark_exhausted();
        anyhow!("Баланс Gemini закончился — пополните его в AI Studio (Billing) и введите новую сумму в окне")
    } else if usage::is_quota(error) {
        anyhow!("Gemini: превышен лимит запросов или закончился баланс (429) — проверьте AI Studio")
    } else {
        anyhow!("{error}")
    }
}

async fn reader(session: Session, mut stream: SplitStream<Socket>, shared: Arc<Shared>, events: mpsc::UnboundedSender<Event>) {
    while let Some(message) = stream.next().await {
        match message {
            Ok(Message::Close(frame)) => {
                // Google closes a session with a reason when the money runs out.
                let reason = frame.map(|f| f.reason.to_string()).unwrap_or_default();
                if usage::is_out_of_money(&reason) || usage::is_quota(&reason) {
                    let error = refusal(&shared, &reason);
                    shared.listener.error(&format!("{error:#}"));
                }
                break;
            }
            Err(_) => break,
            Ok(other) => {
                if let Some(text) = message_text(other) {
                    handle_message(&session, &text, &shared, &events);
                }
            }
        }
    }
    session.state.open.store(false, Ordering::Release);
    let _ = events.send(Event::Ended(session.id));
}

fn handle_message(session: &Session, text: &str, shared: &Shared, events: &mpsc::UnboundedSender<Event>) {
    let Ok(message) = serde_json::from_str::<Value>(text) else {
        return;
    };
    // What this message's seconds of audio cost (an increment, about once a
    // second) — the warm-up and unused spares included: Google bills them too.
    if let Some(usage) = message.get("usageMetadata") {
        let tokens = |field| usage.get(field).and_then(Value::as_u64).unwrap_or(0);
        // A session that served a press is talk; before that, its warm-up.
        let kind = if session.state.used.load(Ordering::Acquire) { Kind::Talk } else { Kind::Service };
        shared.meter.charge(tokens("promptTokenCount"), tokens("responseTokenCount"), kind);
    }
    if message.get("goAway").is_some() {
        session.state.closing.store(true, Ordering::Release); // the next press goes to the spare
    }
    let Some(content) = message.get("serverContent") else {
        return;
    };
    if !session.state.used.load(Ordering::Acquire) {
        // The answer to the warm-up: not for the user, only timed.
        if content.get("modelTurn").is_some() || content.get("outputTranscription").is_some() {
            *session.state.last_output.lock() = Some(Instant::now());
        }
        return;
    }
    let parts = content.pointer("/modelTurn/parts").and_then(Value::as_array);
    for part in parts.into_iter().flatten() {
        let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str) else {
            continue;
        };
        let Ok(bytes) = BASE64.decode(data) else {
            continue;
        };
        let pcm: Vec<i16> = bytes.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect();
        if pcm.is_empty() {
            continue;
        }
        if dsp::peak(&pcm) >= SPEECH_PEAK {
            let _ = events.send(Event::Speech(pcm.len() as f64 / VOICE_RATE as f64));
        }
        shared.listener.voice(&pcm);
    }
    if let Some(text) = content.pointer("/outputTranscription/text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
        let _ = events.send(Event::Heard(text.to_owned()));
        shared.listener.heard(text);
    }
    if let Some(text) = content.pointer("/inputTranscription/text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
        let _ = events.send(Event::Said(text.to_owned()));
        shared.listener.said(text);
    }
}

async fn writer(
    session: Session,
    mut sink: Sink,
    mut outgoing: mpsc::UnboundedReceiver<Outgoing>,
    shared: Arc<Shared>,
    events: mpsc::UnboundedSender<Event>,
    ready: oneshot::Sender<Result<()>>,
) {
    let mut hush = Hush::new(session.id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5);
    let warmed = if shared.prime.is_empty() { Ok(()) } else { warm_up(&mut sink, &session, &shared, &mut hush).await };
    let warmed = warmed.and_then(|()| if shared.closed() { Err(anyhow!("closed")) } else { Ok(()) });
    if let Err(e) = warmed {
        let _ = sink.close().await;
        let _ = ready.send(Err(e));
        return;
    }
    let _ = ready.send(Ok(()));

    let mut ping = interval_at(tokio::time::Instant::now() + PING_EVERY, PING_EVERY);
    let result: Result<()> = async {
        loop {
            tokio::select! {
                message = outgoing.recv() => match message {
                    Some(Outgoing::Audio(pcm)) => sink.send(audio_message(&pcm)).await?,
                    Some(Outgoing::EndOfPress) => {
                        // A pause after the press, so the last phrase is voiced
                        // in full — cut short as soon as anything else is queued
                        // (the next press, or close).
                        for _ in 0..(FINISH_SILENCE.as_millis() / PIECE_TIME.as_millis()) {
                            if !outgoing.is_empty() {
                                break;
                            }
                            sink.send(audio_message(&hush.samples(PIECE, 20.0))).await?;
                            sleep(PIECE_TIME).await;
                        }
                    }
                    Some(Outgoing::Close) | None => {
                        sink.close().await?;
                        return Ok(());
                    }
                },
                _ = ping.tick() => sink.send(Message::Ping(Default::default())).await?,
            }
        }
    }
    .await;
    if let Err(e) = result {
        session.state.open.store(false, Ordering::Release);
        let _ = events.send(Event::SendFailed(session.id, format!("{e:#}")));
    }
}

/// Sends the warm-up voice, then a pause in real time so the model answers
/// it now (it answers only after hearing a pause), and waits until that
/// answer is over — it is dropped, never played.
async fn warm_up(sink: &mut Sink, session: &Session, shared: &Shared, hush: &mut Hush) -> Result<()> {
    let mut audio = shared.prime.clone();
    audio.extend(hush.samples(MIC_RATE as usize * 3, 20.0));
    let started = Instant::now();
    let mut pace = interval(PIECE_TIME);
    for piece in audio.chunks(PIECE) {
        pace.tick().await;
        if shared.closed() {
            bail!("closed");
        }
        sink.send(audio_message(piece)).await?;
    }
    while started.elapsed() < WARM_UP_LIMIT {
        if shared.closed() {
            bail!("closed");
        }
        let last = *session.state.last_output.lock();
        if last.is_some_and(|t| t.elapsed() > Duration::from_secs(1)) {
            break;
        }
        sleep(PIECE_TIME).await;
    }
    Ok(())
}

// ── The controller: the one owner of the session logic ───────────────────────

enum Spare {
    None,
    Opening(u64),
    Ready(Session),
}

/// What the press in progress (or just ended) has produced so far.
struct Turn {
    said: String,
    heard: String,
    speech: f64,
    after_end: f64,
    ended: Option<Instant>,
    last_speech: Option<f64>,
    last_activity: Option<Instant>,
}

impl Turn {
    fn new() -> Self {
        Self { said: String::new(), heard: String::new(), speech: 0.0, after_end: 0.0, ended: None, last_speech: None, last_activity: None }
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
    /// Where presses go.
    active: Option<Session>,
    /// A warm replacement.
    spare: Spare,
    /// The session being opened for a press that is waiting for it.
    adopting: Option<u64>,
    /// Messages of that press, queued meanwhile.
    backlog: Vec<Outgoing>,
    turn: Option<Turn>,
    /// Every session with a live socket, to close them all at the end.
    sessions: HashMap<u64, Session>,
    opening: HashMap<u64, tokio::task::JoinHandle<()>>,
    next_id: u64,
    /// The first session: connect() waits for it.
    first: Option<(u64, std::sync::mpsc::Sender<Result<()>>)>,
}

impl Controller {
    fn new(shared: Arc<Shared>, commands: mpsc::UnboundedReceiver<Command>) -> Self {
        let (events, event_rx) = mpsc::unbounded_channel();
        Self {
            shared,
            commands,
            events,
            event_rx,
            active: None,
            spare: Spare::None,
            adopting: None,
            backlog: Vec::new(),
            turn: None,
            sessions: HashMap::new(),
            opening: HashMap::new(),
            next_id: 1,
            first: None,
        }
    }

    async fn run(mut self, ready: std::sync::mpsc::Sender<Result<()>>) {
        let first = self.open();
        self.first = Some((first, ready));
        let mut tick = interval(Duration::from_millis(200));
        loop {
            tokio::select! {
                command = self.commands.recv() => match command {
                    Some(Command::Begin) => self.begin(),
                    Some(Command::Audio(pcm)) => self.put(Outgoing::Audio(pcm)),
                    Some(Command::End) => self.put(Outgoing::EndOfPress),
                    Some(Command::Close) | None => break,
                },
                Some(event) = self.event_rx.recv() => self.event(event),
                _ = tick.tick() => self.report_when_quiet(),
            }
        }
        self.shutdown().await;
    }

    fn log_error(&self, message: &str) {
        self.shared.listener.error(message);
    }

    fn open(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let (shared, events) = (self.shared.clone(), self.events.clone());
        let task = tokio::spawn(async move {
            let result = open_session(id, shared, events.clone()).await.map_err(|e| format!("{e:#}"));
            let _ = events.send(Event::Opened(id, result));
        });
        self.opening.insert(id, task);
        id
    }

    fn begin(&mut self) {
        self.report_turn(); // the previous press, if not reported yet
        let active_ok = self.active.as_ref().is_some_and(Session::usable);
        if self.adopting.is_none() && !active_ok {
            self.switch_to_spare();
        }
        self.turn = Some(Turn::new());
    }

    fn put(&mut self, message: Outgoing) {
        let Some(turn) = self.turn.as_mut() else {
            return; // between presses nothing is sent
        };
        if matches!(message, Outgoing::EndOfPress) {
            turn.ended = Some(Instant::now());
        }
        match &self.active {
            Some(session) => {
                session.state.used.store(true, Ordering::Release);
                session.send(message);
            }
            None => self.backlog.push(message),
        }
    }

    fn switch_to_spare(&mut self) {
        let old = self.active.take();
        match std::mem::replace(&mut self.spare, Spare::None) {
            Spare::Ready(session) if session.usable() => self.adopt(session),
            Spare::Opening(id) => self.adopting = Some(id),
            _ => self.adopting = Some(self.open()), // none yet, or one that expired
        }
        self.spare = Spare::Opening(self.open()); // the next replacement
        if let Some(old) = old {
            tokio::spawn(async move {
                sleep(RETIRE_AFTER).await; // let its last translation finish
                old.send(Outgoing::Close);
            });
        }
    }

    fn adopt(&mut self, session: Session) {
        session.state.used.store(true, Ordering::Release);
        for message in self.backlog.drain(..) {
            session.send(message); // audio queued while it was opening
        }
        self.active = Some(session);
        self.adopting = None;
    }

    fn event(&mut self, event: Event) {
        match event {
            Event::Started(session) => {
                self.sessions.insert(session.id, session);
            }
            Event::Opened(id, result) => self.opened(id, result),
            Event::Ended(id) => self.ended(id),
            Event::SendFailed(id, error) => {
                if self.active.as_ref().is_some_and(|s| s.id == id) {
                    self.log_error(&format!("связь с Gemini прервалась: {error}"));
                }
            }
            Event::Speech(seconds) => {
                if let Some(turn) = self.turn.as_mut() {
                    let now = Instant::now();
                    turn.speech += seconds;
                    turn.last_activity = Some(now);
                    if let Some(ended) = turn.ended {
                        turn.after_end += seconds;
                        turn.last_speech = Some((now - ended).as_secs_f64());
                    }
                }
            }
            Event::Heard(text) => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.heard.push_str(&text);
                    turn.last_activity = Some(Instant::now());
                }
            }
            Event::Said(text) => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.said.push_str(&text);
                }
            }
            Event::ReopenSpare => {
                if matches!(self.spare, Spare::None) {
                    self.spare = Spare::Opening(self.open());
                }
            }
        }
    }

    fn opened(&mut self, id: u64, result: Result<Session, String>) {
        self.opening.remove(&id);
        if self.first.as_ref().is_some_and(|(first, _)| *first == id) {
            let (_, reply) = self.first.take().expect("checked");
            match result {
                Ok(session) => {
                    self.active = Some(session);
                    self.spare = Spare::Opening(self.open());
                    let _ = reply.send(Ok(()));
                }
                Err(e) => {
                    let _ = reply.send(Err(anyhow!(e)));
                }
            }
        } else if self.adopting == Some(id) {
            match result {
                Ok(session) => self.adopt(session),
                Err(e) => {
                    self.log_error(&format!("нет связи с Gemini: {e}"));
                    self.adopting = None;
                    self.backlog.clear();
                }
            }
        } else if matches!(self.spare, Spare::Opening(spare) if spare == id) {
            self.spare = match result {
                Ok(session) => Spare::Ready(session),
                Err(e) => {
                    self.log_error(&format!("запасная сессия Gemini не открылась: {e}"));
                    Spare::None // the next press opens a fresh one
                }
            };
        } else if let Ok(session) = result {
            session.send(Outgoing::Close); // no longer needed
        }
    }

    fn ended(&mut self, id: u64) {
        self.sessions.remove(&id);
        if matches!(&self.spare, Spare::Ready(s) if s.id == id) {
            // A spare that expired unused (Google closes idle ones too).
            self.spare = Spare::None;
            let events = self.events.clone();
            tokio::spawn(async move {
                sleep(Duration::from_secs(1)).await;
                let _ = events.send(Event::ReopenSpare);
            });
        }
        let pressing = self.turn.as_ref().is_some_and(|t| t.ended.is_none());
        if pressing && self.adopting.is_none() && self.active.as_ref().is_some_and(|s| s.id == id) {
            // Closed in the middle of a press: the rest of it goes to the spare.
            self.log_error("Gemini закрыл сессию посреди фразы — продолжаю в запасной");
            self.switch_to_spare();
        }
    }

    fn report_when_quiet(&mut self) {
        let Some(turn) = self.turn.as_ref() else {
            return;
        };
        let Some(ended) = turn.ended else {
            return;
        };
        let last = turn.last_activity.unwrap_or(ended);
        if ended.elapsed() > REPORT_AFTER && last.elapsed() > REPORT_AFTER {
            self.report_turn();
        }
    }

    fn report_turn(&mut self) {
        let Some(turn) = self.turn.take() else {
            return;
        };
        let Some(ended) = turn.ended else {
            return;
        };
        self.shared.listener.turn_done(&TurnReport {
            said: clean(&turn.said),
            heard: clean(&turn.heard),
            speech: turn.speech,
            speech_after_end: turn.after_end,
            last_speech_after_end: turn.last_speech,
            closed_after: ended.elapsed().as_secs_f64(),
        });
    }

    /// Closes every session (a spare still warming up included) and waits
    /// until their sockets have closed, at most CLOSE_TIMEOUT.
    async fn shutdown(&mut self) {
        self.shared.closed.store(true, Ordering::Release);
        self.report_turn(); // the last press, if its report is still pending
        for (_, task) in self.opening.drain() {
            task.abort(); // still connecting; a warming one stops by itself
        }
        for session in self.sessions.values() {
            session.send(Outgoing::Close);
        }
        let deadline = tokio::time::Instant::now() + CLOSE_TIMEOUT;
        while !self.sessions.is_empty() {
            match timeout_at(deadline, self.event_rx.recv()).await {
                Ok(Some(Event::Ended(id))) => {
                    self.sessions.remove(&id);
                }
                Ok(Some(Event::Started(session))) => {
                    session.send(Outgoing::Close);
                    self.sessions.insert(session.id, session);
                }
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_warm_up_voice_is_built_in() {
        let prime = load_prime();
        assert!(prime.len() > MIC_RATE as usize, "{} samples", prime.len());
    }

    #[test]
    fn audio_goes_out_as_base64_pcm() {
        let Message::Text(text) = audio_message(&[1, -2]) else { panic!("text expected") };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["realtimeInput"]["audio"]["mimeType"], "audio/pcm;rate=16000");
        assert_eq!(BASE64.decode(v["realtimeInput"]["audio"]["data"].as_str().unwrap()).unwrap(), vec![1, 0, 0xFE, 0xFF]);
    }

    #[test]
    fn setup_asks_for_the_target_language_and_both_texts() {
        let Message::Text(text) = setup_message("uk") else { panic!("text expected") };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["setup"]["model"], format!("models/{MODEL}"));
        assert_eq!(v["setup"]["generationConfig"]["translationConfig"]["targetLanguageCode"], "uk");
        assert!(v["setup"].get("inputAudioTranscription").is_some());
    }

    #[test]
    fn spaces_are_tidied() {
        assert_eq!(clean("  Привет,   как \n дела "), "Привет, как дела");
    }
}

//! The microphone and the headphones (CoreAudio through cpal).
//!
//! Both devices run at their own rate — opening them at the model's rates
//! would make CoreAudio switch the device's rate for every other app too.
//! Audio is converted here: the microphone to 16 kHz in 20 ms chunks for the
//! model, the model's 24 kHz voice to the headphones' rate.
//!
//! Audio callbacks only move samples through a lock-free ring or a short
//! lock; all processing runs on ordinary threads.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use coreaudio::audio_unit::macos_helpers::audio_unit_from_device_id;
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::AudioUnit;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
use parking_lot::Mutex;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapProd, HeapRb};

use crate::dsp::{self, resample::StreamResampler};
use crate::echo::FarTap;
use crate::voice_mix::VoiceMixer;
use crate::journal;
use crate::virtual_mic;

/// The rate the model takes; the microphone is converted to it.
pub const MIC_RATE: u32 = 16000;
/// 20 ms: audio leaves for the model 20 ms after it is spoken.
pub const MIC_CHUNK: usize = (MIC_RATE / 50) as usize;
/// The rate the model speaks at.
pub const VOICE_RATE: u32 = 24000;

// ── Devices ──────────────────────────────────────────────────────────────────

fn device_name(device: &Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_owned())
}

/// Names of the microphones (`input`) or outputs, as the system lists them.
pub fn device_names(input: bool) -> Vec<String> {
    let host = cpal::default_host();
    let devices = if input { host.input_devices() } else { host.output_devices() };
    devices.map(|list| list.filter_map(|d| device_name(&d)).collect()).unwrap_or_default()
}

fn find_device(name: &str, input: bool) -> Option<Device> {
    let host = cpal::default_host();
    let devices = if input { host.input_devices() } else { host.output_devices() };
    devices.ok()?.find(|d| device_name(d).as_deref() == Some(name))
}

fn default_device(input: bool) -> Result<Device> {
    let host = cpal::default_host();
    let device = if input { host.default_input_device() } else { host.default_output_device() };
    device.ok_or_else(|| anyhow!(if input { "нет микрофона" } else { "нет устройства вывода" }))
}

/// The named device if it is there, then the system default: a device that
/// was unplugged (or won't open) must not stop the translator.
fn candidates(name: Option<&str>, input: bool) -> Vec<Result<Device>> {
    let mut list = Vec::new();
    if let Some(device) = name.and_then(|n| find_device(n, input)) {
        list.push(Ok(device));
    } else if let Some(name) = name {
        journal::log(&format!("device «{name}» not found; using the system default"));
    }
    list.push(default_device(input));
    list
}

fn first_that_opens<T>(name: Option<&str>, input: bool, mut open: impl FnMut(&Device) -> Result<T>) -> Result<T> {
    let mut last_error = anyhow!("нет устройства");
    for device in candidates(name, input) {
        match device.and_then(|d| open(&d)) {
            Ok(opened) => return Ok(opened),
            Err(e) => {
                journal::log(&format!("audio device failed: {e:#}"));
                last_error = e;
            }
        }
    }
    Err(last_error)
}

// ── Microphone ───────────────────────────────────────────────────────────────

/// Called with every 20 ms chunk of 16 kHz audio (it may change it in place).
pub type ChunkHandler = Box<dyn FnMut(&mut [f32]) + Send>;
/// Called with the microphone exactly as captured, at the device's own rate.
pub type RawHandler = Box<dyn FnMut(&[f32], u32) + Send>;

/// The microphone as 20 ms chunks of 16 kHz mono, handed to a handler on a
/// worker thread. Closed when dropped.
pub struct Microphone {
    pub name: String,
    // Field order matters: the stream stops first, then the worker is joined.
    _stream: Stream,
    _worker: Worker,
}

impl Microphone {
    /// The named microphone (`None` = the system default). `on_raw` also gets
    /// the audio before it is converted for the model.
    pub fn open(name: Option<&str>, on_chunk: ChunkHandler, on_raw: Option<RawHandler>) -> Result<Self> {
        // Shared by the attempts; only one of them ends up running.
        let on_chunk = Arc::new(Mutex::new(on_chunk));
        let on_raw = Arc::new(Mutex::new(on_raw));
        first_that_opens(name, true, |device| Self::open_device(device, on_chunk.clone(), on_raw.clone()))
    }

    fn open_device(device: &Device, on_chunk: Arc<Mutex<ChunkHandler>>, on_raw: Arc<Mutex<Option<RawHandler>>>) -> Result<Self> {
        let supported = device.default_input_config().context("микрофон недоступен")?;
        let config: StreamConfig = supported.config();
        let rate = config.sample_rate;
        let (producer, mut consumer) = HeapRb::<f32>::new(rate as usize).split();

        // device rate → 16 kHz → 20 ms chunks → the handler
        let worker = Worker::spawn("microphone", move |stop| {
            let mut resampler = StreamResampler::new(rate, MIC_RATE);
            let mut raw = vec![0.0f32; rate as usize / 50];
            let mut ready: Vec<f32> = Vec::with_capacity(MIC_CHUNK * 4);
            while !stop.load(Ordering::Acquire) {
                thread::park_timeout(Duration::from_millis(50));
                loop {
                    let n = consumer.pop_slice(&mut raw);
                    if n == 0 {
                        break;
                    }
                    if let Some(on_raw) = on_raw.lock().as_mut() {
                        on_raw(&raw[..n], rate);
                    }
                    resampler.process(&raw[..n], &mut ready);
                }
                let whole = ready.len() / MIC_CHUNK * MIC_CHUNK;
                if whole > 0 {
                    let mut handler = on_chunk.lock();
                    for chunk in ready[..whole].chunks_mut(MIC_CHUNK) {
                        handler(chunk);
                    }
                }
                ready.drain(..whole);
            }
        })?;

        let wake = worker.thread();
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build_input::<f32>(device, &config, producer, wake),
            SampleFormat::I16 => build_input::<i16>(device, &config, producer, wake),
            SampleFormat::I32 => build_input::<i32>(device, &config, producer, wake),
            other => Err(anyhow!("unsupported microphone format {other:?}")),
        }?;
        stream.play().context("микрофон не запускается")?;
        let name = device_name(device).unwrap_or_else(|| "?".into());
        Ok(Self { name, _stream: stream, _worker: worker })
    }
}

fn build_input<T>(device: &Device, config: &StreamConfig, mut ring: HeapProd<f32>, wake: thread::Thread) -> Result<Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let stream = device.build_input_stream::<T, _, _>(
        *config,
        move |data: &[T], _| {
            // The first channel, as the Python version (PortAudio, mono) took it.
            ring.push_iter(data.chunks(channels).map(|frame| frame[0].to_sample::<f32>() * 32768.0));
            wake.unpark();
        },
        |e| journal::log(&format!("microphone error: {e}")),
        None,
    )?;
    Ok(stream)
}

/// A thread that runs until told to stop (and is joined then).
pub(crate) struct Worker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub(crate) fn spawn(name: &str, body: impl FnOnce(&AtomicBool) + Send + 'static) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::Builder::new().name(name.into()).spawn(move || body(&flag))?;
        Ok(Self { stop, handle: Some(handle) })
    }

    pub(crate) fn thread(&self) -> thread::Thread {
        self.handle.as_ref().expect("running").thread().clone()
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Headphones ───────────────────────────────────────────────────────────────

/// A translated-voice chunk this quiet throughout is silence (about −40 dB).
const SILENCE_PEAK: i32 = 300;
/// Gemini's voice already comes in around −19 dBFS: played as it is.
const GAIN: f32 = 1.0;
/// Loudest sample allowed after the gain.
const CEILING: f32 = 32000.0;
/// The model streams chunks at about the pace they play, so one a few tens
/// of ms late would leave a gap mid-word. Holding this much back whenever
/// playback (re)starts absorbs that jitter.
const PREBUFFER: Duration = Duration::from_millis(200);
/// A safety net only: on a 60 s press the queue peaked at 3.5 s.
const MAX_LAG_SECONDS: f32 = 15.0;

/// When translated speech last went into the call (ms since START).
static CALL_VOICE_AT_MS: AtomicU64 = AtomicU64::new(0);
static START: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

/// The translation has been sounding in the call within `within` (the other
/// side's microphone may echo it back — it must not be translated again).
pub fn call_voice_within(within: Duration) -> bool {
    let at = CALL_VOICE_AT_MS.load(Ordering::Relaxed);
    at != 0 && START.elapsed().as_millis() as u64 <= at + within.as_millis() as u64
}

struct Queue {
    samples: VecDeque<f32>,
    playing: bool,       // false: waiting for the prebuffer to fill
    start_at: Instant,   // when a waiting queue may start playing
}

/// Plays the translated voice as it arrives.
///
/// Silent chunks are not queued at all: the model streams long stretches of
/// silence — between sentences and for many seconds after a phrase — and
/// queued silence would delay the next translation by as much.
pub struct Player {
    pub name: String,
    rate: u32,
    queue: Arc<Mutex<Queue>>,
    level_bits: Arc<AtomicU32>,
    resampler: Mutex<StreamResampler>,
    on_trim: Box<dyn Fn(f32) + Send + Sync>,
    // The Mutex only makes the CoreAudio unit shareable (it is Send, not Sync).
    _backend: Mutex<Backend>,
}

/// What actually plays: a device from the system's list, or the hidden input
/// of «PressVoice Microphone» (a CoreAudio output unit opened on it — the
/// device list doesn't show hidden devices).
/// Kept only to keep playing: dropping it stops the sound.
enum Backend {
    Device { _stream: Stream },
    VirtualMic { _unit: AudioUnit },
}

/// An output chosen but not started yet: its rate is needed first (the echo
/// canceller is built for it).
pub enum Output {
    Device { device: Device, config: StreamConfig, format: SampleFormat },
    VirtualMic { unit: AudioUnit, rate: u32 },
}

impl Output {
    /// The named output (`None` = the system default; `virtual_mic::NAME` =
    /// into the call, through the virtual microphone).
    pub fn find(name: Option<&str>) -> Result<Self> {
        if name == Some(virtual_mic::NAME) {
            return Self::virtual_mic();
        }
        first_that_opens(name, false, |device| {
            let supported = device.default_output_config().context("устройство вывода недоступно")?;
            Ok(Self::Device { device: device.clone(), config: supported.config(), format: supported.sample_format() })
        })
    }

    /// No fallback to the speakers here: the translation meant for the call
    /// must not quietly play in the room instead.
    fn virtual_mic() -> Result<Self> {
        let id = virtual_mic::feed_device()
            .ok_or_else(|| anyhow!("виртуальный микрофон «{}» не установлен (driver/install.sh)", virtual_mic::NAME))?;
        let unit = audio_unit_from_device_id(id, false)
            .map_err(|e| anyhow!("«{}» не открывается: {e}", virtual_mic::NAME))?;
        let rate = unit.sample_rate().map_err(|e| anyhow!("«{}»: {e}", virtual_mic::NAME))? as u32;
        Ok(Self::VirtualMic { unit, rate })
    }

    pub fn rate(&self) -> u32 {
        match self {
            Self::Device { config, .. } => config.sample_rate,
            Self::VirtualMic { rate, .. } => *rate,
        }
    }

    fn name(&self) -> String {
        match self {
            Self::Device { device, .. } => device_name(device).unwrap_or_else(|| "?".into()),
            Self::VirtualMic { .. } => virtual_mic::NAME.to_owned(),
        }
    }
}

impl Player {
    /// `far`: where every played block goes for the echo canceller.
    /// `voice`: the speaker's own voice, mixed under the translation.
    /// `on_trim(seconds)` reports if the safety net ever cuts old audio.
    pub fn start(
        output: Output,
        far: Option<FarTap>,
        voice: Option<VoiceMixer>,
        on_trim: impl Fn(f32) + Send + Sync + 'static,
    ) -> Result<Self> {
        let queue = Arc::new(Mutex::new(Queue { samples: VecDeque::new(), playing: false, start_at: Instant::now() }));
        let level_bits = Arc::new(AtomicU32::new((-120f32).to_bits()));
        let into_call = matches!(output, Output::VirtualMic { .. });
        let feed = Feed { queue: queue.clone(), level_bits: level_bits.clone(), far, voice, into_call, mono: Vec::new() };
        let (name, rate) = (output.name(), output.rate());
        let backend = match output {
            Output::Device { device, config, format } => {
                let stream = match format {
                    SampleFormat::F32 => build_output::<f32>(&device, &config, feed),
                    SampleFormat::I16 => build_output::<i16>(&device, &config, feed),
                    SampleFormat::I32 => build_output::<i32>(&device, &config, feed),
                    other => Err(anyhow!("unsupported output format {other:?}")),
                }?;
                stream.play().context("устройство вывода не запускается")?;
                Backend::Device { _stream: stream }
            }
            Output::VirtualMic { unit, .. } => Backend::VirtualMic { _unit: start_virtual_mic(unit, feed)? },
        };
        Ok(Self {
            name,
            rate,
            queue,
            level_bits,
            resampler: Mutex::new(StreamResampler::new(VOICE_RATE, rate)),
            on_trim: Box::new(on_trim),
            _backend: Mutex::new(backend),
        })
    }

    /// A chunk of the model's 24 kHz voice.
    pub fn play(&self, pcm: &[i16]) {
        let peak = dsp::peak(pcm);
        if peak < SILENCE_PEAK {
            return;
        }
        let gain = GAIN.min(CEILING / peak as f32);
        let voice: Vec<f32> = pcm.iter().map(|&x| x as f32 * gain).collect();
        let mut converted = Vec::with_capacity(voice.len() * self.rate as usize / VOICE_RATE as usize + 64);
        self.resampler.lock().process(&voice, &mut converted);

        let max = (MAX_LAG_SECONDS * self.rate as f32) as usize;
        let trimmed = {
            let mut q = self.queue.lock();
            if !q.playing && q.samples.is_empty() {
                q.start_at = Instant::now() + PREBUFFER;
            }
            q.samples.extend(converted);
            let extra = q.samples.len().saturating_sub(max);
            q.samples.drain(..extra);
            extra
        };
        if trimmed > 0 {
            (self.on_trim)(trimmed as f32 / self.rate as f32);
        }
    }

    /// Seconds of translation still waiting to be heard.
    pub fn queued_seconds(&self) -> f32 {
        self.queue.lock().samples.len() as f32 / self.rate as f32
    }

    /// Loudness of what is playing now (for the meter).
    pub fn level_db(&self) -> f32 {
        f32::from_bits(self.level_bits.load(Ordering::Relaxed))
    }
}

/// What the playback callback owns.
struct Feed {
    queue: Arc<Mutex<Queue>>,
    level_bits: Arc<AtomicU32>,
    far: Option<FarTap>,
    voice: Option<VoiceMixer>,
    /// Plays into the call (the virtual microphone).
    into_call: bool,
    mono: Vec<f32>,
}

impl Feed {
    /// Fills `mono` with the next `frames` samples (silence when idle).
    fn next_block(&mut self, frames: usize) {
        if self.mono.len() < frames {
            self.mono.resize(frames, 0.0); // only if the device asks for a bigger block than before
        }
        let got = {
            let mut q = self.queue.lock();
            if !q.playing && !q.samples.is_empty() && Instant::now() >= q.start_at {
                q.playing = true;
            }
            if q.playing {
                let n = frames.min(q.samples.len());
                for (dst, src) in self.mono.iter_mut().zip(q.samples.drain(..n)) {
                    *dst = src;
                }
                if n < frames {
                    q.playing = false; // ran dry: the next audio re-primes
                }
                n
            } else {
                0
            }
        };
        self.mono[got..frames].fill(0.0);
        if got > 0 && self.into_call {
            CALL_VOICE_AT_MS.store((START.elapsed().as_millis() as u64).max(1), Ordering::Relaxed);
        }
        if let Some(voice) = self.voice.as_mut() {
            voice.mix_into(&mut self.mono[..frames], got > 0);
        }
        let level = dsp::dbfs(&self.mono[..frames]);
        self.level_bits.store(level.to_bits(), Ordering::Relaxed);
        if let Some(far) = self.far.as_mut() {
            far.push(&self.mono[..frames]); // silence too: it keeps the timing
        }
    }
}

fn build_output<T>(device: &Device, config: &StreamConfig, mut feed: Feed) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let stream = device.build_output_stream::<T, _, _>(
        *config,
        move |data: &mut [T], _| {
            let frames = data.len() / channels;
            feed.next_block(frames);
            // The voice on every channel: both ears of the headphones.
            for (frame, &sample) in data.chunks_mut(channels).zip(feed.mono.iter()) {
                frame.fill(T::from_sample(sample / 32768.0));
            }
        },
        |e| journal::log(&format!("playback error: {e}")),
        None,
    )?;
    Ok(stream)
}

fn start_virtual_mic(mut unit: AudioUnit, mut feed: Feed) -> Result<AudioUnit> {
    type Args = render_callback::Args<data::NonInterleaved<f32>>;
    unit.set_render_callback(move |args: Args| {
        let Args { num_frames, mut data, .. } = args;
        feed.next_block(num_frames);
        // The same voice on both channels of the microphone.
        for channel in data.channels_mut() {
            for (dst, &sample) in channel.iter_mut().zip(&feed.mono[..num_frames]) {
                *dst = sample / 32768.0;
            }
        }
        Ok(())
    })
    .map_err(|e| anyhow!("«{}»: {e}", virtual_mic::NAME))?;
    unit.start().map_err(|e| anyhow!("«{}» не запускается: {e}", virtual_mic::NAME))?;
    Ok(unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs real devices: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn the_microphone_delivers_20_ms_chunks() {
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let seen = chunks.clone();
        let mic = Microphone::open(None, Box::new(move |chunk: &mut [f32]| seen.lock().push((chunk.len(), dsp::dbfs(chunk)))), None)
            .expect("microphone");
        thread::sleep(Duration::from_secs(1));
        drop(mic);
        let chunks = chunks.lock();
        println!("default microphone: {} chunks, loudest {:.1} dBFS", chunks.len(),
                 chunks.iter().map(|c| c.1).fold(-120.0, f32::max));
        assert!(chunks.len() >= 40, "only {} chunks in 1 s", chunks.len());
        assert!(chunks.iter().all(|c| c.0 == MIC_CHUNK));
    }

    /// Needs the driver installed and microphone access: what is played into the
    /// virtual microphone comes out of «PressVoice Microphone», as Teams hears it.
    #[test]
    #[ignore]
    fn the_virtual_mic_carries_the_translation() {
        let player = Player::start(Output::find(Some(virtual_mic::NAME)).expect("virtual mic"), None, None, |_| {}).expect("player");
        let heard = Arc::new(Mutex::new(Vec::new()));
        let sink = heard.clone();
        let mic = Microphone::open(Some(virtual_mic::NAME), Box::new(move |chunk: &mut [f32]| sink.lock().push(dsp::dbfs(chunk))), None)
            .expect("PressVoice Microphone as an input");
        assert_eq!(mic.name, virtual_mic::NAME);
        // 1 s of a 440 Hz tone at about −20 dBFS, as the model would send it.
        let tone: Vec<i16> = (0..VOICE_RATE)
            .map(|i| (3300.0 * (std::f32::consts::TAU * 440.0 * i as f32 / VOICE_RATE as f32).sin()) as i16)
            .collect();
        player.play(&tone);
        thread::sleep(Duration::from_millis(1300));
        drop(mic);
        let levels = heard.lock();
        let loudest = levels.iter().copied().fold(-120.0, f32::max);
        println!("PressVoice Microphone: {} chunks, loudest {loudest:.1} dBFS", levels.len());
        assert!(loudest > -26.0 && loudest < -20.0, "loudest {loudest:.1} dBFS");
    }

    /// Needs the driver: my own voice reaches the call at the chosen volume.
    #[test]
    #[ignore]
    fn my_voice_goes_into_the_call_at_its_volume() {
        use crate::voice_mix::{MixSettings, VoiceMix, live_voice};
        let output = Output::find(Some(virtual_mic::NAME)).expect("virtual mic");
        let rate = output.rate();
        let mix = MixSettings { level: 0.5, auto_duck: true, duck_to: 0.3, hold: Duration::from_millis(700) };
        let (mut tap, mixer) = live_voice(rate, VoiceMix::new(mix));
        let _player = Player::start(output, None, Some(mixer), |_| {}).expect("player");
        let heard = Arc::new(Mutex::new(Vec::new()));
        let sink = heard.clone();
        let mic = Microphone::open(Some(virtual_mic::NAME), Box::new(move |chunk: &mut [f32]| sink.lock().push(dsp::dbfs(chunk))), None)
            .expect("PressVoice Microphone as an input");
        // 1 s of "my voice": a 200 Hz tone at about −20 dBFS, in 10 ms blocks, in real time.
        let block = rate as usize / 100;
        for n in 0..100 {
            let voice: Vec<f32> = (0..block)
                .map(|i| 3300.0 * (std::f32::consts::TAU * 200.0 * (n * block + i) as f32 / rate as f32).sin())
                .collect();
            tap.push(&voice, rate, true);
            thread::sleep(Duration::from_millis(10));
        }
        thread::sleep(Duration::from_millis(200));
        drop(mic);
        let levels = heard.lock();
        let loudest = levels.iter().copied().fold(-120.0, f32::max);
        println!("my voice in the call: loudest {loudest:.1} dBFS (sent −22.9 at 50% → −28.9)");
        assert!((loudest + 28.9).abs() < 1.5, "loudest {loudest:.1} dBFS");
    }

    #[test]
    #[ignore]
    fn the_player_plays_and_reports_its_level() {
        let player = Player::start(Output::find(None).expect("output"), None, None, |_| {}).expect("player");
        // 0.5 s of a quiet 440 Hz tone (−40 dBFS), as the model would send it.
        let tone: Vec<i16> = (0..VOICE_RATE / 2)
            .map(|i| (330.0 * (std::f32::consts::TAU * 440.0 * i as f32 / VOICE_RATE as f32).sin()) as i16)
            .collect();
        player.play(&tone);
        assert!(player.queued_seconds() > 0.4);
        thread::sleep(Duration::from_millis(400));
        let level = player.level_db();
        assert!(level > -50.0 && level < -35.0, "level {level}");
        thread::sleep(Duration::from_millis(500));
        assert!(player.queued_seconds() < 0.01);
    }
}

//! `--simulate in.wav [--to xx] [--deeper n] [--save out.wav]`: one press
//! made of a recording instead of the microphone — the same path to Gemini
//! (voice lowered, 20 ms chunks in real time, release tail, report), with
//! the translation's text printed and its voice optionally saved.

use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;

use crate::audio::{MIC_CHUNK, MIC_RATE, VOICE_RATE};
use crate::dsp::pitch::{PitchShifter, deeper_ratio};
use crate::dsp::resample::StreamResampler;
use crate::dsp::{self, Hush};
use crate::gemini::{self, GeminiInterpreter, Listener, TurnReport};

#[derive(Default)]
struct Recorder {
    voice: Mutex<Vec<i16>>,
    report: Mutex<Option<TurnReport>>,
    /// When the first text of the speaker, of the translation, and the
    /// first translated speech (not silence) arrived.
    first_said: Mutex<Option<Instant>>,
    first_heard: Mutex<Option<Instant>>,
    first_speech: Mutex<Option<Instant>>,
    last_speech: Mutex<Option<Instant>>,
    /// When each piece of translated speech (not silence) arrived, and how long it is.
    speech_log: Mutex<Vec<(Instant, f64)>>,
}

fn mark_first(slot: &Mutex<Option<Instant>>) {
    slot.lock().get_or_insert_with(Instant::now);
}

impl Listener for Recorder {
    fn voice(&self, pcm: &[i16]) {
        if dsp::peak(pcm) >= 300 {
            mark_first(&self.first_speech);
            *self.last_speech.lock() = Some(Instant::now());
            self.speech_log.lock().push((Instant::now(), pcm.len() as f64 / VOICE_RATE as f64));
        }
        self.voice.lock().extend_from_slice(pcm);
    }
    fn said(&self, _text: &str) {
        mark_first(&self.first_said);
    }
    fn heard(&self, _text: &str) {
        mark_first(&self.first_heard);
    }
    fn turn_done(&self, report: &TurnReport) {
        *self.report.lock() = Some(report.clone());
    }
    fn error(&self, message: &str) {
        eprintln!("error: {message}");
    }
}

fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// Any WAV as 16 kHz mono on the 16-bit scale.
fn read_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("не читается {path}"))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 32768.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.map(|v| v * 32768.0)).collect::<Result<_, _>>()?,
    };
    let mono: Vec<f32> = samples.chunks(spec.channels as usize).map(|f| f[0]).collect();
    let mut out = Vec::new();
    StreamResampler::new(spec.sample_rate, MIC_RATE).process(&mono, &mut out);
    if out.is_empty() {
        bail!("пустой файл {path}");
    }
    Ok(out)
}

pub fn run(api_key: &str, args: &[String]) -> ExitCode {
    match simulate(api_key, args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e:#}");
            ExitCode::FAILURE
        }
    }
}

fn simulate(api_key: &str, args: &[String]) -> Result<()> {
    let input = args.first().context("нужен WAV-файл")?;
    let target = option(args, "--to").unwrap_or("en");
    let deeper: f32 = option(args, "--deeper").unwrap_or("3").parse()?;
    let speech = read_wav(input)?;

    let recorder = Arc::new(Recorder::default());
    let meter = crate::usage::Meter::detached();
    meter.set_balance(100.0);
    let started = Instant::now();
    let interp = GeminiInterpreter::connect(api_key, target, recorder.clone(), true, meter.clone())?;
    println!("сессия готова за {:.1}s (с разогревом)", started.elapsed().as_secs_f64());

    let mut shifter = PitchShifter::new(deeper_ratio(deeper), MIC_RATE);
    let mut hush = Hush::new(1);
    interp.start_turn();
    let pressed = Instant::now();
    // Real time by the clock (sleeping 20 ms per chunk drifts behind: a 7 s
    // phrase took 9.4 s, which made every delay look longer).
    let chunk_time = Duration::from_millis(20);
    let mut due = Instant::now();
    for chunk in speech.chunks(MIC_CHUNK) {
        interp.feed(&dsp::to_i16(&shifter.process(chunk)));
        due += chunk_time;
        thread::sleep(due.saturating_duration_since(Instant::now()));
    }
    let spoken = Instant::now(); // the end of the speech itself
    // A real microphone keeps delivering quiet room sound during the tail.
    let tail_chunks = (gemini::RELEASE_TAIL.as_millis() / 20) as usize;
    for _ in 0..tail_chunks {
        interp.feed(&hush.samples(MIC_CHUNK, 20.0));
        due += chunk_time;
        thread::sleep(due.saturating_duration_since(Instant::now()));
    }
    interp.end_turn();
    let voice_at_release = recorder.voice.lock().len();
    println!("нажатие {:.1}s; перевода во время нажатия: {:.1}s звука",
             pressed.elapsed().as_secs_f64(), voice_at_release as f64 / VOICE_RATE as f64);

    // The report comes once the translation has been quiet for 4 s.
    for _ in 0..200 {
        if recorder.report.lock().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    interp.close();

    let since = |slot: &Mutex<Option<Instant>>| {
        slot.lock().map_or("—".to_owned(), |t| format!("{:.2}s", (t - pressed).as_secs_f64()))
    };
    println!("от начала речи: текст речи {}, текст перевода {}, голос перевода {}",
             since(&recorder.first_said), since(&recorder.first_heard), since(&recorder.first_speech));
    if let Some(last) = *recorder.last_speech.lock() {
        println!("последний звук перевода через {:.2}s после конца речи", (last - spoken).as_secs_f64());
    }
    let snap = meter.snapshot();
    let spent = 100.0 - snap.balance.unwrap_or(100.0);
    let per_second = crate::usage::COST_PER_SECOND;
    println!("списано по отчётам Gemini: ${spent:.4} ≈ {:.1} с — из них разговор {:.1} с, служебное (разогрев) {:.1} с",
             spent / per_second, (spent - snap.service) / per_second, snap.service / per_second);
    match recorder.report.lock().as_ref() {
        Some(r) => {
            println!("сказал:  {}", r.said);
            println!("перевод: {}", r.heard);
            println!("речь перевода {:.1}s, после отпускания {:.1}s", r.speech, r.speech_after_end);
        }
        None => println!("отчёта нет (перевод не затих за 20 с)"),
    }
    if args.iter().any(|a| a == "--json") {
        // Everything measured, on one line (for comparing the engines).
        let secs = |slot: &Mutex<Option<Instant>>| slot.lock().map(|t| (t - pressed).as_secs_f64());
        let log = recorder.speech_log.lock();
        let during: f64 = log.iter().filter(|(t, _)| *t <= spoken).map(|(_, s)| s).sum();
        let total: f64 = log.iter().map(|(_, s)| s).sum();
        let line = serde_json::json!({
            "phrase": (spoken - pressed).as_secs_f64(),
            "first_said": secs(&recorder.first_said),
            "first_heard": secs(&recorder.first_heard),
            "first_voice": secs(&recorder.first_speech),
            "end_after": recorder.last_speech.lock().map(|t| (t - spoken).as_secs_f64()),
            "voice_during": during,
            "voice_total": total,
            "said": recorder.report.lock().as_ref().map_or(String::new(), |r| r.said.clone()),
            "heard": recorder.report.lock().as_ref().map_or(String::new(), |r| r.heard.clone()),
            "cost": meter.snapshot().balance.map_or(0.0, |b| 100.0 - b),
        });
        println!("JSON {line}");
    }
    if let Some(path) = option(args, "--save") {
        let spec = hound::WavSpec { channels: 1, sample_rate: VOICE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for &s in recorder.voice.lock().iter() {
            writer.write_sample(s)?;
        }
        writer.finalize()?;
        println!("голос перевода сохранён: {path}");
    }
    Ok(())
}

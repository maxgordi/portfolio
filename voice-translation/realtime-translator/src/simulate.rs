//! `--simulate in.wav [--from ru] [--to en] [--voice Adrian | --text-only] [--speed 1.2] [--save out.wav]`:
//! one press made of a recording instead of the microphone — the same path
//! to Soniox (20 ms chunks in real time, release tail, finalize), with the
//! timing of the text, the translation and the voice, and what it cost.

use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;

use crate::audio::{MIC_CHUNK, MIC_RATE, VOICE_RATE};
use crate::dsp::{self, Hush};
use crate::dsp::resample::StreamResampler;
use crate::soniox::{self, Interpreter, Listener, Setup, TurnReport};
use crate::usage::Meter;

#[derive(Default)]
struct Recorder {
    voice: Mutex<Vec<i16>>,
    said: Mutex<String>,
    heard: Mutex<String>,
    report: Mutex<Option<TurnReport>>,
    first_said: Mutex<Option<Instant>>,
    first_heard: Mutex<Option<Instant>>,
    first_voice: Mutex<Option<Instant>>,
    last_voice: Mutex<Option<Instant>>,
    /// When each piece of translated speech (not silence) arrived, and how long it is.
    speech_log: Mutex<Vec<(Instant, f64)>>,
    /// When each piece of text arrived: (when, "said" | "heard", text).
    text_log: Mutex<Vec<(Instant, &'static str, String)>>,
}

fn mark_first(slot: &Mutex<Option<Instant>>) {
    slot.lock().get_or_insert_with(Instant::now);
}

impl Listener for Recorder {
    fn voice(&self, pcm: &[i16]) {
        if dsp::peak(pcm) >= 300 {
            mark_first(&self.first_voice);
            *self.last_voice.lock() = Some(Instant::now());
            self.speech_log.lock().push((Instant::now(), pcm.len() as f64 / VOICE_RATE as f64));
        }
        self.voice.lock().extend_from_slice(pcm);
    }
    fn said(&self, text: &str) {
        mark_first(&self.first_said);
        self.said.lock().push_str(text);
        self.text_log.lock().push((Instant::now(), "said", text.to_owned()));
    }
    fn heard(&self, text: &str) {
        mark_first(&self.first_heard);
        self.heard.lock().push_str(text);
        self.text_log.lock().push((Instant::now(), "heard", text.to_owned()));
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
    let setup = Setup {
        target: option(args, "--to").unwrap_or("en").to_owned(),
        source: option(args, "--from").unwrap_or("ru").to_owned(),
        voice: if args.iter().any(|a| a == "--text-only") {
            None
        } else {
            Some(option(args, "--voice").unwrap_or(soniox::DEFAULT_VOICE).to_owned())
        },
        speed: option(args, "--speed").map_or(Ok(1.0), str::parse)?,
    };
    let speech = read_wav(input)?;
    let recorder = Arc::new(Recorder::default());
    let meter = Meter::detached();
    meter.set_balance(100.0);
    let started = Instant::now();
    let interp = Interpreter::connect(api_key, setup.clone(), recorder.clone(), meter.clone())?;
    println!("сессия готова за {:.1}s · {} → {} · {}", started.elapsed().as_secs_f64(), setup.source, setup.target,
             setup.voice.as_deref().map_or("только текст".to_owned(), |v| format!("голос {v}")));

    let mut hush = Hush::new(1);
    interp.start_turn();
    let pressed = Instant::now();
    // Real time by the clock (sleeping 20 ms per chunk drifts behind: a 7 s
    // phrase took 9.4 s, which made every delay look longer).
    let chunk_time = Duration::from_millis(20);
    let mut due = Instant::now();
    for chunk in speech.chunks(MIC_CHUNK) {
        interp.feed(&dsp::to_i16(chunk));
        due += chunk_time;
        thread::sleep(due.saturating_duration_since(Instant::now()));
    }
    let spoken = Instant::now(); // the end of the speech itself
    for _ in 0..(soniox::RELEASE_TAIL.as_millis() / 20) {
        interp.feed(&hush.samples(MIC_CHUNK, 20.0));
        due += chunk_time;
        thread::sleep(due.saturating_duration_since(Instant::now()));
    }
    interp.end_turn();
    for _ in 0..200 {
        if recorder.report.lock().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    interp.close();

    let since = |slot: &Mutex<Option<Instant>>| slot.lock().map_or("—".to_owned(), |t| format!("{:.2}s", (t - pressed).as_secs_f64()));
    println!("фраза {:.1}s; от начала речи: текст {}, перевод {}, голос {}",
             (spoken - pressed).as_secs_f64(), since(&recorder.first_said), since(&recorder.first_heard), since(&recorder.first_voice));
    if let Some(last) = *recorder.last_voice.lock() {
        println!("голос перевода закончился через {:.2}s после конца речи", (last - spoken).as_secs_f64());
    }
    println!("сказал:  {}", recorder.said.lock().trim());
    println!("перевод: {}", recorder.heard.lock().trim());
    let spent = 100.0 - meter.snapshot().balance.unwrap_or(100.0);
    println!("потрачено: ${spent:.5} (голоса {:.1}s)", recorder.voice.lock().len() as f64 / VOICE_RATE as f64);
    if args.iter().any(|a| a == "--json") {
        // Everything measured, on one line (for comparing the engines).
        let secs = |slot: &Mutex<Option<Instant>>| slot.lock().map(|t| (t - pressed).as_secs_f64());
        let log = recorder.speech_log.lock();
        let during: f64 = log.iter().filter(|(t, _)| *t <= spoken).map(|(_, s)| s).sum();
        let total: f64 = log.iter().map(|(_, s)| s).sum();
        let text: Vec<_> = recorder.text_log.lock().iter()
            .map(|(t, side, piece)| serde_json::json!([(*t - pressed).as_secs_f64(), side, piece]))
            .collect();
        let speech: Vec<_> = log.iter().map(|(t, s)| serde_json::json!([(*t - pressed).as_secs_f64(), s])).collect();
        let line = serde_json::json!({
            "phrase": (spoken - pressed).as_secs_f64(),
            "first_said": secs(&recorder.first_said),
            "first_heard": secs(&recorder.first_heard),
            "first_voice": secs(&recorder.first_voice),
            "end_after": recorder.last_voice.lock().map(|t| (t - spoken).as_secs_f64()),
            "voice_during": during,
            "voice_total": total,
            "said": recorder.said.lock().trim().to_owned(),
            "heard": recorder.heard.lock().trim().to_owned(),
            "cost": meter.snapshot().balance.map_or(0.0, |b| 100.0 - b),
            "text": text,
            "speech": speech,
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

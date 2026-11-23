//! The speaker's own voice in the call, under the translation.
//!
//! When the translation goes into «PressVoice Microphone», the other side can
//! also hear the speaker's real voice at a chosen volume — only while the key
//! is held. While the translator speaks, the real voice is turned down further
//! ("auto-ducking"), so the translation stays clear; once it falls silent the
//! voice comes back after a hold time. The translation itself is never touched.
//!
//! The microphone worker pushes the raw voice, converted to the output's rate,
//! into a lock-free ring (`VoiceTap`); the output callback takes it, applies the
//! volume and the ducking, and adds it to the translation (`VoiceMixer`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};

use crate::dsp::resample::StreamResampler;

/// The voice is ducked within ~20 ms of the translation starting…
const ATTACK: Duration = Duration::from_millis(20);
/// …and, after the hold, comes back over ~120 ms (no jump in loudness).
const RELEASE: Duration = Duration::from_millis(120);
/// The voice may lag the microphone by at most this much (clock drift between
/// the microphone and the output is dropped here rather than piling up).
const MAX_BACKLOG: Duration = Duration::from_millis(60);

/// The window's choices for the voice in the call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixSettings {
    /// Volume of the speaker's voice in the call, 0 = off (translation only).
    pub level: f32,
    pub auto_duck: bool,
    /// While the translator speaks, the voice is at this fraction of `level`.
    pub duck_to: f32,
    /// How long after the translation falls silent the voice stays ducked.
    pub hold: Duration,
}

/// The settings, changeable while audio plays.
pub struct VoiceMix {
    level: AtomicU32,
    auto_duck: AtomicBool,
    duck_to: AtomicU32,
    hold_ms: AtomicU32,
}

impl VoiceMix {
    pub fn new(settings: MixSettings) -> Arc<Self> {
        let mix = Arc::new(Self {
            level: AtomicU32::new(0),
            auto_duck: AtomicBool::new(false),
            duck_to: AtomicU32::new(0),
            hold_ms: AtomicU32::new(0),
        });
        mix.set(settings);
        mix
    }

    pub fn set(&self, s: MixSettings) {
        self.level.store(s.level.to_bits(), Ordering::Relaxed);
        self.auto_duck.store(s.auto_duck, Ordering::Relaxed);
        self.duck_to.store(s.duck_to.to_bits(), Ordering::Relaxed);
        self.hold_ms.store(s.hold.as_millis() as u32, Ordering::Relaxed);
    }

    fn get(&self) -> MixSettings {
        MixSettings {
            level: f32::from_bits(self.level.load(Ordering::Relaxed)),
            auto_duck: self.auto_duck.load(Ordering::Relaxed),
            duck_to: f32::from_bits(self.duck_to.load(Ordering::Relaxed)),
            hold: Duration::from_millis(self.hold_ms.load(Ordering::Relaxed) as u64),
        }
    }
}

/// The two ends: the microphone side and the output side, at `out_rate`.
pub fn live_voice(out_rate: u32, mix: Arc<VoiceMix>) -> (VoiceTap, VoiceMixer) {
    let (producer, consumer) = HeapRb::<f32>::new(out_rate as usize).split();
    let tap = VoiceTap { ring: producer, resampler: None, out_rate, converted: Vec::new() };
    let per_second = |d: Duration| 1.0 - (-1.0 / (d.as_secs_f32() * out_rate as f32)).exp();
    let mixer = VoiceMixer {
        ring: consumer,
        mix,
        rate: out_rate,
        gain: 1.0,
        quiet_for: usize::MAX,
        attack: per_second(ATTACK),
        release: per_second(RELEASE),
        voice: Vec::new(),
    };
    (tap, mixer)
}

/// The microphone side (runs on the microphone's worker thread).
pub struct VoiceTap {
    ring: HeapProd<f32>,
    /// Built on the first block, once the microphone's rate is known.
    resampler: Option<(u32, StreamResampler)>,
    out_rate: u32,
    converted: Vec<f32>,
}

impl VoiceTap {
    /// Raw microphone audio at its own `rate`; `open` = the key is held (only
    /// then does the voice go into the call).
    pub fn push(&mut self, samples: &[f32], rate: u32, open: bool) {
        if !open {
            return;
        }
        if self.resampler.as_ref().is_none_or(|(r, _)| *r != rate) {
            self.resampler = Some((rate, StreamResampler::new(rate, self.out_rate)));
        }
        let (_, resampler) = self.resampler.as_mut().expect("just set");
        self.converted.clear();
        resampler.process(samples, &mut self.converted);
        self.ring.push_slice(&self.converted);
    }
}

/// The output side (runs in the output's real-time callback: no locks, no
/// allocation once warmed up).
pub struct VoiceMixer {
    ring: HeapCons<f32>,
    mix: Arc<VoiceMix>,
    rate: u32,
    /// The ducking envelope: 1 = full voice, `duck_to` = ducked.
    gain: f32,
    /// Samples since the translation last sounded.
    quiet_for: usize,
    attack: f32,
    release: f32,
    voice: Vec<f32>,
}

impl VoiceMixer {
    /// Adds the voice to `block` (the translation, 16-bit scale).
    /// `translating`: the block holds translated speech.
    pub fn mix_into(&mut self, block: &mut [f32], translating: bool) {
        let max_backlog = (MAX_BACKLOG.as_secs_f32() * self.rate as f32) as usize;
        let backlog = self.ring.occupied_len();
        if backlog > max_backlog + block.len() {
            self.ring.skip(backlog - max_backlog);
        }
        if self.voice.len() < block.len() {
            self.voice.resize(block.len(), 0.0);
        }
        let got = self.ring.pop_slice(&mut self.voice[..block.len()]);
        self.voice[got..block.len()].fill(0.0);

        let s = self.mix.get();
        if translating {
            self.quiet_for = 0;
        } else {
            self.quiet_for = self.quiet_for.saturating_add(block.len());
        }
        let hold = (s.hold.as_secs_f32() * self.rate as f32) as usize;
        let ducked = s.auto_duck && self.quiet_for < hold.max(1);
        let target = if ducked { s.duck_to } else { 1.0 };
        let speed = if target < self.gain { self.attack } else { self.release };
        let n = block.len();
        for (out, &voice) in block.iter_mut().zip(&self.voice[..n]) {
            self.gain += (target - self.gain) * speed;
            *out = (*out + voice * s.level * self.gain).clamp(-32768.0, 32767.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48000;
    const BLOCK: usize = 480; // 10 ms

    fn settings() -> MixSettings {
        MixSettings { level: 0.5, auto_duck: true, duck_to: 0.3, hold: Duration::from_millis(300) }
    }

    /// Runs `blocks` 10 ms blocks of a constant voice (1000) through the mixer;
    /// returns the voice level in the last block.
    fn run(tap: &mut VoiceTap, mixer: &mut VoiceMixer, blocks: usize, translating: bool) -> f32 {
        let mut last = 0.0;
        for _ in 0..blocks {
            tap.push(&[1000.0; BLOCK], RATE, true);
            let mut block = [0.0f32; BLOCK];
            mixer.mix_into(&mut block, translating);
            last = block[BLOCK - 1];
        }
        last
    }

    #[test]
    fn the_voice_plays_at_its_volume_and_ducks_under_the_translation() {
        let (mut tap, mut mixer) = live_voice(RATE, VoiceMix::new(settings()));
        let alone = run(&mut tap, &mut mixer, 20, false);
        assert!((alone - 500.0).abs() < 5.0, "voice alone {alone}");
        let ducked = run(&mut tap, &mut mixer, 10, true);
        assert!((ducked - 150.0).abs() < 5.0, "under the translation {ducked}");
        // Still ducked within the hold (0.3 s) after the translation stops…
        let held = run(&mut tap, &mut mixer, 20, false);
        assert!((held - 150.0).abs() < 5.0, "during the hold {held}");
        // …then back to full.
        let back = run(&mut tap, &mut mixer, 80, false);
        assert!((back - 500.0).abs() < 5.0, "after the hold {back}");
    }

    #[test]
    fn without_auto_ducking_the_voice_stays_level() {
        let (mut tap, mut mixer) = live_voice(RATE, VoiceMix::new(MixSettings { auto_duck: false, ..settings() }));
        let under = run(&mut tap, &mut mixer, 20, true);
        assert!((under - 500.0).abs() < 5.0, "{under}");
    }

    #[test]
    fn nothing_goes_into_the_call_while_the_key_is_up() {
        let (mut tap, mut mixer) = live_voice(RATE, VoiceMix::new(settings()));
        tap.push(&[1000.0; BLOCK], RATE, false);
        let mut block = [0.0f32; BLOCK];
        mixer.mix_into(&mut block, false);
        assert!(block.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn a_lagging_voice_does_not_pile_up() {
        let (mut tap, mut mixer) = live_voice(RATE, VoiceMix::new(settings()));
        // The microphone delivered 0.5 s while the output was stalled.
        for _ in 0..50 {
            tap.push(&[1000.0; BLOCK], RATE, true);
        }
        let mut block = [0.0f32; BLOCK];
        mixer.mix_into(&mut block, false);
        assert!(mixer.ring.occupied_len() <= (MAX_BACKLOG.as_secs_f32() * RATE as f32) as usize);
    }
}

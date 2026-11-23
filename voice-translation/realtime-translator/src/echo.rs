//! Echo cancellation: takes out of the microphone what the headphones play.
//!
//! The translation in the headphones can reach the microphone — through the
//! air from loud ear cups, or as vibration through the headset itself — and
//! the model then translates its own voice again. This is WebRTC's echo
//! canceller (AEC3, as in Chrome and Google Meet, via livekit's libwebrtc —
//! the same one the Python version uses). It is given every block the
//! headphones play (the "far" signal) and every block the microphone
//! captures, finds the delay between them itself and subtracts the echo.
//! Nothing else is switched on: no noise suppression, no automatic gain, no
//! high-pass — the voice stays as it is.
//!
//! The playback callback only pushes what it played into a lock-free ring
//! (`FarTap`); all the work happens on the microphone thread in `process`.

use libwebrtc::native::apm::AudioProcessingModule;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};

use crate::dsp::resample::StreamResampler;

/// The rate the canceller works at (the microphone's rate for the model).
const RATE: u32 = 16000;
const FRAME: usize = (RATE / 100) as usize; // 10 ms
/// Starting guess of playback → capture delay; AEC3 refines it.
const DELAY_MS: i32 = 40;

/// The playback side's end: every block the headphones play goes in here.
pub struct FarTap(HeapProd<f32>);

impl FarTap {
    /// Real-time safe: never blocks or allocates; drops audio if the
    /// microphone thread has fallen 2 s behind.
    pub fn push(&mut self, samples: &[f32]) {
        self.0.push_slice(samples);
    }
}

pub struct EchoCanceller {
    apm: AudioProcessingModule,
    far: HeapCons<f32>,
    far_resampler: StreamResampler,
    far_raw: Vec<f32>,
    far_16k: Vec<f32>,
    frame: Vec<i16>,
}

impl EchoCanceller {
    /// `far_rate`: the rate the headphones play at.
    pub fn new(far_rate: u32) -> (Self, FarTap) {
        let (producer, consumer) = HeapRb::<f32>::new(far_rate as usize * 2).split();
        let canceller = Self {
            apm: AudioProcessingModule::new(true, false, false, false),
            far: consumer,
            far_resampler: StreamResampler::new(far_rate, RATE),
            far_raw: vec![0.0; far_rate as usize / 50],
            far_16k: Vec::new(),
            frame: vec![0; FRAME],
        };
        (canceller, FarTap(producer))
    }

    /// 16 kHz microphone audio, whole 10 ms frames; the echo is removed in place.
    pub fn process(&mut self, mic: &mut [f32]) {
        debug_assert_eq!(mic.len() % FRAME, 0, "the microphone delivers whole 10 ms frames");
        self.feed_far();
        for block in mic.chunks_mut(FRAME) {
            for (dst, &src) in self.frame.iter_mut().zip(block.iter()) {
                *dst = src.round().clamp(-32768.0, 32767.0) as i16;
            }
            let _ = self.apm.set_stream_delay_ms(DELAY_MS);
            if self.apm.process_stream(&mut self.frame, RATE as i32, 1).is_ok() {
                for (dst, &src) in block.iter_mut().zip(self.frame.iter()) {
                    *dst = src as f32;
                }
            }
        }
    }

    /// Everything the headphones played since the last call, at 16 kHz.
    fn feed_far(&mut self) {
        loop {
            let n = self.far.pop_slice(&mut self.far_raw);
            if n == 0 {
                break;
            }
            self.far_resampler.process(&self.far_raw[..n], &mut self.far_16k);
        }
        let whole = self.far_16k.len() / FRAME * FRAME;
        for block in self.far_16k[..whole].chunks(FRAME) {
            for (dst, &src) in self.frame.iter_mut().zip(block.iter()) {
                *dst = src.round().clamp(-32768.0, 32767.0) as i16;
            }
            let _ = self.apm.process_reverse_stream(&mut self.frame, RATE as i32, 1);
        }
        self.far_16k.drain(..whole);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::dbfs;

    /// The headphones play speech-like noise at 48 kHz; the microphone hears
    /// it 20 ms later, 10 dB quieter. After the canceller has adapted, the
    /// echo must be much quieter than it came in.
    #[test]
    fn removes_a_delayed_echo() {
        let far_rate = 48000;
        let (mut aec, mut tap) = EchoCanceller::new(far_rate);
        let mut hush = crate::dsp::Hush::new(3);
        let seconds = 6;
        let far: Vec<f32> = hush.samples(far_rate as usize * seconds, 3000.0).iter().map(|&x| x as f32).collect();
        // What the microphone hears: the far signal at 16 kHz, delayed, -10 dB.
        let mut far_16k = Vec::new();
        StreamResampler::new(far_rate, RATE).process(&far, &mut far_16k);
        let delay = (RATE / 50) as usize;
        let mic: Vec<f32> = (0..far_16k.len())
            .map(|i| if i >= delay { far_16k[i - delay] * 0.316 } else { 0.0 })
            .collect();
        let (mut before, mut after) = (Vec::new(), Vec::new());
        for (n, chunk) in mic.as_chunks::<320>().0.iter().enumerate() {
            let played = &far[n * 960..(n + 1) * 960];
            tap.push(played);
            let mut block = chunk.to_vec();
            aec.process(&mut block);
            if n >= 150 {
                before.extend_from_slice(chunk);
                after.extend_from_slice(&block);
            }
        }
        let removed = dbfs(&before) - dbfs(&after);
        assert!(removed > 10.0, "echo only {removed:.1} dB quieter");
    }
}

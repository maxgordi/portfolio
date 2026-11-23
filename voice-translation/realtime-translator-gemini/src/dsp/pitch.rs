//! Real-time pitch shifter for the microphone (lower = a deeper voice).
//!
//! The Gemini translator copies the voice it hears; a voice that is a little
//! lower comes back as a steadier male voice. Two read heads move through a
//! short delay line at `ratio` of the input speed and are cross-faded (Hann
//! windows that always sum to 1), so pitch and formants drop together like a
//! slowed tape while the duration stays the same. Added delay: at most
//! `WINDOW`. Works block by block (any block length), keeping its state
//! between calls.

/// Seconds: the longest delay a read head reaches.
const WINDOW: f64 = 0.04;

/// Pitch ratio for lowering a voice by `semitones`.
pub fn deeper_ratio(semitones: f32) -> f32 {
    2f32.powf(-semitones / 12.0)
}

pub struct PitchShifter {
    ratio: f32,
    window: usize,
    history: Vec<f32>,
    phase: f64,
}

impl PitchShifter {
    pub fn new(ratio: f32, rate: u32) -> Self {
        let window = (rate as f64 * WINDOW) as usize;
        Self { ratio, window, history: vec![0.0; window + 2], phase: 0.0 }
    }

    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio;
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let h = self.history.len();
        let mut buf = Vec::with_capacity(h + input.len());
        buf.extend_from_slice(&self.history);
        buf.extend_from_slice(input);
        // The history is kept even when nothing is shifted, so switching
        // the shift on mid-stream starts from real audio, not zeros.
        self.history.copy_from_slice(&buf[buf.len() - h..]);
        if self.ratio == 1.0 {
            return input.to_vec();
        }
        let w = self.window as f64;
        let step = (1.0 - self.ratio as f64) / w;
        let mut out = Vec::with_capacity(input.len());
        for k in 0..input.len() {
            let phase = (self.phase + step * (k + 1) as f64).rem_euclid(1.0);
            let now = (h + k) as f64;
            let mut y = 0.0;
            for head in [phase, (phase + 0.5) % 1.0] {
                let pos = now - 1.0 - head * w;
                let i = pos.floor();
                let frac = pos - i;
                let i = i as usize;
                let sample = buf[i] as f64 * (1.0 - frac) + buf[i + 1] as f64 * frac;
                y += sample * (std::f64::consts::PI * head).sin().powi(2);
            }
            out.push(y.clamp(-32768.0, 32767.0) as f32);
        }
        self.phase = (self.phase + step * input.len() as f64).rem_euclid(1.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dominant frequency by counting upward zero crossings.
    fn frequency(x: &[f32], rate: f32) -> f32 {
        let crossings = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        crossings as f32 * rate / x.len() as f32
    }

    #[test]
    fn three_semitones_lower() {
        let rate = 16000.0;
        let tone: Vec<f32> =
            (0..32000).map(|i| 8000.0 * (std::f32::consts::TAU * 200.0 * i as f32 / rate).sin()).collect();
        let mut shifter = PitchShifter::new(deeper_ratio(3.0), 16000);
        // In 20 ms blocks, as the microphone delivers it.
        let out: Vec<f32> = tone.chunks(320).flat_map(|c| shifter.process(c)).collect();
        assert_eq!(out.len(), tone.len());
        let f = frequency(&out[4000..], rate);
        let expected = 200.0 * deeper_ratio(3.0);
        assert!((f - expected).abs() < 6.0, "got {f} Hz, expected {expected}");
    }

    #[test]
    fn no_shift_passes_audio_through() {
        let mut shifter = PitchShifter::new(1.0, 16000);
        let x: Vec<f32> = (0..320).map(|i| i as f32).collect();
        assert_eq!(shifter.process(&x), x);
    }
}

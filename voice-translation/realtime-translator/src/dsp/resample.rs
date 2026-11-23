//! Streaming sample-rate conversion for a mono signal, in blocks of any
//! length. The audio devices run at their own rate (44.1 or 48 kHz); the
//! model takes 16 kHz and speaks 24 kHz.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler as _, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

pub struct StreamResampler {
    /// `None` when both rates are the same: samples pass through.
    inner: Option<Async<f32>>,
    chunk: usize,
    pending: Vec<f32>,
    scratch: Vec<f32>,
}

impl StreamResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        // 10 ms of input per step; a short sinc keeps the delay ~1–3 ms.
        let chunk = (from_rate / 100).max(1) as usize;
        let inner = (from_rate != to_rate).then(|| {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(128)
                .interpolation(SincInterpolationType::Linear);
            let ratio = to_rate as f64 / from_rate as f64;
            Async::<f32>::new_sinc(ratio, 1.0, &params, chunk, 1, FixedAsync::Input)
                .expect("valid resampler settings")
        });
        let scratch = vec![0.0; inner.as_ref().map_or(0, |r| r.output_frames_max())];
        Self { inner, chunk, pending: Vec::new(), scratch }
    }

    /// Converts `input` and appends what is ready to `out` (the rest of a
    /// step waits for the next call).
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let Some(resampler) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return;
        };
        self.pending.extend_from_slice(input);
        let mut start = 0;
        while self.pending.len() - start >= self.chunk {
            let block = &self.pending[start..start + self.chunk];
            let src = InterleavedSlice::new(block, 1, self.chunk).expect("mono block");
            let capacity = self.scratch.len();
            let mut dst = InterleavedSlice::new_mut(&mut self.scratch, 1, capacity).expect("mono buffer");
            let (used, written) =
                resampler.process_into_buffer(&src, &mut dst, None).expect("buffers are sized by the resampler");
            out.extend_from_slice(&self.scratch[..written]);
            start += used;
        }
        self.pending.drain(..start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_follows_the_rate() {
        let mut r = StreamResampler::new(48000, 16000);
        let mut out = Vec::new();
        // Odd block sizes, like device callbacks.
        for _ in 0..100 {
            r.process(&[0.5; 517], &mut out);
        }
        let expected = 517.0 * 100.0 / 3.0;
        assert!((out.len() as f64 - expected).abs() < 500.0, "{} vs {expected}", out.len());
    }

    #[test]
    fn a_tone_survives() {
        let mut r = StreamResampler::new(24000, 44100);
        let tone: Vec<f32> =
            (0..24000).map(|i| 10000.0 * (std::f32::consts::TAU * 440.0 * i as f32 / 24000.0).sin()).collect();
        let mut out = Vec::new();
        r.process(&tone, &mut out);
        let tail = &out[out.len() / 2..];
        let peak = tail.iter().fold(0f32, |m, &x| m.max(x.abs()));
        assert!((peak - 10000.0).abs() < 300.0, "peak {peak}");
    }

    #[test]
    fn same_rate_passes_through() {
        let mut r = StreamResampler::new(16000, 16000);
        let mut out = Vec::new();
        r.process(&[1.0, 2.0, 3.0], &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }
}

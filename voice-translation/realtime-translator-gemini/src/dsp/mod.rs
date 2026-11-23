//! Small audio helpers. Samples are `f32` on the 16-bit scale
//! (−32768…32767), so levels and thresholds read as in the Python version.

pub mod pitch;
pub mod resample;

/// Loudness of a block in dBFS (−120 for an empty one).
pub fn dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -120.0;
    }
    let mean_square = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
    20.0 * (mean_square.sqrt() / 32768.0 + 1e-9).log10()
}

pub fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&x| x.round().clamp(-32768.0, 32767.0) as i16).collect()
}

pub fn peak(samples: &[i16]) -> i32 {
    samples.iter().map(|&x| (x as i32).abs()).max().unwrap_or(0)
}

/// Gaussian noise with standard deviation `sigma`: the "near-silence" a
/// quiet room makes. The model hears a pause in it, not a cut-off stream.
pub struct Hush {
    state: u64,
}

impl Hush {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn uniform(&mut self) -> f64 {
        // xorshift64*: plenty for noise, no dependency.
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let x = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn samples(&mut self, n: usize, sigma: f64) -> Vec<i16> {
        (0..n)
            .map(|_| {
                // Box–Muller
                let (u1, u2) = (self.uniform(), self.uniform());
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                (z * sigma).round().clamp(-32768.0, 32767.0) as i16
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_sine_is_minus_3_db() {
        let x: Vec<f32> = (0..1600).map(|i| 32767.0 * (i as f32 * 0.1).sin()).collect();
        assert!((dbfs(&x) + 3.0).abs() < 0.1);
        assert_eq!(dbfs(&[]), -120.0);
    }

    #[test]
    fn hush_is_quiet_noise_of_the_asked_spread() {
        let x = Hush::new(7).samples(16000, 20.0);
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
        let sd = (x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        assert!(mean.abs() < 1.0, "mean {mean}");
        assert!((sd - 20.0).abs() < 1.0, "sd {sd}");
    }
}

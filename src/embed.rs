//! YAMNet: the log-mel frontend (a port of the TensorFlow one, see model/README.md) and the
//! ONNX network, which maps one 96×64 log-mel patch to 521 class scores and a 1024-d embedding.

use anyhow::{Context, Result};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;
use tract_onnx::prelude::*;

const MODEL: &[u8] = include_bytes!("../model/yamnet.onnx");
const CLASS_MAP: &str = include_str!("../model/yamnet_class_map.csv");

const RATE: f64 = 16_000.0;
const WINDOW: usize = 400; // 25 ms
const HOP: usize = 160; // 10 ms
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 64;
const MEL_LOW_HZ: f64 = 125.0;
const MEL_HIGH_HZ: f64 = 7_500.0;
const LOG_OFFSET: f32 = 0.001;
const PATCH_FRAMES: usize = 96; // 0.96 s
const PATCH_HOP_FRAMES: usize = 48; // 0.48 s
/// Shortest waveform that yields one patch: 0.96 s + 25 ms window − 10 ms hop.
pub const MIN_SAMPLES: usize = 15_600;
const PATCH_HOP_SAMPLES: usize = PATCH_HOP_FRAMES * HOP;

/// Per-patch class scores and embeddings.
type PatchOutputs = (Vec<Vec<f32>>, Vec<Vec<f32>>);

pub struct Tags {
    /// Mean over patches, one per class.
    pub scores: Vec<f32>,
    /// Mean over patches, L2-normalised.
    pub embedding: Vec<f32>,
}

pub struct Tagger {
    model: Arc<TypedRunnableModel>,
    classes: Vec<String>,
    mel: Vec<[f32; MELS]>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Tagger {
    pub fn load() -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut &MODEL[..])?
            .with_input_fact(0, f32::fact([1, PATCH_FRAMES, MELS]).into())?
            .into_optimized()?
            .into_runnable()
            .context("loading the YAMNet model")?;
        let window = (0..WINDOW)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / WINDOW as f64).cos()) as f32)
            .collect();
        Ok(Self {
            model,
            classes: parse_class_map(CLASS_MAP),
            mel: mel_matrix(),
            window,
            fft: FftPlanner::new().plan_fft_forward(FFT),
        })
    }

    pub fn class_name(&self, index: usize) -> &str {
        &self.classes[index]
    }

    /// The `n` highest-scoring classes, best first.
    pub fn top_classes(&self, scores: &[f32], n: usize) -> Vec<(String, f32)> {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        order.into_iter().take(n).map(|i| (self.class_name(i).to_string(), scores[i])).collect()
    }

    pub fn tag(&self, wave: &[f32]) -> Result<Tags> {
        let (scores, embeddings) = self.run_patches(&self.log_mel(&pad(wave)))?;
        let mean = |rows: &[Vec<f32>]| {
            let mut sum = vec![0.0; rows[0].len()];
            for row in rows {
                sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
            }
            sum.iter_mut().for_each(|s| *s /= rows.len() as f32);
            sum
        };
        let mut embedding = mean(&embeddings);
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        embedding.iter_mut().for_each(|v| *v /= norm);
        Ok(Tags { scores: mean(&scores), embedding })
    }

    /// Per-patch (scores, embeddings) for a log-mel spectrogram.
    pub fn run_patches(&self, log_mel: &[[f32; MELS]]) -> Result<PatchOutputs> {
        let mut scores = Vec::new();
        let mut embeddings = Vec::new();
        let mut start = 0;
        while start + PATCH_FRAMES <= log_mel.len() {
            let patch: Vec<f32> = log_mel[start..start + PATCH_FRAMES].iter().flatten().copied().collect();
            let input = Tensor::from_shape(&[1, PATCH_FRAMES, MELS], &patch)?;
            let out = self.model.run(tvec!(input.into()))?;
            scores.push(out[0].try_as_plain_ram()?.as_slice::<f32>()?.to_vec());
            embeddings.push(out[1].try_as_plain_ram()?.as_slice::<f32>()?.to_vec());
            start += PATCH_HOP_FRAMES;
        }
        Ok((scores, embeddings))
    }

    /// Log-mel frames (10 ms hop) of an already padded waveform.
    pub fn log_mel(&self, wave: &[f32]) -> Vec<[f32; MELS]> {
        let frames = if wave.len() < WINDOW { 0 } else { 1 + (wave.len() - WINDOW) / HOP };
        let mut buf = vec![Complex::new(0.0, 0.0); FFT];
        (0..frames)
            .map(|f| {
                let frame = &wave[f * HOP..f * HOP + WINDOW];
                buf.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
                for (c, (s, w)) in buf.iter_mut().zip(frame.iter().zip(&self.window)) {
                    c.re = s * w;
                }
                self.fft.process(&mut buf);
                let mut mel = [0.0f32; MELS];
                for (bin, weights) in self.mel.iter().enumerate() {
                    let magnitude = buf[bin].norm();
                    mel.iter_mut().zip(weights).for_each(|(m, w)| *m += magnitude * w);
                }
                mel.map(|m| (m + LOG_OFFSET).ln())
            })
            .collect()
    }
}

/// Zero-pads to at least one patch and then to a whole number of patch hops, as YAMNet does.
pub fn pad(wave: &[f32]) -> Vec<f32> {
    let len = wave.len().max(MIN_SAMPLES);
    let hops = (len - MIN_SAMPLES).div_ceil(PATCH_HOP_SAMPLES);
    let mut out = wave.to_vec();
    out.resize(MIN_SAMPLES + hops * PATCH_HOP_SAMPLES, 0.0);
    out
}

fn hz_to_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// tf.signal.linear_to_mel_weight_matrix: HTK mel scale, triangles in mel space, DC bin zeroed.
fn mel_matrix() -> Vec<[f32; MELS]> {
    let (low, high) = (hz_to_mel(MEL_LOW_HZ), hz_to_mel(MEL_HIGH_HZ));
    let edge = |i: usize| low + (high - low) * i as f64 / (MELS + 1) as f64;
    let mut matrix = vec![[0.0f32; MELS]; BINS];
    for (bin, row) in matrix.iter_mut().enumerate().skip(1) {
        let mel = hz_to_mel(RATE / 2.0 * bin as f64 / (BINS - 1) as f64);
        for (m, weight) in row.iter_mut().enumerate() {
            let (lower, center, upper) = (edge(m), edge(m + 1), edge(m + 2));
            let rising = (mel - lower) / (center - lower);
            let falling = (upper - mel) / (upper - center);
            *weight = rising.min(falling).max(0.0) as f32;
        }
    }
    matrix
}

fn parse_class_map(csv: &str) -> Vec<String> {
    csv.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let name = line.splitn(3, ',').nth(2).unwrap_or_default();
            name.trim().trim_matches('"').to_string()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        log_mel: Vec<Vec<f32>>,
        scores: Vec<Vec<f32>>,
        embeddings: Vec<Vec<f32>>,
    }

    fn fixture() -> (Vec<f32>, Reference) {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let wave = hound::WavReader::open(format!("{dir}/synth.wav"))
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let text = std::fs::read_to_string(format!("{dir}/synth_reference.json")).unwrap();
        (wave, serde_json::from_str(&text).unwrap())
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn class_map_has_all_classes() {
        let classes = parse_class_map(CLASS_MAP);
        assert_eq!(classes.len(), 521);
        assert_eq!(classes[0], "Speech");
    }

    #[test]
    fn padding_matches_yamnet() {
        assert_eq!(pad(&[0.0; 100]).len(), MIN_SAMPLES);
        assert_eq!(pad(&[0.0; MIN_SAMPLES + 1]).len(), MIN_SAMPLES + PATCH_HOP_SAMPLES);
    }

    #[test]
    fn matches_tensorflow_reference() {
        let tagger = Tagger::load().unwrap();
        let (wave, reference) = fixture();
        let log_mel = tagger.log_mel(&pad(&wave));
        assert_eq!(log_mel.len(), reference.log_mel.len());
        let worst = log_mel.iter().zip(&reference.log_mel).map(|(a, b)| max_diff(a, b)).fold(0.0, f32::max);
        assert!(worst < 1e-3, "log-mel max diff {worst}");

        let (scores, embeddings) = tagger.run_patches(&log_mel).unwrap();
        assert_eq!(scores.len(), reference.scores.len());
        for (i, (s, e)) in scores.iter().zip(&embeddings).enumerate() {
            assert!(max_diff(s, &reference.scores[i]) < 1e-3, "patch {i} scores");
            let cosine = crate::classify::dot(e, &reference.embeddings[i])
                / (crate::classify::dot(e, e).sqrt()
                    * crate::classify::dot(&reference.embeddings[i], &reference.embeddings[i]).sqrt());
            assert!(cosine > 0.999, "patch {i} embedding cosine {cosine}");
        }
    }
}

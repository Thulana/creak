//! The audio models: a log-mel frontend for each (ports of their own, see model/README.md) and
//! the ONNX network, which maps one 96×64 log-mel patch (0.96 s) to class scores and an embedding.

use anyhow::{Context, Result};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tract_onnx::prelude::*;

/// The model that tags clips. Each has its own embedding space, so exemplars are only compared
/// with clips analysed by the same model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Model {
    /// YAMNet: MobileNetV1, 521 AudioSet classes, 1024-d embedding.
    #[default]
    Yamnet,
    /// CED-tiny: a transformer distilled from an ensemble, 527 AudioSet classes, 192-d embedding.
    CedTiny,
}

impl Model {
    /// The name stored with every analysis.
    pub fn name(self) -> &'static str {
        match self {
            Model::Yamnet => "yamnet",
            Model::CedTiny => "ced-tiny",
        }
    }

    fn assets(self) -> (&'static [u8], &'static str) {
        match self {
            Model::Yamnet => (include_bytes!("../model/yamnet.onnx"), include_str!("../model/yamnet_class_map.csv")),
            Model::CedTiny => {
                (include_bytes!("../model/ced_tiny.onnx"), include_str!("../model/ced_tiny_class_map.csv"))
            }
        }
    }
}

const RATE: f64 = 16_000.0;
const HOP: usize = 160; // 10 ms
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 64;
const PATCH_FRAMES: usize = 96; // 0.96 s
const PATCH_HOP_FRAMES: usize = 48; // 0.48 s
/// Shortest waveform that yields one patch: 0.96 s + 25 ms window − 10 ms hop.
pub const MIN_SAMPLES: usize = 15_600;
const PATCH_HOP_SAMPLES: usize = PATCH_HOP_FRAMES * HOP;

/// YAMNet's frontend (tf.signal): a 25 ms window, magnitude, natural log with an offset.
const YAMNET_WINDOW: usize = 400;
const YAMNET_MEL_HZ: (f64, f64) = (125.0, 7_500.0);
const YAMNET_LOG_OFFSET: f32 = 0.001;
/// CED's frontend (torchaudio): a centred 32 ms window, power, decibels at most 120 dB below the loudest.
const CED_MEL_HZ: (f64, f64) = (0.0, 8_000.0);
const CED_AMIN: f32 = 1e-10;
const CED_TOP_DB: f32 = 120.0;

/// Per-patch class scores and embeddings.
type PatchOutputs = (Vec<Vec<f32>>, Vec<Vec<f32>>);

pub struct Tags {
    /// Mean over patches, one per class.
    pub scores: Vec<f32>,
    /// Mean over patches, L2-normalised.
    pub embedding: Vec<f32>,
    /// One score per class for each 0.96 s patch, in time order.
    pub patch_scores: Vec<Vec<f32>>,
    /// The embedding of each patch, in time order, not normalised.
    pub patch_embeddings: Vec<Vec<f32>>,
}

pub struct Tagger {
    model: Model,
    net: Arc<TypedRunnableModel>,
    classes: Vec<String>,
    mel: Vec<[f32; MELS]>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Tagger {
    pub fn load(model: Model) -> Result<Self> {
        let (onnx, class_map) = model.assets();
        let net = tract_onnx::onnx()
            .model_for_read(&mut &onnx[..])?
            .with_input_fact(0, f32::fact([1, PATCH_FRAMES, MELS]).into())?
            .into_optimized()?
            .into_runnable()
            .with_context(|| format!("loading the {} model", model.name()))?;
        let (window, mel) = match model {
            Model::Yamnet => (hann(YAMNET_WINDOW), mel_matrix(YAMNET_MEL_HZ, false)),
            Model::CedTiny => (hann(FFT), mel_matrix(CED_MEL_HZ, true)),
        };
        Ok(Self {
            model,
            net,
            classes: parse_class_map(class_map),
            mel,
            window,
            fft: FftPlanner::new().plan_fft_forward(FFT),
        })
    }

    pub fn model(&self) -> Model {
        self.model
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
        Ok(Tags {
            scores: mean(&scores),
            embedding: normalized(mean(&embeddings)),
            patch_scores: scores,
            patch_embeddings: embeddings,
        })
    }

    /// Per-patch (scores, embeddings) for a log-mel spectrogram.
    pub fn run_patches(&self, log_mel: &[[f32; MELS]]) -> Result<PatchOutputs> {
        let mut scores = Vec::new();
        let mut embeddings = Vec::new();
        let mut start = 0;
        while start + PATCH_FRAMES <= log_mel.len() {
            let patch: Vec<f32> = log_mel[start..start + PATCH_FRAMES].iter().flatten().copied().collect();
            let input = Tensor::from_shape(&[1, PATCH_FRAMES, MELS], &patch)?;
            let out = self.net.run(tvec!(input.into()))?;
            scores.push(out[0].try_as_plain_ram()?.as_slice::<f32>()?.to_vec());
            embeddings.push(out[1].try_as_plain_ram()?.as_slice::<f32>()?.to_vec());
            start += PATCH_HOP_FRAMES;
        }
        Ok((scores, embeddings))
    }

    /// Log-mel frames (10 ms hop) of an already padded waveform, as the model's frontend makes them.
    pub fn log_mel(&self, wave: &[f32]) -> Vec<[f32; MELS]> {
        let centred;
        let wave = match self.model {
            Model::Yamnet => wave,
            Model::CedTiny => {
                centred = reflect_pad(wave, FFT / 2);
                &centred[..]
            }
        };
        let width = self.window.len();
        let frames = if wave.len() < width { 0 } else { 1 + (wave.len() - width) / HOP };
        let mut buf = vec![Complex::new(0.0, 0.0); FFT];
        let mut rows: Vec<[f32; MELS]> = (0..frames)
            .map(|f| {
                let frame = &wave[f * HOP..f * HOP + width];
                buf.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
                for (c, (s, w)) in buf.iter_mut().zip(frame.iter().zip(&self.window)) {
                    c.re = s * w;
                }
                self.fft.process(&mut buf);
                let mut mel = [0.0f32; MELS];
                for (bin, weights) in self.mel.iter().enumerate() {
                    let value = match self.model {
                        Model::Yamnet => buf[bin].norm(),
                        Model::CedTiny => buf[bin].norm_sqr(),
                    };
                    mel.iter_mut().zip(weights).for_each(|(m, w)| *m += value * w);
                }
                match self.model {
                    Model::Yamnet => mel.map(|m| (m + YAMNET_LOG_OFFSET).ln()),
                    Model::CedTiny => mel.map(|m| 10.0 * m.max(CED_AMIN).log10()),
                }
            })
            .collect();
        if self.model == Model::CedTiny {
            let floor = rows.iter().flatten().copied().fold(f32::NEG_INFINITY, f32::max) - CED_TOP_DB;
            rows.iter_mut().flatten().for_each(|v| *v = v.max(floor));
        }
        rows
    }
}

/// Scaled to unit length, so a dot product is a cosine similarity.
pub fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Zero-pads to at least one patch and then to a whole number of patch hops, as YAMNet does.
pub fn pad(wave: &[f32]) -> Vec<f32> {
    let len = wave.len().max(MIN_SAMPLES);
    let hops = (len - MIN_SAMPLES).div_ceil(PATCH_HOP_SAMPLES);
    let mut out = wave.to_vec();
    out.resize(MIN_SAMPLES + hops * PATCH_HOP_SAMPLES, 0.0);
    out
}

/// Periodic Hann window.
fn hann(len: usize) -> Vec<f32> {
    (0..len).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / len as f64).cos()) as f32).collect()
}

/// Mirrors `pad` samples onto each end without repeating the edge sample, as torch.stft's centring does.
fn reflect_pad(wave: &[f32], pad: usize) -> Vec<f32> {
    let n = wave.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| wave[i]));
    out.extend_from_slice(wave);
    out.extend((n - 1 - pad..n - 1).rev().map(|i| wave[i]));
    out
}

fn hz_to_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// HTK-mel filters, one row per FFT bin, DC bin zeroed, no area normalisation. Band edges are
/// spaced evenly in mel; tf.signal's triangles (YAMNet) are linear in mel, torchaudio's (CED) in Hz.
fn mel_matrix((low_hz, high_hz): (f64, f64), triangles_in_hz: bool) -> Vec<[f32; MELS]> {
    let (low, high) = (hz_to_mel(low_hz), hz_to_mel(high_hz));
    let edge = |i: usize| {
        let mel = low + (high - low) * i as f64 / (MELS + 1) as f64;
        if triangles_in_hz { mel_to_hz(mel) } else { mel }
    };
    let mut matrix = vec![[0.0f32; MELS]; BINS];
    for (bin, row) in matrix.iter_mut().enumerate().skip(1) {
        let hz = RATE / 2.0 * bin as f64 / (BINS - 1) as f64;
        let x = if triangles_in_hz { hz } else { hz_to_mel(hz) };
        for (m, weight) in row.iter_mut().enumerate() {
            let (lower, center, upper) = (edge(m), edge(m + 1), edge(m + 2));
            let rising = (x - lower) / (center - lower);
            let falling = (upper - x) / (upper - center);
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

    fn fixture(reference: &str) -> (Vec<f32>, Reference) {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let wave = hound::WavReader::open(format!("{dir}/synth.wav"))
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let text = std::fs::read_to_string(format!("{dir}/{reference}")).unwrap();
        (wave, serde_json::from_str(&text).unwrap())
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn class_maps_have_all_classes() {
        for (model, count) in [(Model::Yamnet, 521), (Model::CedTiny, 527)] {
            let classes = parse_class_map(model.assets().1);
            assert_eq!(classes.len(), count);
            assert_eq!(classes[0], "Speech");
        }
    }

    #[test]
    fn padding_matches_yamnet() {
        assert_eq!(pad(&[0.0; 100]).len(), MIN_SAMPLES);
        assert_eq!(pad(&[0.0; MIN_SAMPLES + 1]).len(), MIN_SAMPLES + PATCH_HOP_SAMPLES);
    }

    #[test]
    fn reflect_pad_mirrors_without_the_edge() {
        assert_eq!(reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2), [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    /// The Rust frontend and the ONNX network against the original pipeline on the same audio.
    /// `log_mel_tolerance` is in the frontend's unit: natural log for YAMNet, decibels for CED.
    fn matches_reference(model: Model, reference: &str, log_mel_tolerance: f32) {
        let tagger = Tagger::load(model).unwrap();
        let (wave, reference) = fixture(reference);
        let log_mel = tagger.log_mel(&pad(&wave));
        assert_eq!(log_mel.len(), reference.log_mel.len());
        let worst = log_mel.iter().zip(&reference.log_mel).map(|(a, b)| max_diff(a, b)).fold(0.0, f32::max);
        assert!(worst < log_mel_tolerance, "log-mel max diff {worst}");

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

    #[test]
    fn yamnet_matches_tensorflow_reference() {
        matches_reference(Model::Yamnet, "synth_reference.json", 1e-3);
    }

    #[test]
    fn ced_tiny_matches_pytorch_reference() {
        matches_reference(Model::CedTiny, "synth_ced_reference.json", 1e-2);
    }

    #[test]
    fn model_names_are_the_config_values() {
        for model in [Model::Yamnet, Model::CedTiny] {
            assert_eq!(serde_json::to_value(model).unwrap(), model.name());
        }
    }
}

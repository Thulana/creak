//! Microphone input: opens the device, downmixes to mono and hands raw chunks to the
//! processing thread, which resamples them to 16 kHz.

use crate::trigger::SAMPLE_RATE;
use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Sample, SampleFormat, SizedSample, Stream};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

pub struct Input {
    /// Keeps the stream alive; capture stops when this is dropped.
    _stream: Stream,
    pub rate: u32,
    pub name: String,
    pub chunks: Receiver<Vec<f32>>,
}

pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default_id = host.default_input_device().and_then(|d| d.id().ok());
    for device in host.input_devices()? {
        let Ok(id) = device.id() else { continue };
        let marker = if Some(&id) == default_id.as_ref() { "*" } else { " " };
        let desc = device.description().map(|d| d.to_string()).unwrap_or_default();
        let config = device
            .default_input_config()
            .map(|c| format!("{} Hz, {} ch, {}", c.sample_rate(), c.channels(), c.sample_format()))
            .unwrap_or_else(|e| format!("no input config: {e}"));
        println!("{marker} {id}\n    {desc}\n    {config}");
    }
    println!("\n* = default. Put an id in [audio] device to use it.");
    Ok(())
}

fn find_device(name: &str) -> Result<Device> {
    let host = cpal::default_host();
    if name == "default" {
        return host.default_input_device().ok_or_else(|| anyhow!("no default input device"));
    }
    let id = name.parse().map_err(|e| anyhow!("bad device id {name:?}: {e:?}"))?;
    host.device_by_id(&id).ok_or_else(|| anyhow!("input device {name:?} not found; see `creak devices`"))
}

pub fn open(device_name: &str) -> Result<Input> {
    let error = match open_device(find_device(device_name)?, device_name) {
        Ok(input) => return Ok(input),
        Err(e) if device_name != "default" => return Err(e),
        Err(e) => e,
    };
    // Without a sound server, ALSA's default capture device is card 0 shared through dsnoop. It
    // fails while the card itself would open when card 0 has no input (an HDMI card), or when
    // /dev/snd/timer is missing, as in a container given only the card's own nodes. A plughw
    // device opens the card directly and converts the format and rate.
    let cards = cpal::default_host()
        .input_devices()
        .into_iter()
        .flatten()
        .filter(|d| d.id().is_ok_and(|id| id.id().starts_with("plughw:")));
    for device in cards {
        if let Ok(input) = open_device(device, device_name) {
            eprintln!("audio: the default device failed ({error:#}); using {}", input.name);
            return Ok(input);
        }
    }
    Err(error)
}

fn open_device(device: Device, device_name: &str) -> Result<Input> {
    let name = device.id().map(|id| id.to_string()).unwrap_or_else(|_| device_name.to_string());
    // Prefer a native 16 kHz config so no resampling is needed.
    let config = device
        .supported_input_configs()
        .context("listing input configs")?
        .filter_map(|c| c.try_with_sample_rate(SAMPLE_RATE))
        .find(|c| matches!(c.sample_format(), SampleFormat::F32 | SampleFormat::I16))
        .map(Ok)
        .unwrap_or_else(|| device.default_input_config())
        .context("no usable input config")?;
    let rate = config.sample_rate();
    let channels = config.channels() as usize;
    let (tx, rx) = sync_channel(64);
    let stream = match config.sample_format() {
        SampleFormat::F32 => build::<f32>(&device, &config.config(), channels, tx),
        SampleFormat::I16 => build::<i16>(&device, &config.config(), channels, tx),
        SampleFormat::I32 => build::<i32>(&device, &config.config(), channels, tx),
        SampleFormat::U16 => build::<u16>(&device, &config.config(), channels, tx),
        other => bail!("unsupported sample format {other}"),
    }?;
    stream.play()?;
    Ok(Input { _stream: stream, rate, name, chunks: rx })
}

fn build<T>(device: &Device, config: &cpal::StreamConfig, channels: usize, tx: SyncSender<Vec<f32>>) -> Result<Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let stream = device.build_input_stream(
        *config,
        move |data: &[T], _: &_| {
            let mono = data
                .chunks(channels)
                .map(|frame| frame.iter().map(|&s| f32::from_sample(s)).sum::<f32>() / channels as f32)
                .collect();
            // A full queue means the processing thread is stuck; dropping audio beats blocking
            // the audio callback, and the processing thread's watchdog handles a real stall.
            let _ = tx.try_send(mono);
        },
        |err| eprintln!("audio stream: {err}"),
        None,
    )?;
    Ok(stream)
}

/// Streaming resampler to 16 kHz: a windowed-sinc low-pass at the input rate followed by
/// linear interpolation. Good enough for a 64-band mel frontend that stops at 7.5 kHz.
pub struct Resampler {
    step: f64,
    taps: Vec<f32>,
    input: Vec<f32>,
    filtered: Vec<f32>,
    pos: f64,
}

impl Resampler {
    pub fn new(in_rate: u32) -> Self {
        const TAPS: usize = 127;
        let cutoff = 7_400.0 / in_rate as f64;
        let mid = (TAPS / 2) as f64;
        let mut taps: Vec<f32> = (0..TAPS)
            .map(|i| {
                let x = i as f64 - mid;
                let sinc = if x == 0.0 {
                    2.0 * cutoff
                } else {
                    (2.0 * std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (TAPS - 1) as f64).cos();
                (sinc * hann) as f32
            })
            .collect();
        let gain: f32 = taps.iter().sum();
        taps.iter_mut().for_each(|t| *t /= gain);
        Self {
            step: in_rate as f64 / SAMPLE_RATE as f64,
            taps,
            input: vec![0.0; TAPS - 1],
            filtered: Vec::new(),
            pos: 0.0,
        }
    }

    pub fn passthrough(&self) -> bool {
        self.step == 1.0
    }

    pub fn process(&mut self, chunk: &[f32], out: &mut Vec<f32>) {
        if self.passthrough() {
            out.extend_from_slice(chunk);
            return;
        }
        self.input.extend_from_slice(chunk);
        let n = self.taps.len();
        for window in self.input.windows(n) {
            self.filtered.push(window.iter().zip(&self.taps).map(|(a, b)| a * b).sum());
        }
        self.input.drain(..self.input.len() + 1 - n);
        while self.pos + 1.0 < self.filtered.len() as f64 {
            let i = self.pos as usize;
            let frac = (self.pos - i as f64) as f32;
            out.push(self.filtered[i] * (1.0 - frac) + self.filtered[i + 1] * frac);
            self.pos += self.step;
        }
        let used = (self.pos as usize).min(self.filtered.len());
        self.filtered.drain(..used);
        self.pos -= used as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn resample(rate: u32, input: &[f32]) -> Vec<f32> {
        let mut r = Resampler::new(rate);
        let mut out = Vec::new();
        for chunk in input.chunks(441) {
            r.process(chunk, &mut out);
        }
        out
    }

    #[test]
    fn keeps_length_and_passband() {
        for rate in [44_100, 48_000] {
            let out = resample(rate, &tone(rate, 1_000.0, 1.0));
            assert!((out.len() as i32 - 16_000).abs() < 100, "{rate}: {}", out.len());
            let level = rms(&out[1000..15000]);
            assert!((level - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02, "{rate}: {level}");
        }
    }

    #[test]
    fn removes_content_above_nyquist() {
        let out = resample(48_000, &tone(48_000, 12_000.0, 1.0));
        assert!(rms(&out[1000..15000]) < 0.03);
    }
}

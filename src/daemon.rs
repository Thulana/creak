//! `creak run`: capture → trigger → clip → tag → store, plus the web UI.

use crate::capture::{self, Resampler};
use crate::classify;
use crate::config::{Config, Tuning};
use crate::embed::{self, Tagger};
use crate::store::{NewEvent, Store, now_ms};
use crate::trigger::{self, Event, FRAME, SAMPLE_RATE, Trigger};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PREROLL_SAMPLES: usize = SAMPLE_RATE as usize * 3 / 2;
const RING_SAMPLES: usize = SAMPLE_RATE as usize * 2;
/// Context kept before the trigger point when embedding, so the onset is not cut off.
const ONSET_MARGIN_SAMPLES: usize = SAMPLE_RATE as usize / 10;
const LIVE_FRAMES: usize = 30 * 1000 / trigger::FRAME_MS as usize;
const STALL: Duration = Duration::from_secs(10);
const RETRY: Duration = Duration::from_secs(10);
const EXPIRE_EVERY: Duration = Duration::from_secs(3600);

/// What the Live view shows: recent frame levels and the current trigger lines.
#[derive(Default, Serialize)]
pub struct Live {
    pub device: String,
    pub device_rate: u32,
    /// Index of the frame after the last one in `levels`.
    pub frame: u64,
    pub levels: VecDeque<f32>,
    pub ambient_db: f32,
    pub open_db: f32,
    pub close_db: f32,
    pub recording: bool,
    /// The last capture error while the device is being retried.
    pub error: Option<String>,
    /// The 10th-percentile room level before the floor is applied.
    pub measured_ambient_db: Option<f32>,
    /// Trigger values in effect, and whether they come from the Live view rather than the config file.
    pub tuning: Tuning,
    pub tuned: bool,
    /// Set by the web UI after changing `tuning`; the capture loop applies it.
    #[serde(skip)]
    pub retune: bool,
}

pub type SharedLive = Arc<Mutex<Live>>;

pub struct Clip {
    pub samples: Vec<f32>,
    /// Where the triggering sound starts within `samples`; everything before is pre-roll.
    pub onset: usize,
    pub ts: i64,
    pub peak_db: f32,
    pub ambient_db: f32,
}

pub fn run(config: Config) -> Result<()> {
    let store = Store::open(&config.storage.dir)?;
    let tagger = Tagger::load()?;
    eprintln!("model loaded; storage at {}", config.storage.dir.display());
    let live = SharedLive::default();
    let file_tuning = config.trigger.tuning();
    {
        let mut l = live.lock().unwrap();
        l.tuning = file_tuning;
        if let Some(t) = store.trigger_tuning()?.filter(|t| config.trigger.with_tuning(*t).validate().is_ok()) {
            eprintln!("using trigger values set in the web UI: {t:?}");
            (l.tuning, l.tuned) = (t, true);
        }
    }

    if config.web.enabled {
        let web_store = Store::open(&config.storage.dir)?;
        let (web, live) = (config.web.clone(), live.clone());
        std::thread::spawn(move || {
            if let Err(e) = crate::web::serve(&web, file_tuning, web_store, live) {
                eprintln!("web UI stopped: {e:#}");
                std::process::exit(1);
            }
        });
    }

    let (clips_tx, clips_rx) = channel();
    let classify_config = config.clone();
    std::thread::spawn(move || classify_loop(&classify_config, store, tagger, clips_rx));

    // A missing or failing microphone is retried rather than fatal, so the web UI stays up
    // and shows what is wrong.
    loop {
        let error = match capture::open(&config.audio.device) {
            Ok(input) => {
                eprintln!("capturing from {} at {} Hz", input.name, input.rate);
                {
                    let mut l = live.lock().unwrap();
                    (l.device, l.device_rate, l.error) = (input.name.clone(), input.rate, None);
                }
                capture_loop(&config, &input.chunks, input.rate, &live, &clips_tx).unwrap_err()
            }
            Err(e) => e,
        };
        eprintln!("audio: {error:#}; retrying in {}s", RETRY.as_secs());
        live.lock().unwrap().error = Some(format!("{error:#}"));
        std::thread::sleep(RETRY);
    }
}

fn capture_loop(
    config: &Config,
    chunks: &Receiver<Vec<f32>>,
    rate: u32,
    live: &SharedLive,
    clips: &Sender<Clip>,
) -> Result<()> {
    let mut resampler = Resampler::new(rate);
    let mut trigger = Trigger::new(&config.trigger.with_tuning(live.lock().unwrap().tuning));
    let mut ring: VecDeque<f32> = VecDeque::with_capacity(RING_SAMPLES);
    let mut pending: Vec<f32> = Vec::new();
    let mut resampled = Vec::new();
    let mut clip: Option<Clip> = None;
    let mut frame_index: u64 = 0;

    loop {
        let chunk = match chunks.recv_timeout(STALL) {
            Ok(chunk) => chunk,
            Err(RecvTimeoutError::Timeout) => bail!("no audio from the device for {}s", STALL.as_secs()),
            Err(RecvTimeoutError::Disconnected) => bail!("audio stream closed"),
        };
        resampled.clear();
        resampler.process(&chunk, &mut resampled);
        pending.extend_from_slice(&resampled);

        let whole = pending.len() / FRAME * FRAME;
        for frame in pending[..whole].as_chunks::<FRAME>().0 {
            if ring.len() + FRAME > RING_SAMPLES {
                ring.drain(..ring.len() + FRAME - RING_SAMPLES);
            }
            ring.extend(frame);
            if let Some(c) = clip.as_mut() {
                c.samples.extend_from_slice(frame);
            }

            let level = trigger::level_db(frame);
            match trigger.push(level) {
                Some(Event::Open { start }) => {
                    // The ring ends at frame `frame_index`; the run started `back` samples ago.
                    let back = ((frame_index + 1 - start) as usize * FRAME).min(ring.len());
                    let take = (back + PREROLL_SAMPLES).min(ring.len());
                    let samples: Vec<f32> = ring.range(ring.len() - take..).copied().collect();
                    let ts = now_ms() - (take as i64 * 1000 / SAMPLE_RATE as i64);
                    clip = Some(Clip {
                        samples,
                        onset: take - back,
                        ts,
                        peak_db: level,
                        ambient_db: trigger.ambient_db(),
                    });
                }
                Some(Event::Close { peak_db, ambient_db, .. }) => {
                    if let Some(mut c) = clip.take() {
                        c.peak_db = peak_db;
                        c.ambient_db = ambient_db;
                        clips.send(c).context("classifier thread stopped")?;
                    }
                }
                None => {}
            }
            frame_index += 1;

            let mut l = live.lock().unwrap();
            if l.retune {
                trigger.retune(&config.trigger.with_tuning(l.tuning));
                l.retune = false;
            }
            let (open_db, close_db) = trigger.thresholds_db();
            if l.levels.len() == LIVE_FRAMES {
                l.levels.pop_front();
            }
            l.levels.push_back(level);
            l.frame = frame_index;
            l.ambient_db = trigger.ambient_db();
            l.measured_ambient_db = trigger.measured_ambient_db();
            (l.open_db, l.close_db) = (open_db, close_db);
            l.recording = trigger.is_open();
        }
        pending.drain(..whole);
    }
}

fn classify_loop(config: &Config, store: Store, tagger: Tagger, clips: Receiver<Clip>) {
    let mut last_expiry: Option<Instant> = None;
    let mut last: Option<Latest> = None;
    loop {
        if last_expiry.is_none_or(|t| t.elapsed() >= EXPIRE_EVERY) {
            match store.expire(config.storage.retention_days) {
                Ok(0) => {}
                Ok(n) => eprintln!("expired {n} clips"),
                Err(e) => eprintln!("expiring clips: {e:#}"),
            }
            last_expiry = Some(Instant::now());
        }
        match clips.recv_timeout(Duration::from_secs(60)) {
            Ok(clip) => {
                if let Err(e) = handle_clip(config, &store, &tagger, &clip, &mut last) {
                    eprintln!("clip at {}: {e:#}", clip.ts);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// The part of the clip that is embedded: from just before the onset to the end, widened
/// into the pre-roll when that is shorter than one model patch.
pub fn analysis_window(clip: &Clip) -> &[f32] {
    let start = clip.onset.saturating_sub(ONSET_MARGIN_SAMPLES);
    let start = start.min(clip.samples.len().saturating_sub(embed::MIN_SAMPLES));
    &clip.samples[start..]
}

/// The most recent event and when it ended, for grouping the next clip into it.
#[derive(Clone, Copy)]
struct Latest {
    id: i64,
    end_ts: i64,
}

fn handle_clip(config: &Config, store: &Store, tagger: &Tagger, clip: &Clip, last: &mut Option<Latest>) -> Result<()> {
    let started = Instant::now();
    let tags = tagger.tag(analysis_window(clip))?;
    let sounds = classify::sounds(tagger.top_classes(&tags.scores, 20));
    let top = &sounds[..sounds.len().min(3)];
    let c = &config.classify;
    let decision = classify::decide(&tags.embedding, &store.exemplars()?, top, c.match_threshold, c.suggest_threshold);
    if decision.status == classify::Status::Silent {
        eprintln!("dropped a clip with nothing but silence or room tone");
        return Ok(());
    }
    let ms = |samples: usize| samples as i64 * 1000 / SAMPLE_RATE as i64;
    let (onset_ts, end_ts) = (clip.ts + ms(clip.onset), clip.ts + ms(clip.samples.len()));

    if decision.status == classify::Status::Ignored {
        *last = None;
    } else if let Some(prev) = *last
        && c.group_gap_s > 0
        && decision.conflict.is_none()
        && onset_ts - prev.end_ts <= c.group_gap_s as i64 * 1000
        && store.event(prev.id)?.is_some_and(|e| e.label == decision.label)
    {
        store.extend_event(prev.id, end_ts, clip.peak_db, decision.status == classify::Status::Match)?;
        *last = Some(Latest { id: prev.id, end_ts });
        eprintln!("event {}: extended with another {:?} clip", prev.id, decision.label);
        return Ok(());
    }

    let clip_path = if decision.status == classify::Status::Ignored {
        None
    } else {
        let day = chrono::DateTime::from_timestamp_millis(clip.ts).unwrap_or_default().format("%Y-%m-%d");
        let rel = format!("clips/{day}/{}.wav", clip.ts);
        write_wav(&store.dir().join(&rel), &clip.samples)?;
        Some(rel)
    };
    let id = store.insert_event(&NewEvent {
        ts: clip.ts,
        duration_ms: end_ts - clip.ts,
        peak_db: clip.peak_db,
        ambient_db: clip.ambient_db,
        decision: &decision,
        top_classes: top,
        clip_path: clip_path.as_deref(),
        embedding: &tags.embedding,
    })?;
    if decision.status != classify::Status::Ignored {
        *last = Some(Latest { id, end_ts });
    }
    let nearest = decision.nearest.map(|(l, s)| format!(", nearest {l} {s:.2}")).unwrap_or_default();
    let conflict = decision.conflict.map(|c| format!(", model hears {c}: asking")).unwrap_or_default();
    eprintln!(
        "event {id}: {:?} {:?} ({}{nearest}{conflict}) in {} ms",
        decision.status,
        decision.label,
        top.iter().map(|(n, s)| format!("{n} {s:.2}")).collect::<Vec<_>>().join(", "),
        started.elapsed().as_millis()
    );
    Ok(())
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(len: usize, onset: usize) -> Clip {
        Clip { samples: vec![0.0; len], onset, ts: 0, peak_db: 0.0, ambient_db: 0.0 }
    }

    #[test]
    fn window_skips_preroll_for_long_sounds() {
        let c = clip(PREROLL_SAMPLES + 40_000, PREROLL_SAMPLES);
        assert_eq!(analysis_window(&c).len(), 40_000 + ONSET_MARGIN_SAMPLES);
    }

    #[test]
    fn window_widens_into_preroll_for_short_sounds() {
        let c = clip(PREROLL_SAMPLES + 4_000, PREROLL_SAMPLES);
        assert_eq!(analysis_window(&c).len(), embed::MIN_SAMPLES);
    }

    #[test]
    fn back_to_back_clips_with_the_same_label_are_grouped() {
        let dir = std::env::temp_dir().join(format!("creak-group-{}", std::process::id()));
        let mut config = Config::default();
        config.storage.dir = dir.clone();
        let store = Store::open(&dir).unwrap();
        let tagger = Tagger::load().unwrap();
        let sound: Vec<f32> = hound::WavReader::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synth.wav"))
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let at = |ts: i64| Clip { samples: sound.clone(), onset: 0, ts, peak_db: -20.0, ambient_db: -60.0 };
        let mut last = None;
        // 2.5 s clips: the second starts 2 s after the first ends, the third 60 s later.
        for ts in [0, 4_500, 67_000] {
            handle_clip(&config, &store, &tagger, &at(1_000_000 + ts), &mut last).unwrap();
        }
        let events = store.events(&crate::store::Filter { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(events.iter().map(|e| (e.parts, e.duration_ms)).collect::<Vec<_>>(), vec![(1, 2_500), (2, 7_000)]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_match_the_model_contradicts_goes_to_review_and_is_not_grouped() {
        let dir = std::env::temp_dir().join(format!("creak-conflict-{}", std::process::id()));
        let mut config = Config::default();
        config.storage.dir = dir.clone();
        let mut store = Store::open(&dir).unwrap();
        let tagger = Tagger::load().unwrap();
        let sound: Vec<f32> =
            (0..40_000).map(|i| (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin() * 0.5).collect();
        let tags = tagger.tag(&sound).unwrap();
        let heard = classify::sounds(tagger.top_classes(&tags.scores, 20)).remove(0);
        assert!(
            heard.1 >= classify::MIN_CONFLICT_SCORE && heard.0 != "Music",
            "fixture must be heard as a sound: {heard:?}"
        );

        // A "music" exemplar with this very embedding, but which the model heard as Music.
        let taught = classify::Decision {
            status: classify::Status::Pending,
            label: "music".into(),
            nearest: None,
            conflict: None,
        };
        let music = store
            .insert_event(&NewEvent {
                ts: 1_000_000,
                duration_ms: 2_500,
                peak_db: -20.0,
                ambient_db: -60.0,
                decision: &taught,
                top_classes: &[("Music".into(), 0.9)],
                clip_path: None,
                embedding: &tags.embedding,
            })
            .unwrap();
        store.label_event(music, "music").unwrap();

        let clip = Clip { samples: sound, onset: 0, ts: 1_003_000, peak_db: -20.0, ambient_db: -60.0 };
        let mut last = Some(Latest { id: music, end_ts: 1_002_500 });
        handle_clip(&config, &store, &tagger, &clip, &mut last).unwrap();
        let newest = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_ne!(newest.id, music);
        assert_eq!((newest.status.as_str(), newest.label.as_str()), ("pending", "music"));
        assert_eq!(newest.conflict.as_deref(), Some(heard.0.as_str()));
        assert_eq!(store.event(music).unwrap().unwrap().parts, 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Quiet noise, then the fixture sound, then quiet again: one clip, tagged and stored as
    /// pending; after labelling it, the same sound is a match.
    #[test]
    fn pipeline_from_audio_to_event() {
        let dir = std::env::temp_dir().join(format!("creak-pipeline-{}", std::process::id()));
        let mut config = Config::default();
        config.storage.dir = dir.clone();
        let mut seed = 1u32;
        let mut noise = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 * 0.002 - 0.001
                })
                .collect()
        };
        let sound: Vec<f32> = hound::WavReader::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synth.wav"))
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();

        let feed = |audio: Vec<f32>| {
            let (chunks_tx, chunks_rx) = channel();
            let (clips_tx, clips_rx) = channel();
            for chunk in audio.chunks(1000) {
                chunks_tx.send(chunk.to_vec()).unwrap();
            }
            drop(chunks_tx);
            let err = capture_loop(&config, &chunks_rx, SAMPLE_RATE, &SharedLive::default(), &clips_tx).unwrap_err();
            assert!(err.to_string().contains("closed"));
            clips_rx.try_iter().collect::<Vec<Clip>>()
        };
        let audio = [noise(16_000 * 6), sound.clone(), noise(16_000 * 3)].concat();
        let clips = feed(audio.clone());
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].onset, PREROLL_SAMPLES);

        let mut store = Store::open(&dir).unwrap();
        let tagger = Tagger::load().unwrap();
        handle_clip(&config, &store, &tagger, &clips[0], &mut None).unwrap();
        let first = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_eq!(first.status, "pending");
        assert!(dir.join(first.clip_path.as_ref().unwrap()).exists());

        store.label_event(first.id, "chirp").unwrap();
        handle_clip(&config, &store, &tagger, &feed(audio).remove(0), &mut None).unwrap();
        let second = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_eq!((second.status.as_str(), second.label.as_str()), ("match", "chirp"));
        let _ = std::fs::remove_dir_all(dir);
    }
}

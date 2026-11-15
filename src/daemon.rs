//! `creak run`: capture → trigger → clip → tag → store, plus the web UI.

use crate::capture::{self, Resampler};
use crate::classify::{self, MixPart};
use crate::config::{Config, Mode, Overrides};
use crate::embed::{self, Tagger, Tags};
use crate::store::{NewClassification, NewEvent, Store, now_ms};
use crate::trigger::{self, Event, FRAME, FRAME_MS, SAMPLE_RATE, Trigger};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::VecDeque;
use std::hash::{BuildHasher, RandomState};
use std::ops::Range;
use std::path::Path;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Ring buffer room beyond the pre-roll, for the run that opened the clip (`min_ms` is at most 2 s).
const RING_EXTRA_SAMPLES: usize = SAMPLE_RATE as usize * 2;
/// Context kept before the trigger point when embedding, so the onset is not cut off.
const ONSET_MARGIN_SAMPLES: usize = SAMPLE_RATE as usize / 10;
const LIVE_FRAMES: usize = 30 * 1000 / trigger::FRAME_MS as usize;
/// Classes ranked per patch for the sound mix: more than ever reach `sound_score` in one patch.
const PATCH_CLASSES: usize = 20;
const STALL: Duration = Duration::from_secs(10);
const RETRY: Duration = Duration::from_secs(10);
const EXPIRE_EVERY: Duration = Duration::from_secs(600);

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
}

pub type SharedLive = Arc<Mutex<Live>>;

/// What the web UI can change while the daemon runs.
pub struct Settings {
    /// The config file with `overrides` applied.
    pub config: Config,
    pub overrides: Overrides,
    pub mode: Mode,
    /// Bumped on every change, so the capture loop knows to apply it.
    pub version: u64,
}

pub type SharedSettings = Arc<Mutex<Settings>>;

pub struct Clip {
    pub samples: Vec<f32>,
    /// Where the triggering sound starts within `samples`; everything before is pre-roll.
    pub onset: usize,
    /// Where the sound is over within `samples`; everything after is tail.
    pub sound_end: usize,
    pub ts: i64,
    pub peak_db: f32,
    pub ambient_db: f32,
}

pub fn run(file: Config) -> Result<()> {
    file.validate().context("config file")?;
    let store = Store::open(&file.storage.dir)?;
    let tagger = Tagger::load()?;
    eprintln!("model loaded; storage at {}", file.storage.dir.display());
    let mut overrides = store.overrides()?;
    let config = match file.with_overrides(&overrides) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("ignoring the settings changed in the web UI: {e:#}");
            overrides.clear();
            file.clone()
        }
    };
    if !overrides.is_empty() {
        eprintln!("using settings changed in the web UI: {}", serde_json::Value::Object(overrides.clone()));
    }
    let mode = store.mode()?;
    if mode == Mode::Deaf {
        eprintln!("deaf: nothing is recorded until the mode is switched to active");
    }
    let settings = Arc::new(Mutex::new(Settings { config, overrides, mode, version: 0 }));
    let live = SharedLive::default();

    if file.web.enabled {
        let web_store = Store::open(&file.storage.dir)?;
        let (file, live, settings) = (file.clone(), live.clone(), settings.clone());
        std::thread::spawn(move || {
            if let Err(e) = crate::web::serve(&file, web_store, live, settings) {
                eprintln!("web UI stopped: {e:#}");
                std::process::exit(1);
            }
        });
    }

    let (clips_tx, clips_rx) = channel();
    let classify_settings = settings.clone();
    std::thread::spawn(move || classify_loop(&classify_settings, store, tagger, clips_rx));

    // A missing or failing microphone is retried rather than fatal, so the web UI stays up
    // and shows what is wrong.
    loop {
        let error = match capture::open(&file.audio.device) {
            Ok(input) => {
                eprintln!("capturing from {} at {} Hz", input.name, input.rate);
                {
                    let mut l = live.lock().unwrap();
                    (l.device, l.device_rate, l.error) = (input.name.clone(), input.rate, None);
                }
                capture_loop(&settings, &input.chunks, input.rate, &live, &clips_tx).unwrap_err()
            }
            Err(e) => e,
        };
        eprintln!("audio: {error:#}; retrying in {}s", RETRY.as_secs());
        live.lock().unwrap().error = Some(format!("{error:#}"));
        std::thread::sleep(RETRY);
    }
}

fn capture_loop(
    settings: &SharedSettings,
    chunks: &Receiver<Vec<f32>>,
    rate: u32,
    live: &SharedLive,
    clips: &Sender<Clip>,
) -> Result<()> {
    let mut resampler = Resampler::new(rate);
    let mut trigger = Trigger::new(&settings.lock().unwrap().config.trigger);
    let mut version = None;
    let mut pre_roll = 0;
    let mut ring: VecDeque<f32> = VecDeque::new();
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
        {
            let s = settings.lock().unwrap();
            if version != Some(s.version) {
                version = Some(s.version);
                trigger.retune(&s.config.trigger);
                trigger.set_armed(s.mode == Mode::Active);
                if s.mode == Mode::Deaf {
                    clip = None;
                }
                pre_roll = s.config.trigger.pre_roll_ms.div_ceil(FRAME_MS) as usize * FRAME;
            }
        }
        resampled.clear();
        resampler.process(&chunk, &mut resampled);
        pending.extend_from_slice(&resampled);

        let whole = pending.len() / FRAME * FRAME;
        for frame in pending[..whole].as_chunks::<FRAME>().0 {
            let ring_samples = pre_roll + RING_EXTRA_SAMPLES;
            if ring.len() + FRAME > ring_samples {
                ring.drain(..ring.len() + FRAME - ring_samples);
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
                    let take = (back + pre_roll).min(ring.len());
                    let samples: Vec<f32> = ring.range(ring.len() - take..).copied().collect();
                    let ts = now_ms() - (take as i64 * 1000 / SAMPLE_RATE as i64);
                    clip = Some(Clip {
                        samples,
                        onset: take - back,
                        sound_end: 0,
                        ts,
                        peak_db: level,
                        ambient_db: trigger.ambient_db(),
                    });
                }
                Some(Event::Close { end, sound_end, peak_db, ambient_db }) => {
                    if let Some(mut c) = clip.take() {
                        // `samples` ends with the frame before `end`.
                        c.sound_end = c.samples.len().saturating_sub((end - sound_end) as usize * FRAME);
                        c.peak_db = peak_db;
                        c.ambient_db = ambient_db;
                        clips.send(c).context("classifier thread stopped")?;
                    }
                }
                None => {}
            }
            frame_index += 1;

            let mut l = live.lock().unwrap();
            let (open_db, close_db) = trigger.thresholds_db();
            if l.levels.len() == LIVE_FRAMES {
                l.levels.pop_front();
            }
            l.levels.push_back(level);
            l.frame = frame_index;
            l.ambient_db = trigger.ambient_db();
            l.measured_ambient_db = trigger.measured_ambient_db();
            (l.open_db, l.close_db) = (open_db, close_db);
            l.recording = trigger.is_recording();
        }
        pending.drain(..whole);
    }
}

fn classify_loop(settings: &SharedSettings, store: Store, tagger: Tagger, clips: Receiver<Clip>) {
    let mut last_expiry: Option<Instant> = None;
    let mut last: Option<Latest> = None;
    loop {
        let config = settings.lock().unwrap().config.clone();
        if last_expiry.is_none_or(|t| t.elapsed() >= EXPIRE_EVERY) {
            match store.expire(config.storage.retention_days) {
                Ok(0) => {}
                Ok(n) => eprintln!("expired {n} clips"),
                Err(e) => eprintln!("expiring clips: {e:#}"),
            }
            match store.cap_clips(config.storage.max_mb) {
                Ok(0) => {}
                Ok(n) => eprintln!("deleted the {n} oldest clips to stay under {} MB", config.storage.max_mb),
                Err(e) => eprintln!("capping clip space: {e:#}"),
            }
            last_expiry = Some(Instant::now());
        }
        match clips.recv_timeout(Duration::from_secs(60)) {
            Ok(clip) => {
                let config = settings.lock().unwrap().config.clone();
                if let Err(e) = handle_clip(&config, &store, &tagger, &clip, &mut last) {
                    eprintln!("clip at {}: {e:#}", clip.ts);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// The part of the clip that is embedded: from just before the onset to the end of the sound,
/// widened into the pre-roll, and then the tail, when that is shorter than one model patch.
pub fn analysis_window(clip: &Clip) -> Range<usize> {
    let start = clip.onset.saturating_sub(ONSET_MARGIN_SAMPLES);
    let start = start.min(clip.sound_end.saturating_sub(embed::MIN_SAMPLES));
    let end = clip.sound_end.max((start + embed::MIN_SAMPLES).min(clip.samples.len()));
    start..end
}

/// What the model made of a clip.
pub struct Analysis {
    pub window: Range<usize>,
    pub tags: Tags,
    /// The ranked classes of each 0.96 s patch of the window.
    pub patches: Vec<Vec<(String, f32)>>,
    /// The model's view of the window without room tone, best first.
    pub sounds: Vec<(String, f32)>,
    /// The sounds heard in the pre-roll, or None when the clip has none.
    pub background: Option<Vec<String>>,
    pub mix: Vec<MixPart>,
}

/// Runs the model on the clip's window, and on its pre-roll to tell what was there before the
/// onset. The pre-roll is never embedded.
pub fn analyse(tagger: &Tagger, clip: &Clip, sound_score: f32) -> Result<Analysis> {
    let ranked = |tags: &Tags| -> Vec<Vec<(String, f32)>> {
        tags.patch_scores.iter().map(|p| tagger.top_classes(p, PATCH_CLASSES)).collect()
    };
    let window = analysis_window(clip);
    let tags = tagger.tag(&clip.samples[window.clone()])?;
    let patches = ranked(&tags);
    let before = &clip.samples[..clip.onset.saturating_sub(ONSET_MARGIN_SAMPLES)];
    let background = if before.is_empty() {
        None
    } else {
        let parts = classify::mix(&ranked(&tagger.tag(before)?), &[], sound_score);
        Some(parts.into_iter().map(|p| p.class).collect::<Vec<_>>())
    };
    let mix = classify::mix(&patches, background.as_deref().unwrap_or_default(), sound_score);
    let sounds = classify::sounds(tagger.top_classes(&tags.scores, 20));
    Ok(Analysis { window, tags, patches, sounds, background, mix })
}

/// The settings a classification was made with, stored next to it.
#[derive(Serialize)]
struct SettingsInEffect<'a> {
    trigger: &'a crate::config::Trigger,
    classify: &'a crate::config::Classify,
}

/// The most recent event and when it ended, for grouping the next clip into it.
#[derive(Clone, Copy)]
struct Latest {
    id: i64,
    end_ts: i64,
}

fn handle_clip(config: &Config, store: &Store, tagger: &Tagger, clip: &Clip, last: &mut Option<Latest>) -> Result<()> {
    let started = Instant::now();
    let c = &config.classify;
    let Analysis { window, tags, patches, sounds, background, mix } = analyse(tagger, clip, c.sound_score)?;
    let top = &sounds[..sounds.len().min(3)];
    let exemplars = store.exemplars()?;
    let decision = classify::decide(&tags.embedding, &exemplars, top, &mix, c);
    let candidates = classify::candidates(&tags.embedding, &exemplars, 5);
    let classify_ms = started.elapsed().as_millis() as i64;
    let ms = |samples: usize| samples as i64 * 1000 / SAMPLE_RATE as i64;
    let (onset_ts, end_ts) = (clip.ts + ms(clip.onset), clip.ts + ms(clip.samples.len()));

    let event_id = 'event: {
        if decision.status == classify::Status::Silent {
            eprintln!("dropped a clip with nothing but silence or room tone");
            break 'event None;
        }
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
            break 'event Some(prev.id);
        }
        let found = Found { decision: &decision, top, embedding: &tags.embedding, candidates: &candidates, mix: &mix };
        Some(new_event(store, clip, &found, last, started)?)
    };

    store.insert_classification(&NewClassification {
        event_id,
        ts: clip.ts,
        duration_ms: ms(clip.samples.len()),
        onset_ms: ms(clip.onset),
        sound_end_ms: ms(clip.sound_end),
        window_ms: (ms(window.start), ms(window.end)),
        peak_db: clip.peak_db,
        ambient_db: clip.ambient_db,
        model: embed::MODEL_NAME,
        scores: &tags.scores,
        embedding: &tags.embedding,
        patches: &patches.iter().map(|p| p[..p.len().min(5)].to_vec()).collect::<Vec<_>>(),
        top_sounds: &sounds[..sounds.len().min(10)],
        decision: &decision,
        candidates: &candidates,
        settings: &serde_json::to_string(&SettingsInEffect { trigger: &config.trigger, classify: c })?,
        classify_ms,
        mix: &mix,
        background: background.as_deref(),
    })?;
    Ok(())
}

/// What the classifier found for a clip that starts a new event.
struct Found<'a> {
    decision: &'a classify::Decision,
    top: &'a [(String, f32)],
    embedding: &'a [f32],
    candidates: &'a [(String, f32, i64)],
    mix: &'a [MixPart],
}

/// Starts an event for a clip that is not grouped into the previous one, keeping its audio
/// unless it is ignored. A match may be sampled for a "was this right?" check.
fn new_event(store: &Store, clip: &Clip, found: &Found, last: &mut Option<Latest>, started: Instant) -> Result<i64> {
    let Found { decision, top, .. } = *found;
    let end_ts = clip.ts + clip.samples.len() as i64 * 1000 / SAMPLE_RATE as i64;
    let clip_path = if decision.status == classify::Status::Ignored {
        None
    } else {
        let day = chrono::DateTime::from_timestamp_millis(clip.ts).unwrap_or_default().format("%Y-%m-%d");
        let rel = format!("clips/{day}/{}.wav", clip.ts);
        write_wav(&store.dir().join(&rel), &clip.samples)?;
        Some(rel)
    };
    let check = decision.status == classify::Status::Match
        && classify::check_due(decision.nearest.as_ref().map_or(0.0, |n| n.1), RandomState::new().hash_one(clip.ts));
    let id = store.insert_event(&NewEvent {
        ts: clip.ts,
        duration_ms: end_ts - clip.ts,
        peak_db: clip.peak_db,
        ambient_db: clip.ambient_db,
        decision,
        top_classes: top,
        clip_path: clip_path.as_deref(),
        embedding: found.embedding,
        nearest_event: found.candidates.first().map(|(.., event)| *event),
        mix: found.mix,
        check,
    })?;
    if decision.status != classify::Status::Ignored {
        *last = Some(Latest { id, end_ts });
    }
    let nearest = decision.nearest.as_ref().map(|(l, s)| format!(", nearest {l} {s:.2}")).unwrap_or_default();
    let conflict = decision.conflict.as_ref().map(|c| format!(", model hears {c}: asking")).unwrap_or_default();
    eprintln!(
        "event {id}: {:?} {:?} ({}{nearest}{conflict}) in {} ms",
        decision.status,
        decision.label,
        top.iter().map(|(n, s)| format!("{n} {s:.2}")).collect::<Vec<_>>().join(", "),
        started.elapsed().as_millis()
    );
    Ok(id)
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

    const PRE_ROLL: usize = 24_000;

    fn clip(len: usize, onset: usize) -> Clip {
        Clip { samples: vec![0.0; len], onset, sound_end: len, ts: 0, peak_db: 0.0, ambient_db: 0.0 }
    }

    fn shared(config: &Config) -> SharedSettings {
        let settings = Settings { config: config.clone(), overrides: Overrides::new(), mode: Mode::Active, version: 0 };
        Arc::new(Mutex::new(settings))
    }

    #[test]
    fn window_skips_preroll_for_long_sounds() {
        let c = clip(PRE_ROLL + 40_000, PRE_ROLL);
        assert_eq!(analysis_window(&c).len(), 40_000 + ONSET_MARGIN_SAMPLES);
    }

    #[test]
    fn window_skips_the_tail() {
        let c = Clip { sound_end: PRE_ROLL + 40_000, ..clip(PRE_ROLL + 72_000, PRE_ROLL) };
        assert_eq!(analysis_window(&c).end, PRE_ROLL + 40_000);
    }

    #[test]
    fn window_widens_into_preroll_for_short_sounds() {
        let c = clip(PRE_ROLL + 4_000, PRE_ROLL);
        assert_eq!(analysis_window(&c).len(), embed::MIN_SAMPLES);
        let without_preroll = Clip { sound_end: 4_000, ..clip(40_000, 0) };
        assert_eq!(analysis_window(&without_preroll).len(), embed::MIN_SAMPLES);
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
        let at = |ts: i64| Clip {
            samples: sound.clone(),
            onset: 0,
            sound_end: sound.len(),
            ts,
            peak_db: -20.0,
            ambient_db: -60.0,
        };
        let mut last = None;
        // 2.5 s clips: the second starts 2 s after the first ends, the third 60 s later.
        for ts in [0, 4_500, 67_000] {
            handle_clip(&config, &store, &tagger, &at(1_000_000 + ts), &mut last).unwrap();
        }
        let events = store.events(&crate::store::Filter { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(events.iter().map(|e| (e.parts, e.duration_ms)).collect::<Vec<_>>(), vec![(1, 2_500), (2, 7_000)]);
        // every clip keeps its own classification, the grouped one included
        let kept = crate::store::classifications(&store);
        let (first, second) = (Some(events[1].id), Some(events[0].id));
        assert_eq!(kept.iter().map(|k| k.0).collect::<Vec<_>>(), vec![first, first, second]);
        assert!(kept.iter().all(|k| k.3 >= 1 && k.4 == 521), "patches and all class scores: {kept:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_silent_clip_makes_no_event_but_its_classification_is_kept() {
        let dir = std::env::temp_dir().join(format!("creak-silent-{}", std::process::id()));
        let mut config = Config::default();
        config.storage.dir = dir.clone();
        let store = Store::open(&dir).unwrap();
        let tagger = Tagger::load().unwrap();
        let quiet = vec![0.0; 40_000];
        let clip =
            Clip { samples: quiet, onset: 0, sound_end: 40_000, ts: 1_000_000, peak_db: -50.0, ambient_db: -60.0 };
        handle_clip(&config, &store, &tagger, &clip, &mut None).unwrap();
        assert!(store.events(&crate::store::Filter { limit: 10, ..Default::default() }).unwrap().is_empty());
        let kept = crate::store::classifications(&store);
        assert_eq!(
            kept.iter().map(|k| (k.0, k.1.as_str(), k.2.as_str())).collect::<Vec<_>>(),
            vec![(None, "silent", "")]
        );
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
            heard.1 >= config.classify.conflict_score && heard.0 != "Music",
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
                nearest_event: None,
                mix: &[],
                check: false,
            })
            .unwrap();
        store.label_event(music, "music").unwrap();

        let clip =
            Clip { sound_end: sound.len(), samples: sound, onset: 0, ts: 1_003_000, peak_db: -20.0, ambient_db: -60.0 };
        let mut last = Some(Latest { id: music, end_ts: 1_002_500 });
        handle_clip(&config, &store, &tagger, &clip, &mut last).unwrap();
        let newest = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_ne!(newest.id, music);
        assert_eq!((newest.status.as_str(), newest.label.as_str()), ("pending", "music"));
        assert_eq!(newest.conflict.as_deref(), Some(heard.0.as_str()));
        assert_eq!(store.event(music).unwrap().unwrap().parts, 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Steady noise, which the model hears as running water, then a square-wave alarm over it.
    /// Its whole clip is close to a water clip and the model's best sound is still water, so
    /// without the pre-roll it would be matched as water; the alarm is what came in.
    #[test]
    fn a_sound_over_the_background_is_not_tagged_as_the_background() {
        let dir = std::env::temp_dir().join(format!("creak-foreground-{}", std::process::id()));
        let mut config = Config::default();
        config.storage.dir = dir.clone();
        let mut store = Store::open(&dir).unwrap();
        let tagger = Tagger::load().unwrap();
        let mut seed = 7u32;
        let water: Vec<f32> = (0..PRE_ROLL + 48_000)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 * 0.6 - 0.3
            })
            .collect();
        let alarm = |i: usize| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin().signum() * 0.3;
        let mut with_alarm = water.clone();
        with_alarm[PRE_ROLL + 16_000..PRE_ROLL + 32_000].iter_mut().enumerate().for_each(|(i, s)| *s += alarm(i));
        let clip = |samples: Vec<f32>, ts: i64| Clip {
            onset: PRE_ROLL,
            sound_end: samples.len(),
            samples,
            ts,
            peak_db: -10.0,
            ambient_db: -20.0,
        };

        handle_clip(&config, &store, &tagger, &clip(water, 1_000_000), &mut None).unwrap();
        let taught = store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0].id;
        store.label_event(taught, "water").unwrap();

        let analysis = analyse(&tagger, &clip(with_alarm.clone(), 0), config.classify.sound_score).unwrap();
        assert_eq!(analysis.sounds[0].0, "Waterfall", "the model's best sound is the background");
        assert!(analysis.mix.iter().any(|p| p.class == "Waterfall" && p.background));
        let without_preroll = Clip { onset: 0, ..clip(with_alarm[PRE_ROLL..].to_vec(), 0) };
        let blind = analyse(&tagger, &without_preroll, config.classify.sound_score).unwrap();
        let exemplars = store.exemplars().unwrap();
        let d = classify::decide(&blind.tags.embedding, &exemplars, &blind.sounds, &blind.mix, &config.classify);
        assert_eq!((d.status, d.label.as_str()), (classify::Status::Match, "water"), "{d:?}");

        handle_clip(&config, &store, &tagger, &clip(with_alarm, 1_100_000), &mut None).unwrap();
        let e = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_eq!((e.status.as_str(), e.nearest_label.as_deref()), ("pending", Some("water")));
        let heard = e.conflict.clone().expect("the model's sound over the water");
        assert_eq!(e.label, heard.to_lowercase(), "proposed as what came in, not as water");
        assert!(e.mix.iter().any(|p| p.class == heard && !p.background));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Quiet noise, then the fixture sound, then quiet again: one clip with pre-roll that runs
    /// to the shortest clip length, tagged and stored as pending; after labelling it, the same sound is a match. Deaf, the
    /// same audio gives no clip.
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

        let feed = |audio: Vec<f32>, mode: Mode| {
            let (chunks_tx, chunks_rx) = channel();
            let (clips_tx, clips_rx) = channel();
            for chunk in audio.chunks(1000) {
                chunks_tx.send(chunk.to_vec()).unwrap();
            }
            drop(chunks_tx);
            let settings = shared(&config);
            settings.lock().unwrap().mode = mode;
            let err = capture_loop(&settings, &chunks_rx, SAMPLE_RATE, &SharedLive::default(), &clips_tx).unwrap_err();
            assert!(err.to_string().contains("closed"));
            clips_rx.try_iter().collect::<Vec<Clip>>()
        };
        let audio = [noise(16_000 * 6), sound.clone(), noise(16_000 * 15)].concat();
        assert!(feed(audio.clone(), Mode::Deaf).is_empty());
        let clips = feed(audio.clone(), Mode::Active);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].onset, PRE_ROLL);
        let from_onset = clips[0].samples.len() - clips[0].onset;
        assert_eq!(from_onset, config.trigger.min_clip_s as usize * 16_000);
        assert!(clips[0].sound_end < clips[0].samples.len(), "the padding after the sound is not part of it");

        let mut store = Store::open(&dir).unwrap();
        let tagger = Tagger::load().unwrap();
        handle_clip(&config, &store, &tagger, &clips[0], &mut None).unwrap();
        let first = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_eq!(first.status, "pending");
        assert!(dir.join(first.clip_path.as_ref().unwrap()).exists());

        store.label_event(first.id, "chirp").unwrap();
        handle_clip(&config, &store, &tagger, &feed(audio, Mode::Active).remove(0), &mut None).unwrap();
        let second = &store.events(&crate::store::Filter { limit: 1, ..Default::default() }).unwrap()[0];
        assert_eq!((second.status.as_str(), second.label.as_str()), ("match", "chirp"));
        let _ = std::fs::remove_dir_all(dir);
    }
}

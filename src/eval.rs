//! `creak eval`: replays the audited clips in time order through the tagging policy, starting
//! from no exemplars, so the policy is scored on labels it never suggested (plan §3.8). Where it
//! would ask, it learns the audited label; where it would tag automatically, the tag is scored.
//! Candidate changes to matching are replayed on the same clips next to the current policy.

use crate::classify::{self, Exemplar, MixPart, Status};
use crate::config::{Classify, Config};
use crate::daemon::{self, Clip};
use crate::embed::{self, Tagger};
use crate::store::{MatchStats, Store};
use crate::trigger::SAMPLE_RATE;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write;

/// A clip with two or more sounds, each in at least this share of its patches, is mixed.
const MIXED_SHARE: f32 = 0.25;
/// A change is kept when it lowers wrong tags and asks at most this many times as often.
const MAX_QUESTIONS_RATIO: f64 = 1.25;
/// Audited clips the first test set needs before the numbers mean much.
const ENOUGH_CLIPS: usize = 50;

/// One audited clip, as the policy sees it.
pub struct Sample {
    pub event_id: i64,
    /// What the audit says it is.
    pub truth: String,
    /// Mean over the analysed patches, L2-normalised.
    pub embedding: Vec<f32>,
    pub sounds: Vec<(String, f32)>,
    pub mix: Vec<MixPart>,
    /// The mean embedding of only the patches in which the heard sound is present, normalised.
    pub heard_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variant {
    Current,
    /// Proposal and conflict from the model's best sound overall, without the pre-roll.
    NoForeground,
    /// Tag automatically only when the best label beats the best other label by this much.
    Margin(f32),
    /// Compare embeddings after subtracting the mean embedding of the audited clips.
    Centred,
    /// Never learn from a mixed clip.
    SkipMixed,
    /// Learn only from the patches in which the heard sound is present.
    HeardPatches,
}

const CANDIDATES: &[(Variant, &str)] = &[
    (Variant::NoForeground, "without the foreground"),
    (Variant::Margin(0.03), "margin 0.03"),
    (Variant::Margin(0.06), "margin 0.06"),
    (Variant::Centred, "centred embeddings"),
    (Variant::SkipMixed, "never learn from mixed clips"),
    (Variant::HeardPatches, "learn from the heard patches only"),
];

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Tally {
    pub clips: usize,
    pub asked: usize,
    pub auto: usize,
    pub wrong: usize,
    /// Dropped as silence although the audit heard a sound.
    pub dropped: usize,
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub total: Tally,
    /// Questions whose proposal was already the audited label.
    pub proposed_right: usize,
    pub labels: BTreeMap<String, Tally>,
    /// (audited, tagged) → count, over the wrong automatic tags.
    pub confusion: BTreeMap<(String, String), usize>,
}

enum Kind {
    Dropped,
    Asked,
    Right,
    Wrong,
}

/// Replays `samples`, which are in time order, from no exemplars.
pub fn replay(samples: &[Sample], variant: Variant, c: &Classify) -> Outcome {
    let centre = (variant == Variant::Centred).then(|| mean(samples.iter().map(|s| &s.embedding)));
    let mut exemplars: Vec<Exemplar> = Vec::new();
    let mut out = Outcome::default();
    for s in samples {
        let mix: &[MixPart] = if variant == Variant::NoForeground { &[] } else { &s.mix };
        let embedding = match &centre {
            Some(m) => embed::normalized(s.embedding.iter().zip(m).map(|(x, m)| x - m).collect()),
            None => s.embedding.clone(),
        };
        let mut d = classify::decide(&embedding, &exemplars, &s.sounds, mix, c);
        if let Variant::Margin(margin) = variant
            && matches!(d.status, Status::Match | Status::Ignored)
        {
            let best = classify::candidates(&embedding, &exemplars, 2);
            if best.len() == 2 && best[0].1 - best[1].1 < margin {
                d.status = Status::Pending;
            }
        }

        let kind = match d.status {
            Status::Silent => Kind::Dropped,
            Status::Pending => Kind::Asked,
            Status::Match | Status::Ignored if d.label == s.truth => Kind::Right,
            Status::Match | Status::Ignored => Kind::Wrong,
        };
        for t in [&mut out.total, out.labels.entry(s.truth.clone()).or_default()] {
            t.clips += 1;
            match kind {
                Kind::Dropped => t.dropped += 1,
                Kind::Asked => t.asked += 1,
                Kind::Right => t.auto += 1,
                Kind::Wrong => (t.auto, t.wrong) = (t.auto + 1, t.wrong + 1),
            }
        }
        match kind {
            Kind::Wrong => *out.confusion.entry((s.truth.clone(), d.label)).or_default() += 1,
            Kind::Asked => {
                out.proposed_right += (d.label == s.truth) as usize;
                let learned = match variant {
                    Variant::SkipMixed if is_mixed(&s.mix) => None,
                    Variant::HeardPatches => Some(s.heard_embedding.clone().unwrap_or(embedding)),
                    _ => Some(embedding),
                };
                if let Some(embedding) = learned {
                    let heard = classify::heard(&s.sounds, mix);
                    exemplars.push(Exemplar { event_id: s.event_id, label: s.truth.clone(), embedding, heard });
                }
            }
            Kind::Dropped | Kind::Right => {}
        }
    }
    out
}

/// The replay at every match threshold from `from` to 0.99, in steps of 0.01.
pub fn sweep(samples: &[Sample], variant: Variant, c: &Classify, from: f32) -> Vec<(f32, Tally)> {
    let first = (from * 100.0).round() as u32;
    (first..=99)
        .map(|t| {
            let t = t as f32 / 100.0;
            let c = Classify { match_threshold: t, suggest_threshold: c.suggest_threshold.min(t), ..c.clone() };
            (t, replay(samples, variant, &c).total)
        })
        .collect()
}

/// The plan's rule for keeping a change: fewer wrong tags, and at most a quarter more questions.
pub fn keep(change: &Tally, current: &Tally) -> bool {
    change.wrong < current.wrong && change.asked as f64 <= current.asked as f64 * MAX_QUESTIONS_RATIO
}

/// The point of a sweep with the fewest wrong tags that asks at most a quarter more than
/// `current`; without one, the point that asks least.
fn best_point<'a>(sweep: &'a [(f32, Tally)], current: &Tally) -> &'a (f32, Tally) {
    let cap = current.asked as f64 * MAX_QUESTIONS_RATIO;
    sweep
        .iter()
        .filter(|(_, t)| t.asked as f64 <= cap)
        .min_by_key(|(_, t)| (t.wrong, t.asked))
        .or_else(|| sweep.iter().min_by_key(|(_, t)| t.asked))
        .expect("a sweep has points")
}

fn is_mixed(mix: &[MixPart]) -> bool {
    mix.iter().filter(|p| p.share >= MIXED_SHARE).count() >= 2
}

fn mean<'a>(rows: impl Iterator<Item = &'a Vec<f32>>) -> Vec<f32> {
    let mut sum: Vec<f32> = Vec::new();
    let mut n = 0;
    for row in rows {
        sum.resize(row.len(), 0.0);
        sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
        n += 1;
    }
    sum.iter_mut().for_each(|s| *s /= n.max(1) as f32);
    sum
}

/// Everything `creak eval` reports on.
#[derive(Default)]
pub struct Input {
    pub samples: Vec<Sample>,
    /// Audited clips that could not be told.
    pub unclear: usize,
    /// Clips without a stored onset, for which the pre-roll setting was assumed.
    pub assumed_onset: usize,
    /// Exemplars whose audit says otherwise: (event, taught label, audited label).
    pub disagreements: Vec<(i64, String, String)>,
    pub matches: Vec<MatchStats>,
}

/// Replays the audited clips and prints the report, with the settings in effect in the daemon.
pub fn run(file: &Config) -> Result<()> {
    let store = Store::open(&file.storage.dir)?;
    let config = file.with_overrides(&store.overrides()?).unwrap_or_else(|_| file.clone());
    let input = load(&store, &Tagger::load()?, &config)?;
    print!("{}", report(&input, &config.classify));
    Ok(())
}

/// Runs the model again on every audited clip that still has its audio.
pub fn load(store: &Store, tagger: &Tagger, config: &Config) -> Result<Input> {
    let audited = store.audited()?;
    let mut input = Input::default();
    for a in &audited {
        let path = store.dir().join(&a.clip_path);
        let wave = match read_wav(&path) {
            Ok(wave) => wave,
            Err(e) => {
                eprintln!("skipping event {}: {e:#}", a.event_id);
                continue;
            }
        };
        let at = |ms: i64| (ms.max(0) as usize * SAMPLE_RATE as usize / 1000).min(wave.len());
        input.assumed_onset += a.onset_ms.is_none() as usize;
        let onset = at(a.onset_ms.unwrap_or(config.trigger.pre_roll_ms as i64));
        let sound_end = a.sound_end_ms.map_or(wave.len(), at).max(onset);
        let clip = Clip { samples: wave, onset, sound_end, ts: a.ts, peak_db: 0.0, ambient_db: 0.0 };
        input.samples.push(sample(tagger, &clip, a.event_id, &a.label, config.classify.sound_score)?);
    }
    input.unclear = store.audit_counts()?.iter().filter(|(label, _)| label.is_none()).map(|(_, n)| *n as usize).sum();
    input.disagreements = audited
        .iter()
        .filter_map(|a| {
            a.exemplar.as_ref().filter(|x| **x != a.label).map(|x| (a.event_id, x.clone(), a.label.clone()))
        })
        .collect();
    input.matches = store.match_stats()?;
    Ok(input)
}

fn sample(tagger: &Tagger, clip: &Clip, event_id: i64, truth: &str, sound_score: f32) -> Result<Sample> {
    let a = daemon::analyse(tagger, clip, sound_score)?;
    let heard_embedding = classify::heard(&a.sounds, &a.mix).and_then(|h| {
        let rows: Vec<&Vec<f32>> = a
            .patches
            .iter()
            .zip(&a.tags.patch_embeddings)
            .filter(|(patch, _)| patch.iter().any(|(class, score)| *class == h.class && *score >= sound_score))
            .map(|(_, e)| e)
            .collect();
        (!rows.is_empty()).then(|| embed::normalized(mean(rows.into_iter())))
    });
    Ok(Sample {
        event_id,
        truth: truth.to_string(),
        embedding: a.tags.embedding,
        sounds: a.sounds,
        mix: a.mix,
        heard_embedding,
    })
}

fn read_wav(path: &std::path::Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(reader.into_samples::<i16>().map(|s| s.map(|s| s as f32 / 32768.0)).collect::<Result<_, _>>()?)
}

fn percent(part: usize, whole: usize) -> String {
    if whole == 0 { "–".into() } else { format!("{:.0} %", part as f64 * 100.0 / whole as f64) }
}

/// The text `creak eval` prints.
pub fn report(input: &Input, c: &Classify) -> String {
    let samples = &input.samples;
    let mut r = String::new();
    let _ = writeln!(
        r,
        "{} audited clips{}, replayed in time order from no taught examples.",
        samples.len(),
        if input.unclear > 0 { format!(" ({} more could not be told)", input.unclear) } else { String::new() }
    );
    if input.assumed_onset > 0 {
        let _ = writeln!(
            r,
            "{} clips have no stored onset; their sound is taken to start after the pre-roll.",
            input.assumed_onset
        );
    }
    if samples.len() < ENOUGH_CLIPS {
        let _ = writeln!(
            r,
            "A first test set needs {ENOUGH_CLIPS}–100 audited clips across your common sounds (the Audit view); \
             with fewer, these numbers are mostly noise."
        );
    }
    if samples.is_empty() {
        write_matches(&mut r, &input.matches);
        return r;
    }

    let current = replay(samples, Variant::Current, c);
    let t = &current.total;
    let _ = writeln!(
        r,
        "\nCurrent policy (match threshold {:.2}, suggest threshold {:.2}):",
        c.match_threshold, c.suggest_threshold
    );
    let _ = writeln!(
        r,
        "  asked about {} of {} ({}) · tagged automatically {} · wrong {} ({} of automatic tags) · dropped as silence {}",
        t.asked,
        t.clips,
        percent(t.asked, t.clips),
        t.auto,
        t.wrong,
        percent(t.wrong, t.auto),
        t.dropped
    );
    let _ = writeln!(r, "  the proposal was already right for {} of the {} questions", current.proposed_right, t.asked);
    let _ =
        writeln!(r, "\n  {:<24} {:>6} {:>6} {:>6} {:>6} {:>8}", "label", "clips", "asked", "auto", "wrong", "dropped");
    let mut labels: Vec<_> = current.labels.iter().collect();
    labels.sort_by(|a, b| b.1.clips.cmp(&a.1.clips).then(a.0.cmp(b.0)));
    for (label, t) in labels {
        let _ =
            writeln!(r, "  {label:<24} {:>6} {:>6} {:>6} {:>6} {:>8}", t.clips, t.asked, t.auto, t.wrong, t.dropped);
    }
    if !current.confusion.is_empty() {
        let _ = writeln!(r, "\nWrong automatic tags (audited → tagged):");
        for ((truth, tagged), n) in &current.confusion {
            let _ = writeln!(r, "  {truth} → {tagged}: {n}");
        }
    }

    let _ = writeln!(
        r,
        "\nMatch threshold sweep, current policy:\n  {:>9} {:>6} {:>6} {:>6}",
        "threshold", "asked", "auto", "wrong"
    );
    for (threshold, t) in sweep(samples, Variant::Current, c, 0.70) {
        let _ = writeln!(r, "  {threshold:>9.2} {:>6} {:>6} {:>6}", t.asked, t.auto, t.wrong);
    }

    let _ = writeln!(
        r,
        "\nCandidate changes, one at a time. Kept when wrong tags drop and questions rise by at most a quarter.\n  \
         {:<34} {:>9} {:>6} {:>6} {:>6}  verdict",
        "change", "threshold", "asked", "auto", "wrong"
    );
    let row = |r: &mut String, name: &str, threshold: f32, t: &Tally, verdict: &str| {
        let _ = writeln!(r, "  {name:<34} {threshold:>9.2} {:>6} {:>6} {:>6}  {verdict}", t.asked, t.auto, t.wrong);
    };
    row(&mut r, "current", c.match_threshold, &current.total, "");
    for (variant, name) in CANDIDATES {
        // Centring moves every similarity, so it is compared at its own best threshold.
        let (threshold, t) = if *variant == Variant::Centred {
            best_point(&sweep(samples, *variant, c, 0.30), &current.total).clone()
        } else {
            (c.match_threshold, replay(samples, *variant, c).total)
        };
        let verdict = if keep(&t, &current.total) { "keep" } else { "not better" };
        row(&mut r, name, threshold, &t, verdict);
    }

    if !input.disagreements.is_empty() {
        let _ = writeln!(r, "\nTaught examples the audit disagrees with:");
        for (id, taught, audited) in &input.disagreements {
            let _ = writeln!(
                r,
                "  #{id} taught as {taught:?}, audited as {audited:?}: `creak label {id} {audited:?}` or `creak skip {id}`"
            );
        }
    }
    write_matches(&mut r, &input.matches);
    r
}

fn write_matches(r: &mut String, matches: &[MatchStats]) {
    if matches.is_empty() {
        return;
    }
    let _ = writeln!(
        r,
        "\nAutomatic tags in use, from the sampled checks:\n  {:<24} {:>8} {:>10} {:>6} {:>6} {:>10}",
        "label", "matches", "corrected", "right", "wrong", "precision"
    );
    for m in matches {
        let precision = m.precision.map_or("–".to_string(), |p| format!("{:.0} %", p * 100.0));
        let _ = writeln!(
            r,
            "  {:<24} {:>8} {:>10} {:>6} {:>6} {:>10}",
            m.label, m.matches, m.corrected, m.right, m.wrong, precision
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(class: &str, share: f32, background: bool) -> MixPart {
        MixPart { class: class.into(), share, peak: 0.6, background }
    }

    fn s(id: i64, truth: &str, v: [f32; 2], class: &str) -> Sample {
        Sample {
            event_id: id,
            truth: truth.into(),
            embedding: embed::normalized(v.to_vec()),
            sounds: vec![(class.into(), 0.5)],
            mix: vec![part(class, 1.0, false)],
            heard_embedding: None,
        }
    }

    fn c() -> Classify {
        Classify { match_threshold: 0.9, suggest_threshold: 0.6, ..Default::default() }
    }

    #[test]
    fn asks_once_per_sound_and_learns_the_audited_label() {
        let samples = [
            s(1, "door", [1.0, 0.0], "Door"),
            s(2, "bell", [0.0, 1.0], "Doorbell"),
            s(3, "door", [0.99, 0.05], "Door"),
            s(4, "bell", [0.05, 0.99], "Doorbell"),
        ];
        let out = replay(&samples, Variant::Current, &c());
        assert_eq!(out.total, Tally { clips: 4, asked: 2, auto: 2, wrong: 0, dropped: 0 });
        assert_eq!(out.proposed_right, 1, "the model's Door is the label door; Doorbell is not bell");
        assert_eq!(out.labels["door"], Tally { clips: 2, asked: 1, auto: 1, wrong: 0, dropped: 0 });
    }

    #[test]
    fn wrong_tags_are_scored_against_the_audit() {
        // A cough that sounds like music to the embedding, and the model hears music too.
        let samples = [s(1, "music", [1.0, 0.0], "Music"), s(2, "cough", [0.98, 0.2], "Music")];
        let out = replay(&samples, Variant::Current, &c());
        assert_eq!((out.total.auto, out.total.wrong), (1, 1));
        assert_eq!(out.confusion[&("cough".into(), "music".into())], 1);
        // A margin does not help when there is no other label to beat.
        assert_eq!(replay(&samples, Variant::Margin(0.06), &c()).total.wrong, 1);
    }

    #[test]
    fn a_margin_asks_when_two_labels_are_close() {
        // The third clip is 0.951 from music and 0.947 from tv.
        let close =
            [s(1, "music", [1.0, 0.0], "Music"), s(2, "tv", [0.8, 0.6], "Music"), s(3, "tv", [0.95, 0.31], "Music")];
        let plain = replay(&close, Variant::Current, &c()).total;
        let margin = replay(&close, Variant::Margin(0.06), &c()).total;
        assert_eq!((plain.asked, plain.wrong), (2, 1));
        assert_eq!((margin.asked, margin.wrong), (3, 0));
        assert!(!keep(&margin, &plain), "three questions instead of two is more than a quarter more");
    }

    #[test]
    fn silence_is_a_miss_when_the_audit_heard_something() {
        let mut quiet = s(1, "tick", [1.0, 0.0], "Tick");
        quiet.sounds = vec![("Tick".into(), 0.02)];
        assert_eq!(replay(&[quiet], Variant::Current, &c()).total.dropped, 1);
    }

    #[test]
    fn skipping_mixed_clips_never_learns_them() {
        let mut mixed = s(1, "music", [1.0, 0.0], "Music");
        mixed.mix = vec![part("Music", 1.0, false), part("Speech", 0.5, false)];
        let samples = [mixed, s(2, "music", [1.0, 0.0], "Music")];
        assert_eq!(replay(&samples, Variant::Current, &c()).total.asked, 1);
        assert_eq!(replay(&samples, Variant::SkipMixed, &c()).total.asked, 2);
    }

    #[test]
    fn the_keep_rule_allows_a_quarter_more_questions() {
        let current = Tally { clips: 50, asked: 20, auto: 30, wrong: 3, dropped: 0 };
        assert!(keep(&Tally { asked: 25, wrong: 2, ..current.clone() }, &current));
        assert!(!keep(&Tally { asked: 26, wrong: 2, ..current.clone() }, &current));
        assert!(!keep(&Tally { asked: 20, wrong: 3, ..current.clone() }, &current));
    }

    /// Running water (noise) and an alarm (a square wave), each after 1.5 s of quiet, written
    /// as clips of a store without stored onsets, and audited.
    #[test]
    fn audited_clips_are_replayed_from_their_audio() {
        use crate::store::{NewEvent, now_ms};
        let dir = std::env::temp_dir().join(format!("creak-eval-{}", std::process::id()));
        let mut config = Config::default();
        config.storage.dir = dir.clone();
        let mut store = Store::open(&dir).unwrap();
        let mut seed = 7u32;
        let mut water = || -> Vec<f32> {
            (0..32_000)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 * 0.6 - 0.3
                })
                .collect()
        };
        let alarm: Vec<f32> = (0..32_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin().signum() * 0.3)
            .collect();
        let clips =
            [("water", water()), ("alarm", alarm.clone()), ("water", water()), ("alarm", alarm), ("water", water())];
        let mut ids = Vec::new();
        for (i, (label, sound)) in clips.into_iter().enumerate() {
            let rel = format!("clips/{i}.wav");
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(dir.join(&rel), spec).unwrap();
            for s in [vec![0.0; 24_000], sound].concat() {
                w.write_sample((s * i16::MAX as f32) as i16).unwrap();
            }
            w.finalize().unwrap();
            let decision =
                classify::Decision { status: Status::Pending, label: "x".into(), nearest: None, conflict: None };
            let event = NewEvent {
                ts: now_ms() + i as i64,
                duration_ms: 3_500,
                peak_db: -10.0,
                ambient_db: -60.0,
                decision: &decision,
                top_classes: &[],
                clip_path: Some(&rel),
                embedding: &[1.0],
                nearest_event: None,
                mix: &[],
                check: false,
            };
            let id = store.insert_event(&event).unwrap();
            store.audit(id, (i < 4).then_some(label)).unwrap();
            ids.push(id);
        }
        store.label_event(ids[1], "siren").unwrap();

        let input = load(&store, &Tagger::load().unwrap(), &config).unwrap();
        assert_eq!((input.samples.len(), input.assumed_onset, input.unclear), (4, 4, 1));
        assert_eq!(input.disagreements, vec![(ids[1], "siren".to_string(), "alarm".to_string())]);
        assert_eq!(
            input.samples.iter().map(|s| s.truth.as_str()).collect::<Vec<_>>(),
            ["water", "alarm", "water", "alarm"]
        );
        assert!(input.samples.iter().all(|s| s.heard_embedding.is_some() && !s.mix.is_empty()));
        let out = replay(&input.samples, Variant::Current, &config.classify);
        assert_eq!(out.total, Tally { clips: 4, asked: 2, auto: 2, wrong: 0, dropped: 0 }, "{out:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_report_covers_the_plan() {
        let samples = [s(1, "music", [1.0, 0.0], "Music"), s(2, "cough", [0.98, 0.2], "Music")];
        let input = Input {
            samples: samples.into(),
            unclear: 1,
            assumed_onset: 0,
            disagreements: vec![(7, "music".to_string(), "cough".to_string())],
            matches: vec![MatchStats {
                label: "music".into(),
                matches: 4,
                right: 1,
                precision: Some(1.0),
                ..Default::default()
            }],
        };
        let text = report(&input, &c());
        for expected in [
            "2 audited clips (1 more could not be told)",
            "needs 50–100 audited clips",
            "asked about 1 of 2",
            "cough → music: 1",
            "Match threshold sweep",
            "margin 0.03",
            "centred embeddings",
            "#7 taught as \"music\", audited as \"cough\"",
            "100 %",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }
}

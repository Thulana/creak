//! Nearest-exemplar matching and the proposal rules from plan §3.2. Pure.

use crate::config;
use serde::{Deserialize, Serialize};

/// Reserved label: clips that match it are logged without keeping the audio.
pub const IGNORE: &str = "ignore";
/// Automatic matches below this similarity all go to a "was this right?" check …
pub const CHECK_BELOW: f32 = 0.9;
/// … and one in this many above it.
pub const CHECK_ONE_IN: u64 = 10;
/// Not accepted as a label: a clip with nothing in it is dropped, not labelled.
pub const SILENCE: &str = "silence";
/// Model classes that describe the room or the recording rather than a sound in it. They are
/// never proposed or shown, and a clip with nothing else in it is dropped.
const NOT_A_SOUND: &[&str] = &[
    "Silence",
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
    "Outside, urban or manmade",
    "Outside, rural or natural",
    "Reverberation",
    "Echo",
    "Noise",
    "Environmental noise",
    "Static",
    "Mains hum",
    "White noise",
    "Pink noise",
    "Field recording",
];

pub struct Exemplar {
    /// The event whose clip this exemplar is.
    pub event_id: i64,
    pub label: String,
    /// L2-normalised.
    pub embedding: Vec<f32>,
    /// What the model heard in this exemplar's clip, from [`heard`].
    pub heard: Option<Heard>,
}

/// One sound in a clip: the share of its 0.96 s patches in which the model hears it at
/// `sound_score` or more, and its best score in any patch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixPart {
    pub class: String,
    pub share: f32,
    pub peak: f32,
    /// Already heard in the pre-roll, so it is not what triggered the clip.
    pub background: bool,
}

/// The sound the model takes to be the one that triggered a clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Heard {
    pub class: String,
    pub score: f32,
    /// It came in with the onset over a sound that was already there.
    pub foreground: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Match,
    Pending,
    Ignored,
    /// Nothing but silence or room tone: not an event, nothing is stored.
    Silent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub status: Status,
    /// The applied label for a match, the proposal for a pending clip.
    pub label: String,
    /// Nearest exemplar and its cosine similarity, recorded even below the thresholds.
    pub nearest: Option<(String, f32)>,
    /// The model's class when it is sure of a sound that none of the label's exemplars was
    /// heard as. Such a clip goes to review instead of being matched.
    pub conflict: Option<String>,
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn is_sound(class: &str) -> bool {
    !NOT_A_SOUND.contains(&class)
}

/// The model's ranked classes without the ones that are not a sound, best first.
pub fn sounds(ranked: Vec<(String, f32)>) -> Vec<(String, f32)> {
    ranked.into_iter().filter(|(name, _)| is_sound(name)).collect()
}

/// The sound mix of a clip from the ranked classes of each of its patches, most present first.
/// `background` names the sounds heard before the onset.
pub fn mix(patches: &[Vec<(String, f32)>], background: &[String], sound_score: f32) -> Vec<MixPart> {
    let mut parts: Vec<MixPart> = Vec::new();
    for patch in patches {
        for (class, score) in patch.iter().filter(|(class, score)| *score >= sound_score && is_sound(class)) {
            match parts.iter_mut().find(|p| p.class == *class) {
                Some(p) => (p.share, p.peak) = (p.share + 1.0, p.peak.max(*score)),
                None => parts.push(MixPart {
                    class: class.clone(),
                    share: 1.0,
                    peak: *score,
                    background: background.contains(class),
                }),
            }
        }
    }
    parts.iter_mut().for_each(|p| p.share /= patches.len() as f32);
    parts.sort_by(|a, b| b.share.total_cmp(&a.share).then(b.peak.total_cmp(&a.peak)));
    parts
}

/// What the model heard as the sound that triggered a clip: its best sound overall, unless that
/// sound was already there before the onset. Then it is the sound that came in with the onset
/// and peaked highest, scored by that peak.
pub fn heard(sounds: &[(String, f32)], mix: &[MixPart]) -> Option<Heard> {
    let (class, score) = sounds.first()?;
    if mix.iter().any(|p| p.background && p.class == *class)
        && let Some(p) = mix.iter().filter(|p| !p.background).max_by(|a, b| a.peak.total_cmp(&b.peak))
    {
        return Some(Heard { class: p.class.clone(), score: p.peak, foreground: true });
    }
    Some(Heard { class: class.clone(), score: *score, foreground: false })
}

/// `sounds` is the model's view from [`sounds`] and `mix` the clip's sound mix from [`mix`].
/// What the model heard ([`heard`]) is the first guess when no exemplar is close; a clip with
/// no sound strong enough is [`Status::Silent`]. A close exemplar is applied without asking only
/// when the model does not contradict it. When it does with a sound that came in over the
/// background, that sound is proposed instead of the exemplar's label.
pub fn decide(
    embedding: &[f32],
    exemplars: &[Exemplar],
    sounds: &[(String, f32)],
    mix: &[MixPart],
    c: &config::Classify,
) -> Decision {
    let nearest = exemplars.iter().map(|e| (e, dot(embedding, &e.embedding))).max_by(|a, b| a.1.total_cmp(&b.1));
    let audible = sounds.first().is_some_and(|(_, score)| *score >= c.sound_score);
    let heard = heard(sounds, mix).filter(|_| audible);
    let conflict_with = |label: &str| {
        heard
            .as_ref()
            .filter(|h| h.score >= c.conflict_score)
            .filter(|h| {
                !exemplars.iter().any(|e| {
                    e.label == label && e.heard.as_ref().is_some_and(|k| k.class == h.class && k.score >= c.sound_score)
                })
            })
            .map(|h| h.class.clone())
    };
    let proposal = |label: &str, conflict: &Option<String>| match (conflict, &heard) {
        (Some(class), Some(h)) if h.foreground => normalize_label(class).unwrap_or_default(),
        _ => label.to_string(),
    };

    let (status, label, conflict) = match (nearest, &heard) {
        (Some((e, sim)), _) if sim >= c.match_threshold => match conflict_with(&e.label) {
            Some(class) => (Status::Pending, proposal(&e.label, &Some(class.clone())), Some(class)),
            None if e.label == IGNORE => (Status::Ignored, IGNORE.to_string(), None),
            None => (Status::Match, e.label.clone(), None),
        },
        (_, None) => (Status::Silent, String::new(), None),
        (Some((e, sim)), _) if sim >= c.suggest_threshold => {
            let conflict = conflict_with(&e.label);
            (Status::Pending, proposal(&e.label, &conflict), conflict)
        }
        (_, Some(h)) => (Status::Pending, normalize_label(&h.class).unwrap_or_default(), None),
    };
    Decision { status, label, nearest: nearest.map(|(e, sim)| (e.label.clone(), sim)), conflict }
}

/// Whether an automatic match goes to a "was this right?" check; `draw` is a random number.
pub fn check_due(similarity: f32, draw: u64) -> bool {
    similarity < CHECK_BELOW || draw.is_multiple_of(CHECK_ONE_IN)
}

/// How many automatic matches one checked match stands for, given how they were sampled.
pub fn check_weight(similarity: f32) -> f64 {
    if similarity < CHECK_BELOW { 1.0 } else { CHECK_ONE_IN as f64 }
}

/// The best similarity of each label's exemplars to `embedding`, with the event of that
/// exemplar, best first; at most `n`.
pub fn candidates(embedding: &[f32], exemplars: &[Exemplar], n: usize) -> Vec<(String, f32, i64)> {
    let mut best: Vec<(String, f32, i64)> = Vec::new();
    for e in exemplars {
        let sim = dot(embedding, &e.embedding);
        match best.iter_mut().find(|(label, ..)| *label == e.label) {
            Some(b) if sim > b.1 => (b.1, b.2) = (sim, e.event_id),
            Some(_) => {}
            None => best.push((e.label.clone(), sim, e.event_id)),
        }
    }
    best.sort_by(|a, b| b.1.total_cmp(&a.1));
    best.truncate(n);
    best
}

/// Labels are free text: trimmed, lower-case, single spaces, at most 64 characters.
pub fn normalize_label(raw: &str) -> Option<String> {
    let label = raw.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    (!label.is_empty() && label.chars().count() <= 64).then_some(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(label: &str, v: [f32; 2]) -> Exemplar {
        Exemplar { event_id: 0, label: label.into(), embedding: v.to_vec(), heard: None }
    }

    fn heard_as(label: &str, v: [f32; 2], class: &str) -> Exemplar {
        Exemplar { heard: Some(Heard { class: class.into(), score: 0.5, foreground: false }), ..ex(label, v) }
    }

    fn classify() -> config::Classify {
        config::Classify { match_threshold: 0.8, suggest_threshold: 0.6, ..Default::default() }
    }

    fn decide(embedding: &[f32], exemplars: &[Exemplar], sounds: &[(String, f32)]) -> Decision {
        super::decide(embedding, exemplars, sounds, &[], &classify())
    }

    fn patch(classes: &[(&str, f32)]) -> Vec<(String, f32)> {
        classes.iter().map(|(c, s)| (c.to_string(), *s)).collect()
    }

    /// Four patches of music, one with a cough in it; music was already playing before the onset.
    fn cough_over_music() -> (Vec<(String, f32)>, Vec<MixPart>) {
        let music = patch(&[("Music", 0.8), ("Silence", 0.2)]);
        let patches = [music.clone(), patch(&[("Music", 0.6), ("Cough", 0.55)]), music.clone(), music];
        let sounds = vec![("Music".into(), 0.75), ("Cough".into(), 0.14)];
        (sounds, mix(&patches, &["Music".into()], 0.1))
    }

    /// The model hears a door, but not surely enough to contradict anything.
    fn run(v: [f32; 2], exemplars: &[Exemplar]) -> Decision {
        decide(&v, exemplars, &[("Door".into(), 0.2)])
    }

    #[test]
    fn no_exemplars_proposes_model_guess() {
        let d = run([1.0, 0.0], &[]);
        assert_eq!(d, Decision { status: Status::Pending, label: "door".into(), nearest: None, conflict: None });
    }

    #[test]
    fn close_exemplar_matches() {
        let d = run([1.0, 0.0], &[ex("bell", [0.0, 1.0]), ex("front door", [0.9, 0.436])]);
        assert_eq!(d.status, Status::Match);
        assert_eq!(d.label, "front door");
        assert_eq!(d.nearest.unwrap().0, "front door");
    }

    #[test]
    fn middling_exemplar_is_proposed_not_applied() {
        let d = run([1.0, 0.0], &[ex("bell", [0.7, 0.714])]);
        assert_eq!((d.status, d.label.as_str()), (Status::Pending, "bell"));
    }

    #[test]
    fn far_exemplar_falls_back_to_model_but_is_recorded() {
        let d = run([1.0, 0.0], &[ex("bell", [0.3, 0.954])]);
        assert_eq!((d.status, d.label.as_str()), (Status::Pending, "door"));
        assert!((d.nearest.unwrap().1 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn ignore_match_is_ignored() {
        let d = run([1.0, 0.0], &[ex(IGNORE, [1.0, 0.0])]);
        assert_eq!(d.status, Status::Ignored);
    }

    #[test]
    fn room_and_silence_classes_are_not_sounds() {
        let ranked = vec![("Silence".into(), 0.55), ("Inside, small room".into(), 0.3), ("Cough".into(), 0.23)];
        assert_eq!(sounds(ranked), vec![("Cough".to_string(), 0.23)]);
    }

    #[test]
    fn any_real_sound_beats_silence() {
        let d = decide(&[1.0, 0.0], &[], &[("Cough".into(), 0.23)]);
        assert_eq!((d.status, d.label.as_str()), (Status::Pending, "cough"));
    }

    #[test]
    fn nothing_but_silence_is_dropped_unless_taught() {
        let faint = [("Speech".into(), 0.03)];
        let d = decide(&[1.0, 0.0], &[ex("bell", [0.7, 0.714])], &faint);
        assert_eq!(d.status, Status::Silent);
        assert_eq!(decide(&[1.0, 0.0], &[], &[]).status, Status::Silent);
        let d = decide(&[1.0, 0.0], &[ex("tick", [1.0, 0.0])], &faint);
        assert_eq!((d.status, d.label.as_str()), (Status::Match, "tick"));
    }

    #[test]
    fn a_sure_model_holds_a_conflicting_match_for_review() {
        let music = [heard_as("music", [1.0, 0.0], "Music")];
        let d = decide(&[1.0, 0.0], &music, &[("Cough".into(), 0.67)]);
        assert_eq!((d.status, d.label.as_str(), d.conflict.as_deref()), (Status::Pending, "music", Some("Cough")));
        let d = decide(&[1.0, 0.0], &music, &[("Music".into(), 0.7)]);
        assert_eq!((d.status, d.conflict), (Status::Match, None));
        let d = decide(&[1.0, 0.0], &music, &[("Cough".into(), 0.2)]);
        assert_eq!((d.status, d.conflict), (Status::Match, None));
    }

    #[test]
    fn a_confirmed_conflict_teaches_the_label_that_class() {
        let door = [heard_as("door", [1.0, 0.0], "Door"), heard_as("door", [0.0, 1.0], "Slam")];
        let d = decide(&[1.0, 0.0], &door, &[("Slam".into(), 0.6)]);
        assert_eq!((d.status, d.label.as_str()), (Status::Match, "door"));
    }

    #[test]
    fn ignore_does_not_swallow_a_different_sound() {
        let ignored = [heard_as(IGNORE, [1.0, 0.0], "Typing")];
        let d = decide(&[1.0, 0.0], &ignored, &[("Cough".into(), 0.6)]);
        assert_eq!((d.status, d.label.as_str(), d.conflict.as_deref()), (Status::Pending, IGNORE, Some("Cough")));
        let d = decide(&[1.0, 0.0], &ignored, &[("Typing".into(), 0.6)]);
        assert_eq!(d.status, Status::Ignored);
    }

    #[test]
    fn a_suggested_label_carries_the_conflict() {
        let d = decide(&[1.0, 0.0], &[heard_as("bell", [0.7, 0.714], "Bell")], &[("Cough".into(), 0.5)]);
        assert_eq!((d.status, d.label.as_str(), d.conflict.as_deref()), (Status::Pending, "bell", Some("Cough")));
    }

    #[test]
    fn the_mix_is_the_share_and_peak_of_each_sound() {
        let (_, mix) = cough_over_music();
        let parts: Vec<_> = mix.iter().map(|p| (p.class.as_str(), p.share, p.peak, p.background)).collect();
        assert_eq!(parts, vec![("Music", 1.0, 0.8, true), ("Cough", 0.25, 0.55, false)]);
    }

    #[test]
    fn a_sound_over_the_background_is_what_was_heard() {
        let (sounds, mix) = cough_over_music();
        assert_eq!(heard(&sounds, &mix), Some(Heard { class: "Cough".into(), score: 0.55, foreground: true }));
        // A quiet pre-roll: the best sound overall is what triggered the clip.
        let quiet = super::mix(&[patch(&[("Music", 0.8), ("Cough", 0.55)])], &[], 0.1);
        assert_eq!(heard(&sounds, &quiet), Some(Heard { class: "Music".into(), score: 0.75, foreground: false }));
        // Nothing new came in: the background is all there is.
        let steady = super::mix(&[patch(&[("Music", 0.8)])], &["Music".into()], 0.1);
        assert_eq!(heard(&sounds, &steady).unwrap().class, "Music");
    }

    #[test]
    fn a_cough_over_background_music_is_tagged_as_a_cough() {
        let (sounds, mix) = cough_over_music();
        let decide = |exemplars: &[Exemplar]| super::decide(&[1.0, 0.0], exemplars, &sounds, &mix, &classify());
        // Nothing taught yet: proposed as a cough, not as music.
        let d = decide(&[]);
        assert_eq!((d.status, d.label.as_str()), (Status::Pending, "cough"));
        // Close to a music exemplar: not matched as music, but asked about as a cough.
        let d = decide(&[heard_as("music", [1.0, 0.0], "Music")]);
        assert_eq!((d.status, d.label.as_str(), d.conflict.as_deref()), (Status::Pending, "cough", Some("Cough")));
        // Once a cough over music has been confirmed, the next one is tagged without asking.
        let taught = [heard_as("music", [0.0, 1.0], "Music"), heard_as("cough", [0.95, 0.312], "Cough")];
        let d = decide(&taught);
        assert_eq!((d.status, d.label.as_str()), (Status::Match, "cough"));
        // Without knowing the background, the same clip would have been matched as music.
        let d = super::decide(&[1.0, 0.0], &[heard_as("music", [1.0, 0.0], "Music")], &sounds, &[], &classify());
        assert_eq!((d.status, d.label.as_str()), (Status::Match, "music"));
    }

    #[test]
    fn matches_below_0_9_are_all_checked_and_one_in_ten_above() {
        assert!((0..10).all(|draw| check_due(0.86, draw)));
        assert_eq!((0..100).filter(|&draw| check_due(0.95, draw)).count(), 10);
        assert_eq!((check_weight(0.86), check_weight(0.95)), (1.0, 10.0));
    }

    #[test]
    fn candidates_keep_each_labels_best_exemplar() {
        let exemplars = [
            Exemplar { event_id: 1, ..ex("bell", [0.6, 0.8]) },
            Exemplar { event_id: 2, ..ex("door", [0.8, 0.6]) },
            Exemplar { event_id: 3, ..ex("bell", [1.0, 0.0]) },
        ];
        let c = candidates(&[1.0, 0.0], &exemplars, 5);
        assert_eq!(c, vec![("bell".into(), 1.0, 3), ("door".into(), 0.8, 2)]);
        assert_eq!(candidates(&[1.0, 0.0], &exemplars, 1).len(), 1);
    }

    #[test]
    fn labels_are_normalised() {
        assert_eq!(normalize_label("  Front   Door "), Some("front door".into()));
        assert_eq!(normalize_label("   "), None);
        assert_eq!(normalize_label(&"x".repeat(65)), None);
    }
}

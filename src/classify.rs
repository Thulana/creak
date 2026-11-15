//! Nearest-exemplar matching and the proposal rules from plan §3.2. Pure.

use crate::config;
use serde::Serialize;

/// Reserved label: clips that match it are logged without keeping the audio.
pub const IGNORE: &str = "ignore";
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
    pub label: String,
    /// L2-normalised.
    pub embedding: Vec<f32>,
    /// The model's top sound class for this exemplar's clip, with its score.
    pub top_sound: Option<(String, f32)>,
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

/// The model's ranked classes without the ones that are not a sound, best first.
pub fn sounds(ranked: Vec<(String, f32)>) -> Vec<(String, f32)> {
    ranked.into_iter().filter(|(name, _)| !NOT_A_SOUND.contains(&name.as_str())).collect()
}

/// `sounds` is the model's view from [`sounds`]: its best entry is the first guess when no
/// exemplar is close, and a clip without one strong enough is [`Status::Silent`]. A close
/// exemplar is applied without asking only when the model does not contradict it.
pub fn decide(embedding: &[f32], exemplars: &[Exemplar], sounds: &[(String, f32)], c: &config::Classify) -> Decision {
    let nearest = exemplars.iter().map(|e| (e, dot(embedding, &e.embedding))).max_by(|a, b| a.1.total_cmp(&b.1));
    let heard = sounds.first().filter(|(_, score)| *score >= c.sound_score);
    let conflict_with = |label: &str| {
        heard
            .filter(|(_, score)| *score >= c.conflict_score)
            .filter(|(class, _)| {
                !exemplars.iter().any(|e| {
                    e.label == label && e.top_sound.as_ref().is_some_and(|(k, s)| k == class && *s >= c.sound_score)
                })
            })
            .map(|(class, _)| class.clone())
    };

    let (status, label, conflict) = match (nearest, heard) {
        (Some((e, sim)), _) if sim >= c.match_threshold => match conflict_with(&e.label) {
            Some(class) => (Status::Pending, e.label.clone(), Some(class)),
            None if e.label == IGNORE => (Status::Ignored, IGNORE.to_string(), None),
            None => (Status::Match, e.label.clone(), None),
        },
        (_, None) => (Status::Silent, String::new(), None),
        (Some((e, sim)), _) if sim >= c.suggest_threshold => {
            (Status::Pending, e.label.clone(), conflict_with(&e.label))
        }
        (_, Some((class, _))) => (Status::Pending, normalize_label(class).unwrap_or_default(), None),
    };
    Decision { status, label, nearest: nearest.map(|(e, sim)| (e.label.clone(), sim)), conflict }
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
        Exemplar { label: label.into(), embedding: v.to_vec(), top_sound: None }
    }

    fn heard_as(label: &str, v: [f32; 2], class: &str) -> Exemplar {
        Exemplar { top_sound: Some((class.into(), 0.5)), ..ex(label, v) }
    }

    fn decide(embedding: &[f32], exemplars: &[Exemplar], sounds: &[(String, f32)]) -> Decision {
        let c = config::Classify { match_threshold: 0.8, suggest_threshold: 0.6, ..Default::default() };
        super::decide(embedding, exemplars, sounds, &c)
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
    fn labels_are_normalised() {
        assert_eq!(normalize_label("  Front   Door "), Some("front door".into()));
        assert_eq!(normalize_label("   "), None);
        assert_eq!(normalize_label(&"x".repeat(65)), None);
    }
}

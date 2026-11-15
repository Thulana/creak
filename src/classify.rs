//! Nearest-exemplar matching and the proposal rules from plan §3.2. Pure.

use serde::Serialize;

/// Reserved label: clips that match it are logged without keeping the audio.
pub const IGNORE: &str = "ignore";
/// Not accepted as a label: a clip with nothing in it is dropped, not labelled.
pub const SILENCE: &str = "silence";
/// The weakest model score that still counts as hearing a sound.
pub const MIN_SOUND_SCORE: f32 = 0.1;
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
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The model's ranked classes without the ones that are not a sound, best first.
pub fn sounds(ranked: Vec<(String, f32)>) -> Vec<(String, f32)> {
    ranked.into_iter().filter(|(name, _)| !NOT_A_SOUND.contains(&name.as_str())).collect()
}

/// `sounds` is the model's view from [`sounds`]: its best entry is the first guess when no
/// exemplar is close, and a clip without one strong enough is [`Status::Silent`].
pub fn decide(
    embedding: &[f32],
    exemplars: &[Exemplar],
    sounds: &[(String, f32)],
    match_threshold: f32,
    suggest_threshold: f32,
) -> Decision {
    let nearest = exemplars.iter().map(|e| (e, dot(embedding, &e.embedding))).max_by(|a, b| a.1.total_cmp(&b.1));
    let guess =
        sounds.first().filter(|(_, score)| *score >= MIN_SOUND_SCORE).and_then(|(name, _)| normalize_label(name));

    let (status, label) = match (nearest, guess) {
        (Some((e, sim)), _) if sim >= match_threshold && e.label == IGNORE => (Status::Ignored, IGNORE.to_string()),
        (Some((e, sim)), _) if sim >= match_threshold => (Status::Match, e.label.clone()),
        (_, None) => (Status::Silent, String::new()),
        (Some((e, sim)), _) if sim >= suggest_threshold => (Status::Pending, e.label.clone()),
        (_, Some(guess)) => (Status::Pending, guess),
    };
    Decision { status, label, nearest: nearest.map(|(e, sim)| (e.label.clone(), sim)) }
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
        Exemplar { label: label.into(), embedding: v.to_vec() }
    }

    fn run(v: [f32; 2], exemplars: &[Exemplar]) -> Decision {
        decide(&v, exemplars, &[("Door".into(), 0.5)], 0.8, 0.6)
    }

    #[test]
    fn no_exemplars_proposes_model_guess() {
        let d = run([1.0, 0.0], &[]);
        assert_eq!(d, Decision { status: Status::Pending, label: "door".into(), nearest: None });
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
        let d = decide(&[1.0, 0.0], &[], &[("Cough".into(), 0.23)], 0.8, 0.6);
        assert_eq!((d.status, d.label.as_str()), (Status::Pending, "cough"));
    }

    #[test]
    fn nothing_but_silence_is_dropped_unless_taught() {
        let faint = [("Speech".into(), 0.03)];
        let d = decide(&[1.0, 0.0], &[ex("bell", [0.7, 0.714])], &faint, 0.8, 0.6);
        assert_eq!(d.status, Status::Silent);
        assert_eq!(decide(&[1.0, 0.0], &[], &[], 0.8, 0.6).status, Status::Silent);
        let d = decide(&[1.0, 0.0], &[ex("tick", [1.0, 0.0])], &faint, 0.8, 0.6);
        assert_eq!((d.status, d.label.as_str()), (Status::Match, "tick"));
    }

    #[test]
    fn labels_are_normalised() {
        assert_eq!(normalize_label("  Front   Door "), Some("front door".into()));
        assert_eq!(normalize_label("   "), None);
        assert_eq!(normalize_label(&"x".repeat(65)), None);
    }
}

//! Nearest-exemplar matching and the proposal rules from plan §3.2. Pure.

use serde::Serialize;

/// Reserved label: clips that match it are logged without keeping the audio.
pub const IGNORE: &str = "ignore";

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

pub fn decide(
    embedding: &[f32],
    exemplars: &[Exemplar],
    model_guess: &str,
    match_threshold: f32,
    suggest_threshold: f32,
) -> Decision {
    let nearest = exemplars.iter().map(|e| (e, dot(embedding, &e.embedding))).max_by(|a, b| a.1.total_cmp(&b.1));

    let (status, label) = match nearest {
        Some((e, sim)) if sim >= match_threshold && e.label == IGNORE => (Status::Ignored, IGNORE.to_string()),
        Some((e, sim)) if sim >= match_threshold => (Status::Match, e.label.clone()),
        Some((e, sim)) if sim >= suggest_threshold => (Status::Pending, e.label.clone()),
        _ => (Status::Pending, normalize_label(model_guess).unwrap_or_default()),
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
        decide(&v, exemplars, "Door", 0.8, 0.6)
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
    fn labels_are_normalised() {
        assert_eq!(normalize_label("  Front   Door "), Some("front door".into()));
        assert_eq!(normalize_label("   "), None);
        assert_eq!(normalize_label(&"x".repeat(65)), None);
    }
}

use crate::embed::Model;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub const DEFAULT_PATH: &str = "/etc/creak/config.toml";

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub audio: Audio,
    pub trigger: Trigger,
    pub classify: Classify,
    pub storage: Storage,
    pub web: Web,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Audio {
    /// "default", or a device id as printed by `creak devices`.
    pub device: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Trigger {
    pub ratio: f32,
    pub close_ratio: f32,
    pub min_ms: u32,
    pub hang_ms: u32,
    /// Recording goes on this long after the sound is over; a new sound in that time joins the clip.
    pub tail_ms: u32,
    /// A clip runs at least this long from its onset, even when the sound is shorter.
    pub min_clip_s: u32,
    pub max_clip_s: u32,
    pub cooldown_s: u32,
    /// Audio kept from before the onset, for listening only.
    pub pre_roll_ms: u32,
    pub ambient_floor_db: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Classify {
    /// The model that tags clips. Changing it needs a restart; the daemon then analyses the clips
    /// it still has again, so taught examples carry over. The thresholds were set for YAMNet.
    pub model: Model,
    pub match_threshold: f32,
    pub suggest_threshold: f32,
    /// The weakest model score that still counts as hearing a sound.
    pub sound_score: f32,
    /// How sure the model must be of a sound before it can overrule a match and send the clip to review.
    pub conflict_score: f32,
    /// A clip with the same label that starts within this many seconds of the previous
    /// event's end extends that event instead of starting a new one. 0 turns grouping off.
    pub group_gap_s: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Storage {
    pub dir: PathBuf,
    pub retention_days: u32,
    /// The most space all clips together may take; beyond it, the oldest that are not exemplars are deleted.
    pub max_mb: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Web {
    pub enabled: bool,
    pub listen: String,
    /// Required as `?token=` or `Authorization: Bearer` when non-empty.
    pub token: String,
}

impl Default for Audio {
    fn default() -> Self {
        Self { device: "default".into() }
    }
}

impl Default for Trigger {
    fn default() -> Self {
        Self {
            ratio: 3.0,
            close_ratio: 1.5,
            min_ms: 60,
            hang_ms: 800,
            tail_ms: 2000,
            min_clip_s: 15,
            max_clip_s: 60,
            cooldown_s: 2,
            pre_roll_ms: 1500,
            ambient_floor_db: -60.0,
        }
    }
}

impl Default for Classify {
    fn default() -> Self {
        Self {
            model: Model::Yamnet,
            match_threshold: 0.85,
            suggest_threshold: 0.60,
            sound_score: 0.1,
            conflict_score: 0.3,
            group_gap_s: 10,
        }
    }
}

impl Default for Storage {
    fn default() -> Self {
        Self { dir: "/var/lib/creak".into(), retention_days: 14, max_mb: 2000 }
    }
}

impl Default for Web {
    fn default() -> Self {
        Self { enabled: true, listen: "127.0.0.1:8377".into(), token: String::new() }
    }
}

/// Active records as usual; deaf records nothing, for the times you do not want to be heard.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Active,
    Deaf,
}

/// A setting that can be changed in the web UI while the daemon runs. `key` is `section.name`
/// as in the config file; the range applies to the config file too.
#[derive(Debug, Serialize)]
pub struct Field {
    pub key: &'static str,
    pub name: &'static str,
    pub unit: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub help: &'static str,
}

pub const FIELDS: &[Field] = &[
    Field {
        key: "trigger.ratio",
        name: "Open ratio",
        unit: "×",
        min: 1.1,
        max: 100.0,
        step: 0.1,
        help: "A clip opens when the level rises above the ambient level times this (3 is about +9.5 dB).",
    },
    Field {
        key: "trigger.close_ratio",
        name: "Close ratio",
        unit: "×",
        min: 1.0,
        max: 100.0,
        step: 0.1,
        help: "The sound is over once the level stays below the ambient level times this for the hang time. \
               Below the open ratio.",
    },
    Field {
        key: "trigger.ambient_floor_db",
        name: "Ambient floor",
        unit: "dBFS",
        min: -100.0,
        max: -10.0,
        step: 1.0,
        help: "The ambient level is the quietest tenth of the last 5 minutes, but never below this. \
               Raise it if a very quiet room triggers on tiny sounds.",
    },
    Field {
        key: "trigger.min_ms",
        name: "Shortest sound",
        unit: "ms",
        min: 20.0,
        max: 2000.0,
        step: 20.0,
        help: "The level has to stay above the open line this long to open a clip; shorter blips are ignored.",
    },
    Field {
        key: "trigger.hang_ms",
        name: "Hang",
        unit: "ms",
        min: 20.0,
        max: 10000.0,
        step: 20.0,
        help: "How long the level has to stay below the close line before the sound counts as over.",
    },
    Field {
        key: "trigger.tail_ms",
        name: "Tail",
        unit: "ms",
        min: 0.0,
        max: 30000.0,
        step: 100.0,
        help: "Recording goes on this long after the sound is over. A new sound in that time joins the same clip.",
    },
    Field {
        key: "trigger.min_clip_s",
        name: "Shortest clip",
        unit: "s",
        min: 0.0,
        max: 120.0,
        step: 1.0,
        help: "Every clip runs at least this long from its onset, so a short sound comes with what followed it.",
    },
    Field {
        key: "trigger.max_clip_s",
        name: "Longest clip",
        unit: "s",
        min: 1.0,
        max: 300.0,
        step: 1.0,
        help: "A clip is cut here even if the sound goes on. The next clip of the same sound extends the same \
               event, but only the first clip's audio is kept.",
    },
    Field {
        key: "trigger.cooldown_s",
        name: "Cooldown",
        unit: "s",
        min: 0.0,
        max: 600.0,
        step: 1.0,
        help: "No new clip opens for this long after one ends.",
    },
    Field {
        key: "trigger.pre_roll_ms",
        name: "Pre-roll",
        unit: "ms",
        min: 0.0,
        max: 5000.0,
        step: 100.0,
        help: "Audio kept from before the onset, for listening. Pre-roll and tail are not sent to the model.",
    },
    Field {
        key: "classify.match_threshold",
        name: "Match threshold",
        unit: "",
        min: 0.0,
        max: 1.0,
        step: 0.01,
        help: "Similarity to a taught clip at which its label is applied without asking. Higher asks more often.",
    },
    Field {
        key: "classify.suggest_threshold",
        name: "Suggest threshold",
        unit: "",
        min: 0.0,
        max: 1.0,
        step: 0.01,
        help: "Above this similarity the nearest taught label is proposed instead of the model's guess. \
               At most the match threshold.",
    },
    Field {
        key: "classify.sound_score",
        name: "Sound score",
        unit: "",
        min: 0.0,
        max: 1.0,
        step: 0.01,
        help: "The weakest model score that counts as hearing a sound. A clip with nothing this strong is \
               dropped as silence.",
    },
    Field {
        key: "classify.conflict_score",
        name: "Conflict score",
        unit: "",
        min: 0.0,
        max: 1.0,
        step: 0.01,
        help: "How sure the model has to be of a different sound before a match goes to review instead.",
    },
    Field {
        key: "classify.group_gap_s",
        name: "Group gap",
        unit: "s",
        min: 0.0,
        max: 600.0,
        step: 1.0,
        help: "A clip with the same label within this many seconds of the previous event extends that event. \
               0 turns grouping off.",
    },
    Field {
        key: "storage.retention_days",
        name: "Keep clips",
        unit: "days",
        min: 1.0,
        max: 3650.0,
        step: 1.0,
        help: "Clips that are not taught examples are deleted after this many days.",
    },
    Field {
        key: "storage.max_mb",
        name: "Space for all clips",
        unit: "MB",
        min: 10.0,
        max: 1_000_000.0,
        step: 10.0,
        help: "When all clips together take more than this, the oldest are deleted first. Taught examples \
               count towards it but are never deleted. Checked every 10 minutes.",
    },
];

/// Settings changed in the web UI, as `{"section.name": value}` for keys in [`FIELDS`]. They
/// are stored in the database and override the config file until reset.
pub type Overrides = Map<String, Value>;

/// The value of a `section.name` key in a serialised [`Config`].
pub fn lookup<'a>(config: &'a Value, key: &str) -> Option<&'a Value> {
    let (section, name) = key.split_once('.')?;
    config.get(section)?.get(name)
}

/// The config file's `text` with `classify.model` set to `model`; comments and the rest of the
/// file stay as written.
pub fn with_model(text: &str, model: Model) -> Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse().context("parsing the config file")?;
    doc["classify"]["model"] = toml_edit::value(model.name());
    let text = doc.to_string();
    let config: Config = toml::from_str(&text).context("the config file after setting the model")?;
    ensure!(config.classify.model == model, "classify.model did not take; set it by hand");
    Ok(text)
}

impl Config {
    /// Reads `path`. A missing file at the default location means all defaults;
    /// a missing file anywhere else is an error.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).with_context(|| format!("parsing {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && path == Path::new(DEFAULT_PATH) => {
                Ok(Self::default())
            }
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Every field within its range, and the open line above the close line.
    pub fn validate(&self) -> Result<()> {
        let value = serde_json::to_value(self)?;
        for f in FIELDS {
            let v = lookup(&value, f.key).and_then(Value::as_f64).context(f.key)?;
            ensure!((f.min..=f.max).contains(&v), "{} must be between {} and {}", f.key, f.min, f.max);
        }
        let (t, c) = (&self.trigger, &self.classify);
        ensure!(t.close_ratio < t.ratio, "trigger.close_ratio must be below trigger.ratio");
        ensure!(t.min_clip_s <= t.max_clip_s, "trigger.min_clip_s must be at most trigger.max_clip_s");
        ensure!(
            c.suggest_threshold <= c.match_threshold,
            "classify.suggest_threshold must be at most classify.match_threshold"
        );
        Ok(())
    }

    /// This config with `overrides` applied, validated.
    pub fn with_overrides(&self, overrides: &Overrides) -> Result<Self> {
        let mut value = serde_json::to_value(self)?;
        for (key, v) in overrides {
            ensure!(FIELDS.iter().any(|f| f.key == key), "{key} cannot be changed from the web UI");
            let (section, name) = key.split_once('.').expect("field keys are section.name");
            value[section][name] = v.clone();
        }
        let config: Self = serde_json::from_value(value).context("a setting has the wrong type")?;
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn partial_file_keeps_other_defaults() {
        let c: Config = toml::from_str("[trigger]\nratio = 4.0\n").unwrap();
        assert_eq!(c.trigger.ratio, 4.0);
        assert_eq!(c.trigger.hang_ms, 800);
        assert_eq!(c.web.listen, "127.0.0.1:8377");
    }

    #[test]
    fn model_is_chosen_by_name() {
        assert_eq!(Config::default().classify.model, Model::Yamnet);
        let c: Config = toml::from_str("[classify]\nmodel = \"ced-tiny\"\n").unwrap();
        assert_eq!(c.classify.model, Model::CedTiny);
        assert!(toml::from_str::<Config>("[classify]\nmodel = \"ced\"\n").is_err());
    }

    #[test]
    fn setting_the_model_keeps_the_rest_of_the_file() {
        let file = "# mine\n[classify]\nmatch_threshold = 0.9 # tuned\n\n[web]\nenabled = false\n";
        let text = with_model(file, Model::CedTiny).unwrap();
        assert!(text.contains("# mine") && text.contains("match_threshold = 0.9 # tuned"), "{text}");
        let c: Config = toml::from_str(&text).unwrap();
        assert_eq!((c.classify.model, c.classify.match_threshold, c.web.enabled), (Model::CedTiny, 0.9, false));

        let back = with_model(&text, Model::Yamnet).unwrap();
        assert_eq!(back.matches("model").count(), 1, "replaced, not added: {back}");
        assert_eq!(toml::from_str::<Config>(&back).unwrap().classify.model, Model::Yamnet);
        let fresh = with_model("", Model::CedTiny).unwrap();
        assert_eq!(toml::from_str::<Config>(&fresh).unwrap().classify.model, Model::CedTiny);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(toml::from_str::<Config>("[trigger]\nratoi = 4.0\n").is_err());
    }

    #[test]
    fn defaults_are_valid_and_every_field_exists() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn overrides_apply_and_are_checked() {
        let file = Config::default();
        let set = |v: Value| file.with_overrides(v.as_object().unwrap());
        let c = set(json!({"trigger.ratio": 4.0, "classify.group_gap_s": 0})).unwrap();
        assert_eq!((c.trigger.ratio, c.trigger.close_ratio, c.classify.group_gap_s), (4.0, 1.5, 0));
        assert!(set(json!({"trigger.close_ratio": 3.0})).is_err(), "close line above the open line");
        assert!(set(json!({"classify.match_threshold": 1.5})).is_err(), "out of range");
        assert!(set(json!({"trigger.hang_ms": 2.5})).is_err(), "not an integer");
        assert!(set(json!({"web.token": "x"})).is_err(), "not a live setting");
    }
}

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
    pub max_clip_s: u32,
    pub cooldown_s: u32,
    pub ambient_floor_db: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Classify {
    pub match_threshold: f32,
    pub suggest_threshold: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Storage {
    pub dir: PathBuf,
    pub retention_days: u32,
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
            max_clip_s: 15,
            cooldown_s: 2,
            ambient_floor_db: -60.0,
        }
    }
}

impl Default for Classify {
    fn default() -> Self {
        Self { match_threshold: 0.80, suggest_threshold: 0.60 }
    }
}

impl Default for Storage {
    fn default() -> Self {
        Self { dir: "/var/lib/creak".into(), retention_days: 14 }
    }
}

impl Default for Web {
    fn default() -> Self {
        Self { enabled: true, listen: "127.0.0.1:8377".into(), token: String::new() }
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_file_keeps_other_defaults() {
        let c: Config = toml::from_str("[trigger]\nratio = 4.0\n").unwrap();
        assert_eq!(c.trigger.ratio, 4.0);
        assert_eq!(c.trigger.hang_ms, 800);
        assert_eq!(c.web.listen, "127.0.0.1:8377");
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(toml::from_str::<Config>("[trigger]\nratoi = 4.0\n").is_err());
    }
}

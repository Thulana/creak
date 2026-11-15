//! SQLite event log, exemplars and clip files under the storage directory.

use crate::classify::{self, Exemplar, MixPart};
use crate::config::{Mode, Overrides};
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY,
    ts            INTEGER NOT NULL,   -- unix ms at clip start
    duration_ms   INTEGER NOT NULL,
    peak_db       REAL    NOT NULL,
    ambient_db    REAL    NOT NULL,
    label         TEXT    NOT NULL,   -- applied label, or the proposal while pending
    status        TEXT    NOT NULL,   -- match | pending | ignored | confirmed | corrected | skipped
    similarity    REAL,               -- to the nearest exemplar at capture time
    nearest_label TEXT,
    top_classes   TEXT    NOT NULL,   -- JSON [[class, score], ...] from the model
    clip_path     TEXT,               -- relative to the storage dir; NULL once dropped
    embedding     BLOB    NOT NULL,   -- empty once the clip has expired
    parts         INTEGER NOT NULL DEFAULT 1, -- clips grouped into this event; only the first is kept
    conflict      TEXT,               -- the model's class when it contradicted the label and the clip went to review
    nearest_event INTEGER,            -- the event whose clip is the nearest exemplar
    mix           TEXT    NOT NULL DEFAULT '[]', -- JSON [{class, share, peak, background}, ...], most present first
    checked       TEXT                -- a sampled match: due | right | wrong; NULL when not sampled
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS exemplars (
    id         INTEGER PRIMARY KEY,
    event_id   INTEGER NOT NULL UNIQUE REFERENCES events(id),
    label      TEXT    NOT NULL,
    embedding  BLOB    NOT NULL,
    created_ts INTEGER NOT NULL
);
-- What a clip is, said in the Audit view without seeing any label. Never an exemplar.
CREATE TABLE IF NOT EXISTS audits (
    event_id   INTEGER PRIMARY KEY REFERENCES events(id),
    label      TEXT,                  -- NULL when it could not be told
    created_ts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL               -- JSON
);
-- Everything the classifier worked out for one clip, kept for good: one row per clip,
-- including clips grouped into an earlier event and clips dropped as silence.
CREATE TABLE IF NOT EXISTS classifications (
    id            INTEGER PRIMARY KEY,
    event_id      INTEGER REFERENCES events(id), -- the event it started or extended; NULL when dropped as silence
    ts            INTEGER NOT NULL,   -- unix ms at clip start, pre-roll included
    duration_ms   INTEGER NOT NULL,   -- the whole clip: pre-roll, sound and tail
    onset_ms      INTEGER NOT NULL,   -- offsets within the clip: where the sound starts …
    sound_end_ms  INTEGER NOT NULL,   -- … where it is over …
    window_start_ms INTEGER NOT NULL, -- … and the part the model was given
    window_end_ms INTEGER NOT NULL,
    peak_db       REAL    NOT NULL,
    ambient_db    REAL    NOT NULL,
    model         TEXT    NOT NULL,   -- the model behind scores and embedding
    scores        BLOB    NOT NULL,   -- mean score per model class, f32 LE, in the class map's order
    embedding     BLOB    NOT NULL,   -- mean embedding, L2-normalised, f32 LE
    patches       TEXT    NOT NULL,   -- JSON: for each 0.96 s patch, its top classes [[class, score], ...]
    top_sounds    TEXT    NOT NULL,   -- JSON [[class, score], ...]: the model's view without room tone, best first
    status        TEXT    NOT NULL,   -- the decision: match | pending | ignored | silent
    label         TEXT    NOT NULL,   -- applied or proposed label; empty when silent
    similarity    REAL,               -- to the nearest exemplar
    nearest_label TEXT,
    nearest_event INTEGER,            -- the event whose clip that exemplar is
    candidates    TEXT    NOT NULL,   -- JSON [[label, similarity, event], ...]: each label's best exemplar, best first
    conflict      TEXT,
    settings      TEXT    NOT NULL,   -- JSON: the trigger and classify settings in effect
    classify_ms   INTEGER NOT NULL,
    mix           TEXT    NOT NULL DEFAULT '[]', -- JSON: the sound mix of the analysed part, as on events
    background    TEXT                -- JSON [class, ...]: sounds heard in the pre-roll; NULL without one
);
CREATE INDEX IF NOT EXISTS classifications_ts ON classifications(ts);
CREATE INDEX IF NOT EXISTS classifications_event ON classifications(event_id);
";

const OVERRIDES_KEY: &str = "overrides";
const MODE_KEY: &str = "mode";
/// Databases from creak 0.1 hold the Live view's trigger values under this key.
const OLD_TRIGGER_KEY: &str = "trigger";

pub struct Store {
    conn: Connection,
    dir: PathBuf,
}

pub struct NewEvent<'a> {
    pub ts: i64,
    pub duration_ms: i64,
    pub peak_db: f32,
    pub ambient_db: f32,
    pub decision: &'a classify::Decision,
    pub top_classes: &'a [(String, f32)],
    pub clip_path: Option<&'a str>,
    pub embedding: &'a [f32],
    pub nearest_event: Option<i64>,
    pub mix: &'a [MixPart],
    /// A match sampled for a "was this right?" check.
    pub check: bool,
}

pub struct NewClassification<'a> {
    pub event_id: Option<i64>,
    pub ts: i64,
    pub duration_ms: i64,
    pub onset_ms: i64,
    pub sound_end_ms: i64,
    pub window_ms: (i64, i64),
    pub peak_db: f32,
    pub ambient_db: f32,
    pub model: &'a str,
    pub scores: &'a [f32],
    pub embedding: &'a [f32],
    pub patches: &'a [Vec<(String, f32)>],
    pub top_sounds: &'a [(String, f32)],
    pub decision: &'a classify::Decision,
    pub candidates: &'a [(String, f32, i64)],
    /// JSON of the settings in effect.
    pub settings: &'a str,
    pub classify_ms: i64,
    pub mix: &'a [MixPart],
    pub background: Option<&'a [String]>,
}

#[derive(Debug, Serialize)]
pub struct Event {
    pub id: i64,
    pub ts: i64,
    pub duration_ms: i64,
    pub peak_db: f64,
    pub ambient_db: f64,
    pub label: String,
    pub status: String,
    pub similarity: Option<f64>,
    pub nearest_label: Option<String>,
    pub top_classes: serde_json::Value,
    pub clip_path: Option<String>,
    pub exemplar: bool,
    pub parts: i64,
    pub conflict: Option<String>,
    pub nearest_event: Option<i64>,
    pub mix: Vec<MixPart>,
    pub checked: Option<String>,
}

/// An event after a user's label, and the exemplar that caused it when it had been tagged
/// automatically with a different label.
#[derive(Debug, Serialize)]
pub struct Labelled {
    pub event: Event,
    pub cause: Option<Event>,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckAnswer {
    /// The automatic tag was right.
    Right,
    /// Can't tell from the clip: the check is dropped.
    Unsure,
}

#[derive(Default)]
pub struct Filter {
    pub status: Option<String>,
    pub label: Option<String>,
    pub since_ms: Option<i64>,
    pub before_id: Option<i64>,
    pub hide_ignored: bool,
    /// Only what the Review view asks about: pending events and matches due for a check.
    pub review: bool,
    pub limit: i64,
}

struct KeptClip {
    id: i64,
    path: String,
    /// An exemplar or audited clip, which is never deleted.
    kept: bool,
    bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct LabelCount {
    pub label: String,
    pub exemplars: i64,
}

/// A taught label, and how often the clips tagged with it automatically were corrected later.
#[derive(Debug, PartialEq, Serialize)]
pub struct LabelStats {
    pub label: String,
    pub exemplars: i64,
    pub matches: i64,
    pub corrected: i64,
}

/// An audited clip, for `creak eval`.
pub struct Audited {
    pub event_id: i64,
    pub ts: i64,
    /// What the audit said the clip is.
    pub label: String,
    pub clip_path: String,
    /// Where the sound starts and ends within the clip, when the classification was kept.
    pub onset_ms: Option<i64>,
    pub sound_end_ms: Option<i64>,
    /// The clip's label when it is an exemplar.
    pub exemplar: Option<String>,
}

/// Automatic matches of one label, and how the sampled ones were checked.
#[derive(Debug, Default, Serialize)]
pub struct MatchStats {
    pub label: String,
    pub matches: i64,
    /// Matches later corrected to another label.
    pub corrected: i64,
    pub right: i64,
    pub wrong: i64,
    /// Share of matches that are right, from the checks weighted by how they were sampled.
    pub precision: Option<f64>,
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir.join("clips")).with_context(|| format!("creating {}", dir.display()))?;
        let conn = Connection::open(dir.join("creak.db"))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        for (table, column, definition) in [
            ("events", "parts", "INTEGER NOT NULL DEFAULT 1"),
            ("events", "conflict", "TEXT"),
            ("events", "nearest_event", "INTEGER"),
            ("events", "mix", "TEXT NOT NULL DEFAULT '[]'"),
            ("events", "checked", "TEXT"),
            ("classifications", "mix", "TEXT NOT NULL DEFAULT '[]'"),
            ("classifications", "background", "TEXT"),
        ] {
            let exists: bool =
                conn.query_row("SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name = ?2", [table, column], |r| {
                    r.get(0)
                })?;
            if !exists {
                conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"))?;
            }
        }
        let store = Self { conn, dir: dir.to_path_buf() };
        if let Some(old) = store.setting::<Overrides>(OLD_TRIGGER_KEY)? {
            let mut overrides = store.overrides()?;
            for (name, value) in old {
                overrides.entry(format!("trigger.{name}")).or_insert(value);
            }
            store.set_overrides(&overrides)?;
            store.conn.execute("DELETE FROM settings WHERE key = ?1", [OLD_TRIGGER_KEY])?;
        }
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn insert_event(&self, e: &NewEvent) -> Result<i64> {
        let d = e.decision;
        self.conn.execute(
            "INSERT INTO events (ts, duration_ms, peak_db, ambient_db, label, status, similarity,
                                 nearest_label, top_classes, clip_path, embedding, conflict, nearest_event, mix, checked)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                e.ts,
                e.duration_ms,
                e.peak_db,
                e.ambient_db,
                d.label,
                serde_json::to_value(d.status)?.as_str(),
                d.nearest.as_ref().map(|n| n.1),
                d.nearest.as_ref().map(|n| &n.0),
                serde_json::to_string(e.top_classes)?,
                e.clip_path,
                to_blob(e.embedding),
                d.conflict,
                e.nearest_event,
                serde_json::to_string(e.mix)?,
                e.check.then_some("due"),
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn insert_classification(&self, c: &NewClassification) -> Result<i64> {
        let d = c.decision;
        let nearest_event = c.candidates.first().map(|(.., event)| *event);
        self.conn.execute(
            "INSERT INTO classifications (event_id, ts, duration_ms, onset_ms, sound_end_ms, window_start_ms,
                 window_end_ms, peak_db, ambient_db, model, scores, embedding, patches, top_sounds, status, label,
                 similarity, nearest_label, nearest_event, candidates, conflict, settings, classify_ms, mix, background)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
                 ?21, ?22, ?23, ?24, ?25)",
            params![
                c.event_id,
                c.ts,
                c.duration_ms,
                c.onset_ms,
                c.sound_end_ms,
                c.window_ms.0,
                c.window_ms.1,
                c.peak_db,
                c.ambient_db,
                c.model,
                to_blob(c.scores),
                to_blob(c.embedding),
                serde_json::to_string(c.patches)?,
                serde_json::to_string(c.top_sounds)?,
                serde_json::to_value(d.status)?.as_str(),
                d.label,
                d.nearest.as_ref().map(|n| n.1),
                d.nearest.as_ref().map(|n| &n.0),
                nearest_event,
                serde_json::to_string(c.candidates)?,
                d.conflict,
                c.settings,
                c.classify_ms,
                serde_json::to_string(c.mix)?,
                c.background.map(serde_json::to_string).transpose()?,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Grows an event by one more clip that ends at `end_ts`. A pending event becomes a match
    /// when the new clip matched an exemplar with the same label.
    pub fn extend_event(&self, id: i64, end_ts: i64, peak_db: f32, matched: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE events SET duration_ms = ?2 - ts, peak_db = max(peak_db, ?3), parts = parts + 1,
                    status = CASE WHEN ?4 AND status = 'pending' THEN 'match' ELSE status END
             WHERE id = ?1",
            params![id, end_ts, peak_db, matched],
        )?;
        Ok(())
    }

    /// Settings changed in the web UI; empty means the config file applies unchanged.
    pub fn overrides(&self) -> Result<Overrides> {
        Ok(self.setting(OVERRIDES_KEY)?.unwrap_or_default())
    }

    pub fn set_overrides(&self, overrides: &Overrides) -> Result<()> {
        self.set_setting(OVERRIDES_KEY, overrides)
    }

    pub fn mode(&self) -> Result<Mode> {
        Ok(self.setting(MODE_KEY)?.unwrap_or_default())
    }

    pub fn set_mode(&self, mode: Mode) -> Result<()> {
        self.set_setting(MODE_KEY, &mode)
    }

    /// A stored setting; one that no longer parses counts as missing.
    fn setting<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let value: Option<String> =
            self.conn.query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0)).optional()?;
        Ok(value.and_then(|v| serde_json::from_str(&v).ok()))
    }

    fn set_setting<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, serde_json::to_string(value)?],
        )?;
        Ok(())
    }

    pub fn exemplars(&self) -> Result<Vec<Exemplar>> {
        let mut stmt = self.conn.prepare(
            "SELECT x.label, x.embedding, e.top_classes, x.event_id, e.mix
             FROM exemplars x JOIN events e ON e.id = x.event_id",
        )?;
        let rows = stmt.query_map([], |r| {
            let top: Vec<(String, f32)> = serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or_default();
            let mix: Vec<MixPart> = serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_default();
            Ok(Exemplar {
                event_id: r.get(3)?,
                label: r.get(0)?,
                embedding: from_blob(&r.get::<_, Vec<u8>>(1)?),
                heard: classify::heard(&classify::sounds(top), &mix),
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn events(&self, f: &Filter) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(&format!(
            "{SELECT_EVENT}
             WHERE (?1 IS NULL OR e.status = ?1) AND (?2 IS NULL OR e.label = ?2)
               AND (?3 IS NULL OR e.ts >= ?3) AND (?4 IS NULL OR e.id < ?4)
               AND (NOT ?6 OR e.status <> 'ignored')
               AND (NOT ?7 OR e.status = 'pending' OR (e.checked = 'due' AND e.clip_path IS NOT NULL))
             ORDER BY e.id DESC LIMIT ?5"
        ))?;
        let rows = stmt.query_map(
            params![f.status, f.label, f.since_ms, f.before_id, f.limit, f.hide_ignored, f.review],
            event_from_row,
        )?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn event(&self, id: i64) -> Result<Option<Event>> {
        let sql = format!("{SELECT_EVENT} WHERE e.id = ?1");
        Ok(self.conn.query_row(&sql, [id], event_from_row).optional()?)
    }

    /// Applies a user's label. The clip becomes (or replaces) an exemplar for that label. A
    /// match due for a check is checked by it: right when the label stays, wrong when it changes.
    pub fn label_event(&mut self, id: i64, raw_label: &str) -> Result<Labelled> {
        let label = user_label(raw_label)?;
        let tx = self.conn.transaction()?;
        let (old_label, old_status, nearest_event, embedding): (String, String, Option<i64>, Vec<u8>) = tx
            .query_row("SELECT label, status, nearest_event, embedding FROM events WHERE id = ?1", [id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })
            .optional()?
            .with_context(|| format!("no event {id}"))?;
        if embedding.is_empty() {
            bail!("event {id} has expired; its embedding is gone");
        }
        let status = if label == old_label { "confirmed" } else { "corrected" };
        tx.execute(
            "UPDATE events SET label = ?1, status = ?2,
                    checked = CASE WHEN checked IS NULL THEN NULL WHEN ?2 = 'corrected' THEN 'wrong'
                                   WHEN checked = 'due' THEN 'right' ELSE checked END
             WHERE id = ?3",
            params![label, status, id],
        )?;
        tx.execute(
            "INSERT INTO exemplars (event_id, label, embedding, created_ts) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(event_id) DO UPDATE SET label = excluded.label",
            params![id, label, embedding, now_ms()],
        )?;
        tx.commit()?;
        let automatic = matches!(old_status.as_str(), "match" | "ignored");
        let cause = match nearest_event {
            Some(cause) if automatic && status == "corrected" => self.event(cause)?.filter(|e| e.exemplar),
            _ => None,
        };
        Ok(Labelled { event: self.event(id)?.expect("event exists"), cause })
    }

    /// Answers an event without teaching: it keeps its label but is no longer asked about, and
    /// its clip stops being an exemplar if it was one.
    pub fn skip_event(&self, id: i64) -> Result<Event> {
        let updated = self.conn.execute(
            "UPDATE events SET status = 'skipped', checked = CASE WHEN checked = 'due' THEN NULL ELSE checked END
             WHERE id = ?1",
            [id],
        )?;
        if updated == 0 {
            bail!("no event {id}");
        }
        self.conn.execute("DELETE FROM exemplars WHERE event_id = ?1", [id])?;
        Ok(self.event(id)?.expect("event exists"))
    }

    /// Answers a sampled match's check without relabelling it.
    pub fn answer_check(&self, id: i64, answer: CheckAnswer) -> Result<Event> {
        let checked = match answer {
            CheckAnswer::Right => Some("right"),
            CheckAnswer::Unsure => None,
        };
        let updated = self
            .conn
            .execute("UPDATE events SET checked = ?2 WHERE id = ?1 AND checked IS NOT NULL", params![id, checked])?;
        if updated == 0 {
            bail!("event {id} is not a match sampled for a check");
        }
        Ok(self.event(id)?.expect("event exists"))
    }

    pub fn labels(&self) -> Result<Vec<LabelCount>> {
        let mut stmt = self.conn.prepare("SELECT label, COUNT(*) FROM exemplars GROUP BY label ORDER BY label")?;
        let rows = stmt.query_map([], |r| Ok(LabelCount { label: r.get(0)?, exemplars: r.get(1)? }))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Every taught label, alphabetically, with its automatic tags and how many were corrected.
    pub fn label_stats(&self) -> Result<Vec<LabelStats>> {
        let mut matches: BTreeMap<String, MatchStats> =
            self.match_stats()?.into_iter().map(|m| (m.label.clone(), m)).collect();
        let stats = self.labels()?.into_iter().map(|l| {
            let m = matches.remove(&l.label).unwrap_or_default();
            LabelStats { label: l.label, exemplars: l.exemplars, matches: m.matches, corrected: m.corrected }
        });
        Ok(stats.collect())
    }

    /// Renames a taught label wherever it appears: exemplars, events, audits and classifications.
    /// Renaming onto another label merges the two. Returns the number of exemplars renamed.
    pub fn rename_label(&mut self, raw_label: &str, raw_to: &str) -> Result<usize> {
        let label = classify::normalize_label(raw_label).context("empty or too long label")?;
        let to = user_label(raw_to)?;
        if label == classify::IGNORE || to == classify::IGNORE {
            bail!("{} is reserved; relabel single events instead", classify::IGNORE);
        }
        if label == to {
            bail!("{label:?} already has that name");
        }
        let tx = self.conn.transaction()?;
        let renamed = tx.execute("UPDATE exemplars SET label = ?2 WHERE label = ?1", params![label, to])?;
        if renamed == 0 {
            bail!("no taught label {label:?}");
        }
        for sql in [
            "UPDATE events SET label = ?2 WHERE label = ?1",
            "UPDATE events SET nearest_label = ?2 WHERE nearest_label = ?1",
            "UPDATE audits SET label = ?2 WHERE label = ?1",
            "UPDATE classifications SET label = ?2 WHERE label = ?1",
            "UPDATE classifications SET nearest_label = ?2 WHERE nearest_label = ?1",
            "UPDATE classifications SET candidates = (
                 SELECT json_group_array(CASE WHEN c.value ->> 0 = ?1 THEN json_set(c.value, '$[0]', ?2)
                                              ELSE json(c.value) END)
                 FROM json_each(candidates) c)
             WHERE EXISTS (SELECT 1 FROM json_each(candidates) c WHERE c.value ->> 0 = ?1)",
        ] {
            tx.execute(sql, params![label, to])?;
        }
        tx.commit()?;
        Ok(renamed)
    }

    /// Forgets a taught label: its exemplars are removed, so nothing is tagged with it any more.
    /// Events keep the label they were given. Returns the number of exemplars removed.
    pub fn delete_label(&self, raw_label: &str) -> Result<usize> {
        let label = classify::normalize_label(raw_label).context("empty or too long label")?;
        let deleted = self.conn.execute("DELETE FROM exemplars WHERE label = ?1", [&label])?;
        if deleted == 0 {
            bail!("no taught label {label:?}");
        }
        Ok(deleted)
    }

    /// Events that started at or after `since_ms`.
    pub fn events_since(&self, since_ms: i64) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM events WHERE ts >= ?1", [since_ms], |r| r.get(0))?)
    }

    /// Events waiting in the Review view: pending ones and matches due for a check.
    pub fn review_count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM events WHERE status = 'pending' OR (checked = 'due' AND clip_path IS NOT NULL)",
            [],
            |r| r.get(0),
        )?)
    }

    /// A random clip that has not been audited yet.
    pub fn next_audit(&self) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id FROM events WHERE clip_path IS NOT NULL AND id NOT IN (SELECT event_id FROM audits)
                 ORDER BY random() LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Records what a clip is, or with `None` that it could not be told.
    pub fn audit(&self, id: i64, raw_label: Option<&str>) -> Result<()> {
        let label = raw_label.map(user_label).transpose()?;
        let has_clip: Option<bool> = self
            .conn
            .query_row("SELECT clip_path IS NOT NULL FROM events WHERE id = ?1", [id], |r| r.get(0))
            .optional()?;
        match has_clip {
            None => bail!("no event {id}"),
            Some(false) => bail!("event {id} has no clip to listen to"),
            Some(true) => {}
        }
        self.conn.execute(
            "INSERT INTO audits (event_id, label, created_ts) VALUES (?1, ?2, ?3)
             ON CONFLICT(event_id) DO UPDATE SET label = excluded.label, created_ts = excluded.created_ts",
            params![id, label, now_ms()],
        )?;
        Ok(())
    }

    /// Audited clips per label, most first; `None` counts the clips that could not be told.
    pub fn audit_counts(&self) -> Result<Vec<(Option<String>, i64)>> {
        let mut stmt =
            self.conn.prepare("SELECT label, COUNT(*) FROM audits GROUP BY label ORDER BY COUNT(*) DESC, label")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Audited clips with a label and audio, oldest first.
    pub fn audited(&self) -> Result<Vec<Audited>> {
        let mut stmt = self.conn.prepare(
            "SELECT a.event_id, e.ts, a.label, e.clip_path, c.onset_ms, c.sound_end_ms, x.label
             FROM audits a JOIN events e ON e.id = a.event_id
             LEFT JOIN classifications c ON c.id = (SELECT MIN(id) FROM classifications WHERE event_id = a.event_id)
             LEFT JOIN exemplars x ON x.event_id = a.event_id
             WHERE a.label IS NOT NULL AND e.clip_path IS NOT NULL
             ORDER BY e.ts",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Audited {
                event_id: r.get(0)?,
                ts: r.get(1)?,
                label: r.get(2)?,
                clip_path: r.get(3)?,
                onset_ms: r.get(4)?,
                sound_end_ms: r.get(5)?,
                exemplar: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Automatic matches per label, most first, from the classifications that started an event.
    pub fn match_stats(&self) -> Result<Vec<MatchStats>> {
        let mut stmt = self.conn.prepare(
            // A match that ends up with the label it was matched as, changed back or merged into
            // it, was right after all.
            "SELECT c.label, e.similarity, e.status = 'corrected' AND e.label <> c.label,
                    CASE WHEN e.checked = 'wrong' AND e.label = c.label THEN 'right' ELSE e.checked END
             FROM classifications c JOIN events e ON e.id = c.event_id
             WHERE c.status = 'match' AND c.id = (SELECT MIN(id) FROM classifications WHERE event_id = c.event_id)",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<f64>>(1)?,
                r.get::<_, bool>(2)?,
                r.get::<_, Option<String>>(3)?,
            ))
        })?;
        // Per label: the stats, and the sampling weight of the right and of the wrong checks.
        let mut by_label: BTreeMap<String, (MatchStats, f64, f64)> = BTreeMap::new();
        for row in rows {
            let (label, similarity, corrected, checked) = row?;
            let (stats, right_w, wrong_w) = by_label.entry(label.clone()).or_default();
            stats.label = label;
            stats.matches += 1;
            stats.corrected += corrected as i64;
            let weight = classify::check_weight(similarity.unwrap_or_default() as f32);
            match checked.as_deref() {
                Some("right") => (stats.right, *right_w) = (stats.right + 1, *right_w + weight),
                Some("wrong") => (stats.wrong, *wrong_w) = (stats.wrong + 1, *wrong_w + weight),
                _ => {}
            }
        }
        let mut stats: Vec<MatchStats> = by_label
            .into_values()
            .map(|(s, right_w, wrong_w)| MatchStats {
                precision: (right_w + wrong_w > 0.0).then(|| right_w / (right_w + wrong_w)),
                ..s
            })
            .collect();
        stats.sort_by_key(|s| std::cmp::Reverse(s.matches));
        Ok(stats)
    }

    /// Deletes clips older than `days` that are neither exemplars nor audited, and drops their
    /// embeddings.
    pub fn expire(&self, days: u32) -> Result<usize> {
        let cutoff = now_ms() - days as i64 * 86_400_000;
        let mut stmt = self.conn.prepare(
            "SELECT id, clip_path FROM events
             WHERE ts < ?1 AND length(embedding) > 0 AND id NOT IN (SELECT event_id FROM exemplars)
               AND id NOT IN (SELECT event_id FROM audits WHERE label IS NOT NULL)",
        )?;
        let expired: Vec<(i64, Option<String>)> =
            stmt.query_map([cutoff], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        for (id, clip) in &expired {
            self.drop_clip(*id, clip.as_deref())?;
        }
        Ok(expired.len())
    }

    /// Deletes the oldest clips that are neither exemplars nor audited until all clips take at
    /// most `max_mb`.
    pub fn cap_clips(&self, max_mb: u32) -> Result<usize> {
        let clips = self.kept_clips()?;
        let mut total: u64 = clips.iter().map(|c| c.bytes).sum();
        let mut dropped = 0;
        for clip in clips.iter().filter(|c| !c.kept) {
            if total <= max_mb as u64 * 1_000_000 {
                break;
            }
            self.drop_clip(clip.id, Some(&clip.path))?;
            total -= clip.bytes;
            dropped += 1;
        }
        Ok(dropped)
    }

    /// Bytes on disk of every clip still kept.
    pub fn clip_bytes(&self) -> Result<u64> {
        Ok(self.kept_clips()?.iter().map(|c| c.bytes).sum())
    }

    /// Clips on disk, oldest first.
    fn kept_clips(&self) -> Result<Vec<KeptClip>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.clip_path, x.id IS NOT NULL OR a.label IS NOT NULL
             FROM events e LEFT JOIN exemplars x ON x.event_id = e.id LEFT JOIN audits a ON a.event_id = e.id
             WHERE e.clip_path IS NOT NULL ORDER BY e.ts",
        )?;
        let rows = stmt.query_map([], |r| {
            let path: String = r.get(1)?;
            let bytes = std::fs::metadata(self.dir.join(&path)).map_or(0, |m| m.len());
            Ok(KeptClip { id: r.get(0)?, path, kept: r.get(2)?, bytes })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Deletes an event's clip and its embedding; the event row stays in the log.
    fn drop_clip(&self, id: i64, clip: Option<&str>) -> Result<()> {
        if let Some(clip) = clip {
            let _ = std::fs::remove_file(self.dir.join(clip));
        }
        self.conn.execute("UPDATE events SET clip_path = NULL, embedding = x'' WHERE id = ?1", [id])?;
        Ok(())
    }
}

const SELECT_EVENT: &str = "
    SELECT e.id, e.ts, e.duration_ms, e.peak_db, e.ambient_db, e.label, e.status, e.similarity,
           e.nearest_label, e.top_classes, e.clip_path, x.id IS NOT NULL, e.parts, e.conflict,
           e.nearest_event, e.mix, e.checked
    FROM events e LEFT JOIN exemplars x ON x.event_id = e.id";

/// A label typed by the user: normalised, and never `silence`.
fn user_label(raw: &str) -> Result<String> {
    let Some(label) = classify::normalize_label(raw) else { bail!("empty or too long label") };
    if label == classify::SILENCE {
        bail!("silence is not a sound to label; use ignore for a sound you never want to hear about");
    }
    Ok(label)
}

fn event_from_row(r: &Row) -> rusqlite::Result<Event> {
    Ok(Event {
        id: r.get(0)?,
        ts: r.get(1)?,
        duration_ms: r.get(2)?,
        peak_db: r.get(3)?,
        ambient_db: r.get(4)?,
        label: r.get(5)?,
        status: r.get(6)?,
        similarity: r.get(7)?,
        nearest_label: r.get(8)?,
        top_classes: serde_json::from_str(&r.get::<_, String>(9)?).unwrap_or_default(),
        clip_path: r.get(10)?,
        exemplar: r.get(11)?,
        parts: r.get(12)?,
        conflict: r.get(13)?,
        nearest_event: r.get(14)?,
        mix: serde_json::from_str(&r.get::<_, String>(15)?).unwrap_or_default(),
        checked: r.get(16)?,
    })
}

/// (event, status, label, number of patches, number of class scores) of every classification, oldest first.
#[cfg(test)]
pub fn classifications(store: &Store) -> Vec<(Option<i64>, String, String, usize, usize)> {
    let mut stmt =
        store.conn.prepare("SELECT event_id, status, label, patches, scores FROM classifications ORDER BY id").unwrap();
    let rows = stmt.query_map([], |r| {
        let patches: Vec<serde_json::Value> = serde_json::from_str(&r.get::<_, String>(3)?).unwrap();
        Ok((r.get(0)?, r.get(1)?, r.get(2)?, patches.len(), r.get::<_, Vec<u8>>(4)?.len() / 4))
    });
    rows.unwrap().map(Result::unwrap).collect()
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::{Decision, Status};

    fn temp_store() -> Store {
        let dir = std::env::temp_dir().join(format!("creak-test-{}-{}", std::process::id(), rand_suffix()));
        Store::open(&dir).unwrap()
    }

    fn rand_suffix() -> u128 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    }

    fn insert(store: &Store, ts: i64, label: &str, embedding: &[f32]) -> i64 {
        let decision = Decision { status: Status::Pending, label: label.into(), nearest: None, conflict: None };
        store
            .insert_event(&NewEvent {
                ts,
                duration_ms: 500,
                peak_db: -20.0,
                ambient_db: -50.0,
                decision: &decision,
                top_classes: &[("Door".into(), 0.5)],
                clip_path: Some("clips/x.wav"),
                embedding,
                nearest_event: None,
                mix: &[],
                check: false,
            })
            .unwrap()
    }

    #[test]
    fn labelling_creates_and_replaces_one_exemplar() {
        let mut store = temp_store();
        let id = insert(&store, now_ms(), "door", &[0.6, 0.8]);
        let e = store.label_event(id, "Door").unwrap().event;
        assert_eq!((e.status.as_str(), e.exemplar), ("confirmed", true));
        let e = store.label_event(id, "back door").unwrap().event;
        assert_eq!(e.status, "corrected");
        let exemplars = store.exemplars().unwrap();
        assert_eq!(exemplars.len(), 1);
        assert_eq!((exemplars[0].label.as_str(), exemplars[0].embedding.clone()), ("back door", vec![0.6, 0.8]));
        let heard = exemplars[0].heard.as_ref().map(|h| (h.class.as_str(), h.score));
        assert_eq!(heard, Some(("Door", 0.5)));
    }

    #[test]
    fn silence_is_not_a_label() {
        let mut store = temp_store();
        let id = insert(&store, now_ms(), "door", &[1.0]);
        assert!(store.label_event(id, " Silence ").is_err());
        assert!(store.exemplars().unwrap().is_empty());
    }

    #[test]
    fn expiry_spares_exemplars() {
        let mut store = temp_store();
        let old = now_ms() - 20 * 86_400_000;
        let kept = insert(&store, old, "door", &[1.0]);
        let dropped = insert(&store, old, "door", &[1.0]);
        let recent = insert(&store, now_ms(), "door", &[1.0]);
        store.label_event(kept, "door").unwrap();
        assert_eq!(store.expire(14).unwrap(), 1);
        assert!(store.event(dropped).unwrap().unwrap().clip_path.is_none());
        assert!(store.label_event(dropped, "door").is_err());
        assert!(store.event(recent).unwrap().unwrap().clip_path.is_some());
        assert!(store.event(kept).unwrap().unwrap().clip_path.is_some());
    }

    #[test]
    fn the_size_cap_deletes_the_oldest_clips_but_not_exemplars_or_audited_ones() {
        let mut store = temp_store();
        let clip = |store: &Store, ts: i64| {
            let id = insert(store, ts, "door", &[1.0]);
            let rel = format!("clips/{id}.wav");
            std::fs::write(store.dir().join(&rel), vec![0u8; 400_000]).unwrap();
            store.conn.execute("UPDATE events SET clip_path = ?1 WHERE id = ?2", params![rel, id]).unwrap();
            id
        };
        let ids: Vec<i64> = (0..4).map(|i| clip(&store, 1_000 + i)).collect();
        store.label_event(ids[0], "door").unwrap();
        store.audit(ids[1], Some("door")).unwrap();
        assert_eq!(store.clip_bytes().unwrap(), 1_600_000);
        // 1.6 MB against a 1 MB cap: the oldest is taught and the next audited, so the last two go
        assert_eq!(store.cap_clips(1).unwrap(), 2);
        let kept: Vec<bool> = ids.iter().map(|&id| store.event(id).unwrap().unwrap().clip_path.is_some()).collect();
        assert_eq!(kept, vec![true, true, false, false]);
        assert!(!store.dir().join(format!("clips/{}.wav", ids[2])).exists());
        assert_eq!(store.clip_bytes().unwrap(), 800_000);
        assert_eq!(store.cap_clips(1).unwrap(), 0);
    }

    #[test]
    fn extending_grows_duration_and_upgrades_pending() {
        let store = temp_store();
        let id = insert(&store, 1_000, "music", &[1.0]);
        store.extend_event(id, 31_000, -10.0, false).unwrap();
        let e = store.event(id).unwrap().unwrap();
        assert_eq!((e.duration_ms, e.parts, e.peak_db, e.status.as_str()), (30_000, 2, -10.0, "pending"));
        store.extend_event(id, 41_000, -30.0, true).unwrap();
        let e = store.event(id).unwrap().unwrap();
        assert_eq!((e.duration_ms, e.parts, e.peak_db, e.status.as_str()), (40_000, 3, -10.0, "match"));
    }

    #[test]
    fn settings_round_trip() {
        let store = temp_store();
        assert!(store.overrides().unwrap().is_empty());
        assert_eq!(store.mode().unwrap(), Mode::Active);
        let overrides = serde_json::json!({"trigger.ratio": 4.0}).as_object().unwrap().clone();
        store.set_overrides(&overrides).unwrap();
        store.set_mode(Mode::Deaf).unwrap();
        assert_eq!((store.overrides().unwrap(), store.mode().unwrap()), (overrides, Mode::Deaf));
    }

    #[test]
    fn live_view_values_from_0_1_become_overrides() {
        let dir = std::env::temp_dir().join(format!("creak-old-trigger-{}", rand_suffix()));
        Store::open(&dir)
            .unwrap()
            .set_setting(
                OLD_TRIGGER_KEY,
                &serde_json::json!({"ratio": 4.0, "close_ratio": 2.0, "ambient_floor_db": -50.0}),
            )
            .unwrap();
        let store = Store::open(&dir).unwrap();
        let expected =
            serde_json::json!({"trigger.ratio": 4.0, "trigger.close_ratio": 2.0, "trigger.ambient_floor_db": -50.0});
        assert_eq!(serde_json::Value::Object(store.overrides().unwrap()), expected);
        assert!(store.setting::<Overrides>(OLD_TRIGGER_KEY).unwrap().is_none());
    }

    #[test]
    fn old_databases_gain_the_new_columns() {
        let dir = std::env::temp_dir().join(format!("creak-migrate-{}", rand_suffix()));
        std::fs::create_dir_all(&dir).unwrap();
        Connection::open(dir.join("creak.db"))
            .unwrap()
            .execute_batch("CREATE TABLE events (id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, duration_ms INTEGER NOT NULL, peak_db REAL NOT NULL, ambient_db REAL NOT NULL, label TEXT NOT NULL, status TEXT NOT NULL, similarity REAL, nearest_label TEXT, top_classes TEXT NOT NULL, clip_path TEXT, embedding BLOB NOT NULL);
                            INSERT INTO events VALUES (1, 1000, 500, -20, -50, 'door', 'pending', NULL, NULL, '[]', NULL, x'');
                            CREATE TABLE classifications (id INTEGER PRIMARY KEY, event_id INTEGER, ts INTEGER NOT NULL)")
            .unwrap();
        let store = Store::open(&dir).unwrap();
        let e = store.event(1).unwrap().unwrap();
        assert_eq!((e.parts, e.conflict, e.nearest_event, e.mix, e.checked), (1, None, None, vec![], None));
        let columns: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('classifications') WHERE name IN ('mix', 'background')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(columns, 2);
    }

    #[test]
    fn filters_and_paging() {
        let store = temp_store();
        let ids: Vec<i64> =
            (0..5).map(|i| insert(&store, now_ms(), if i % 2 == 0 { "door" } else { "bell" }, &[1.0])).collect();
        let doors = store.events(&Filter { label: Some("door".into()), limit: 10, ..Default::default() }).unwrap();
        assert_eq!(doors.len(), 3);
        let page = store.events(&Filter { before_id: Some(ids[3]), limit: 2, ..Default::default() }).unwrap();
        assert_eq!(page.iter().map(|e| e.id).collect::<Vec<_>>(), vec![ids[2], ids[1]]);
    }

    /// An automatic match of `label` because of `cause`, with the classification that made it.
    fn insert_match(store: &Store, label: &str, similarity: f32, cause: i64, check: bool) -> i64 {
        let decision = Decision {
            status: Status::Match,
            label: label.into(),
            nearest: Some((label.into(), similarity)),
            conflict: None,
        };
        let id = store
            .insert_event(&NewEvent {
                ts: now_ms(),
                duration_ms: 500,
                peak_db: -20.0,
                ambient_db: -50.0,
                decision: &decision,
                top_classes: &[],
                clip_path: Some("clips/x.wav"),
                embedding: &[1.0],
                nearest_event: Some(cause),
                mix: &[],
                check,
            })
            .unwrap();
        store
            .insert_classification(&NewClassification {
                event_id: Some(id),
                ts: now_ms(),
                duration_ms: 500,
                onset_ms: 0,
                sound_end_ms: 500,
                window_ms: (0, 500),
                peak_db: -20.0,
                ambient_db: -50.0,
                model: "yamnet",
                scores: &[],
                embedding: &[1.0],
                patches: &[],
                top_sounds: &[],
                decision: &decision,
                candidates: &[(label.into(), similarity, cause)],
                settings: "{}",
                classify_ms: 1,
                mix: &[],
                background: None,
            })
            .unwrap();
        id
    }

    fn review(store: &Store) -> Vec<i64> {
        store.events(&Filter { review: true, limit: 10, ..Default::default() }).unwrap().iter().map(|e| e.id).collect()
    }

    #[test]
    fn skipping_answers_without_teaching() {
        let mut store = temp_store();
        let id = insert(&store, now_ms(), "music", &[1.0]);
        store.label_event(id, "music").unwrap();
        let e = store.skip_event(id).unwrap();
        assert_eq!((e.status.as_str(), e.label.as_str(), e.exemplar), ("skipped", "music", false));
        assert!(store.exemplars().unwrap().is_empty());
        assert!(review(&store).is_empty());
        assert!(store.skip_event(9_999).is_err());
    }

    #[test]
    fn correcting_a_match_points_at_the_exemplar_that_caused_it() {
        let mut store = temp_store();
        let taught = insert(&store, now_ms(), "music", &[1.0]);
        store.label_event(taught, "music").unwrap();
        let matched = insert_match(&store, "music", 0.93, taught, false);
        let corrected = store.label_event(matched, "cough").unwrap();
        assert_eq!(corrected.event.status, "corrected");
        assert_eq!(corrected.cause.map(|c| (c.id, c.label)), Some((taught, "music".into())));
        // Confirming a match, or correcting a clip that was only proposed, has no cause.
        let confirmed = insert_match(&store, "music", 0.93, taught, false);
        assert!(store.label_event(confirmed, "music").unwrap().cause.is_none());
        let pending = insert(&store, now_ms(), "music", &[1.0]);
        assert!(store.label_event(pending, "cough").unwrap().cause.is_none());
        // Once the cause is removed, there is nothing left to point at.
        store.skip_event(taught).unwrap();
        let again = insert_match(&store, "music", 0.93, taught, false);
        assert!(store.label_event(again, "cough").unwrap().cause.is_none());
    }

    #[test]
    fn sampled_matches_wait_in_review_until_checked() {
        let mut store = temp_store();
        let taught = insert(&store, now_ms(), "door", &[1.0]);
        store.label_event(taught, "door").unwrap();
        let right = insert_match(&store, "door", 0.86, taught, true);
        let wrong = insert_match(&store, "door", 0.86, taught, true);
        let unsure = insert_match(&store, "door", 0.86, taught, true);
        let unsampled = insert_match(&store, "door", 0.95, taught, false);
        assert_eq!(review(&store), vec![unsure, wrong, right]);
        assert_eq!(store.review_count().unwrap(), 3);

        assert_eq!(store.answer_check(right, CheckAnswer::Right).unwrap().checked.as_deref(), Some("right"));
        assert_eq!(store.label_event(wrong, "window").unwrap().event.checked.as_deref(), Some("wrong"));
        assert_eq!(store.answer_check(unsure, CheckAnswer::Unsure).unwrap().checked, None);
        assert!(store.answer_check(unsampled, CheckAnswer::Right).is_err(), "only sampled matches are checked");
        assert!(review(&store).is_empty());
        // Right stays right when the label is confirmed, and turns wrong when it is changed.
        assert_eq!(store.label_event(right, "door").unwrap().event.checked.as_deref(), Some("right"));
        assert_eq!(store.label_event(right, "gate").unwrap().event.checked.as_deref(), Some("wrong"));
    }

    #[test]
    fn match_precision_weighs_checks_by_how_they_were_sampled() {
        let mut store = temp_store();
        let taught = insert(&store, now_ms(), "door", &[1.0]);
        store.label_event(taught, "door").unwrap();
        // Two checked below 0.9, one right and one wrong; one checked above, right, standing for ten.
        let low_right = insert_match(&store, "door", 0.86, taught, true);
        let low_wrong = insert_match(&store, "door", 0.86, taught, true);
        let high_right = insert_match(&store, "door", 0.95, taught, true);
        insert_match(&store, "door", 0.95, taught, false);
        store.answer_check(low_right, CheckAnswer::Right).unwrap();
        store.label_event(low_wrong, "gate").unwrap();
        store.answer_check(high_right, CheckAnswer::Right).unwrap();
        insert_match(&store, "bell", 0.95, taught, false);

        let stats = store.match_stats().unwrap();
        let door = &stats[0];
        assert_eq!((door.label.as_str(), door.matches, door.corrected, door.right, door.wrong), ("door", 4, 1, 2, 1));
        assert!((door.precision.unwrap() - 11.0 / 12.0).abs() < 1e-9);
        assert_eq!((stats[1].label.as_str(), stats[1].matches, stats[1].precision), ("bell", 1, None));
    }

    fn label_names(store: &Store) -> Vec<(String, i64)> {
        store.labels().unwrap().into_iter().map(|l| (l.label, l.exemplars)).collect()
    }

    #[test]
    fn renaming_a_label_renames_it_everywhere_and_can_merge_two() {
        let mut store = temp_store();
        let door = insert(&store, now_ms(), "door", &[1.0]);
        store.label_event(door, "door").unwrap();
        let gate = insert(&store, now_ms(), "gate", &[1.0]);
        store.label_event(gate, "gate").unwrap();
        let matched = insert_match(&store, "door", 0.9, door, false);
        store.audit(matched, Some("door")).unwrap();
        let called_gate = insert_match(&store, "door", 0.86, door, true);
        store.label_event(called_gate, "gate").unwrap();

        assert_eq!(store.rename_label(" Door ", "Front  Door").unwrap(), 1);
        let e = store.event(matched).unwrap().unwrap();
        assert_eq!((e.label.as_str(), e.nearest_label.as_deref()), ("front door", Some("front door")));
        assert_eq!(store.audited().unwrap()[0].label, "front door");
        assert_eq!(classifications(&store)[0].2, "front door");
        let (nearest, candidates): (String, String) = store
            .conn
            .query_row("SELECT nearest_label, candidates FROM classifications", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!((nearest.as_str(), candidates.as_str()), ("front door", r#"[["front door",0.9,1]]"#));

        assert_eq!(store.rename_label("gate", "front door").unwrap(), 2);
        assert_eq!(label_names(&store), vec![("front door".into(), 3)]);
        // The match corrected to gate now carries the label it was matched as: it was right.
        let m = &store.match_stats().unwrap()[0];
        assert_eq!((m.matches, m.corrected, m.right, m.wrong), (2, 0, 1, 0));
        assert!(store.rename_label("gate", "door").is_err(), "no longer taught");
        assert!(store.rename_label("front door", "front door").is_err());
        assert!(store.rename_label("front door", "silence").is_err());
        assert!(store.rename_label("front door", "ignore").is_err(), "reserved");
        let ignored = insert(&store, now_ms(), "fan", &[1.0]);
        store.label_event(ignored, "ignore").unwrap();
        assert!(store.rename_label("ignore", "fan").is_err(), "reserved");
    }

    #[test]
    fn label_stats_count_corrected_matches_and_deleting_a_label_forgets_it() {
        let mut store = temp_store();
        let door = insert(&store, now_ms(), "door", &[1.0]);
        store.label_event(door, "door").unwrap();
        let kept = insert_match(&store, "door", 0.9, door, false);
        let wrong = insert_match(&store, "door", 0.9, door, false);
        store.label_event(wrong, "gate").unwrap();
        let stats = |label: &str, exemplars, matches, corrected| LabelStats {
            label: label.into(),
            exemplars,
            matches,
            corrected,
        };
        assert_eq!(store.label_stats().unwrap(), vec![stats("door", 1, 2, 1), stats("gate", 1, 0, 0)]);

        assert_eq!(store.delete_label("Door").unwrap(), 1);
        assert_eq!(label_names(&store), vec![("gate".into(), 1)]);
        assert_eq!(store.event(kept).unwrap().unwrap().label, "door", "events keep their label");
        assert!(!store.event(door).unwrap().unwrap().exemplar);
        assert!(store.delete_label("door").is_err());
    }

    #[test]
    fn audits_are_kept_apart_and_keep_their_clips() {
        let mut store = temp_store();
        let old = now_ms() - 20 * 86_400_000;
        let audited = insert(&store, old, "music", &[1.0]);
        let unclear = insert(&store, old, "music", &[1.0]);
        let other = insert(&store, old, "door", &[1.0]);
        store.audit(audited, Some(" Cough ")).unwrap();
        store.audit(unclear, None).unwrap();
        assert!(store.audit(other, Some("silence")).is_err());
        assert!(store.audit(9_999, Some("cough")).is_err());
        assert_eq!(store.next_audit().unwrap(), Some(other));
        assert!(store.exemplars().unwrap().is_empty(), "an audit never teaches");
        assert_eq!(store.audit_counts().unwrap(), vec![(None, 1), (Some("cough".into()), 1)]);

        // Past retention, the audited clip stays; one that could not be told goes like any other.
        assert_eq!(store.expire(14).unwrap(), 2);
        assert!(store.event(audited).unwrap().unwrap().clip_path.is_some());
        assert!(store.event(unclear).unwrap().unwrap().clip_path.is_none());
        assert!(store.audit(unclear, Some("cough")).is_err(), "no clip left to listen to");

        store.label_event(audited, "music").unwrap();
        let listed = store.audited().unwrap();
        assert_eq!(listed.len(), 1);
        let a = &listed[0];
        assert_eq!((a.event_id, a.label.as_str(), a.exemplar.as_deref()), (audited, "cough", Some("music")));
        assert_eq!((a.onset_ms, a.sound_end_ms), (None, None));
    }
}

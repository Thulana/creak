//! SQLite event log, exemplars and clip files under the storage directory.

use crate::classify::{self, Exemplar};
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Serialize;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY,
    ts            INTEGER NOT NULL,   -- unix ms at clip start
    duration_ms   INTEGER NOT NULL,
    peak_db       REAL    NOT NULL,
    ambient_db    REAL    NOT NULL,
    label         TEXT    NOT NULL,   -- applied label, or the proposal while pending
    status        TEXT    NOT NULL,   -- match | pending | ignored | confirmed | corrected
    similarity    REAL,               -- to the nearest exemplar at capture time
    nearest_label TEXT,
    top_classes   TEXT    NOT NULL,   -- JSON [[class, score], ...] from the model
    clip_path     TEXT,               -- relative to the storage dir; NULL once dropped
    embedding     BLOB    NOT NULL    -- empty once the clip has expired
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS exemplars (
    id         INTEGER PRIMARY KEY,
    event_id   INTEGER NOT NULL UNIQUE REFERENCES events(id),
    label      TEXT    NOT NULL,
    embedding  BLOB    NOT NULL,
    created_ts INTEGER NOT NULL
);
";

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
}

#[derive(Default)]
pub struct Filter {
    pub status: Option<String>,
    pub label: Option<String>,
    pub since_ms: Option<i64>,
    pub before_id: Option<i64>,
    pub limit: i64,
}

#[derive(Debug, Serialize)]
pub struct LabelCount {
    pub label: String,
    pub exemplars: i64,
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir.join("clips")).with_context(|| format!("creating {}", dir.display()))?;
        let conn = Connection::open(dir.join("creak.db"))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn, dir: dir.to_path_buf() })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn insert_event(&self, e: &NewEvent) -> Result<i64> {
        let d = e.decision;
        self.conn.execute(
            "INSERT INTO events (ts, duration_ms, peak_db, ambient_db, label, status, similarity,
                                 nearest_label, top_classes, clip_path, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
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
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn exemplars(&self) -> Result<Vec<Exemplar>> {
        let mut stmt = self.conn.prepare("SELECT label, embedding FROM exemplars")?;
        let rows =
            stmt.query_map([], |r| Ok(Exemplar { label: r.get(0)?, embedding: from_blob(&r.get::<_, Vec<u8>>(1)?) }))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn events(&self, f: &Filter) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(&format!(
            "{SELECT_EVENT}
             WHERE (?1 IS NULL OR e.status = ?1) AND (?2 IS NULL OR e.label = ?2)
               AND (?3 IS NULL OR e.ts >= ?3) AND (?4 IS NULL OR e.id < ?4)
             ORDER BY e.id DESC LIMIT ?5"
        ))?;
        let rows = stmt.query_map(params![f.status, f.label, f.since_ms, f.before_id, f.limit], event_from_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn event(&self, id: i64) -> Result<Option<Event>> {
        let sql = format!("{SELECT_EVENT} WHERE e.id = ?1");
        Ok(self.conn.query_row(&sql, [id], event_from_row).optional()?)
    }

    /// Applies a user's label. The clip becomes (or replaces) an exemplar for that label.
    pub fn label_event(&mut self, id: i64, raw_label: &str) -> Result<Event> {
        let Some(label) = classify::normalize_label(raw_label) else { bail!("empty or too long label") };
        let tx = self.conn.transaction()?;
        let (old_label, embedding): (String, Vec<u8>) = tx
            .query_row("SELECT label, embedding FROM events WHERE id = ?1", [id], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?
            .with_context(|| format!("no event {id}"))?;
        if embedding.is_empty() {
            bail!("event {id} has expired; its embedding is gone");
        }
        let status = if label == old_label { "confirmed" } else { "corrected" };
        tx.execute("UPDATE events SET label = ?1, status = ?2 WHERE id = ?3", params![label, status, id])?;
        tx.execute(
            "INSERT INTO exemplars (event_id, label, embedding, created_ts) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(event_id) DO UPDATE SET label = excluded.label",
            params![id, label, embedding, now_ms()],
        )?;
        tx.commit()?;
        Ok(self.event(id)?.expect("event exists"))
    }

    pub fn labels(&self) -> Result<Vec<LabelCount>> {
        let mut stmt = self.conn.prepare("SELECT label, COUNT(*) FROM exemplars GROUP BY label ORDER BY label")?;
        let rows = stmt.query_map([], |r| Ok(LabelCount { label: r.get(0)?, exemplars: r.get(1)? }))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn pending_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM events WHERE status = 'pending'", [], |r| r.get(0))?)
    }

    /// Deletes clips older than `days` that are not exemplars, and drops their embeddings.
    pub fn expire(&self, days: u32) -> Result<usize> {
        let cutoff = now_ms() - days as i64 * 86_400_000;
        let mut stmt = self.conn.prepare(
            "SELECT id, clip_path FROM events
             WHERE ts < ?1 AND length(embedding) > 0 AND id NOT IN (SELECT event_id FROM exemplars)",
        )?;
        let expired: Vec<(i64, Option<String>)> =
            stmt.query_map([cutoff], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        for (id, clip) in &expired {
            if let Some(clip) = clip {
                let _ = std::fs::remove_file(self.dir.join(clip));
            }
            self.conn.execute("UPDATE events SET clip_path = NULL, embedding = x'' WHERE id = ?1", [id])?;
        }
        Ok(expired.len())
    }
}

const SELECT_EVENT: &str = "
    SELECT e.id, e.ts, e.duration_ms, e.peak_db, e.ambient_db, e.label, e.status, e.similarity,
           e.nearest_label, e.top_classes, e.clip_path, x.id IS NOT NULL
    FROM events e LEFT JOIN exemplars x ON x.event_id = e.id";

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
    })
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
        let decision = Decision { status: Status::Pending, label: label.into(), nearest: None };
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
            })
            .unwrap()
    }

    #[test]
    fn labelling_creates_and_replaces_one_exemplar() {
        let mut store = temp_store();
        let id = insert(&store, now_ms(), "door", &[0.6, 0.8]);
        let e = store.label_event(id, "Door").unwrap();
        assert_eq!((e.status.as_str(), e.exemplar), ("confirmed", true));
        let e = store.label_event(id, "back door").unwrap();
        assert_eq!(e.status, "corrected");
        let exemplars = store.exemplars().unwrap();
        assert_eq!(exemplars.len(), 1);
        assert_eq!((exemplars[0].label.as_str(), exemplars[0].embedding.clone()), ("back door", vec![0.6, 0.8]));
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
    fn filters_and_paging() {
        let store = temp_store();
        let ids: Vec<i64> =
            (0..5).map(|i| insert(&store, now_ms(), if i % 2 == 0 { "door" } else { "bell" }, &[1.0])).collect();
        let doors = store.events(&Filter { label: Some("door".into()), limit: 10, ..Default::default() }).unwrap();
        assert_eq!(doors.len(), 3);
        let page = store.events(&Filter { before_id: Some(ids[3]), limit: 2, ..Default::default() }).unwrap();
        assert_eq!(page.iter().map(|e| e.id).collect::<Vec<_>>(), vec![ids[2], ids[1]]);
    }
}

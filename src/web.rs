//! The web UI: one embedded page and a small JSON API on the same port.

use crate::config::{self, Config, Mode, Overrides};
use crate::daemon::{SharedLive, SharedSettings};
use crate::store::{Filter, Store};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Read;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const PAGE: &str = include_str!("../ui/index.html");

type Reply = Response<std::io::Cursor<Vec<u8>>>;

/// `file` is the config file as read, which resetting a setting returns to.
pub fn serve(file: &Config, mut store: Store, live: SharedLive, settings: SharedSettings) -> Result<()> {
    let listen = &file.web.listen;
    let server = Server::http(listen).map_err(|e| anyhow!("listening on {listen}: {e}"))?;
    eprintln!("web UI on http://{listen}/");
    for mut request in server.incoming_requests() {
        let reply =
            handle(&mut request, file, &mut store, &live, &settings).unwrap_or_else(|e| text(500, &format!("{e:#}")));
        let _ = request.respond(reply);
    }
    Ok(())
}

fn handle(
    request: &mut Request,
    file: &Config,
    store: &mut Store,
    live: &SharedLive,
    settings: &SharedSettings,
) -> Result<Reply> {
    let url = request.url().to_string();
    let (path, query) = split_url(&url);
    let path: Vec<&str> = path.trim_matches('/').split('/').collect();
    let method = request.method().clone();

    if method == Method::Get && path == [""] {
        return Ok(with_type(Response::from_data(PAGE.as_bytes().to_vec()), "text/html; charset=utf-8"));
    }
    if !authorized(request, &query, &file.web.token) {
        return Ok(text(401, "missing or wrong token"));
    }
    Ok(match (method, path.as_slice()) {
        (Method::Get, ["api", "live"]) => {
            let since: u64 = query.get("since").and_then(|s| s.parse().ok()).unwrap_or(0);
            let mode = settings.lock().unwrap().mode;
            let l = live.lock().unwrap();
            let fresh = (l.frame.saturating_sub(since) as usize).min(l.levels.len());
            json_reply(&json!({
                "device": l.device, "device_rate": l.device_rate, "frame": l.frame,
                "levels": l.levels.iter().skip(l.levels.len() - fresh).collect::<Vec<_>>(),
                "ambient_db": l.ambient_db, "open_db": l.open_db, "close_db": l.close_db, "recording": l.recording, "error": l.error,
                "measured_ambient_db": l.measured_ambient_db, "mode": mode,
            }))
        }
        (Method::Get, ["api", "events"]) => {
            let filter = Filter {
                status: query.get("status").filter(|s| !s.is_empty()).cloned(),
                label: query.get("label").filter(|s| !s.is_empty()).cloned(),
                since_ms: None,
                before_id: query.get("before").and_then(|s| s.parse().ok()),
                hide_ignored: query.get("hide_ignored").is_some_and(|v| v == "1"),
                limit: query.get("limit").and_then(|s| s.parse().ok()).unwrap_or(50).clamp(1, 500),
            };
            json_reply(&json!({ "events": store.events(&filter)? }))
        }
        (Method::Get, ["api", "labels"]) => {
            json_reply(&json!({ "labels": store.labels()?, "pending": store.pending_count()? }))
        }
        (Method::Get, ["api", "settings"]) => settings_reply(file, store, settings)?,
        (Method::Patch, ["api", "settings"]) => {
            let Ok(patch) = serde_json::from_str::<Overrides>(&read_body(request)?) else {
                return Ok(text(400, "expected {\"section.name\": value, …}; null resets a setting"));
            };
            let mut overrides = settings.lock().unwrap().overrides.clone();
            for (key, value) in patch {
                if value.is_null() {
                    overrides.remove(&key);
                } else {
                    overrides.insert(key, value);
                }
            }
            override_settings(file, store, settings, overrides)?
        }
        (Method::Delete, ["api", "settings"]) => override_settings(file, store, settings, Overrides::new())?,
        (Method::Get, ["api", "mode"]) => json_reply(&json!({ "mode": settings.lock().unwrap().mode })),
        (Method::Post, ["api", "mode"]) => {
            #[derive(Deserialize)]
            struct Body {
                mode: Mode,
            }
            let Ok(Body { mode }) = serde_json::from_str(&read_body(request)?) else {
                return Ok(text(400, "expected {\"mode\": \"active\"} or {\"mode\": \"deaf\"}"));
            };
            store.set_mode(mode)?;
            let mut s = settings.lock().unwrap();
            if s.mode != mode {
                eprintln!("mode: {}", if mode == Mode::Deaf { "deaf, not recording" } else { "active" });
            }
            s.mode = mode;
            s.version += 1;
            json_reply(&json!({ "mode": mode }))
        }
        (Method::Post, ["api", "events", id, "label"]) => {
            #[derive(Deserialize)]
            struct Body {
                label: String,
            }
            let Ok(body) = serde_json::from_str::<Body>(&read_body(request)?) else {
                return Ok(text(400, "expected {\"label\": \"…\"}"));
            };
            let Ok(id) = id.parse() else { return Ok(text(404, "no such event")) };
            match store.label_event(id, &body.label) {
                Ok(event) => json_reply(&json!({ "event": event })),
                Err(e) => text(400, &format!("{e:#}")),
            }
        }
        (Method::Get, ["api", "clips", id]) => {
            let clip = id
                .trim_end_matches(".wav")
                .parse()
                .ok()
                .and_then(|id| store.event(id).ok().flatten())
                .and_then(|e| e.clip_path);
            match clip {
                Some(rel) => wav(request, &std::fs::read(store.dir().join(rel))?),
                None => text(404, "no clip for this event"),
            }
        }
        _ => text(404, "not found"),
    })
}

/// Applies `overrides` to the config file, and stores them only if the result is valid.
fn override_settings(file: &Config, store: &Store, settings: &SharedSettings, overrides: Overrides) -> Result<Reply> {
    let config = match file.with_overrides(&overrides) {
        Ok(config) => config,
        Err(e) => return Ok(text(400, &format!("{e:#}"))),
    };
    store.set_overrides(&overrides)?;
    {
        let mut s = settings.lock().unwrap();
        (s.config, s.overrides) = (config, overrides);
        s.version += 1;
    }
    settings_reply(file, store, settings)
}

/// Every live setting with its range, help, value in effect and config-file value, the
/// settings that only the config file can change, and the space clips take now.
fn settings_reply(file: &Config, store: &Store, settings: &SharedSettings) -> Result<Reply> {
    let clip_mb = store.clip_bytes()? as f64 / 1e6;
    let s = settings.lock().unwrap();
    // Through a string, so f32 values read as written (0.85) rather than widened (0.8500000238).
    let as_written = |c: &Config| serde_json::from_str::<Value>(&serde_json::to_string(c)?);
    let (file_values, values) = (as_written(file)?, as_written(&s.config)?);
    let mut fields = Vec::new();
    for f in config::FIELDS {
        let mut field = serde_json::to_value(f)?;
        field["value"] = config::lookup(&values, f.key).cloned().unwrap_or(Value::Null);
        field["file"] = config::lookup(&file_values, f.key).cloned().unwrap_or(Value::Null);
        field["overridden"] = s.overrides.contains_key(f.key).into();
        fields.push(field);
    }
    Ok(json_reply(&json!({
        "fields": fields,
        "mode": s.mode,
        "file_only": { "audio.device": file.audio.device, "storage.dir": file.storage.dir, "web.listen": file.web.listen },
        "clip_mb": clip_mb,
    })))
}

fn read_body(request: &mut Request) -> Result<String> {
    let mut body = String::new();
    request.as_reader().take(4096).read_to_string(&mut body)?;
    Ok(body)
}

fn authorized(request: &Request, query: &HashMap<String, String>, token: &str) -> bool {
    if token.is_empty() {
        return true;
    }
    let bearer = format!("Bearer {token}");
    query.get("token").is_some_and(|t| t == token)
        || request.headers().iter().any(|h| h.field.equiv("Authorization") && h.value.as_str() == bearer)
}

/// Serves a WAV, honouring a single `Range: bytes=a-b`; Safari will not play audio without it.
fn wav(request: &Request, data: &[u8]) -> Reply {
    let range = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Range"))
        .and_then(|h| parse_range(h.value.as_str(), data.len()));
    let response = match range {
        Some((start, end)) => Response::from_data(data[start..=end].to_vec())
            .with_status_code(StatusCode(206))
            .with_header(header("Content-Range", &format!("bytes {start}-{end}/{}", data.len()))),
        None => Response::from_data(data.to_vec()),
    };
    with_type(response.with_header(header("Accept-Ranges", "bytes")), "audio/wav")
}

fn parse_range(value: &str, len: usize) -> Option<(usize, usize)> {
    let (start, end) = value.strip_prefix("bytes=")?.split_once('-')?;
    let (start, end) = match (start.parse::<usize>(), end.parse::<usize>()) {
        (Ok(s), Ok(e)) => (s, e.min(len - 1)),
        (Ok(s), Err(_)) => (s, len - 1),
        (Err(_), Ok(suffix)) => (len.saturating_sub(suffix), len - 1),
        _ => return None,
    };
    (start <= end && end < len).then_some((start, end))
}

fn split_url(url: &str) -> (&str, HashMap<String, String>) {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let params = query.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (decode(k), decode(v))).collect();
    (path, params)
}

fn decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => match s.get(i + 1..i + 3).and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
                Some(b) => {
                    out.push(b);
                    i += 2;
                }
                None => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

fn with_type(response: Reply, content_type: &str) -> Reply {
    response.with_header(header("Content-Type", content_type)).with_header(header("Cache-Control", "no-store"))
}

fn json_reply(value: &serde_json::Value) -> Reply {
    with_type(Response::from_data(value.to_string().into_bytes()), "application/json")
}

fn text(code: u16, message: &str) -> Reply {
    with_type(
        Response::from_data(message.as_bytes().to_vec()).with_status_code(StatusCode(code)),
        "text/plain; charset=utf-8",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        assert_eq!(parse_range("bytes=0-1", 10), Some((0, 1)));
        assert_eq!(parse_range("bytes=4-", 10), Some((4, 9)));
        assert_eq!(parse_range("bytes=-3", 10), Some((7, 9)));
        assert_eq!(parse_range("bytes=0-99", 10), Some((0, 9)));
        assert_eq!(parse_range("bytes=12-", 10), None);
    }

    #[test]
    fn query_decoding() {
        let (path, q) = split_url("/api/events?label=front%20door&status=pending&x=a+b&bad=%zz");
        assert_eq!(path, "/api/events");
        assert_eq!(q["label"], "front door");
        assert_eq!(q["x"], "a b");
        assert_eq!(q["bad"], "%zz");
    }
}

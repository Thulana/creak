//! `creak doctor`: checks the config, the storage, the model, the microphone, the service and
//! the web UI, one line each.

use crate::capture;
use crate::config::{Config, DEFAULT_PATH};
use crate::embed::Tagger;
use crate::setup::{self, Service};
use crate::store::{Store, now_ms};
use crate::trigger::SAMPLE_RATE;
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::{Duration, Instant};

const LISTEN_FOR: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Ok,
    Warn,
    Fail,
}

#[derive(Default)]
struct Report {
    failed: usize,
}

impl Report {
    fn line(&mut self, mark: Mark, what: &str, detail: impl AsRef<str>) {
        let symbol = match mark {
            Mark::Ok => "ok  ",
            Mark::Warn => "warn",
            Mark::Fail => {
                self.failed += 1;
                "FAIL"
            }
        };
        println!("{symbol}  {what:<11} {}", detail.as_ref());
    }
}

pub fn run(config_path: &Path) -> Result<()> {
    let mut report = Report::default();
    let file = match Config::load(config_path).and_then(|c| c.validate().map(|()| c)) {
        Ok(c) if !config_path.exists() && config_path == Path::new(DEFAULT_PATH) => {
            report.line(Mark::Ok, "config", format!("no {DEFAULT_PATH}; built-in defaults"));
            c
        }
        Ok(c) => {
            report.line(Mark::Ok, "config", format!("{} is valid", config_path.display()));
            c
        }
        Err(e) => {
            report.line(Mark::Fail, "config", format!("{e:#}"));
            bail!("the config file has to be fixed first");
        }
    };

    let config = match Store::open(&file.storage.dir) {
        Ok(store) => storage(&mut report, &file, &store),
        Err(e) => {
            let hint = "; run as the service's user: sudo -u creak creak doctor";
            report.line(Mark::Fail, "storage", format!("{}: {e:#}{hint}", file.storage.dir.display()));
            file.clone()
        }
    };

    match test_model() {
        Ok(ms) => report.line(Mark::Ok, "model", format!("loads and tags 1 s of audio in {ms} ms")),
        Err(e) => report.line(Mark::Fail, "model", format!("{e:#}")),
    }

    let service = setup::service();
    match &service {
        None => {}
        Some(Service::Missing) => report.line(Mark::Warn, "service", "not installed; sudo creak setup installs it"),
        Some(Service::Stopped(state)) => {
            report.line(Mark::Fail, "service", format!("{state}, not running; journalctl -u creak says why"))
        }
        Some(Service::Running { since, rss_mb }) => {
            let memory = rss_mb.map(|mb| format!(", using {mb:.0} MB")).unwrap_or_default();
            report.line(Mark::Ok, "service", format!("running since {since}{memory}"));
        }
    }

    // While the daemon runs it has the microphone, so its own view of it is asked for instead.
    let daemon = if config.web.enabled {
        match get(&config.web.listen, "/api/live", &config.web.token) {
            Ok(live) => {
                let url = setup::web_url(&config.web).unwrap_or_default();
                report.line(Mark::Ok, "web UI", format!("answers at {}", url.split('?').next().unwrap_or("")));
                Some(live)
            }
            Err(e) => {
                report.line(Mark::Fail, "web UI", format!("{}: {e:#}", config.web.listen));
                None
            }
        }
    } else {
        report.line(Mark::Ok, "web UI", "turned off in the config file");
        None
    };
    match daemon {
        Some(live) => daemon_microphone(&mut report, &live),
        None => microphone(&mut report, &config.audio.device, matches!(service, Some(Service::Running { .. }))),
    }

    if report.failed > 0 {
        bail!("{} of the checks failed", report.failed);
    }
    Ok(())
}

/// Reports the database, the space clips take and what is left on the disk, and returns the
/// config with the settings changed in the web UI applied.
fn storage(report: &mut Report, file: &Config, store: &Store) -> Config {
    let config = match store.overrides().and_then(|o| file.with_overrides(&o)) {
        Ok(config) => config,
        Err(e) => {
            report.line(Mark::Warn, "settings", format!("the daemon ignores the web UI's settings: {e:#}"));
            file.clone()
        }
    };
    let dir = file.storage.dir.display();
    let (Ok(clip_bytes), Ok(day), Ok(review)) =
        (store.clip_bytes(), store.events_since(now_ms() - 86_400_000), store.review_count())
    else {
        report.line(Mark::Fail, "storage", format!("{dir}: the database cannot be read"));
        return config;
    };
    let (clip_mb, max_mb) = (clip_bytes as f64 / 1e6, config.storage.max_mb as f64);
    let free_mb = setup::free_bytes(&file.storage.dir).map(|b| b as f64 / 1e6);
    let detail = format!(
        "{dir}: clips take {clip_mb:.0} of {max_mb:.0} MB{}",
        free_mb.map(|mb| format!(", {:.1} GB free on the disk", mb / 1e3)).unwrap_or_default()
    );
    match free_mb {
        Some(free) if free < max_mb - clip_mb => {
            report.line(Mark::Warn, "storage", format!("{detail}: the disk fills up before the clip space does"))
        }
        _ => report.line(Mark::Ok, "storage", detail),
    }
    report.line(Mark::Ok, "events", format!("{day} in the last 24 h, {review} waiting for review"));
    config
}

/// Loads the embedded model and tags a second of a test tone; returns how long tagging took.
fn test_model() -> Result<u128> {
    let tagger = Tagger::load()?;
    let tone: Vec<f32> = (0..SAMPLE_RATE as usize)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();
    let started = Instant::now();
    let tags = tagger.tag(&tone)?;
    ensure!(!tags.scores.is_empty() && !tags.embedding.is_empty(), "the model returned nothing");
    Ok(started.elapsed().as_millis())
}

/// The microphone as the running daemon sees it, from the Live view's data.
fn daemon_microphone(report: &mut Report, live: &Value) {
    let device = live["device"].as_str().unwrap_or_default();
    if let Some(error) = live["error"].as_str() {
        report.line(Mark::Fail, "microphone", format!("the daemon cannot record: {error}"));
    } else if device.is_empty() {
        report.line(Mark::Warn, "microphone", "the daemon has not opened a device yet");
    } else {
        let levels: Vec<f64> = live["levels"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
        let peak = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let detail = format!(
            "the daemon records from {device} at {} Hz; ambient {:.1} dBFS",
            live["device_rate"],
            live["ambient_db"].as_f64().unwrap_or_default()
        );
        if peak <= -119.0 {
            report.line(Mark::Warn, "microphone", format!("{detail}, but the last 30 s were digital silence"));
        } else {
            report.line(Mark::Ok, "microphone", detail);
        }
    }
}

/// Opens the microphone and listens for a moment.
fn microphone(report: &mut Report, device: &str, service_running: bool) {
    match listen(device) {
        Ok((name, rate, _, peak)) if peak <= -119.0 => report.line(
            Mark::Warn,
            "microphone",
            format!("{name} at {rate} Hz gives digital silence: muted, or the wrong device (`creak devices`)"),
        ),
        Ok((name, rate, level, peak)) => report.line(
            Mark::Ok,
            "microphone",
            format!("{name} at {rate} Hz: {level:.1} dBFS, peak {peak:.1} over {} s", LISTEN_FOR.as_secs()),
        ),
        Err(e) => {
            let busy = if service_running { "; the running service may hold the device" } else { "" };
            report.line(Mark::Fail, "microphone", format!("{e:#}{busy}"));
        }
    }
}

/// (device name, rate, level, loudest 20 ms frame) of a short listen, levels in dBFS.
fn listen(device: &str) -> Result<(String, u32, f32, f32)> {
    let input = capture::open(device)?;
    let (mut sum, mut count, mut peak) = (0.0f64, 0usize, f32::NEG_INFINITY);
    let frame = input.rate as usize / 50;
    let started = Instant::now();
    let mut samples = Vec::new();
    while started.elapsed() < LISTEN_FOR {
        let chunk = input.chunks.recv_timeout(Duration::from_secs(3)).context("no audio from the device")?;
        samples.extend(chunk);
        let whole = samples.len() / frame * frame;
        for f in samples[..whole].chunks(frame) {
            peak = peak.max(crate::trigger::level_db(f));
        }
        sum += samples[..whole].iter().map(|s| (s * s) as f64).sum::<f64>();
        count += whole;
        samples.drain(..whole);
    }
    let level = (10.0 * (sum / count.max(1) as f64).max(1e-12).log10()) as f32;
    Ok((input.name, input.rate, level, peak))
}

/// GETs `path` from the daemon's web server, which answers HTTP/1.0 and closes the connection.
fn get(listen: &str, path: &str, token: &str) -> Result<Value> {
    let addr = local_addr(listen)?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let auth = if token.is_empty() { String::new() } else { format!("Authorization: Bearer {token}\r\n") };
    write!(stream, "GET {path} HTTP/1.0\r\n{auth}\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let (head, body) = response.split_once("\r\n\r\n").context("not an HTTP response")?;
    let status = head.split_whitespace().nth(1).unwrap_or_default();
    ensure!(status == "200", "HTTP {status}: {}", body.trim());
    Ok(serde_json::from_str(body)?)
}

/// Where to reach a server listening on `listen` from this machine.
fn local_addr(listen: &str) -> Result<SocketAddr> {
    let mut addr = listen.to_socket_addrs()?.next().with_context(|| format!("{listen} resolves to nothing"))?;
    match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => addr.set_ip(Ipv4Addr::LOCALHOST.into()),
        IpAddr::V6(ip) if ip.is_unspecified() => addr.set_ip(Ipv6Addr::LOCALHOST.into()),
        _ => {}
    }
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_server_on_every_interface_is_reached_on_loopback() {
        assert_eq!(local_addr("0.0.0.0:8377").unwrap(), "127.0.0.1:8377".parse().unwrap());
        assert_eq!(local_addr("[::]:8377").unwrap(), "[::1]:8377".parse().unwrap());
        assert_eq!(local_addr("192.168.1.20:80").unwrap(), "192.168.1.20:80".parse().unwrap());
    }

    #[test]
    fn asks_the_web_server_for_json() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let listen = server.server_addr().to_ip().unwrap().to_string();
        std::thread::spawn(move || {
            for request in server.incoming_requests().take(2) {
                let authorized = request.headers().iter().any(|h| h.value.as_str() == "Bearer secret");
                let (code, body) = if authorized { (200, r#"{"device":"mic"}"#) } else { (401, "wrong token") };
                request.respond(tiny_http::Response::from_string(body).with_status_code(code)).unwrap();
            }
        });
        assert_eq!(get(&listen, "/api/live", "secret").unwrap()["device"], "mic");
        let err = get(&listen, "/api/live", "nope").unwrap_err().to_string();
        assert_eq!(err, "HTTP 401: wrong token");
    }
}

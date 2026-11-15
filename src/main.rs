mod capture;
mod classify;
mod config;
mod daemon;
mod doctor;
mod embed;
mod eval;
mod setup;
mod store;
mod trigger;
mod web;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// It heard that: learns the sounds of your home from your answers.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Config file. Missing at the default path means built-in defaults.
    #[arg(long, global = true, default_value = config::DEFAULT_PATH)]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Install or upgrade the service: user, directories, default config, systemd unit (root).
    Setup,
    /// List capture devices.
    Devices,
    /// Run the daemon (systemd runs this).
    Run,
    /// Show the event log, newest first.
    Events {
        /// Only events newer than this: e.g. 30m, 12h, 7d.
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        label: Option<String>,
        /// match, pending, ignored, confirmed, corrected or skipped.
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: i64,
        #[arg(long)]
        json: bool,
    },
    /// Confirm or correct an event's label; use `ignore` to stop keeping clips like it.
    Label { id: i64, label: String },
    /// Answer an event without teaching: it is no longer asked about and stops being an example.
    Skip { id: i64 },
    /// Score tagging against the clips you audited, and compare candidate matching changes.
    Eval {
        /// Replay with this model instead of the one in the config file, to compare them.
        #[arg(long)]
        model: Option<embed::Model>,
    },
    /// What waits for an answer, and where to give it.
    Review,
    /// Taught labels: their examples and how often their automatic tags were corrected.
    Labels {
        #[command(subcommand)]
        action: Option<LabelAction>,
    },
    /// Live meter: the microphone level, the ambient level and the line a sound has to cross.
    Calibrate {
        /// A device id from `creak devices`, instead of the one in the config file.
        #[arg(long)]
        device: Option<String>,
    },
    /// Check the config, storage, model, service, web UI and microphone.
    Doctor,
    /// Stop and remove the service; keeps config and data unless --purge (root).
    Uninstall {
        #[arg(long)]
        purge: bool,
    },
}

#[derive(Subcommand)]
enum LabelAction {
    /// Rename a label everywhere; renaming it to another label merges the two.
    Rename { label: String, to: String },
    /// Forget a label: its taught clips stop being examples, so nothing is tagged with it any more.
    Delete { label: String },
}

/// `creak review` lists at most this many of the events waiting.
const REVIEW_LIST: i64 = 20;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Setup => setup::setup(&cli.config),
        Command::Uninstall { purge } => setup::uninstall(&cli.config, purge),
        Command::Devices => capture::list_devices(),
        Command::Run => daemon::run(config::Config::load(&cli.config)?),
        Command::Events { since, label, status, limit, json } => {
            let store = open_store(&cli.config)?;
            let since_ms = since.map(|s| parse_age(&s).map(|ms| store::now_ms() - ms)).transpose()?;
            let events = store.events(&store::Filter {
                status,
                label,
                since_ms,
                before_id: None,
                hide_ignored: false,
                review: false,
                limit,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&events)?);
            } else {
                for e in events {
                    print_event(&e, store.dir());
                }
            }
            Ok(())
        }
        Command::Label { id, label } => {
            let mut store = open_store(&cli.config)?;
            let labelled = store.label_event(id, &label)?;
            print_event(&labelled.event, store.dir());
            if let Some(cause) = labelled.cause {
                println!(
                    "It had been tagged automatically because of event {0}, taught as {1:?}. \
                     `creak label {0} <label>` relabels that example; `creak skip {0}` removes it.",
                    cause.id, cause.label
                );
            }
            Ok(())
        }
        Command::Skip { id } => {
            let store = open_store(&cli.config)?;
            print_event(&store.skip_event(id)?, store.dir());
            Ok(())
        }
        Command::Eval { model } => eval::run(&config::Config::load(&cli.config)?, model),
        Command::Review => {
            let config = config::Config::load(&cli.config)?;
            let store = store::Store::open(&config.storage.dir)?;
            let waiting = store.events(&store::Filter { review: true, limit: REVIEW_LIST, ..Default::default() })?;
            for e in &waiting {
                print_question(e, store.dir());
            }
            let count = store.review_count()?;
            if count == 0 {
                println!("Nothing waits for review.");
                return Ok(());
            }
            let listed = if count > REVIEW_LIST { format!(", the {REVIEW_LIST} newest above") } else { String::new() };
            match setup::web_url(&config.web) {
                Some(url) => println!("{count} waiting for review{listed}. Play and answer them in the web UI: {url}"),
                None => println!(
                    "{count} waiting for review{listed}. Answer with `creak label <id> <label>` (the same label \
                     confirms it), `creak label <id> ignore` or `creak skip <id>`."
                ),
            }
            Ok(())
        }
        Command::Labels { action } => {
            let mut store = open_store(&cli.config)?;
            match action {
                None => print_labels(&store.label_stats()?),
                Some(LabelAction::Rename { label, to }) => {
                    let n = store.rename_label(&label, &to)?;
                    let to = classify::normalize_label(&to).unwrap_or(to);
                    println!("renamed {label:?} to {to:?}; taught clips with it: {n}");
                }
                Some(LabelAction::Delete { label }) => {
                    let n = store.delete_label(&label)?;
                    println!("forgot {label:?}; taught clips that are no longer examples: {n}");
                }
            }
            Ok(())
        }
        Command::Calibrate { device } => {
            let file = config::Config::load(&cli.config)?;
            // The trigger as the daemon runs it: with the settings changed in the web UI.
            let overrides = store::Store::open(&file.storage.dir).and_then(|s| s.overrides());
            let config = match overrides.and_then(|o| file.with_overrides(&o)) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!("using the config file without the web UI's settings: {e:#}");
                    file
                }
            };
            capture::calibrate(device.as_deref().unwrap_or(&config.audio.device), &config.trigger)
        }
        Command::Doctor => doctor::run(&cli.config),
    }
}

fn open_store(config_path: &std::path::Path) -> Result<store::Store> {
    store::Store::open(&config::Config::load(config_path)?.storage.dir)
}

fn local_time(ts: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ts)
        .map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default()
}

fn clip_text(e: &store::Event, dir: &std::path::Path) -> String {
    e.clip_path.as_ref().map(|c| format!("  {}", dir.join(c).display())).unwrap_or_default()
}

fn print_event(e: &store::Event, dir: &std::path::Path) {
    let time = local_time(e.ts);
    let nearest = match (&e.nearest_label, e.similarity) {
        (Some(l), Some(s)) => format!("  nearest {l} {s:.2}"),
        _ => String::new(),
    };
    let clip = clip_text(e, dir);
    println!("{:>6}  {time}  {:<9}  {:<20}{nearest}{clip}", e.id, e.status, e.label);
}

/// An event waiting for review, asked as the Review view asks it.
fn print_question(e: &store::Event, dir: &std::path::Path) {
    let question = if e.status == "match" { format!("{}, right?", e.label) } else { format!("{}?", e.label) };
    let note = match (&e.conflict, &e.nearest_label) {
        (Some(c), Some(nearest)) if classify::normalize_label(c).as_ref() == Some(&e.label) => {
            format!("  came in over {nearest}")
        }
        (Some(c), _) => format!("  the model hears {c}"),
        (None, _) => String::new(),
    };
    println!("{:>6}  {}  {question:<24}{note}{}", e.id, local_time(e.ts), clip_text(e, dir));
}

fn print_labels(labels: &[store::LabelStats]) {
    if labels.is_empty() {
        println!("Nothing taught yet: every clip you confirm or correct teaches its label.");
        return;
    }
    println!("{:<24} {:>8} {:>8} {:>10}", "label", "examples", "matches", "corrected");
    for l in labels {
        let corrected = match l.matches {
            0 => "–".to_string(),
            n => format!("{} ({:.0} %)", l.corrected, 100.0 * l.corrected as f64 / n as f64),
        };
        println!("{:<24} {:>8} {:>8} {:>10}", l.label, l.exemplars, l.matches, corrected);
    }
}

fn parse_age(s: &str) -> Result<i64> {
    let (number, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: i64 = number.parse()?;
    let unit_ms = match unit {
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => bail!("use a number with m, h or d, e.g. 12h"),
    };
    Ok(n * unit_ms)
}

mod capture;
mod classify;
mod config;
mod daemon;
mod embed;
mod setup;
mod store;
mod trigger;
mod web;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Learns the sounds of your home from one confirmation each.
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
        /// match, pending, ignored, confirmed or corrected.
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: i64,
        #[arg(long)]
        json: bool,
    },
    /// Confirm or correct an event's label; use `ignore` to stop keeping clips like it.
    Label { id: i64, label: String },
    /// Stop and remove the service; keeps config and data unless --purge (root).
    Uninstall {
        #[arg(long)]
        purge: bool,
    },
}

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
            let events = store.events(&store::Filter { status, label, since_ms, before_id: None, limit })?;
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
            print_event(&store.label_event(id, &label)?, store.dir());
            Ok(())
        }
    }
}

fn open_store(config_path: &std::path::Path) -> Result<store::Store> {
    store::Store::open(&config::Config::load(config_path)?.storage.dir)
}

fn print_event(e: &store::Event, dir: &std::path::Path) {
    let time = chrono::DateTime::from_timestamp_millis(e.ts)
        .map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default();
    let nearest = match (&e.nearest_label, e.similarity) {
        (Some(l), Some(s)) => format!("  nearest {l} {s:.2}"),
        _ => String::new(),
    };
    let clip = e.clip_path.as_ref().map(|c| format!("  {}", dir.join(c).display())).unwrap_or_default();
    println!("{:>6}  {time}  {:<9}  {:<20}{nearest}{clip}", e.id, e.status, e.label);
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

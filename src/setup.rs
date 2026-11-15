//! Installing and removing the service. This is the platform-specific part: Linux with
//! systemd today; macOS (launchd) and Windows (service) come later.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use anyhow::Result;
use std::path::Path;

#[cfg(target_os = "linux")]
pub use linux::{setup, uninstall};

#[cfg(not(target_os = "linux"))]
pub fn setup(_config_path: &Path) -> Result<()> {
    anyhow::bail!("`creak setup` supports Linux only for now; run `creak run --config <file>` directly")
}

#[cfg(not(target_os = "linux"))]
pub fn uninstall(_config_path: &Path, _purge: bool) -> Result<()> {
    anyhow::bail!("`creak uninstall` supports Linux only for now")
}

/// The config `setup` writes on first install: reachable from the LAN, protected by a token.
pub fn initial_config(token: &str) -> String {
    format!(
        "# creak configuration. Every key is optional; see the README for all of them.\n\
         \n\
         [audio]\n\
         device = \"default\"   # `creak devices` lists the alternatives\n\
         \n\
         [web]\n\
         enabled = true\n\
         # Reachable from other machines on the network; keep it off the internet.\n\
         listen = \"0.0.0.0:8377\"\n\
         token = \"{token}\"\n"
    )
}

pub fn unit_file(exe: &Path, config_path: &Path, data_dir: &Path) -> String {
    format!(
        "[Unit]\n\
         Description=creak: learns the sounds of your home\n\
         After=network-online.target sound.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         ExecStart={exe} run --config {config}\n\
         User=creak\n\
         Group=creak\n\
         SupplementaryGroups=audio\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         NoNewPrivileges=yes\n\
         ProtectSystem=strict\n\
         ProtectHome=yes\n\
         PrivateTmp=yes\n\
         ReadWritePaths={data}\n\
         \n\
         [Install]\n\
         WantedBy=multi-user.target\n",
        exe = exe.display(),
        config = config_path.display(),
        data = data_dir.display(),
    )
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::config::Config;
    use anyhow::{Context, bail};
    use std::fs;
    use std::io::Read;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    const UNIT_PATH: &str = "/etc/systemd/system/creak.service";
    const USER: &str = "creak";

    pub fn setup(config_path: &Path) -> Result<()> {
        require_root()?;
        if !succeeds("id", &["-u", USER]) {
            let mut args = vec!["--system", "--user-group", "--no-create-home", "--shell", "/usr/sbin/nologin"];
            if succeeds("getent", &["group", "audio"]) {
                args.extend(["--groups", "audio"]);
            }
            args.push(USER);
            run("useradd", &args)?;
            println!("created system user {USER}");
        }

        let config_dir = config_path.parent().context("config path has no directory")?;
        fs::create_dir_all(config_dir)?;
        let new_token = if config_path.exists() {
            println!("keeping existing {}", config_path.display());
            None
        } else {
            let token = random_token()?;
            fs::write(config_path, initial_config(&token))?;
            run("chown", &[&format!("root:{USER}"), &config_path.to_string_lossy()])?;
            fs::set_permissions(config_path, fs::Permissions::from_mode(0o640))?;
            println!("wrote {}", config_path.display());
            Some(token)
        };
        let config = Config::load(config_path)?;

        let data_dir = &config.storage.dir;
        fs::create_dir_all(data_dir)?;
        run("chown", &["-R", &format!("{USER}:{USER}"), &data_dir.to_string_lossy()])?;
        fs::set_permissions(data_dir, fs::Permissions::from_mode(0o750))?;

        let exe = std::env::current_exe()?;
        fs::write(UNIT_PATH, unit_file(&exe, config_path, data_dir))?;
        run("systemctl", &["daemon-reload"])?;
        run("systemctl", &["enable", "--quiet", "creak"])?;
        run("systemctl", &["restart", "creak"])?;
        println!("service creak (re)started; logs: journalctl -u creak -f");

        if config.web.enabled {
            let port = config.web.listen.rsplit(':').next().unwrap_or("8377");
            let host = if config.web.listen.starts_with("127.") { "127.0.0.1".to_string() } else { lan_address() };
            let token = new_token.unwrap_or(config.web.token);
            let query = if token.is_empty() { String::new() } else { format!("?token={token}") };
            println!("web UI: http://{host}:{port}/{query}");
        }
        Ok(())
    }

    pub fn uninstall(config_path: &Path, purge: bool) -> Result<()> {
        require_root()?;
        let config = Config::load(config_path).unwrap_or_default();
        let _ = run("systemctl", &["disable", "--now", "--quiet", "creak"]);
        if Path::new(UNIT_PATH).exists() {
            fs::remove_file(UNIT_PATH)?;
            run("systemctl", &["daemon-reload"])?;
        }
        let exe = std::env::current_exe()?;
        if exe.starts_with("/usr/local/bin") {
            fs::remove_file(&exe)?;
        }
        if purge {
            let _ = fs::remove_dir_all(&config.storage.dir);
            if let Some(dir) = config_path.parent() {
                let _ = fs::remove_dir_all(dir);
            }
            if succeeds("id", &["-u", USER]) {
                run("userdel", &[USER])?;
            }
            println!("creak removed, with its config, data and user");
        } else {
            println!(
                "creak removed; kept {} and {} (use --purge to delete them)",
                config_path.display(),
                config.storage.dir.display()
            );
        }
        Ok(())
    }

    fn require_root() -> Result<()> {
        let status = fs::read_to_string("/proc/self/status")?;
        let uid = status.lines().find_map(|l| l.strip_prefix("Uid:")).and_then(|v| v.split_whitespace().nth(1));
        if uid != Some("0") {
            bail!("this needs root; run it with sudo");
        }
        Ok(())
    }

    fn random_token() -> Result<String> {
        let mut bytes = [0u8; 16];
        fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    fn lan_address() -> String {
        Command::new("hostname")
            .arg("-I")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.split_whitespace().next().map(str::to_string))
            .unwrap_or_else(|| "<this-machine>".into())
    }

    fn succeeds(cmd: &str, args: &[&str]) -> bool {
        Command::new(cmd).args(args).output().is_ok_and(|o| o.status.success())
    }

    fn run(cmd: &str, args: &[&str]) -> Result<()> {
        let out = Command::new(cmd).args(args).output().with_context(|| format!("running {cmd}"))?;
        if !out.status.success() {
            bail!("{cmd} {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn initial_config_parses() {
        let c: Config = toml::from_str(&initial_config("abc")).unwrap();
        assert_eq!((c.web.listen.as_str(), c.web.token.as_str()), ("0.0.0.0:8377", "abc"));
    }
}

# creak — implementation plan

_Written 2026-11-15. Status: Phase 0 (minimal deployable version) in progress._

`creak` is a single-binary daemon that listens to one microphone all the time, cuts a clip whenever the sound rises above `n ×` the ambient level, lets a model tag the clip (door, doorbell, party, movie…), logs it, and publishes it to Home Assistant over MQTT. It then asks you whether the tag is right. One confirmation is enough: from then on, anything that sounds like that clip is tagged and published without asking.

Linux comes first. macOS and Windows follow later (Phase 7), so platform-specific code stays in one module.

## 1. Scope

**In v1**

- Always-on capture from one input device, with an ambient-relative trigger.
- Clips and an event log on local disk.
- Every clip gets a tag proposed by the model first. You confirm or correct it, and one confirmation makes that tag automatic. Labels are yours; there is no fixed list.
- A confirm / correct / ignore loop in a web UI that plays each clip, and in the CLI.
- MQTT publishing with Home Assistant discovery.
- `curl | sh` install, systemd service, upgrade and uninstall.

**Out of v1**

- Speech transcription, speaker identification, wake words.
- Several microphones or rooms, direction of arrival.
- Training or fine-tuning a neural network.
- Any cloud service.

## 2. Decisions

| Topic | Choice | Why |
|---|---|---|
| Language | Rust, one binary | Nothing to install besides the binary. Low idle memory for a 24/7 process. |
| Capture | `cpal`, resampled to 16 kHz mono | ALSA on Linux, CoreAudio and WASAPI later, from the same code. |
| Tagging | One pretrained audio model used two ways: its own AudioSet classes give the first guess, and its embedding is matched against your confirmed clips | The first guess needs no setup. A confirmed clip is ground truth, so one is enough to match the same sound again. Runs in milliseconds on CPU. |
| Model | YAMNet (MobileNetV1 on log-mel patches, 521 AudioSet classes, 1024-d embedding), converted to ONNX by `model/convert.py`. Apache-2.0. | Small, well documented, outputs both class scores and an embedding. Whether it separates your sounds well enough is measured in Phase 0 on the real installation, not in a separate spike. CED-tiny stays the alternative if it does not. |
| Inference | `tract-onnx`, pure Rust | No native library to link or ship, so cross-compiling for aarch64, macOS and Windows stays trivial. |
| Frontend | Log-mel computed in Rust, matching YAMNet's TensorFlow frontend | The ONNX graph stays a plain CNN. Checked against the TensorFlow reference on a fixture. |
| Model file | Embedded in the binary | One download, no second fetch at install time. |
| Storage | SQLite (`rusqlite`, bundled) + WAV clips | No database server. |
| Review surface | Web UI served by the daemon (§3.7), plus CLI commands | A home server is headless, and you have to hear a clip to label it. Everything except playback also works from the CLI. |
| Output | MQTT with HA discovery. The SQLite log always exists. | Standard route into Home Assistant and anything else. |
| LLM | Optional and last. Could replace the generic first guess for clips with no close exemplar. | Small audio LLMs are weak on environmental sound. |
| Distribution | GitHub Releases + a short `install.sh` | See §8. |

## 3. How it works

```
mic ─▶ capture ─▶ level meter ─▶ trigger ─▶ clip (WAV) ─▶ embed ─▶ match ─▶ event row ─▶ MQTT
         │            │                                              ▲
         │            └─ ambient = 10th percentile, trailing 5 min   │
         └─ 2 s ring buffer (pre-roll)                    exemplars ◀┴─ confirm / correct / ignore
```

### 3.1 Trigger

- Level is RMS over 20 ms frames, in dBFS.
- **Ambient** is the 10th percentile of frame levels over the trailing 5 minutes, with a configurable minimum.
- **Open** when the level stays above `ambient × ratio` for `min_ms`. Default ratio 3 (about +9.5 dB).
- **Close** when the level stays below `ambient × close_ratio` for `hang_ms`.
- The clip includes 1.5 s of pre-roll from the ring buffer and is cut at `max_clip_s` (default 15 s).
- A cooldown stops one long noise from producing a storm of events.

### 3.2 Tagging

The model always answers first; the user is only asked to check.

1. Take the active part of the clip (from the trigger opening to its close, at least ~1 s, widened into the pre-roll if shorter). The pre-roll stays in the WAV for listening but is not embedded, so room ambient does not dominate the embedding.
2. Run the model on it in 0.96 s patches with a 0.48 s hop. It returns AudioSet class scores and an embedding per patch; mean-pool both and L2-normalise the embedding.
3. Find the nearest confirmed exemplar by cosine similarity (brute force is fine for thousands).
4. Result:
   - **match**: the nearest exemplar is above `match_threshold`. The clip takes that exemplar's label and is published at once. Nobody is asked.
   - **pending**: nothing that close. The clip gets a proposed label and goes to the review queue as a question ("Door?"). The proposal is the nearest exemplar's label if it is above `suggest_threshold`, otherwise the model's top AudioSet class.
   - **ignored**: the nearest match carries the reserved label `ignore`. The event row is kept, the clip is not.

On day one there are no exemplars, so every clip is pending with the model's own guess. Every event records its nearest exemplar and similarity, even below the thresholds, so the thresholds can be tuned from real data.

### 3.3 Feedback loop

- You answer a pending clip with confirm, correct (pick or type another label) or ignore. Any of the three stores the clip as an exemplar for the chosen label.
- **One confirmation is enough.** You know what happened and when, so a confirmed clip is ground truth. There is no minimum count and no trust score: a label with one exemplar already produces matches.
- Reliability comes from `match_threshold`, not from repetition. A sound that differs from the exemplar (the same door closed gently) falls below it, is asked about once, and then matches as well.
- A wrong match can be corrected afterwards in the event log. The clip becomes an exemplar for the right label, so the nearest exemplar is the right one next time.
- Pending clips are not published unless `publish_pending = true`.

### 3.4 Scenes (party, movie, music)

Scenes last for minutes or hours, so the trigger is the wrong tool: the ambient level rises with them.

- A label is either an `event` or a `scene`.
- Teach a scene while it is happening: `creak teach party --for 10m` samples exemplars for that long.
- Once any scene label exists, the daemon classifies the last 10 s every minute.
- A scene turns on after 3 of the last 5 samples match and turns off after 5 misses in a row.

### 3.5 Storage and privacy

- `events(id, ts, duration_ms, peak_db, ambient_db, label, status, similarity, nearest_label, top_classes, clip_path, embedding)`
- `exemplars(id, event_id, label, embedding, clip_path, created_ts)`
- `labels(name, kind)` arrives with scenes; until then labels are derived from exemplars.
- Clips will contain conversations. They never leave the machine.
- Clips that are not exemplars are deleted after `retention_days` (default 14) or when `max_clip_mb` is reached. Exemplar clips are kept, so they can be re-embedded if the model changes.

### 3.6 MQTT

| Topic | Payload |
|---|---|
| `creak/status` | `online` / `offline` (last will) |
| `creak/event` | `{id, ts, label, status, similarity, duration_ms, peak_db, ambient_db, clip_url}` |
| `creak/scene/<label>` | `ON` / `OFF`, retained |
| `creak/ambient` | ambient level in dBFS, once a minute |
| `creak/label/set` | inbound `{id, label}`, so an HA notification action can confirm a tag |

HA discovery creates one `event` entity (event types are your labels), one `binary_sensor` per scene and one `sensor` for the ambient level.

### 3.7 Web UI

Turned on or off with `[web] enabled`. With it off, the daemon, the CLI and MQTT work unchanged; you label by event id and fetch clips from the path `creak events` prints.

| View | What it does |
|---|---|
| **Review** | The pending queue, newest first. Each clip has a play button, the proposed label as a question ("Door?"), the model's top classes, and confirm / correct / ignore. Keyboard shortcuts for all four, so a queue clears quickly. |
| **Events** | The event log, filtered by label and status. Every row plays its clip, shows the nearest exemplar and similarity, and can be relabelled. |
| **Labels** | Exemplar count per label, and how often its matches were corrected. Rename, delete, switch between `event` and `scene`. |
| **Live** | Current level, ambient and the trigger lines, for checking the microphone and tuning `ratio` from a browser. |

- One static HTML file with plain JavaScript, embedded in the binary. No build step, no framework.
- It talks to a small JSON API on the same port; clips are served as WAV with range requests, so every browser's own audio player works.
- The built-in default binds to `127.0.0.1`. `creak setup` writes a config that binds to all interfaces with a random token, because the usual install is a headless box reviewed from another device. Keep it on a trusted network or a VPN; never expose it to the internet.

## 4. CLI

```
creak setup                     # root: user, dirs, default config, systemd unit
creak devices                   # list capture devices
creak calibrate                 # live meter: level, ambient, trigger line
creak run                       # the daemon; systemd runs this
creak events [--since 1d] [--label door] [--json]
creak review                    # pending count, and the web UI URL if it is enabled
creak label <event-id> <label|ignore>
creak labels                    # exemplar count and corrected-match rate per label
creak teach <scene> --for 10m
creak doctor                    # checks mic, model, MQTT, disk
creak uninstall [--purge]
```

The CLI reads and writes the same SQLite file (WAL mode). The daemon reads exemplars from the database for every clip, so a label set from the CLI applies to the next clip.

## 5. Config

`/etc/creak/config.toml` on Linux. Everything has a default; this is the whole surface.

```toml
[audio]
device = "default"

[trigger]
ratio = 3.0
close_ratio = 1.5
min_ms = 60
hang_ms = 800
max_clip_s = 15
cooldown_s = 2
ambient_floor_db = -60

[classify]
match_threshold = 0.80
suggest_threshold = 0.60
publish_pending = false

[storage]
dir = "/var/lib/creak"
retention_days = 14
max_clip_mb = 2000

[web]
enabled = true
listen = "127.0.0.1:8377"
token = ""

[mqtt]
enabled = false
url = "mqtt://127.0.0.1:1883"
discovery = true
```

The two thresholds are starting values. Phase 0 measures the real similarity ranges and sets them.

## 6. Repository layout

```
creak/
├── plan.md
├── Cargo.toml
├── install.sh
├── model/             # convert.py, yamnet.onnx, class map, licence notes
├── src/
│   ├── main.rs        # clap commands
│   ├── config.rs      # config file and defaults
│   ├── capture.rs     # cpal stream, resampling, ring buffer, clip cutting
│   ├── trigger.rs     # level, ambient, open/close; pure and unit-tested
│   ├── embed.rs       # log-mel frontend, model session, pooling
│   ├── classify.rs    # nearest exemplar, proposal rules; pure and unit-tested
│   ├── store.rs       # SQLite, clip files, retention
│   ├── daemon.rs      # wires capture → trigger → classify → store
│   ├── web.rs         # web UI and its JSON API
│   ├── mqtt.rs        # publish, discovery, label/set
│   └── setup.rs       # setup, uninstall, doctor; the platform-specific part
├── ui/index.html      # one page, embedded in the binary
├── tests/fixtures/    # short WAVs and the TensorFlow reference output
└── .github/workflows/ # ci.yml, release.yml
```

## 7. Phases

Each phase ends with a check. Do not start the next one until it passes.

Phase 0 replaces a separate offline spike: the questions it was meant to answer (does the microphone give usable audio, is the model's first guess sensible, where do the thresholds sit) are answered by installing a minimal version on the real machine and using its web UI for a few days.

| # | Phase | Delivers | Check |
|---|---|---|---|
| 0 | **Minimal deployable version** | `devices`, `run`, `events`, `label`, `setup`, `uninstall`. Capture, trigger, clips, YAMNet tagging, exemplar matching, retention by age. Web UI with Review, Events and Live. Release workflow for Linux x86_64 and aarch64, `install.sh`. | A fresh Ubuntu 24.04 machine or container goes from the one-liner to a running service and prints the UI URL; re-running it upgrades and keeps the config. The Live view moves with the room and clips sound right. A clap gives exactly one clip with pre-roll. Rust log-mel and embeddings match the TensorFlow reference on the fixture. After a few days with ~4 real sounds confirmed once each: the model's own top class is a sensible first guess for most clips, and there is a `match_threshold` where wrong matches stay under 5 % and at least half of the same-sound clips match from one exemplar. Model and thresholds are written into this plan. |
| 1 | **Complete the core** | `calibrate`, `review`, `labels`, `doctor`. Labels view (rename, delete, corrected-match rate). Retention by size. 24 h soak. | No event storm and flat memory over 24 h. A wrong match corrected in the event log does not repeat. With `[web] enabled = false` the daemon still runs and `creak label` still works. |
| 2 | **MQTT and Home Assistant** | Publishing, discovery, `label/set`. | Entities appear in HA without YAML. An automation fires on a matched event. An HA notification action confirms a tag. |
| 3 | **Scenes** | `labels` table, `teach`, the one-minute sampler, scene state. | A movie is taught once; the next evening the scene turns on within 5 minutes and off within 5 minutes of the end. |
| 4 | **LLM first guess, optional** | For clips with no close exemplar, ask an OpenAI-compatible endpoint with audio input for the proposed label, in place of the AudioSet class. Off by default. | Only build this if the model's own first guesses are too generic to be useful after Phase 1. |
| 5 | **macOS and Windows** | Release builds for both. `setup` / `uninstall` install a launchd agent on macOS and a Windows service; platform default paths for config and data. Install via a script or a package manager. | On each OS: install, the Live view moves, a clap gives one clip, uninstall leaves nothing behind except data. |

## 8. Packaging and distribution

**Artifacts**, built by GitHub Actions on a version tag:

- `creak-<version>-x86_64-unknown-linux-gnu.tar.gz` and `…-aarch64-unknown-linux-gnu.tar.gz` (Raspberry Pi class).
- `SHA256SUMS`.
- Built on the oldest supported Ubuntu runner, so the binary runs on Debian 12+ and Ubuntu 22.04+. Runtime dependencies: glibc and `libasound2`.

**`install.sh`** is POSIX `sh`, under 100 lines, and does only this:

1. Detect the architecture and refuse anything unsupported.
2. Resolve the version: `CREAK_VERSION` if set, otherwise the latest release.
3. Download the tarball and `SHA256SUMS`, verify the checksum, stop on a mismatch.
4. Install the binary to `/usr/local/bin/creak`.
5. Run `creak setup`, unless `CREAK_NO_SETUP=1`.

Everything stateful lives in `creak setup`, in Rust, where it can be tested: the `creak` system user in the `audio` group, `/etc/creak`, `/var/lib/creak`, a default config that is never overwritten, and a hardened systemd unit (`User=creak`, `NoNewPrivileges`, `ProtectSystem=strict`, `Restart=on-failure`).

**Install**

```sh
curl -fsSL https://raw.githubusercontent.com/Thulana/creak/main/install.sh | sh
```

The README also documents the inspect-first form and a pinned version:

```sh
curl -fsSL https://raw.githubusercontent.com/Thulana/creak/main/install.sh -o install.sh
less install.sh
CREAK_VERSION=v0.1.0 sh install.sh
```

**Upgrade** is re-running the installer; `setup` restarts the service. **Uninstall** is `creak uninstall`, which keeps data unless `--purge` is given.

## 9. Running in a container

- The microphone belongs to the host. In an LXC or Docker container, pass through the capture nodes under `/dev/snd` and make sure the container's `audio` group owns them (or map the host's audio gid).
- systemd hardening (`ProtectSystem`, `PrivateTmp`) needs namespace support; in an unprivileged LXC container enable nesting, or the unit fails with `226/NAMESPACE`.
- No MQTT broker or Home Assistant is needed until Phase 2.

## 10. Risks and open questions

1. **Microphones with built-in processing.** Array microphones that apply voice-tuned noise suppression before the OS sees the signal may flatten the sounds this tool is for. Phase 0's first clips show it; a plain USB microphone is the fallback.
2. **Doors and windows in other rooms.** One microphone hears a distant window poorly, and a contact sensor is more reliable for those. Expect the best results on sounds with no sensor: doorbell, appliances, alarms, party, movie.
3. **Mean-pooling short sounds.** A door slam fills one or two patches; the active-segment rule in §3.2 keeps ambient out, but very short sounds may still need max-pooling or the loudest patch. Phase 0 data decides.
4. **Rust.** Chosen for the single binary. The Python alternative would be a `uv tool install` package, with the same design and a heavier install.
5. **Why not an existing tool.** bark-detector, ha-audio-events, sound-sensor-homeassistant, AudioClassifier-MQTT, Yamcam, Heyra and Frigate all classify into YAMNet's fixed labels. None learns your own sounds or has a confirm loop, and none uses an ambient-relative trigger. bark-detector's clip log and MQTT discovery are worth reading before Phases 1 and 2.

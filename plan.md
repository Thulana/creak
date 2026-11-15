# creak — implementation plan

_Written 2026-11-15. Status: Phase 0 (minimal deployable version) in review; Phase 1 (accuracy) next._

`creak` is a single-binary daemon that listens to one microphone all the time, cuts a clip whenever the sound rises above `n ×` the ambient level, lets a model tag the clip (door, doorbell, party, movie…), logs it, and publishes it to Home Assistant over MQTT. It then asks you whether the tag is right. A confirmation is usually enough: from then on, anything that sounds like that clip is tagged and published without asking, unless the model hears something that contradicts the label (§3.2).

Linux comes first. macOS and Windows follow later (Phase 6), so platform-specific code stays in one module.

## 1. Scope

**In v1**

- Always-on capture from one input device, with an ambient-relative trigger.
- Clips and an event log on local disk.
- Every clip gets a tag proposed by the model first. You confirm or correct it, and a confirmation usually makes that tag automatic. Labels are yours; there is no fixed list.
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
- The clip includes `pre_roll_ms` (1.5 s) of pre-roll from the ring buffer.
- **Tail.** After the close, recording goes on for `tail_ms` (2 s). A sound that crosses the open line during the tail reopens the same clip instead of starting a new one after the cooldown.
- Every clip runs at least `min_clip_s` (4 s) from its onset, so a click or a knock comes with what followed it, and is cut at `max_clip_s` (30 s). Pre-roll, tail and this padding are for listening; only onset to close is embedded (§3.2).
- A cooldown stops one long noise from producing a storm of events.
- **Modes.** `active` records. `deaf` keeps measuring the level and the ambient but opens no clip, and drops one being recorded. The mode is stored in the database and switched from the web UI header or `POST /api/mode` (later also `creak/mode/set`, §3.6), so an automation can make creak listen only while nobody is home.
- **Grouping.** A clip whose label equals the previous event's, and that starts within `group_gap_s` (default 10 s) of that event's end, extends the event instead of creating a new one: its duration grows and its clip count goes up, and only the first clip's audio is kept. Continuous music becomes one event, not one every 17 s.

### 3.2 Tagging

The model always answers first; the user is only asked to check.

1. Take the active part of the clip (from the trigger opening to its close, at least ~1 s, widened into the pre-roll if shorter). The pre-roll and the tail stay in the WAV for listening but are not embedded, so room ambient does not dominate the embedding.
2. Run the model on it in 0.96 s patches with a 0.48 s hop. It returns AudioSet class scores and an embedding per patch; mean-pool both and L2-normalise the embedding.
3. Take the model's view without the classes that are not a sound: Silence, the room and outdoor ambience classes, reverberation, echo and the noise classes. These are never proposed or shown. The best remaining class is the model's guess if it scores at least `sound_score` (0.1).
4. Find the nearest confirmed exemplar by cosine similarity (brute force is fine for thousands).
5. Result, in this order:
   - **conflict → pending**: the nearest exemplar is above `match_threshold`, but the model is sure (score ≥ `conflict_score`, 0.3) of a sound that none of that label's exemplars was heard as. The clip goes to review as "Music? · the model hears Cough" instead of being matched. Confirming it adds that class to the label's exemplars, so the same pair is not asked about again. The same holds for `ignore`, so ignoring typing does not swallow a cough. A conflicting clip is never grouped into the previous event.
   - **ignored**: the nearest exemplar is above `match_threshold` and carries the reserved label `ignore`. The event row is kept, the clip is not.
   - **match**: the nearest exemplar is above `match_threshold`. The clip takes that exemplar's label and is published at once. Nobody is asked.
   - **dropped**: the model has no guess, so it heard nothing but silence or room tone. No event, no clip, no question; only its classification is kept (§3.5). Silence is never an event and `silence` is not accepted as a label.
   - **pending**: the clip gets a proposed label and goes to the review queue as a question ("Door?"). The proposal is the nearest exemplar's label if it is above `suggest_threshold`, otherwise the model's guess. A nearest-exemplar proposal carries the same conflict note when the model disagrees with it.

On day one there are no exemplars, so every clip is pending with the model's own guess. Every event records its nearest exemplar and similarity, even below the thresholds, so the thresholds can be tuned from real data.

### 3.3 Feedback loop

- You answer a pending clip with confirm, correct (pick or type another label) or ignore. Any of the three stores the clip as an exemplar for the chosen label.
- **One confirmation is usually enough.** You know what happened and when, so a confirmed clip is ground truth. There is no minimum count and no trust score: a label with one exemplar already produces matches. It asks again only when the model contradicts the label (§3.2).
- Reliability comes from `match_threshold`, not from repetition. A sound that differs from the exemplar (the same door closed gently) falls below it, is asked about once, and then matches as well.
- A wrong match can be corrected afterwards in the event log. The clip becomes an exemplar for the right label, so the nearest exemplar is the right one next time.
- Pending clips are not published unless `publish_pending = true`.
- **Label the sound that triggered the clip, not what was on in the background.** Background music belongs to scenes (§3.4). A cough confirmed as "music" because music was playing becomes a music exemplar, and the next cough is tagged "music" without a question. Phase 0 data had this for 9 of 16 music exemplars; §3.8 is about catching and preventing it.

### 3.4 Mixed clips and scenes (party, movie, dinner)

A clip often holds more than one sound: music, several voices, a laugh. Its tags are the mean over its patches, so it collapses to the strongest class ("Music") and the rest is lost. A scene such as a party lasts minutes or hours, and the trigger is the wrong tool for it: the ambient level rises with the scene, and the open line with it, exactly while the scene is on. Three layers, each useful on its own:

**1. The sound mix of a clip.** YAMNet already scores every 0.96 s patch. For each sound class (the `sounds` of §3.2, so no room tone), keep the share of the clip's patches in which it scores at least `sound_score`, and its peak. It is stored per event as `mix` (the top eight or so) and shown in Review and Events as "Speech 70 % · Music 55 % · Laughter 20 %". A clip with two or more sounds, each in at least a quarter of its patches, is *mixed*.

- **Foreground and background.** Classes already present in the pre-roll are background; classes that appear only after the onset are what triggered the clip. An event's proposal comes from the foreground, so a cough over music is proposed as a cough (§3.8, risk 3). The pre-roll is run through the model for this comparison only; it is still not embedded.

**2. A scene sampler, independent of the trigger.** Every `scene_every_s` (60 s) the daemon tags the last 10 s of audio, whether or not a clip is open; never in deaf mode. A scene window is the last `scene_window_min` (5) minutes of samples, plus the mixes of the clips cut in it. Its fingerprint is the mean presence of each class over the window and the mean embedding.

**3. Naming a scene: the model guesses, you confirm.** The same loop as for events.

- **First guess** without any teaching, from a small table of AudioSet classes in `classify.rs`:

  | Scene | Needs, across the window |
  |---|---|
  | party | Music, and Chatter / Hubbub / Crowd, and Laughter / Cheering / Clapping |
  | tv or movie | Television, or Speech and Music, steady for most of the window |
  | cooking | Frying / Sizzle / Dishes / Microwave / Water tap |
  | dinner | Speech, and Cutlery / Dishes |
  | cleaning | Vacuum cleaner / Washing machine / Dishwasher |

- **Review** gets a second kind of card for a window that a rule matches but no taught scene does: "Party? · music, chatter, laughter for 12 min", with confirm, correct and ignore. The answer stores the window's fingerprint as a scene exemplar. A scene can also be taught while it happens: `creak teach party --for 10m`.
- **Matching** is cosine similarity of the fingerprint to the scene exemplars, with the event thresholds. A scene turns on after 3 of the last 5 samples match and off after 5 misses in a row.
- Events during a scene carry it (`scene` column), so "a glass breaking during the party" can be found, and the scene's classes count as background when a clip in it is proposed a label.
- `labels(name, kind)` records whether a label is an `event` or a `scene`.

**Considered and not chosen**

- A text description from an LLM ("sounds like a party: music, many voices, laughter"). It can work from the class timeline alone, so no audio leaves the machine, but the rule table and taught scenes cover the need. It stays an option next to Phase 5.
- Zero-shot audio–text models (CLAP), where a scene is described in words. Most general, but the model is about ten times YAMNet's size: too heavy for a small always-on box and the single binary.

### 3.5 Storage and privacy

- `events(id, ts, duration_ms, peak_db, ambient_db, label, status, similarity, nearest_label, top_classes, clip_path, embedding, parts, conflict)`; the sound mix and the scene (§3.4) add `mix` and `scene`.
- `exemplars(id, event_id, label, embedding, created_ts)`; the clip is the event's.
- `classifications`: everything the model and the classifier worked out for each clip, one row per clip, including clips grouped into an earlier event and clips dropped as silence. The class scores (all 521), the embedding, the top classes of every patch, the onset, sound end and analysis window, the decision with each label's best similarity and the nearest exemplar's event, the trigger and classify settings in effect, and the model's name. It is the long-term record: it never expires, so thresholds can be replayed and scenes and sound mixes (§3.4) computed for the past after the clips are gone. 10–16 KB per clip: 6 KB of embedding and scores, and the per-patch classes, which grow with the clip's length.
- `labels(name, kind)` arrives with scenes; until then labels are derived from exemplars.
- Clips will contain conversations. They never leave the machine.
- Clips that are not exemplars are deleted after `retention_days` (default 14), and the oldest go first once all clips take more than `max_clip_mb` (default 2000). Both are checked every 10 minutes. Exemplar clips count towards the cap but are kept, so they can be re-embedded if the model changes.

### 3.6 MQTT

| Topic | Payload |
|---|---|
| `creak/status` | `online` / `offline` (last will) |
| `creak/event` | `{id, ts, label, status, similarity, duration_ms, peak_db, ambient_db, clip_url}` |
| `creak/scene/<label>` | `ON` / `OFF`, retained |
| `creak/ambient` | ambient level in dBFS, once a minute |
| `creak/label/set` | inbound `{id, label}`, so an HA notification action can confirm a tag |
| `creak/mode` | `active` / `deaf`, retained |
| `creak/mode/set` | inbound `active` / `deaf`, so a presence automation can switch it |

HA discovery creates one `event` entity (event types are your labels), one `binary_sensor` per scene and one `sensor` for the ambient level.

### 3.7 Web UI

Turned on or off with `[web] enabled`. With it off, the daemon, the CLI and MQTT work unchanged; you label by event id and fetch clips from the path `creak events` prints.

| View | What it does |
|---|---|
| **Review** | The pending queue, newest first. Each clip has a play button, the proposed label as a question ("Door?"), the model's top classes, and confirm / correct / ignore. Keyboard shortcuts for all four, so a queue clears quickly. |
| **Events** | The event log, filtered by label and status. Every row plays its clip, shows the nearest exemplar and similarity, and can be relabelled. |
| **Labels** | Exemplar count per label, and how often its matches were corrected. Rename, delete, switch between `event` and `scene`. |
| **Live** | Current level, measured ambient and the trigger lines, for checking the microphone. Sliders for `ratio`, `close_ratio` and `ambient_floor_db`, the three settings worth changing while watching the meter. |
| **Settings** | Every setting under `[trigger]` and `[classify]`, and `retention_days` and `max_clip_mb`, with its range and what it does. A change applies while the daemon runs and is stored in the database as an override of the config file until reset. The ranges live in `config.rs` and apply to the config file too. Device, listen address, token and data directory need a restart, so they stay in the file. |

The header carries the active/deaf switch (§3.1).

- One static HTML file with plain JavaScript, embedded in the binary. No build step, no framework.
- It talks to a small JSON API on the same port; clips are served as WAV with range requests, so every browser's own audio player works.
- The built-in default binds to `127.0.0.1`. `creak setup` writes a config that binds to all interfaces with a random token, because the usual install is a headless box reviewed from another device. Keep it on a trusted network or a VPN; never expose it to the internet.

### 3.8 Measuring and protecting accuracy

One confirmation is enough only if confirmations are right. Phase 0 showed they often are not: proposals anchor the answer, and clips that mix a foreground and a background sound get the background's label. So accuracy is measured on labels the system did not suggest, and every change to matching is judged by replaying recorded data, never by trying it live.

**Measuring**

- **Audit labels.** An Audit mode in the web UI plays a random clip without its label, proposal or model guess, and you say what it is. Audit labels are stored apart from exemplars and never become one, so the system is never graded on its own suggestions. 50–100 audited clips across the common sounds is the first test set.
- **Replay: `creak eval`.** Replays the audited clips in time order from an empty exemplar set, simulating the loop: where the policy would ask, it learns the audited label; where it would tag automatically, the tag is scored against it. It reports questions asked, wrong automatic tags per label, the confusion between labels and a sweep over `match_threshold`. Policies are compared on the same recordings.
- **In use.** A sample of automatic matches (every match below 0.9, one in ten above) goes to a quick "was this right?" check. With the corrected-after-match rate already in the log, that gives a running precision per label.

**Protecting the exemplars**

- Review shows the model's view next to the proposal and flags a conflict ("Music? · the model hears Cough"), with the model's class as its own one-key answer. In place since Phase 0, together with the conflict rule in §3.2.
- **Skip** answers a clip without creating an exemplar, for mixed or unclear clips. Today every answer creates one.
- Correcting an automatic match shows the exemplar that caused it, with relabel or remove, so the cause is fixed rather than outvoted.

**Candidate matching changes**, each kept only if the replay shows fewer wrong tags without many more questions:

- A margin: tag automatically only when the best label beats the best other label by a margin.
- Centred embeddings: subtract the mean embedding of recent clips before comparing. YAMNet embeddings share a large common component, so unrelated sounds sit near 0.45 and the useful range is narrow.
- Clean exemplars: a clip whose patches disagree is a poor exemplar; ask, or keep only the matching part.
- Splitting a clip at a sharp new onset (a cough over typing), with the split clip embedded from around its onset, and grouping stitching continuous sounds back together.
- Proposing from the foreground: the classes of the sound mix that are not already in the pre-roll (§3.4).

Already measured on Phase 0 data and rejected: embedding only the loudest patches (29 of 38 nearest labels right, against 35 of 38 for the plain mean).

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
tail_ms = 2000
min_clip_s = 4
max_clip_s = 30
cooldown_s = 2
pre_roll_ms = 1500
ambient_floor_db = -60

[classify]
match_threshold = 0.85
suggest_threshold = 0.60
sound_score = 0.1
conflict_score = 0.3
group_gap_s = 10
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

The two thresholds are starting values. On Phase 0 data, raising `match_threshold` from 0.80 to 0.85 cut wrong automatic tags from 3 of 32 to 2 of 29 for 3 more questions; the wrong matches left sit above 0.92 and come from mislabelled exemplars, which §3.8 addresses. The replay in Phase 1 sets the final values.

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
| 1 | **Accuracy** | §3.8: Audit mode, `creak eval`, the match check sample; conflict display, Skip and correct-to-cause in Review; the sound mix of a clip and its foreground (§3.4, layer 1); candidate matching changes tried one at a time through the replay. | On at least 50 audited clips, the replay reports the wrong-tag rate and question count for the current policy, and every change kept lowers wrong tags without raising questions by more than a quarter. A cough over background music is tagged as a cough. |
| 2 | **Complete the core** | `calibrate`, `review`, `labels`, `doctor`. Labels view (rename, delete, corrected-match rate). 24 h soak. | No event storm and flat memory over 24 h. A wrong match corrected in the event log does not repeat. With `[web] enabled = false` the daemon still runs and `creak label` still works. |
| 3 | **MQTT and Home Assistant** | Publishing, discovery, `label/set`, `mode/set`. | Entities appear in HA without YAML. An automation fires on a matched event. An HA notification action confirms a tag. |
| 4 | **Scenes** | §3.4 layers 2 and 3: `labels` table, the scene sampler, the rule table, scene cards in Review, `teach`, scene state, the `scene` column on events. | A party is proposed by the rule table the first time it happens. A movie is taught once; the next evening the scene turns on within 5 minutes and off within 5 minutes of the end. |
| 5 | **LLM first guess, optional** | For clips with no close exemplar, ask an OpenAI-compatible endpoint with audio input for the proposed label, in place of the AudioSet class. Off by default. | Only build this if the model's own first guesses are too generic to be useful after Phase 1. |
| 6 | **macOS and Windows** | Release builds for both. `setup` / `uninstall` install a launchd agent on macOS and a Windows service; platform default paths for config and data. Install via a script or a package manager. | On each OS: install, the Live view moves, a clap gives one clip, uninstall leaves nothing behind except data. |

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
- No MQTT broker or Home Assistant is needed until Phase 3.

## 10. Risks and open questions

1. **Microphones with built-in processing.** Array microphones that apply voice-tuned noise suppression before the OS sees the signal may flatten the sounds this tool is for. Phase 0's first clips show it; a plain USB microphone is the fallback.
2. **Doors and windows in other rooms.** One microphone hears a distant window poorly, and a contact sensor is more reliable for those. Expect the best results on sounds with no sensor: doorbell, appliances, alarms, party, movie.
3. **Short sounds inside longer clips.** A cough over typing or music is recorded but takes the clip's dominant label. Pooling only the loudest patches made matching worse (§3.8); the foreground of the sound mix (§3.4) and splitting at onsets are the next candidates.
4. **Rust.** Chosen for the single binary. The Python alternative would be a `uv tool install` package, with the same design and a heavier install.
5. **Why not an existing tool.** bark-detector, ha-audio-events, sound-sensor-homeassistant, AudioClassifier-MQTT, Yamcam, Heyra and Frigate all classify into YAMNet's fixed labels. None learns your own sounds or has a confirm loop, and none uses an ambient-relative trigger. bark-detector's clip log and MQTT discovery are worth reading before Phases 2 and 3.

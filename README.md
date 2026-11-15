# creak

*It heard that.*

creak listens to one microphone, cuts a clip whenever something is louder than the room usually is, and tags it: door, doorbell, kettle, whatever you call it. A pretrained audio model makes the first guess and you confirm or correct it. Every answer teaches it: once a sound has been confirmed, clips that sound like it are tagged without asking, unless the model hears something that contradicts the label, in which case it asks again.

It is a single binary with a small web UI. Clips stay on the machine.

**Status:** early. Linux only for now; macOS and Windows are planned. Home Assistant / MQTT publishing is not built yet. See [plan.md](plan.md) for the design and the roadmap.

## Install

On a Linux machine (x86_64 or aarch64, Debian 12+ / Ubuntu 22.04+, systemd) with a microphone:

```sh
curl -fsSL https://raw.githubusercontent.com/Thulana/creak/main/install.sh | sh
```

Or read the script first, and pin a version:

```sh
curl -fsSL https://raw.githubusercontent.com/Thulana/creak/main/install.sh -o install.sh
less install.sh
CREAK_VERSION=v0.1.0 sh install.sh
```

The installer downloads the release for your architecture, checks its SHA-256, installs `/usr/local/bin/creak`, installs the ALSA library if it is missing, and runs `creak setup`. Setup creates a `creak` system user in the `audio` group, `/etc/creak/config.toml` (only if it does not exist), `/var/lib/creak`, and a systemd service, then prints the web UI address:

```
web UI: http://192.168.1.20:8377/?token=3f9c…
```

Open that link from any device on your network. The token is stored in `/etc/creak/config.toml`.

**Upgrade:** run the installer again. Config and data are kept.
**Uninstall:** `sudo creak uninstall` keeps config and data; `sudo creak uninstall --purge` removes them too.

## Using it

- **Live** shows the microphone level, the ambient level and the line a sound has to cross. Check it first: if the level does not move when you talk, the wrong device is selected (`creak devices`, then `[audio] device` in the config). The sliders below the graph move the open line, the close line and the ambient floor while you watch; they are saved and override the config file until you press Reset.
- **Review** lists clips waiting for an answer, each with a proposed label as a question ("Door?"). Play it, then confirm (`Y`), type the right label and correct (`C`), skip (`S`) or ignore (`I`). Skip answers without teaching: use it for a clip with several sounds in it, or one you cannot make out. When the model disagrees with the proposal the card says so ("Music? · the model hears Cough") and `M` answers with the model's sound. Space plays, `J`/`K` move.

  A clip is asked about when no taught clip is close enough to tag it, or when one is close but the model is sure it hears a different sound than that label's taught clips. Everything else is tagged without asking.

  Each card shows the sound mix: how much of the clip each sound fills. Sounds that were already there before the clip started (dashed) are background. When something new comes in over them, that is what the clip is proposed as: a cough over music reads "Cough? · came in over Music", and `M` answers with the label of the nearest taught clip instead.

  A few clips that were tagged without asking come back as "Door, right?": every one below 0.9 similarity and one in ten above. Answer right (`Y`), correct it, or can't tell (`S`). The answers give a running precision per label, shown in the Audit view.
- **Events** is the full log, with the nearest learned sound and its similarity for every clip. A wrong match can be relabelled here. Correcting a clip that was tagged without asking shows the taught clip that caused it, so you can relabel or remove that one too; otherwise the same mistake comes back.
- **Audit** plays random clips without saying what creak thinks they are, and asks you what each one is. These answers never teach creak. They are the test set for `creak eval`, which scores creak on labels it did not suggest: 50–100 audited clips across your common sounds make a first one. Audited clips are kept like taught ones.
- **Labels** lists everything creak has been taught: how many taught clips each label has, how many clips were tagged with it without asking, and how many of those you corrected later. Rename a label here; renaming it to another label's name merges the two. Deleting a label stops its taught clips being examples, so nothing is tagged with it any more; past events keep it.
- **Settings** lists every threshold with what it does. A change applies at once, without a restart, and overrides the config file until you reset that setting (↺) or all of them.
- **Active / deaf** is the switch at the top right. Deaf records nothing: no clips, no tags, no events, though the Live meter still moves. The mode survives a restart. To switch it from a script or Home Assistant, for example when everyone has left the house:

  ```sh
  curl -X POST -H "Authorization: Bearer <token>" -d '{"mode": "active"}' http://<host>:8377/api/mode
  ```

  ```yaml
  # Home Assistant: call rest_command.creak_mode with mode: active or deaf
  rest_command:
    creak_mode:
      url: http://<host>:8377/api/mode
      method: post
      headers:
        Authorization: Bearer <token>
      payload: '{"mode": "{{ mode }}"}'
  ```

One answer per sound is enough. A clip you confirm becomes an example; the next clip whose similarity to it is above `match_threshold` gets its label automatically. `ignore` works the same way, for sounds you never want to hear about again.

Label the sound that triggered the clip, not what was playing in the background: a cough confirmed as "music" teaches creak that coughs are music. Clips in which the model hears nothing but silence or room tone are dropped, so there is no `silence` label.

Clips are deleted after a while, but what the model made of each one is kept for good in the `classifications` table of `creak.db`: all its class scores, its embedding, the top classes of every 0.96 s of it, its sound mix and the sounds heard before it, the decision and the settings it was made with. That includes clips grouped into an earlier event and clips dropped as silence. It takes 10–16 KB per clip, more for longer clips.

## Measuring accuracy

`creak eval` replays your audited clips in the order they were recorded, starting with nothing taught: where creak would ask, it learns your audit answer; where it would tag without asking, the tag is scored against it. It prints how often it asked, the wrong tags per label and which labels get mixed up, how both change with `match_threshold`, and the same numbers for a few candidate changes to matching, each marked keep or not better. It also lists taught clips whose audit says something else, and the precision of automatic tags from the checks in Review.

```sh
sudo -u creak creak eval
```

## Configuration

`/etc/creak/config.toml`. Every key is optional; these are the defaults:

```toml
[audio]
device = "default"        # ALSA's default, else the first card that can record; or an id from `creak devices`

[trigger]
ratio = 3.0               # open when louder than ambient × ratio (about +9.5 dB)
close_ratio = 1.5         # the sound is over when quieter than ambient × close_ratio …
hang_ms = 800             # … for this long
tail_ms = 2000            # keep recording this long after the sound; a sound in the tail joins the clip
min_ms = 60               # ignore blips shorter than this
min_clip_s = 15           # every clip runs at least this long from its onset; a sound in that time joins it
max_clip_s = 60
cooldown_s = 2
pre_roll_ms = 1500        # audio kept from before the onset
ambient_floor_db = -60    # ambient is the 10th percentile of the last 5 minutes, never below this

[classify]
match_threshold = 0.85    # cosine similarity to a confirmed clip that applies its label; higher asks more often
suggest_threshold = 0.60  # above this, the nearest label is proposed instead of the model's guess
sound_score = 0.1         # the weakest model score that counts as a sound; a clip with none is dropped as silence
conflict_score = 0.3      # how sure the model must be of another sound to send a match to review
group_gap_s = 10          # a same-label clip within this many seconds extends the previous event; 0 = off

[storage]
dir = "/var/lib/creak"
retention_days = 14       # clips that are not examples are deleted after this
max_mb = 2000             # all clips together; beyond this the oldest that are not examples go first

[web]
enabled = true
listen = "127.0.0.1:8377" # setup writes 0.0.0.0:8377 with a random token
token = ""
```

Everything in `[trigger]` and `[classify]`, and the two `[storage]` limits, can also be changed on the Settings page, which applies the change at once and overrides this file until reset. A change to the file itself needs a restart: `sudo systemctl restart creak`. Logs: `journalctl -u creak -f`.

The web UI is meant for a home network or a VPN. Do not expose it to the internet.

## CLI

```
creak devices                       list capture devices
creak run [--config FILE]           run the daemon in the foreground
creak events [--since 1d] [--label door] [--status pending] [--json]
creak review                        what waits for an answer, and the web UI address
creak label <event-id> <label>      confirm or correct; `ignore` to stop keeping clips like it
creak skip <event-id>               answer without teaching; also removes a taught clip
creak labels                        taught labels, with how often their automatic tags were corrected
creak labels rename <label> <new>   rename everywhere; another label's name merges the two
creak labels delete <label>         stop tagging with it; past events keep it
creak eval                          score tagging against your audited clips
creak calibrate [--device ID]       live meter in the terminal: level, ambient, the line to cross
creak doctor                        check config, disk, model, service, web UI and microphone
sudo creak setup                    install or upgrade the service
sudo creak uninstall [--purge]
```

The database belongs to the `creak` user, so run everything but `devices`, `setup` and `uninstall` as `sudo -u creak creak …`.

Without the web UI (`[web] enabled = false`), `creak review` lists the clips waiting with their file paths; confirm one with `creak label <id>` and the same label. `creak calibrate` opens the microphone itself, so stop the service first if the device cannot be shared. When something seems wrong, start with `sudo -u creak creak doctor`.

## Running in a container

The microphone belongs to the host. Pass the card's control node and its capture nodes from `/dev/snd` into the container (for card 1, device 0: `controlC1` and `pcmC1D0c`) and make sure the container's `audio` group can open them. ALSA's `default` device also needs `/dev/snd/timer`; without it, creak opens the first card that can record directly and logs which one it chose. In an unprivileged LXC container, enable nesting, or the hardened systemd unit fails to start with `226/NAMESPACE`.

## Building

```sh
sudo apt-get install libasound2-dev   # Linux only
cargo build --release
cargo test --release
```

The YAMNet model is converted once from the official weights by [model/convert.py](model/convert.py) and embedded in the binary; see [model/README.md](model/README.md).

## Licence

MIT, see [LICENSE](LICENSE). The embedded YAMNet weights are Apache-2.0 and the AudioSet class names CC BY-SA 4.0; see [NOTICE](NOTICE).

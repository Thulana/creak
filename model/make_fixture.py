# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "tensorflow", "tf-keras", "h5py"]
# ///
"""Generate tests/fixtures/synth.wav and synth_reference.json with TF YAMNet's own pipeline.

Usage: uv run model/make_fixture.py [cache_dir]
Needs the reference TF code (yamnet.py, params.py, features.py from tensorflow/models) and
yamnet.h5; both are downloaded into cache_dir.
"""
import json
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
fixtures = here.parent / "tests" / "fixtures"
cache = Path(sys.argv[1]) if len(sys.argv) > 1 else here / ".cache"
src = cache / "yamnet_src"
src.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/"
for f in ("yamnet.py", "params.py", "features.py"):
    if not (src / f).exists():
        urllib.request.urlretrieve(base + f, src / f)
h5 = cache / "yamnet.h5"
if not h5.exists():
    urllib.request.urlretrieve("https://storage.googleapis.com/audioset/yamnet.h5", h5)

SR = 16000
rng = np.random.default_rng(1234)
t = np.arange(int(2.5 * SR)) / SR
f0, f1, dur = 200.0, 6000.0, t[-1]
chirp = 0.3 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur)))
noise = 0.05 * rng.standard_normal(t.size)
clicks = np.zeros_like(t)
for c in (0.3, 0.9, 1.0, 1.7):
    i = int(c * SR)
    clicks[i:i + 3] = [0.5, -0.4, 0.2]
x = chirp + noise + clicks
assert np.abs(x).max() < 0.9
pcm = np.round(x * 32767).astype("<i2")
fixtures.mkdir(parents=True, exist_ok=True)
with wave.open(str(fixtures / "synth.wav"), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())

sys.path.insert(0, str(src))
import tensorflow as tf  # noqa: E402
import params as params_lib  # noqa: E402
import yamnet as yamnet_lib  # noqa: E402

with wave.open(str(fixtures / "synth.wav"), "rb") as w:
    samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
waveform = (samples / 32768.0).astype(np.float32)

model = yamnet_lib.yamnet_frames_model(params_lib.Params())
model.load_weights(str(h5))
scores, embeddings, log_mel = (a.numpy() for a in model(waveform))


def r(a):
    return [[float(f"{v:.6g}") for v in row] for row in a]


with open(fixtures / "synth_reference.json", "w") as f:
    json.dump({"log_mel": r(log_mel), "scores": r(scores), "embeddings": r(embeddings)},
              f, separators=(",", ":"))
print("log_mel", log_mel.shape, "scores", scores.shape, "embeddings", embeddings.shape)

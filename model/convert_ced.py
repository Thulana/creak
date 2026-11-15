# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "onnx", "torch", "torchaudio", "transformers>=4.40,<5"]
# ///
"""Export CED-tiny to ced_tiny.onnx, and write its class map and the Rust test's reference.

Usage: uv run model/convert_ced.py [cache_dir]
Writes ced_tiny.onnx and ced_tiny_class_map.csv next to this script, and
tests/fixtures/synth_ced_reference.json.
Input  patches    f32 [1, 96, 64]   (one log-mel patch: frames x mel bands)
Output scores     f32 [1, 527]      (sigmoid)
       embeddings f32 [1, 192]      (mean of the transformer's output tokens)

The weights and the model code come from the Hugging Face release, pinned to one commit.
The reference runs the release's own feature extractor and the PyTorch model on
tests/fixtures/synth.wav, zero-padded the way embed.rs pads a clip.
"""
import csv
import json
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

REPO = "mispeech/ced-tiny"
REVISION = "ace276d29dd0bb3f3517b0fa8cf300738c409019"
LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
FRAMES, BANDS = 96, 64
PATCH_HOP_FRAMES = 48
MIN_SAMPLES, PATCH_HOP_SAMPLES = 15600, 7680

here = Path(__file__).resolve().parent
cache = Path(sys.argv[1]) if len(sys.argv) > 1 else here / ".cache"
cache.mkdir(parents=True, exist_ok=True)


class Patch(torch.nn.Module):
    """One log-mel patch in, class scores and the pooled embedding out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, patches):
        tokens = self.model.encoder(patches.transpose(1, 2)).logits
        embeddings = tokens.mean(1)
        return self.model.outputlayer(embeddings).sigmoid(), embeddings


def pad(x):
    """embed::pad: at least one patch, then a whole number of patch hops."""
    n = max(len(x), MIN_SAMPLES)
    hops = -(-(n - MIN_SAMPLES) // PATCH_HOP_SAMPLES)
    return np.pad(x, (0, MIN_SAMPLES + hops * PATCH_HOP_SAMPLES - len(x)))


def sig(rows):
    return [[float(f"{v:.6g}") for v in row] for row in rows]


model = AutoModelForAudioClassification.from_pretrained(
    REPO, revision=REVISION, trust_remote_code=True, cache_dir=cache).eval()
extractor = AutoFeatureExtractor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True, cache_dir=cache)
patch = Patch(model).eval()

torch.onnx.export(
    patch, (torch.zeros(1, FRAMES, BANDS),), here / "ced_tiny.onnx", input_names=["patches"],
    output_names=["scores", "embeddings"], opset_version=17, dynamo=False)

labels = cache / "class_labels_indices.csv"
if not labels.exists():
    urllib.request.urlretrieve(LABELS_URL, labels)
with open(labels, newline="") as f:
    rows = list(csv.reader(f))
names = [model.config.id2label[i] for i in range(model.config.outputdim)]
assert [r[2] for r in rows[1:]] == names, "CED's classes are not in AudioSet's order"
with open(here / "ced_tiny_class_map.csv", "w", newline="") as f:
    csv.writer(f, lineterminator="\n").writerows(rows)

with wave.open(str(here.parent / "tests/fixtures/synth.wav")) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
features = extractor(torch.from_numpy(pad(x)), sampling_rate=16000)["input_values"][0]
log_mel = features.T.contiguous()
scores, embeddings = [], []
with torch.no_grad():
    for start in range(0, log_mel.shape[0] - FRAMES + 1, PATCH_HOP_FRAMES):
        s, e = patch(log_mel[start:start + FRAMES][None])
        scores.append(s[0].numpy())
        embeddings.append(e[0].numpy())
reference = {"log_mel": sig(log_mel.numpy()), "scores": sig(scores), "embeddings": sig(embeddings)}
(here.parent / "tests/fixtures/synth_ced_reference.json").write_text(json.dumps(reference))
print(f"{len(scores)} patches, {len(names)} classes, embedding {embeddings[0].shape[0]}")

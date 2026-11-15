# Model assets

Two models are embedded; `[classify] model` picks one. Both take one 0.96 s log-mel patch, `patches` f32 [1,96,64] (frames x bands, 10 ms hop), and return sigmoid class scores and an embedding, so everything after the frontend is shared.

## YAMNet

- `yamnet.onnx`: YAMNet (MobileNetV1, AudioSet) converted from the official Keras weights `https://storage.googleapis.com/audioset/yamnet.h5`. BatchNorm is folded into the convs. Input `patches` f32 [1,96,64] (log-mel frames x bands), outputs `scores` [1,521] (sigmoid) and `embeddings` [1,1024]. Opset 13; checked against TF YAMNet (max abs diff ~1e-5) and tract.
- `yamnet_class_map.csv`: 521 class names, from `tensorflow/models/research/audioset/yamnet/yamnet_class_map.csv`.
- Licences: the `tensorflow/models` repo (model code, weights, class map) is Apache-2.0 (its root `LICENSE`, "Copyright 2022 Google LLC", covers `/research`). The AudioSet ontology, the source of the class names, is CC BY-SA 4.0 (github.com/audioset/ontology); attribute Google and keep that in mind for redistribution of the class map.
- Regenerate: `uv run model/convert.py [cache_dir]` (writes `yamnet.onnx` and the class map); `uv run model/make_fixture.py [cache_dir]` (writes `tests/fixtures/synth.wav` and `synth_reference.json`, via TF YAMNet's own pipeline; needs TensorFlow).
- `synth_reference.json` holds `log_mel` [288][64], `scores` [5][521], `embeddings` [5][1024] (6 significant digits) for `synth.wav` decoded as int16 / 32768.

### Frontend the Rust side must reproduce (features.py, params.py)

- Waveform: mono, 16000 Hz, float in [-1, 1]. Zero-pad at the end (pad_waveform): min = int((0.96 + 0.025 - 0.010) * 16000) = 15600 samples; beyond that round the excess over 15600 up to a whole multiple of the 7680-sample patch hop. Total = 15600 + 7680 * ceil((n - 15600) / 7680).
- STFT: window 400 samples (0.025 s), hop 160 (0.010 s), FFT length 512 (next power of two >= 400, window zero-padded), periodic Hann `0.5 - 0.5 cos(2 pi n / 400)`, no centring/padding (frames = (len - 400) / 160 + 1, only full frames), 257 bins. Magnitude `|X|`, not power.
- Mel: `tf.signal.linear_to_mel_weight_matrix`, 64 bands, 125 to 7500 Hz, 257 bins, mel(f) = 1127 ln(1 + f / 700). Bin frequencies are `linspace(0, 8000, 257)`; the DC bin (index 0) is excluded: its weights are all zero. Band edges are `linspace(mel(125), mel(7500), 66)`, triangles evaluated in the mel domain: weight = max(0, min((m - lo) / (c - lo), (hi - m) / (hi - c))), no area normalisation.
- `log_mel = ln(mel_spectrogram + 0.001)` (natural log).
- Patches: 96 frames (0.96 s) with hop 48 frames (0.48 s), shape [96, 64], only complete patches. Model input is one patch at a time.

## CED-tiny

- `ced_tiny.onnx`: CED-tiny (a ViT distilled from an ensemble of larger models, 5.5 M parameters, 48.1 mAP on AudioSet) exported from the Hugging Face release `mispeech/ced-tiny`, pinned to one commit. Input `patches` f32 [1,96,64], outputs `scores` [1,527] (sigmoid) and `embeddings` [1,192], the mean of the transformer's output tokens, which is what the classifier head pools. Opset 17, checked against PyTorch and tract.
- `ced_tiny_class_map.csv`: AudioSet's 527 classes (`class_labels_indices.csv`), which is the order of CED's outputs; the script checks that. YAMNet's 521 are the same names less six: Male speech, Female speech, Male singing, Female singing, Funny music and Battle cry.
- Licence: the Hugging Face release (weights and model code) is Apache-2.0. The training code at github.com/RicherMans/CED is GPL-3.0 and is not used.
- Regenerate: `uv run model/convert_ced.py [cache_dir]` writes `ced_tiny.onnx`, the class map and `tests/fixtures/synth_ced_reference.json`. The reference holds `log_mel` [290][64] from the release's own feature extractor, and `scores` [5][527] and `embeddings` [5][192] from the PyTorch model, for `synth.wav` padded as `embed::pad` pads it. It runs the release's model code (`trust_remote_code`), pinned to the same commit.

### Frontend the Rust side must reproduce (torchaudio MelSpectrogram + AmplitudeToDB, feature_extraction_ced.py)

- Waveform: mono, 16000 Hz, float in [-1, 1], padded as for YAMNet so both models get the same patches.
- STFT: centred: the waveform is reflect-padded by 256 samples at each end (the edge sample is not repeated). Window 512 samples, periodic Hann `0.5 - 0.5 cos(2 pi n / 512)`, hop 160, FFT length 512, 257 bins; frames = 1 + len / 160. Power `|X|^2`.
- Mel: `torchaudio.functional.melscale_fbanks`, 64 bands, 0 to 8000 Hz, HTK scale, no normalisation. Band edges are evenly spaced in mel and converted back to Hz; the triangles are linear in Hz: weight = max(0, min((f - lo) / (c - lo), (hi - f) / (hi - c))).
- `log_mel = 10 log10(max(mel, 1e-10))`, then raised to at least 120 dB below the loudest value of the whole spectrogram.
- The network's first layer is a batch norm over the 64 bands, so no further normalisation happens outside it.

# AnimatAI

**Voice in, facial animation out.** AnimatAI turns speech into 52 ARKit blendshapes and 3 head rotations at 30 fps. It is built to plug into any voice AI (streaming TTS, assistant, VTuber) and into robot faces (mapping to servos), and to be fine-tuned on one person's expressions with a few minutes of video.

```
audio 16 kHz ─► frozen front-end ─► causal TCN + FiLM(style) ─► 55 channels ─► post-process ─► avatar / OSC / servos
                (mHuBERT-147                (~100 ms of                          (blinks, emotion,
                 or log-mel)                 lookahead)                           smoothing, robot rig)
```

This is version 2 (A2B), a full rewrite. Version 1 (T2B, MFCC + LSTM, January 2026) is kept under the [`v1-t2b`](../../tree/v1-t2b) tag and in [`legacy/`](legacy).

## Why the rewrite

| Problem in v1 | Answer in v2 |
|---|---|
| MFCC + LSTM trained on 13 min of a single speaker: did not generalize | Frozen multilingual speech encoder (mHuBERT-147); the head only learns "phonetics → face" |
| No validation, no metric | Train / val / **unseen-speaker test** split, lip-sync metrics (`mouth_corr`, `jaw_corr`, MAE per group), early stopping, EMA |
| Noisy data (voice-over, cutaways, 15 fps, tracker jitter) | 30 fps, Savitzky-Golay denoising, then **cleaning by the model itself** (audio/video offset, voice-over rejection) |
| Head pose = face position in the frame | Real yaw / pitch / roll from the MediaPipe matrix |
| A 1.2 GB emotion model called on every request | Emotion is an **explicit control layer** set by the agent (`happy:0.6`), at no cost |
| Not real-time | **Causal** blocks: the same model runs on files and in streaming (tested: streaming = offline) |
| No embedded deployment | `lite` variant (log-mel in numpy) + **ONNX export**: runs without PyTorch |
| No fine-tuning | FiLM **styles**: a new "mimic" is a new style vector (+ adapters) |

## Install

```bash
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e ".[all]"        # or only [train], [mining], [export], [serve]
pytest                          # 17 tests, no GPU or data needed (including the full pipeline on synthetic data)
```

## Pipeline

```bash
# 1. Data: face-to-camera videos → clips (audio + blendshapes)
animatai mine --add https://youtu.be/XXXX --workers 4
animatai extract my_video.mp4 --name me

# 2. Cleaning (30 fps, tracker denoising, speech filter) + split
animatai prepare

# 3. Training, then cleaning by the model and final training
animatai train --config configs/base.yaml --run runs/base_pass1
animatai prepare --model runs/base_pass1/best.pt
animatai train --config configs/base.yaml --run runs/base     # quality (mHuBERT)
animatai train --config configs/lite.yaml --run runs/lite     # embedded (log-mel)

# 4. Evaluation, offline and through the real-time engine
animatai evaluate --model runs/base/best.pt --streaming

# 5. Use
animatai infer  --model runs/base/best.pt --audio voice.wav --out anim.json --emotion happy:0.6
animatai infer  --model runs/base/best.pt --audio voice.wav --out servos.csv --rig examples/rigs/simple_robot_head.yaml
animatai serve  --model runs/base/best.pt                      # HTTP API: POST /v1/animate
animatai live   --model runs/base/best.pt --osc 127.0.0.1:9000 # microphone → OSC (VRChat, Unity, Unreal…)
animatai export --model runs/lite/best.pt --out models/lite_onnx
```

Any setting can be overridden on the command line: `animatai train --set train.epochs=60 model.d_model=384`.

## Fine-tune on a mimic

Film the person speaking to the camera, 2 to 10 minutes, in good light:

```bash
animatai extract matteo.mp4 --name matteo --out data/mimics
animatai finetune --model runs/base/best.pt --data data/mimics/matteo --name matteo --out models/matteo.pt
animatai infer --model models/matteo.pt --style matteo --audio voice.wav --out anim.json
```

| `--mode` | What is learned | Data |
|---|---|---|
| `style` | Only the style vector (32 values) | ≥ 30 s |
| `adapter` (default) | Style + FiLM generators + output head, anchored to the base weights | ≥ 2 min |
| `full` | All weights, low learning rate | ≥ 10 min |

Styles can be blended at inference: `animator.set_style({"generic": 0.4, "matteo": 0.6})`.

## Streaming in a voice AI

```python
from animatai.runtime.engine import Animator

animator = Animator("runs/base/best.pt", style="generic", emotion="happy:0.5")
for chunk in tts_stream:                      # audio chunks of any size
    for frame in animator.stream(chunk, sr=24000):
        send_to_face(frame)                   # 55 floats, in animator.channels order
animator.stream([], flush=True); animator.reset()
```

Algorithmic latency is `animator.latency_ms` (about 100–180 ms) plus your chunk size. The agent can change emotion at any time: `animator.set_emotion("surprised:0.4")`.

## Robot faces

A YAML *rig* describes each actuator: a mix of channels mapped to a mechanical range, with a **speed limit** to protect the servos. See [examples/rigs/simple_robot_head.yaml](examples/rigs/simple_robot_head.yaml). On a Raspberry Pi, export the `lite` model to ONNX: the runtime only needs `numpy` and `onnxruntime`.

## Results so far

**Provisional**: only the first training pass of the `base` model has run (stopped at epoch 11, best checkpoint at epoch 5). Cleaning by the model, the final `base` / `lite` training and the streaming evaluation of the final model are still to do.

Data: 10 face-to-camera videos (French and English), 59.6 min train, 5.7 min validation, 15.6 min test (speakers never seen in training).

| Correlation between prediction and tracker (1.0 = perfect) | Value |
|---|---|
| Mouth, validation | 0.53 |
| Jaw (`jawOpen`), validation | 0.68 |
| Mouth per clip, training speakers | 0.55 – 0.64 |
| Mouth per clip, **unseen speakers** | 0.37 – 0.40 |
| Articulation amplitude reproduced (before post-process gain) | 0.61 |
| `lite` log-mel trial (12 short epochs), validation | 0.45 |

The head converges in about 5 epochs (~30 s per epoch on an RTX 3050 Laptop GPU). The fixed cost is caching the mHuBERT features (~3 min per hour of audio, done once).

Honest reading: good lip-sync on known voices, weaker on an unknown voice. With only 8 training speakers, that is expected; more speakers is the main lever.

## Layout

```
src/animatai/
  schema.py              output contract (55 channels, groups, mirroring, v1 conversion)
  audio.py features.py   audio I/O, mel and SSL front-ends
  data/                  mining (MediaPipe), prepare (cleaning, split), dataset (feature cache)
  model.py losses.py     TCN-FiLM network, position + velocity loss, metrics
  train.py finetune.py evaluate.py export.py
  runtime/               engine (offline + streaming), postprocess, rig (robots), live (mic → OSC/UDP)
  serve.py cli.py
configs/  tests/  examples/  legacy/ (v1 code, kept as is)
```

## Known limits

- Targets come from MediaPipe: the model cannot be more precise than that tracker (the tongue, for example, is not captured).
- Eyes and gaze cannot be predicted from audio: blinks are procedural.
- Quality on an unknown speaker depends directly on the number of speakers in the training data.
- YouTube videos are used for local training only. Check the rights before redistributing data, or a model trained on it. For that reason, no v2 trained weights and no demo page with third-party audio are published here.

Full documentation in French: [README.fr.md](README.fr.md).

## License

[MIT](LICENSE) © starman-tech

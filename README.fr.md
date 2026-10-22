# AnimatAI — Audio → Visage (A2B)

Modèle qui transforme **de la voix en animation faciale** : 52 blendshapes ARKit + 3 rotations de tête, à 30 fps.
Conçu pour être branché sur **n'importe quelle IA vocale** (TTS en streaming, assistant, VTuber) et sur des
**visages robotiques** (retargeting vers servos), et pour être **ré-entraînable / fine-tunable facilement** sur les
mimiques d'une personne.

```
audio 16 kHz ─► front-end gelé ─► TCN causal + FiLM(style) ─► 55 canaux ─► post-process ─► avatar / OSC / servos
                (mHuBERT-147                (≈100 ms de                     (clignements, émotion,
                 ou log-mel)                 lookahead)                      lissage, gain, rig robot)
```

## Pourquoi cette architecture

| Problème de l'ancien projet | Réponse |
|---|---|
| MFCC + LSTM entraîné sur 13 min d'un seul locuteur → ne généralise pas | Encodeur de parole **pré-entraîné multilingue gelé** (mHuBERT-147) : la tête apprend seulement « phonétique → visage » |
| Aucune validation, aucune métrique | Split train / val / **test locuteur jamais vu**, métriques lip-sync (`mouth_corr`, `jaw_corr`, MAE par groupe), early stopping, EMA |
| Données bruitées (voix off, plans de coupe, 15 fps, jitter du tracker, « .wav » en WebM) | `prepare` : 30 fps, débruitage Savitzky-Golay, puis **nettoyage par le modèle** (décalage A/V + rejet des voix off), décodage ffmpeg |
| Pose de tête = position du visage dans l'image | Vraies rotations (yaw/pitch/roll) depuis la matrice MediaPipe |
| Modèle d'émotion de 1,2 Go appelé à chaque requête | Émotion = **couche de contrôle explicite** pilotée par l'agent (`happy:0.6`), sans coût |
| Pas de temps réel | Blocs **causaux** : le même modèle sert en fichier et en streaming (test : streaming ≡ offline) |
| Pas de déploiement embarqué | Variante `lite` (log-mel numpy) + **export ONNX** : tourne sans PyTorch |
| Fine-tuning impossible | **Styles** FiLM : un « mimic » = un nouveau vecteur de style (+ adaptateurs), quelques minutes de vidéo suffisent |

## Installation

```bash
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e ".[all]"        # ou seulement [train], [mining], [export], [serve]
pytest                          # 17 tests, sans GPU ni données (dont le pipeline complet sur données synthétiques)
```

## Pipeline complet

```bash
# 1. Données : vidéos face caméra → clips (audio + blendshapes)
animatai mine --add https://youtu.be/XXXX https://youtu.be/YYYY --workers 4
animatai extract ma_video.mp4 --name moi            # fichier local

# 2. Nettoyage (30 fps, débruitage du tracker, filtre parole) + split  → data/processed/report.json
animatai prepare

# 3. Entraînement, passe 1
animatai train --config configs/base.yaml --run runs/base_pass1

# 3bis. Nettoyage par le modèle : corrige les décalages audio/vidéo et rejette les plans où le visage
#       à l'écran n'est pas la voix (voix off, plans de coupe), puis ré-entraînement final
animatai prepare --model runs/base_pass1/best.pt
animatai train --config configs/base.yaml --run runs/base     # qualité (mHuBERT)
animatai train --config configs/lite.yaml --run runs/lite     # embarqué (log-mel)

# 4. Évaluation (offline et à travers le moteur temps réel)
animatai evaluate --model runs/base/best.pt
animatai evaluate --model runs/base/best.pt --streaming

# 5. Utilisation
animatai infer  --model runs/base/best.pt --audio voix.wav --out anim.json --emotion happy:0.6
animatai infer  --model runs/base/best.pt --audio voix.wav --out servos.csv --rig examples/rigs/simple_robot_head.yaml
animatai serve  --model runs/base/best.pt            # API HTTP  POST /v1/animate
animatai live   --model runs/base/best.pt --osc 127.0.0.1:9000     # micro → OSC (VRChat, Unity, Unreal…)
animatai export --model runs/lite/best.pt --out models/lite_onnx   # ONNX, sans torch
animatai demo   --model runs/base/best.pt --out demo.html          # page autonome : visage robot animé, modèle vs tracker
animatai demo   --model runs/base/best.pt --audio ma_voix.wav      # … sur tes propres fichiers audio
```

Tout paramètre se surcharge en ligne : `animatai train --set train.epochs=60 model.d_model=384`.

## Fine-tuner sur un « mimic »

Filme la personne (ou le personnage) qui parle face caméra, 2 à 10 minutes, bonne lumière :

```bash
animatai extract matteo.mp4 --name matteo --out data/mimics
animatai finetune --model runs/base/best.pt --data data/mimics/matteo --name matteo --out models/matteo.pt
animatai infer --model models/matteo.pt --style matteo --audio voix.wav --out anim.json
```

| `--mode` | Ce qui est appris | Données conseillées |
|---|---|---|
| `style` | uniquement le vecteur de style (32 valeurs) | ≥ 30 s |
| `adapter` (défaut) | style + générateurs FiLM + tête de sortie, ancrés aux poids de base | ≥ 2 min |
| `full` | tous les poids, LR faible | ≥ 10 min |

Les styles existants restent intacts ; on peut les **mélanger** à l'inférence :
`animator.set_style({"generic": 0.4, "matteo": 0.6})`.

## Intégration dans une IA vocale (streaming)

```python
from animatai.runtime.engine import Animator

animator = Animator("runs/base/best.pt", style="generic", emotion="happy:0.5")
for chunk in tts_stream:                      # morceaux audio de taille quelconque
    for frame in animator.stream(chunk, sr=24000):
        send_to_face(frame)                   # 55 floats, ordre = animator.channels
animator.stream([], flush=True); animator.reset()
```

Latence algorithmique : voir `animator.latency_ms` (≈ 100–180 ms) + la taille de tes chunks.
L'agent peut changer d'émotion à tout moment : `animator.set_emotion("surprised:0.4")`
(émotions : neutral, happy, sad, angry, surprised, fear, disgust, thinking — mélangeables).

## Visages robotiques

Un *rig* YAML décrit chaque actionneur : mélange de canaux → plage mécanique, avec **limite de vitesse** pour
protéger les servos. Voir [examples/rigs/simple_robot_head.yaml](examples/rigs/simple_robot_head.yaml).
Sur Raspberry Pi : exporte le modèle `lite` en ONNX ; le runtime n'a besoin que de `numpy` + `onnxruntime`.

## Résultats

État au 2026-09-20 — **provisoire** : seule la passe 1 du modèle `base` a tourné (interrompue à l'époque 11,
meilleur checkpoint à l'époque 5 : `runs/base_pass1/best.pt`). Le nettoyage par le modèle, l'entraînement final
`base`/`lite` et l'évaluation streaming du modèle final restent à faire.

Données : 10 vidéos face caméra (FR + EN), 59,6 min train / 5,7 min val / 15,6 min test (2 locuteurs jamais vus).

| Mesure (corrélation prédiction ↔ tracker, 1.0 = parfait) | Valeur |
|---|---|
| Bouche, locuteurs vus, validation | 0,53 |
| Mâchoire (`jawOpen`), validation | 0,68 |
| Bouche par clip, locuteurs d'entraînement | 0,55 – 0,64 |
| Bouche par clip, **locuteurs jamais vus** | 0,37 – 0,40 |
| Amplitude d'articulation reproduite (avant gain du post-process) | 0,61 |
| Essai `lite` log-mel (12 époques courtes), validation | 0,45 |

Coût : la tête converge en ~5 époques (≈ 30 s/époque sur RTX 3050 Laptop). Le coût fixe est la mise en cache
des features mHuBERT (~3 min pour 1 h d'audio, faite une seule fois). Le minage vidéo est la partie lente
(≈ temps réel par vidéo sur CPU).

Lecture honnête : bon lip-sync sur des voix connues, correct mais nettement plus faible sur une voix inconnue —
avec seulement 8 locuteurs d'entraînement, c'est attendu. Ajouter des locuteurs est le levier principal.

## Structure

```
src/animatai/
  schema.py            contrat de sortie (55 canaux, groupes, miroir, conversion legacy 58)
  audio.py features.py I/O audio, front-ends mel / SSL
  data/                mining (MediaPipe), prepare (nettoyage/split), dataset (cache de features)
  model.py losses.py   réseau TCN-FiLM, loss position+vitesse, métriques
  train.py finetune.py evaluate.py export.py
  runtime/             engine (offline + streaming), postprocess, rig (robots), live (micro → OSC/UDP)
  serve.py cli.py
configs/  tests/  examples/  legacy/ (ancien code, conservé tel quel)
```

## Limites connues (honnêtes)

- Les cibles viennent de MediaPipe : le modèle ne peut pas être plus précis que ce tracker (la langue, par
  exemple, n'est pas capturée).
- Yeux et regard ne sont pas prédictibles depuis l'audio : les clignements sont procéduraux.
- La qualité sur un locuteur inconnu dépend directement du **nombre de locuteurs** dans `data/raw`. Ajouter des
  vidéos variées (`animatai mine --add …`) est le levier n°1.
- Les vidéos YouTube servent uniquement à l'entraînement local ; vérifie les droits avant de redistribuer des
  données ou un modèle entraîné dessus.

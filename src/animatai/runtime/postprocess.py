"""Stateful per-frame post-processing (numpy only, streaming friendly).

The network gives what the *audio* determines (lips, jaw, a bit of brows and
head). Everything audio cannot know is layered here, explicitly and
controllably: blinks, an emotion chosen by the calling agent, smoothing, gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import schema

_I = schema.INDEX

# Additive blendshape offsets at intensity 1.0. Kept off the lip-closure
# channels so visemes stay readable under any emotion.
EMOTIONS: dict[str, dict[str, float]] = {
    "neutral": {},
    "happy": {"mouthSmileLeft": 0.55, "mouthSmileRight": 0.55, "cheekSquintLeft": 0.35, "cheekSquintRight": 0.35,
              "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2, "browOuterUpLeft": 0.1, "browOuterUpRight": 0.1},
    "sad": {"browInnerUp": 0.55, "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4, "eyeLookDownLeft": 0.2,
            "eyeLookDownRight": 0.2, "mouthShrugLower": 0.15, "headPitch": -0.12},
    "angry": {"browDownLeft": 0.65, "browDownRight": 0.65, "noseSneerLeft": 0.3, "noseSneerRight": 0.3,
              "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3, "mouthPressLeft": 0.15, "mouthPressRight": 0.15,
              "jawForward": 0.1},
    "surprised": {"browInnerUp": 0.5, "browOuterUpLeft": 0.55, "browOuterUpRight": 0.55, "eyeWideLeft": 0.5,
                  "eyeWideRight": 0.5, "jawOpen": 0.08},
    "fear": {"browInnerUp": 0.6, "eyeWideLeft": 0.45, "eyeWideRight": 0.45, "mouthStretchLeft": 0.3,
             "mouthStretchRight": 0.3},
    "disgust": {"noseSneerLeft": 0.6, "noseSneerRight": 0.6, "browDownLeft": 0.35, "browDownRight": 0.35,
                "mouthUpperUpLeft": 0.3, "mouthUpperUpRight": 0.3},
    "thinking": {"browDownLeft": 0.2, "browOuterUpRight": 0.35, "eyeLookUpLeft": 0.3, "eyeLookUpRight": 0.3,
                 "eyeLookOutRight": 0.2, "eyeLookInLeft": 0.2, "mouthPucker": 0.12, "headRoll": 0.08},
}


def parse_emotion(spec: str | dict | None) -> dict[str, float]:
    """'happy', 'happy:0.6', 'happy:0.5,surprised:0.3' or a dict -> {name: intensity}."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        mix = {k: float(v) for k, v in spec.items()}
    else:
        mix = {}
        for part in str(spec).split(","):
            name, _, value = part.strip().partition(":")
            mix[name] = float(value) if value else 1.0
    unknown = set(mix) - set(EMOTIONS)
    if unknown:
        raise ValueError(f"unknown emotion(s) {sorted(unknown)}; available: {sorted(EMOTIONS)}")
    return mix


def emotion_vector(mix: dict[str, float]) -> np.ndarray:
    vec = np.zeros(schema.N_CHANNELS, dtype=np.float32)
    for name, intensity in mix.items():
        for channel, value in EMOTIONS[name].items():
            vec[_I[channel]] += intensity * value
    return vec


@dataclass
class PostProcessor:
    fps: int = 30
    articulation: float = 1.15  # mouth gain around the rest pose; regression under-articulates
    smoothing: float = 0.35  # 0 = off; one-pole low-pass on non-mouth channels
    mouth_smoothing: float = 0.0
    blinks: bool = True
    blink_interval: tuple[float, float] = (2.0, 6.0)  # seconds between blinks
    emotion_rate: float = 3.0  # how fast an emotion change blends in (1/s)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: np.ndarray | None = field(default=None, init=False, repr=False)
    _emotion: np.ndarray = field(init=False, repr=False)
    _emotion_target: np.ndarray = field(init=False, repr=False)
    _next_blink: float = field(default=0.0, init=False, repr=False)
    _blink_phase: float = field(default=-1.0, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._emotion = np.zeros(schema.N_CHANNELS, dtype=np.float32)
        self._emotion_target = self._emotion.copy()
        self._next_blink = self._rng.uniform(*self.blink_interval)
        self._mouth = np.array(schema.GROUPS["mouth"])
        alpha = np.full(schema.N_CHANNELS, 1.0 - self.smoothing, dtype=np.float32)
        alpha[self._mouth] = 1.0 - self.mouth_smoothing
        self._alpha = alpha

    def set_emotion(self, spec: str | dict | None) -> None:
        self._emotion_target = emotion_vector(parse_emotion(spec))

    def reset(self) -> None:
        self._state = None
        self._blink_phase = -1.0

    def _blink_value(self) -> float:
        dt = 1.0 / self.fps
        if self._blink_phase < 0:
            self._next_blink -= dt
            if self._next_blink > 0:
                return 0.0
            self._blink_phase = 0.0
        self._blink_phase += dt / 0.2  # a blink lasts ~200 ms
        if self._blink_phase >= 1.0:
            self._blink_phase = -1.0
            self._next_blink = self._rng.uniform(*self.blink_interval)
            return 0.0
        return float(np.sin(np.pi * self._blink_phase) ** 0.7)

    def __call__(self, frames: np.ndarray, rest: np.ndarray | None = None) -> np.ndarray:
        """frames (T, 55) raw model output -> (T, 55) final animation. Keeps state across calls."""
        out = np.empty_like(frames, dtype=np.float32)
        rest = np.zeros(schema.N_CHANNELS, np.float32) if rest is None else rest
        blend = min(1.0, self.emotion_rate / self.fps)
        for t, frame in enumerate(frames.astype(np.float32)):
            frame = frame.copy()
            frame[self._mouth] = rest[self._mouth] + self.articulation * (frame[self._mouth] - rest[self._mouth])
            self._state = frame if self._state is None else self._state + self._alpha * (frame - self._state)
            self._emotion += blend * (self._emotion_target - self._emotion)
            final = self._state + self._emotion
            if self.blinks:
                final[schema.BLINK] = np.maximum(final[schema.BLINK] * 0.5, self._blink_value())
            final[: schema.N_BLENDSHAPES] = np.clip(final[: schema.N_BLENDSHAPES], 0.0, 1.0)
            final[schema.N_BLENDSHAPES:] = np.clip(final[schema.N_BLENDSHAPES:], -1.0, 1.0)
            out[t] = final
        return out

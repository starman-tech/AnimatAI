"""Retarget the 55 channels to a physical robot face (servos, motors, LEDs).

A rig is a YAML file listing actuators. Each actuator mixes channels, maps the
result onto its mechanical range and is slew-rate limited so a bad frame can
never slam a servo:

    actuators:
      jaw:
        channels: {jawOpen: 1.0, mouthFunnel: 0.2}   # weighted sum, then clamp to [in_min, in_max]
        input: [0.0, 0.8]                            # channel-mix range that maps to...
        output: [92, 140]                            # ...this actuator range (deg, pwm us, 0-255...)
        max_speed: 400                               # output units per second (optional)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .. import schema


@dataclass
class Actuator:
    name: str
    weights: np.ndarray
    in_lo: float
    in_hi: float
    out_lo: float
    out_hi: float
    max_speed: float | None


class Rig:
    def __init__(self, actuators: list[Actuator], fps: int = 30):
        self.actuators = actuators
        self.fps = fps
        self._last: np.ndarray | None = None

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.actuators]

    @classmethod
    def load(cls, path: str | Path, fps: int = 30) -> "Rig":
        spec = yaml.safe_load(Path(path).read_text())
        actuators = []
        for name, a in spec["actuators"].items():
            weights = np.zeros(schema.N_CHANNELS, dtype=np.float32)
            for channel, weight in a["channels"].items():
                if channel not in schema.INDEX:
                    raise KeyError(f"rig actuator {name!r}: unknown channel {channel!r}")
                weights[schema.INDEX[channel]] = weight
            in_lo, in_hi = a.get("input", [0.0, 1.0])
            out_lo, out_hi = a["output"]
            actuators.append(Actuator(name, weights, in_lo, in_hi, out_lo, out_hi, a.get("max_speed")))
        return cls(actuators, fps)

    def reset(self) -> None:
        self._last = None

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """(T, 55) -> (T, n_actuators) in actuator units. Stateful (slew limiter)."""
        out = np.empty((len(frames), len(self.actuators)), dtype=np.float32)
        for t, frame in enumerate(frames):
            values = np.empty(len(self.actuators), dtype=np.float32)
            for k, a in enumerate(self.actuators):
                x = (float(a.weights @ frame) - a.in_lo) / (a.in_hi - a.in_lo)
                values[k] = a.out_lo + np.clip(x, 0.0, 1.0) * (a.out_hi - a.out_lo)
            if self._last is not None:
                for k, a in enumerate(self.actuators):
                    if a.max_speed:
                        step = a.max_speed / self.fps
                        values[k] = np.clip(values[k], self._last[k] - step, self._last[k] + step)
            self._last = values
            out[t] = values
        return out

"""Output schema of AnimatAI: 52 ARKit blendshapes + 3 head rotation channels.

The schema is the contract between the model and every consumer (avatar
engines, voice assistants, robot rigs). It never changes silently: a
checkpoint stores the schema version it was trained with.
"""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "animatai-55-v1"

# ARKit / MediaPipe blendshape names, in MediaPipe FaceLandmarker order.
BLENDSHAPES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)

# Head rotation, radians-ish normalised to roughly [-1, 1] (1.0 == 45 degrees).
HEAD_POSE: tuple[str, ...] = ("headYaw", "headPitch", "headRoll")

CHANNELS: tuple[str, ...] = BLENDSHAPES + HEAD_POSE
N_BLENDSHAPES = len(BLENDSHAPES)  # 52
N_POSE = len(HEAD_POSE)  # 3
N_CHANNELS = len(CHANNELS)  # 55
INDEX = {name: i for i, name in enumerate(CHANNELS)}


def _idx(pred) -> list[int]:
    return [i for i, n in enumerate(CHANNELS) if pred(n)]


# Groups drive loss weighting and the evaluation report. Lip-sync quality is
# decided by the `mouth` group; eyes are mostly not predictable from audio.
GROUPS: dict[str, list[int]] = {
    "mouth": _idx(lambda n: n.startswith(("mouth", "jaw")) or n == "cheekPuff"),
    "brows": _idx(lambda n: n.startswith("brow")),
    "eyes": _idx(lambda n: n.startswith("eye")),
    "face": _idx(lambda n: n.startswith(("cheekSquint", "noseSneer"))),
    "head": _idx(lambda n: n.startswith("head")),
}
BLINK = [INDEX["eyeBlinkLeft"], INDEX["eyeBlinkRight"]]

# Left/right channel pairs, used by the mirror augmentation.
MIRROR_PAIRS: list[tuple[int, int]] = [
    (INDEX[n], INDEX[n[: -len("Left")] + "Right"])
    for n in BLENDSHAPES
    if n.endswith("Left") and (n[: -len("Left")] + "Right") in INDEX
]
MIRROR_NEGATE = [INDEX["headYaw"], INDEX["headRoll"]]


def channel_weights(group_weights: dict[str, float]) -> np.ndarray:
    """Per-channel loss weights from per-group weights (unlisted channels -> 0)."""
    w = np.zeros(N_CHANNELS, dtype=np.float32)
    for group, idxs in GROUPS.items():
        w[idxs] = group_weights.get(group, 1.0)
    return w


def mirror(frames: np.ndarray) -> np.ndarray:
    """Swap left/right channels: a free, label-preserving augmentation."""
    out = frames.copy()
    for left, right in MIRROR_PAIRS:
        out[..., left], out[..., right] = frames[..., right], frames[..., left]
    out[..., MIRROR_NEGATE] *= -1.0
    return out


LEGACY_ASPECT = 4 / 3  # webcam recordings of the legacy web recorder


def from_legacy58(frames58: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert the legacy 58-vector (6 pose + 52 blendshapes) to the schema.

    Legacy pose was [1-cx, cy, scale, 0, yaw_proxy, roll]: the first three are
    the face position in the camera image (not a property of the speech) and
    pitch was never measured. Returns (frames55, valid_mask55).
    """
    frames58 = np.asarray(frames58, dtype=np.float32)
    out = np.zeros((len(frames58), N_CHANNELS), dtype=np.float32)
    out[:, :N_BLENDSHAPES] = np.clip(frames58[:, 6:58], 0.0, 1.0)
    out[:, INDEX["headYaw"]] = np.clip(frames58[:, 4], -1.0, 1.0)
    # Legacy roll was measured in normalised (non-square) image coordinates, with the opposite sign.
    roll = np.arctan(np.tan(frames58[:, 5]) / LEGACY_ASPECT)
    out[:, INDEX["headRoll"]] = np.clip(-roll / (np.pi / 4), -1.0, 1.0)
    valid = np.ones(N_CHANNELS, dtype=np.float32)
    valid[INDEX["headPitch"]] = 0.0
    return out, valid


def to_legacy58(frames55: np.ndarray) -> np.ndarray:
    """Inverse of `from_legacy58` for the old web viewer (face centred)."""
    frames55 = np.asarray(frames55, dtype=np.float32)
    out = np.zeros((len(frames55), 58), dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 2] = 0.5, 0.5, 1.0
    out[:, 4] = frames55[:, INDEX["headYaw"]]
    out[:, 5] = -frames55[:, INDEX["headRoll"]] * (np.pi / 4)
    out[:, 6:58] = frames55[:, :N_BLENDSHAPES]
    return out

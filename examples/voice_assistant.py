"""Drive a face from a streaming TTS: feed audio chunks as they arrive, get frames back.

    python examples/voice_assistant.py runs/base/best.pt some_speech.wav
"""
import sys

from animatai.audio import SAMPLE_RATE, load_audio
from animatai.runtime.engine import Animator

animator = Animator(sys.argv[1], style="generic", emotion="happy:0.5")
audio = load_audio(sys.argv[2])  # stand-in for your TTS stream

chunk = SAMPLE_RATE // 10  # 100 ms, as a TTS engine would deliver
for start in range(0, len(audio), chunk):
    last = start + chunk >= len(audio)
    for frame in animator.stream(audio[start:start + chunk], flush=last):
        # -> send to your avatar / robot here (frame is 55 floats, see animator.channels)
        pass
    if start == 0:
        animator.set_emotion("surprised:0.4")  # the agent can change emotion mid-sentence
animator.reset()  # before the next utterance
print("ok, algorithmic latency:", animator.latency_ms, "ms")

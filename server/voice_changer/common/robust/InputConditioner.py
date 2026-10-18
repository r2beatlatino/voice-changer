"""
Cleans the microphone signal before it reaches the voice conversion model.

    mic (16 kHz) -> GTCRN denoiser -> Silero VAD gate -> embedder / pitch extractor

The browser's own noise suppression (Sup1/Sup2) only applies when audio goes through the
client, so in server-audio mode the model used to receive the raw microphone signal.
This runs on the server, so it covers both modes.
"""

import os
from dataclasses import dataclass

import numpy as np
from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
GTCRN_MODEL = os.path.join(MODEL_DIR, "gtcrn_simple.onnx")
SILERO_MODEL = os.path.join(MODEL_DIR, "silero_vad.onnx")


# Silero's confidence drops on quiet input, so it hears a level-normalised copy.
VAD_TARGET_PEAK = 0.5
VAD_MAX_GAIN = 16.0  # +24 dB
VAD_RELEASE_S = 3.0


@dataclass
class ConditionerResult:
    audio: np.ndarray  # float32, 16 kHz, [-1, 1]
    speech_prob: float
    gate: float  # 1.0 = pass, 0.0 = silence


class InputConditioner:
    def __init__(self):
        self.denoiser = None
        self.vad = None
        self.gate_open = False
        self.hangover_left = 0.0
        self.vad_level = 0.0
        self._load()

    def _load(self):
        try:
            from voice_changer.common.robust.GtcrnDenoiser import GtcrnDenoiser

            self.denoiser = GtcrnDenoiser(GTCRN_MODEL)
        except Exception as e:
            logger.warning(f"[InputConditioner] denoiser unavailable, continuing without it: {e}")
        try:
            from voice_changer.common.robust.SileroVad import SileroVad

            self.vad = SileroVad(SILERO_MODEL)
        except Exception as e:
            logger.warning(f"[InputConditioner] VAD unavailable, continuing without it: {e}")

    def reset(self):
        if self.denoiser is not None:
            self.denoiser.reset()
        if self.vad is not None:
            self.vad.reset()
        self.gate_open = False
        self.hangover_left = 0.0
        self.vad_level = 0.0

    def _vad_input(self, audio: np.ndarray) -> np.ndarray:
        """Copy of the chunk scaled towards VAD_TARGET_PEAK. Only the VAD hears it."""
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if peak >= self.vad_level:
            self.vad_level = peak
        else:
            # slow release so pauses between words don't pump the gain up
            decay = np.exp(-(audio.shape[0] / 16000.0) / VAD_RELEASE_S)
            self.vad_level = max(peak, self.vad_level * decay)
        gain = np.clip(VAD_TARGET_PEAK / max(self.vad_level, 1e-6), 1.0, VAD_MAX_GAIN)
        return np.clip(audio * gain, -1.0, 1.0)

    def process(
        self,
        audio: np.ndarray,
        denoise: bool,
        denoise_strength: float,
        vad: bool,
        vad_threshold: float,
        vad_hangover_ms: float,
    ) -> ConditionerResult:
        if denoise and self.denoiser is not None:
            audio = self.denoiser.process(audio, denoise_strength)
        elif self.denoiser is not None:
            # keep the denoiser's state and delay line moving so switching it on is seamless
            self.denoiser.process(audio, 0.0)

        if not vad or self.vad is None:
            return ConditionerResult(audio, 1.0, 1.0)

        prob = self.vad.process(self._vad_input(audio))
        chunk_ms = audio.shape[0] / 16.0
        # hysteresis: harder to open than to stay open, so the gate doesn't chatter mid-word
        if prob >= vad_threshold or (self.gate_open and prob >= vad_threshold - 0.15):
            self.gate_open = True
            self.hangover_left = vad_hangover_ms
        elif self.hangover_left > 0:
            self.hangover_left -= chunk_ms
        else:
            self.gate_open = False
        return ConditionerResult(audio, prob, 1.0 if self.gate_open else 0.0)

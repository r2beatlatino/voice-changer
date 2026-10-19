"""
Cleans the microphone signal before it reaches the voice conversion model.

    mic (16 kHz) -> GTCRN denoiser -> Silero VAD gate -> embedder / pitch extractor

The browser's own noise suppression (Sup1/Sup2) only applies when audio goes through the
client, so in server-audio mode the model used to receive the raw microphone signal.
This runs on the server, so it covers both modes.
"""

import os
import threading
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt
from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
GTCRN_MODEL = os.path.join(MODEL_DIR, "gtcrn_simple.onnx")
SILERO_MODEL = os.path.join(MODEL_DIR, "silero_vad.onnx")


# Silero's confidence drops on quiet input, so it hears a level-normalised copy.
VAD_TARGET_PEAK = 0.5
VAD_MAX_GAIN = 16.0  # +24 dB
VAD_RELEASE_S = 3.0

# Rumble below the voice (fans, AC, desk vibration) is often most of the noise energy and
# GTCRN only partly removes it under speech. Nothing the model needs lives below 85 Hz.
# It runs both before and after GTCRN: on a real recording, before-only left more rumble
# (GTCRN cleans the band less once it is attenuated) and after-only thinned the voice.
HIGHPASS_HZ = 85.0
HIGHPASS_SOS = butter(4, HIGHPASS_HZ, btype="highpass", fs=16000, output="sos")


@dataclass
class ConditionerResult:
    audio: np.ndarray  # float32, 16 kHz, [-1, 1]
    speech_prob: float
    gate: float  # 1.0 = pass, 0.0 = silence


class InputConditioner:
    def __init__(self):
        self.denoiser = None
        self.strong_denoiser = None  # DeepFilterNet3, loaded the first time it is asked for
        self.strong_failed = False
        self.strong_loader = None
        self.vad = None
        self.gate_open = False
        self.hangover_left = 0.0
        self.vad_level = 0.0
        self.hp_state = np.zeros((2, HIGHPASS_SOS.shape[0], 2))
        # the denoiser and VAD keep streaming state, so overlapping requests would corrupt it
        self.lock = threading.Lock()
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
        with self.lock:
            if self.denoiser is not None:
                self.denoiser.reset()
            if self.strong_denoiser is not None:
                self.strong_denoiser.reset()
            if self.vad is not None:
                self.vad.reset()
            self.gate_open = False
            self.hangover_left = 0.0
            self.vad_level = 0.0
            self.hp_state = np.zeros((2, HIGHPASS_SOS.shape[0], 2))

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

    def _highpass(self, audio: np.ndarray, stage: int) -> np.ndarray:
        out, self.hp_state[stage] = sosfilt(HIGHPASS_SOS, audio, zi=self.hp_state[stage])
        return out.astype(np.float32)

    def process(
        self,
        audio: np.ndarray,
        denoise: bool,
        denoise_strength: float,
        vad: bool,
        vad_threshold: float,
        vad_hangover_ms: float,
        strong: bool = False,
    ) -> ConditionerResult:
        with self.lock:
            return self._process(audio, denoise, denoise_strength, vad, vad_threshold, vad_hangover_ms, strong)

    def _load_strong(self):
        try:
            from voice_changer.common.robust.DfnDenoiser import DfnDenoiser

            dfn = DfnDenoiser()
            # the first CUDA calls are slow (about 200 ms on an RTX 3090); pay that here, not mid-stream
            for _ in range(2):
                dfn.process(np.zeros(1600, dtype=np.float32))
            dfn.reset()
            with self.lock:
                self.strong_denoiser = dfn
            logger.info(f"[InputConditioner] DeepFilterNet3 loaded on {dfn.device}")
        except (Exception, SystemExit) as e:  # df calls exit() when it can't fetch the weights
            self.strong_failed = True
            logger.warning(f"[InputConditioner] DeepFilterNet3 unavailable, using GTCRN: {e}")

    def _strong(self):
        """DeepFilterNet3 once loaded. Loading takes about a second, so it happens in the
        background and GTCRN keeps working until it is ready."""
        if self.strong_denoiser is None and not self.strong_failed and self.strong_loader is None:
            self.strong_loader = threading.Thread(target=self._load_strong, daemon=True)
            self.strong_loader.start()
        return self.strong_denoiser

    def _process(self, audio, denoise, denoise_strength, vad, vad_threshold, vad_hangover_ms, strong=False) -> ConditionerResult:
        if denoise:
            audio = self._highpass(audio, 0)
        use_strong = denoise and strong and self._strong() is not None
        # the denoiser not in use still hears the input, so switching between them is seamless
        if self.strong_denoiser is not None:
            if use_strong:
                strong_out = self.strong_denoiser.process(audio, denoise_strength)
            else:
                self.strong_denoiser.push(audio)
        if self.denoiser is not None:
            gtcrn_out = self.denoiser.process(audio, denoise_strength if denoise and not use_strong else 0.0)
            if denoise and not use_strong:
                audio = gtcrn_out
        if use_strong:
            audio = strong_out
        if denoise:
            audio = self._highpass(audio, 1)

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

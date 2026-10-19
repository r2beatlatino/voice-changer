"""
Setup assistant: picks the noise settings and the pitch shift from two short recordings,
10 s of the room with the speaker silent and about 15 s of the speaker reading aloud.

This automates what was tuned by hand on a real recording: Strong denoise when the room is
noisy, the highest VAD threshold that still lets all of the speech through, and a pitch
shift that moves the speaker's own median pitch to the typical pitch of the target voice.

Input: float32 mono at 16 kHz in [-1, 1]. Tips are codes; the client words them.
"""

import numpy as np
from scipy.signal import welch

from voice_changer.common.robust.InputConditioner import InputConditioner, next_gate

SR = 16000
CHUNK = 1600  # 100 ms, about what the gate sees per block at the default chunk size
VAD_HANGOVER_MS = 500
# Lowest first: the first that keeps the room out wins. A higher threshold than that only
# risks cutting soft word onsets and endings, which a speech mask of loud chunks can't see.
VAD_CANDIDATES = [0.3, 0.4, 0.5]
MAX_NOISE_PASS = 0.05
MIN_SPEECH_PASS = 0.9
# typical speaking pitch of the target voice, used to suggest a pitch shift
TARGET_PITCH_HZ = {"female": 280.0, "male": 125.0}
# Strong when the room is this noisy; on a real rumble recording it removed about 20 dB more
STRONG_BELOW_SNR_DB = 35.0
STRONG_ABOVE_NOISE_DB = -50.0


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-9)))


def _chunks(x: np.ndarray):
    n = x.shape[0] // CHUNK
    return x[: n * CHUNK].reshape(n, CHUNK)


def estimate_pitch(x: np.ndarray, fmin: float = 60.0, fmax: float = 500.0) -> np.ndarray:
    """Per-frame f0 (Hz, 0 = unvoiced) with the YIN difference function, 40 ms frames, 10 ms hop.
    Good enough for the median pitch of read speech; RVC's own extractor is used for conversion."""
    frame, hop = 640, 160
    lag_min, lag_max = int(SR / fmax), int(SR / fmin)
    out = []
    rms_all = np.sqrt(np.mean(x**2)) if x.size else 0.0
    for start in range(0, x.shape[0] - frame - lag_max, hop):
        seg = x[start : start + frame + lag_max]
        if np.sqrt(np.mean(seg[:frame] ** 2)) < 0.2 * rms_all:
            out.append(0.0)
            continue
        d = np.sum((np.lib.stride_tricks.sliding_window_view(seg, frame) - seg[:frame]) ** 2, axis=1)
        cmnd = d[1:] * np.arange(1, lag_max + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
        cand = np.where(cmnd[lag_min - 1 :] < 0.25)[0]
        if cand.size == 0:
            out.append(0.0)
            continue
        lag = cand[0] + lag_min
        # follow the dip to its minimum
        while lag < lag_max and cmnd[lag] < cmnd[lag - 1]:
            lag += 1
        out.append(SR / lag)
    return np.array(out)


def _gate_pass(probs: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    if not mask.any():
        return 0.0
    gate_open, hangover, opened = False, 0.0, []
    for p in probs:
        gate_open, hangover = next_gate(gate_open, hangover, p, threshold, VAD_HANGOVER_MS, CHUNK / 16.0)
        opened.append(gate_open)
    return float(np.mean(np.array(opened)[mask]))


def _condition(conditioner: InputConditioner, x: np.ndarray, strong: bool = False):
    """Speech probability per chunk, and the denoised audio."""
    conditioner.reset()
    probs, audio = [], []
    for c in _chunks(x):
        r = conditioner.process(c, denoise=True, denoise_strength=1.0, vad=True, vad_threshold=1.1, vad_hangover_ms=0, strong=strong)
        probs.append(r.speech_prob)
        audio.append(r.audio)
    return np.array(probs), (np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32))


def calibrate(noise: np.ndarray, voice: np.ndarray, target: str = "keep", strong_available: bool = True, pitch_fn=estimate_pitch, conditioner: InputConditioner | None = None) -> dict:
    """conditioner: one with DeepFilterNet3 already loaded (the voice's own, while live
    conversion is stopped) lets the VAD be measured behind Strong, as it will run."""
    noise = noise.astype(np.float32)
    voice = voice.astype(np.float32)
    tips: list[str] = []

    if conditioner is None:
        conditioner = InputConditioner(preload_strong=False)
    voice_probs, voice_clean = _condition(conditioner, voice)
    noise_probs, noise_clean = _condition(conditioner, noise)

    def rms(x):
        return float(np.sqrt(np.mean(x**2))) if x.size else 0.0

    noise_rms = rms(noise)
    # Chunks clearly above the room noise are the speech the gate must let through. Measured
    # after denoising, or a loud room hides the speech; the denoiser is 32 ms late, which is
    # well inside a 100 ms chunk.
    clean_chunks = _chunks(voice_clean)
    chunk_rms = np.sqrt(np.mean(clean_chunks**2, axis=1)) if clean_chunks.size else np.zeros(0)
    # pauses between words are not speech either: at least -16 dB from the loud part of the reading
    loud = float(np.percentile(chunk_rms, 90)) if chunk_rms.size else 0.0
    speech = chunk_rms > max(rms(noise_clean) * 4.0, loud * 0.15, 1e-5)
    raw_chunks = _chunks(voice)[: chunk_rms.shape[0]]
    raw_rms = np.sqrt(np.mean(raw_chunks**2, axis=1)) if raw_chunks.size else np.zeros(0)
    voice_rms = float(np.median(raw_rms[speech])) if speech.any() else 0.0
    voice_peak = float(np.max(np.abs(voice))) if voice.size else 0.0
    snr = _db(voice_rms) - _db(noise_rms)

    if speech.sum() < 20:  # less than 2 s of speech
        tips.append("no_voice")
    if voice_peak > 0.98:
        tips.append("mic_clipping")
    elif voice_peak < 0.1:  # below -20 dBFS
        tips.append("mic_low")

    low_share = 0.0
    if noise.size >= 4096:
        f, pxx = welch(noise, fs=SR, nperseg=4096)
        low_share = float(pxx[f < 150].sum() / max(pxx.sum(), 1e-20))
        if low_share > 0.5 and _db(noise_rms) > -60:
            tips.append("low_rumble")
    if snr < 20:
        tips.append("noisy_room")

    noisy = snr < STRONG_BELOW_SNR_DB or _db(noise_rms) > STRONG_ABOVE_NOISE_DB
    strong = strong_available and noisy
    if noisy and not strong_available:
        tips.append("strong_missing")

    behind_strong = strong and conditioner.strong_denoiser is not None
    if behind_strong:
        voice_probs, _ = _condition(conditioner, voice, strong=True)
        noise_probs, _ = _condition(conditioner, noise, strong=True)
    everything = np.ones(noise_probs.shape[0], dtype=bool)
    threshold = VAD_CANDIDATES[-1]
    for t in VAD_CANDIDATES:
        if _gate_pass(noise_probs, everything, t) <= MAX_NOISE_PASS:
            threshold = t
            break
    if strong and not behind_strong:
        # Measured behind GTCRN, which lets rumble through that Silero takes for speech; behind
        # Strong (on a real recording) none of it reached the gate, so don't go stricter than 0.4.
        threshold = min(threshold, 0.4)
    voice_pass = _gate_pass(voice_probs, speech, threshold)
    noise_pass = _gate_pass(noise_probs, everything, threshold)
    if noise_pass > MAX_NOISE_PASS and not (strong and not behind_strong):
        tips.append("speech_like_noise")
    if voice_pass < MIN_SPEECH_PASS:
        tips.append("vad_cuts")
    conditioner.reset()

    settings = {"denoise": 1, "denoiseStrength": 1.0, "denoiseStrong": 1 if strong else 0, "vad": 1, "vadThreshold": threshold}

    f0 = pitch_fn(voice_clean)
    voiced = f0[f0 > 0]
    user_pitch = float(np.median(voiced)) if voiced.size >= 50 else 0.0
    if target in TARGET_PITCH_HZ and user_pitch > 0:
        tran = round(12 * np.log2(TARGET_PITCH_HZ[target] / user_pitch))
        # a semitone off the octave: take the octave, which is what tuning by ear landed on
        if abs(abs(tran) - 12) <= 1:
            tran = 12 if tran > 0 else -12
        settings["tran"] = int(np.clip(tran, -24, 24))
    elif target in TARGET_PITCH_HZ:
        tips.append("no_pitch")

    return {
        "settings": settings,
        "measure": {
            "noiseDb": round(_db(noise_rms), 1),
            "voiceDb": round(_db(voice_rms), 1),
            "snrDb": round(snr, 1),
            "voicePeakDb": round(_db(voice_peak), 1),
            "lowFreqShare": round(low_share, 2),
            "userPitchHz": round(user_pitch, 1),
            "speechPass": round(voice_pass, 2),
            "noisePass": round(noise_pass, 2),
        },
        "tips": tips,
    }

"""
Setup assistant: settings picked from a room recording and a reading. Run from the server directory:
    python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest
import resampy
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_changer.common.robust.Calibration import VAD_CANDIDATES, calibrate, estimate_pitch  # noqa: E402

TEST_WAV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test.wav")


@pytest.fixture(scope="module")
def speech():
    x, sr = sf.read(TEST_WAV, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return resampy.resample(x, sr, 16000).astype(np.float32) * 0.5


def _rumble(seconds, rms):
    n = np.random.default_rng(0).standard_normal(16000 * seconds)
    r = sosfilt(butter(2, 120, fs=16000, output="sos"), n).astype(np.float32)
    return r * rms / np.sqrt(np.mean(r**2))


def test_pitch_of_a_harmonic_tone():
    t = np.arange(16000 * 2) / 16000
    x = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 6)).astype(np.float32) * 0.3
    f0 = estimate_pitch(x)
    assert abs(np.median(f0[f0 > 0]) - 150) < 3


def test_quiet_room(speech):
    noise = np.random.default_rng(1).standard_normal(16000 * 10).astype(np.float32) * 3e-4
    r = calibrate(noise, speech + np.resize(noise, speech.shape), target="female")
    s = r["settings"]
    assert s["vadThreshold"] in VAD_CANDIDATES and r["measure"]["speechPass"] >= 0.95
    # the sample speaker talks at about 105 Hz; a female voice sits at about 280 Hz
    assert 15 <= s["tran"] <= 18
    assert s["vadThreshold"] == VAD_CANDIDATES[0]  # nothing in a quiet room to keep out
    assert "low_rumble" not in r["tips"]


def test_a_semitone_off_the_octave_snaps_to_it(speech):
    noise = np.random.default_rng(1).standard_normal(16000 * 10).astype(np.float32) * 3e-4

    def pitch(x):
        return np.full(500, 147.7)  # ben's speaking pitch: 11.1 semitones below 280 Hz

    r = calibrate(noise, speech + np.resize(noise, speech.shape), target="female", pitch_fn=pitch)
    assert r["settings"]["tran"] == 12


def test_rumble_turns_on_strong_and_says_why(speech):
    rum = _rumble(10, 0.01)
    r = calibrate(rum, speech + np.resize(rum, speech.shape), target="keep")
    assert r["settings"]["denoiseStrong"] == 1
    assert "low_rumble" in r["tips"]
    assert "tran" not in r["settings"]


def test_strong_only_when_installed(speech):
    rum = _rumble(10, 0.01)
    r = calibrate(rum, speech + np.resize(rum, speech.shape), strong_available=False)
    assert r["settings"]["denoiseStrong"] == 0
    assert "strong_missing" in r["tips"]


def test_strong_not_loaded_yet_keeps_vad_moderate(speech):
    # without DeepFilterNet3 loaded the gate is measured behind GTCRN, which leaks rumble
    rum = _rumble(10, 0.01)
    r = calibrate(rum, speech + np.resize(rum, speech.shape), strong_available=True)
    assert r["settings"]["denoiseStrong"] == 1
    assert r["settings"]["vadThreshold"] <= 0.4
    assert "speech_like_noise" not in r["tips"]


def test_silence_instead_of_reading_is_reported():
    noise = np.random.default_rng(2).standard_normal(16000 * 10).astype(np.float32) * 3e-4
    r = calibrate(noise, noise.copy(), target="female")
    assert "no_voice" in r["tips"] or "no_pitch" in r["tips"]


def test_calibrate_endpoint(speech):
    import base64
    import json

    from restapi.MMVC_Rest_VoiceChanger import CalibrationRequest, ClipRequest, MMVC_Rest_VoiceChanger

    class _NoVoice:
        voiceChanger = None

    api = MMVC_Rest_VoiceChanger(_NoVoice())

    def b64(x):
        return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()

    noise = np.random.default_rng(3).standard_normal(16000 * 5).astype(np.float32) * 3e-4
    res = api.calibrate(CalibrationRequest(sampleRate=16000, noise=b64(noise), voice=b64(speech), target="male"))
    assert res.status_code == 200 and "tran" in json.loads(res.body)["settings"]
    res = api.convert_clip(ClipRequest(sampleRate=16000, voice=b64(speech)))
    assert res.status_code == 400  # nothing loaded to convert with

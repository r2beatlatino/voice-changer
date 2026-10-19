"""
Tests for the noise-robust front end. Run from the server directory:
    python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_changer.common.robust.GtcrnDenoiser import GtcrnDenoiser  # noqa: E402
from voice_changer.common.robust.InputConditioner import GTCRN_MODEL, SILERO_MODEL, InputConditioner  # noqa: E402
from voice_changer.common.robust.PitchStabilizer import f0_to_coarse, stabilize_f0  # noqa: E402
from voice_changer.common.robust.SileroVad import SileroVad  # noqa: E402

SR = 16000


def tone(seconds: float, hz: float = 200.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def run_chunks(fn, x, sizes):
    out, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out.append(fn(x[i : i + n]))
        i += n
        k += 1
    return np.concatenate(out)


class TestDenoiser:
    def test_output_length_matches_input_for_any_chunk_size(self):
        d = GtcrnDenoiser(GTCRN_MODEL)
        for n in [1, 100, 255, 256, 257, 1600, 4097]:
            assert d.process(np.zeros(n, dtype=np.float32)).shape[0] == n

    def test_result_does_not_depend_on_chunking(self):
        x = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
        a = run_chunks(GtcrnDenoiser(GTCRN_MODEL).process, x, [1600])
        b = run_chunks(GtcrnDenoiser(GTCRN_MODEL).process, x, [333, 1000, 77, 4096])
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_reduces_white_noise(self):
        noise = np.random.default_rng(1).standard_normal(2 * SR).astype(np.float32) * 0.05
        out = run_chunks(GtcrnDenoiser(GTCRN_MODEL).process, noise, [1600])
        # skip the first half second while the recurrent state settles
        assert np.std(out[SR // 2 :]) < 0.3 * np.std(noise[SR // 2 :])

    def test_strength_zero_is_delayed_passthrough(self):
        x = np.random.default_rng(2).standard_normal(SR).astype(np.float32)
        d = GtcrnDenoiser(GTCRN_MODEL)
        out = run_chunks(lambda c: d.process(c, strength=0.0), x, [1000])
        lat = GtcrnDenoiser.LATENCY_SAMPLES
        np.testing.assert_allclose(out[lat:], x[:-lat], atol=1e-6)


class TestVad:
    def test_silence_and_noise_are_not_speech(self):
        v = SileroVad(SILERO_MODEL)
        assert v.process(np.zeros(SR, dtype=np.float32)) < 0.2
        v.reset()
        noise = np.random.default_rng(3).standard_normal(SR).astype(np.float32) * 0.05
        probs = [v.process(noise[i : i + 512]) for i in range(0, len(noise), 512)]
        assert max(probs) < 0.5

    def test_short_chunks_keep_last_probability(self):
        v = SileroVad(SILERO_MODEL)
        p = v.process(np.zeros(100, dtype=np.float32))
        assert p == 0.0


class TestConditioner:
    def test_gate_stays_closed_on_silence(self):
        c = InputConditioner()
        for _ in range(20):
            r = c.process(np.zeros(1600, dtype=np.float32), True, 1.0, True, 0.5, 300)
            assert r.audio.shape[0] == 1600
            assert r.gate == 0.0

    def test_gate_hangover(self):
        c = InputConditioner()
        c.gate_open = True
        c.hangover_left = 300
        gates = [c.process(np.zeros(1600, dtype=np.float32), False, 1.0, True, 0.5, 300).gate for _ in range(6)]
        # 100 ms chunks: open for the 300 ms hangover, then closed
        assert gates[:3] == [1.0, 1.0, 1.0]
        assert gates[-1] == 0.0

    def test_vad_hears_quiet_input_louder_but_output_is_untouched(self):
        c = InputConditioner()
        quiet = tone(0.1, amp=0.05)
        boosted = c._vad_input(quiet)
        assert np.abs(boosted).max() == pytest.approx(0.5, rel=0.01)
        r = c.process(quiet, False, 1.0, True, 0.3, 500)
        np.testing.assert_array_equal(r.audio, quiet)

    def test_vad_gain_is_capped_for_near_silence(self):
        c = InputConditioner()
        hiss = tone(0.1, amp=0.001)
        assert np.abs(c._vad_input(hiss)).max() <= 0.001 * 16.0 + 1e-6

    def test_vad_gain_releases_slowly_in_pauses(self):
        c = InputConditioner()
        c._vad_input(tone(0.1, amp=0.4))
        # a 100 ms pause right after loud speech should not be boosted to full scale
        assert np.abs(c._vad_input(tone(0.1, amp=0.01))).max() < 0.05

    def test_concurrent_calls_do_not_corrupt_state(self):
        import threading

        c = InputConditioner()
        errors = []

        def worker(n):
            x = np.random.default_rng(n).standard_normal(n).astype(np.float32) * 0.1
            try:
                for _ in range(30):
                    assert c.process(x, True, 1.0, True, 0.3, 500).audio.shape[0] == n
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in (4438, 8192, 1600)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_highpass_removes_rumble_but_keeps_voice(self):
        def level_after(hz):
            c = InputConditioner()
            c.denoiser = None  # isolate the high-pass
            x = tone(1.0, hz=hz, amp=0.3)
            out = np.concatenate([c.process(x[i : i + 1600], True, 1.0, False, 0.3, 500).audio for i in range(0, len(x), 1600)])
            return np.std(out[SR // 2 :]) / np.std(x[SR // 2 :])

        assert level_after(40) < 0.1  # -20 dB
        assert level_after(250) > 0.95

    def test_strong_mode_uses_deepfilternet_when_loaded(self):
        c = InputConditioner()
        calls = []

        class Fake:
            def process(self, x, strength):
                calls.append(len(x))
                return x

        c.strong_denoiser = Fake()
        c.process(np.zeros(1600, dtype=np.float32), True, 1.0, False, 0.3, 500, strong=True)
        assert calls == [1600]

    def test_strong_mode_falls_back_to_gtcrn(self, monkeypatch):
        import voice_changer.common.robust.DfnDenoiser as dfn_module

        def broken(self):
            raise SystemExit(1)  # what df does when it can't download the weights

        monkeypatch.setattr(dfn_module.DfnDenoiser, "__init__", broken)
        c = InputConditioner()
        r = c.process(np.zeros(1600, dtype=np.float32), True, 1.0, False, 0.3, 500, strong=True)
        assert r.audio.shape[0] == 1600
        c.strong_loader.join()
        assert c.strong_failed and c.strong_denoiser is None

    def test_strong_mode_loads_in_background(self, monkeypatch):
        import time

        import voice_changer.common.robust.DfnDenoiser as dfn_module

        calls = []

        def slow_init(self):
            time.sleep(0.5)
            self.device = "cpu"
            self.reset()

        monkeypatch.setattr(dfn_module.DfnDenoiser, "__init__", slow_init)
        monkeypatch.setattr(dfn_module.DfnDenoiser, "process", lambda self, x, strength: calls.append(len(x)) or x)
        c = InputConditioner()
        x = np.zeros(1600, dtype=np.float32)
        t0 = time.perf_counter()
        c.process(x, True, 1.0, False, 0.3, 500, strong=True)
        assert time.perf_counter() - t0 < 0.3  # GTCRN answers while the model loads
        assert calls == []
        c.strong_loader.join()
        c.process(x, True, 1.0, False, 0.3, 500, strong=True)
        assert calls == [1600]

    def test_everything_off_is_passthrough(self):
        c = InputConditioner()
        x = np.random.default_rng(4).standard_normal(1600).astype(np.float32)
        r = c.process(x, False, 1.0, False, 0.5, 300)
        np.testing.assert_array_equal(r.audio, x)
        assert r.gate == 1.0


class TestPitchStabilizer:
    def test_removes_short_voiced_blips(self):
        f0 = np.zeros(50)
        f0[20:22] = 300
        assert stabilize_f0(f0).max() == 0

    def test_bridges_short_gaps(self):
        f0 = np.full(40, 200.0)
        f0[18:21] = 0
        out = stabilize_f0(f0)
        assert np.all(out > 0)
        np.testing.assert_allclose(out[18:21], 200.0, rtol=1e-6)

    def test_keeps_long_pauses(self):
        f0 = np.full(60, 200.0)
        f0[20:40] = 0
        assert np.all(stabilize_f0(f0)[20:40] == 0)

    def test_folds_octave_jumps(self):
        f0 = np.full(40, 150.0)
        f0[20] = 300.0
        f0[25] = 75.0
        np.testing.assert_allclose(stabilize_f0(f0), 150.0, rtol=1e-6)

    def test_keeps_real_pitch_glides(self):
        f0 = np.geomspace(120, 240, 100)  # one octave over a second
        out = stabilize_f0(f0)
        assert np.max(np.abs(np.log2(out / f0))) < 0.02

    def test_coarse_matches_extractor_formula(self):
        f0 = np.array([0.0, 100.0, 440.0, 1100.0])
        mel = 1127 * np.log(1 + f0 / 700)
        lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
        expected = mel.copy()
        expected[expected > 0] = (expected[expected > 0] - lo) * 254 / (hi - lo) + 1
        expected = np.rint(np.clip(expected, 1, 255)).astype(int)
        np.testing.assert_array_equal(f0_to_coarse(f0), expected)


class _StubPipeline:
    """Stands in for the real RVC pipeline (no model weights needed)."""

    device = torch.device("cpu")

    def __init__(self, model_sr):
        self.model_sr = model_sr
        self.calls = 0
        self.f0Stabilize = False

    def exec(self, sid, audio, pitchf, feature, f0_up_key, index_rate, if_f0, silence_front, embOutputLayer, useFinalProj, repeat, protect, out_size):
        self.calls += 1
        out = torch.full((out_size,), 1000, dtype=torch.int16)
        return out, torch.from_numpy(pitchf), torch.from_numpy(feature)


@pytest.fixture
def rvc():
    from data.ModelSlot import RVCModelSlot
    from voice_changer.RVC.RVCr2 import RVCr2

    slot = RVCModelSlot()
    slot.samplingRate = 40000
    slot.f0 = True
    slot.embChannels = 768
    params = type("P", (), {})()
    m = RVCr2.__new__(RVCr2)
    # skip __init__'s model managers; only what inference() touches
    from voice_changer.RVC.RVCSettings import RVCSettings

    m.settings = RVCSettings()
    m.settings.silentThreshold = 0.0
    m.slotInfo = slot
    m.params = params
    m.audio_buffer = m.pitchf_buffer = m.feature_buffer = None
    m.prevVol = 0.0
    m.prevGate = 0.0
    m.inputConditioner = InputConditioner()
    m.pipeline = _StubPipeline(slot.samplingRate)
    m.setSamplingRate(48000, 48000)
    return m


class TestRvcIntegration:
    def test_silence_skips_inference(self, rvc):
        for _ in range(10):
            rvc.inference(np.zeros(4800, dtype=np.float32), 1024, 576)
        assert rvc.pipeline.calls == 0

    def test_gate_off_runs_inference_every_block(self, rvc):
        rvc.settings.vad = 0
        rng = np.random.default_rng(5)
        for _ in range(5):
            out = rvc.inference((rng.standard_normal(4800) * 3000).astype(np.float32), 1024, 576)
            assert out is not None and out.shape[0] > 4800
        assert rvc.pipeline.calls == 5

    def test_silent_block_matches_converted_block_length(self, rvc):
        silent = rvc.inference(np.zeros(4800, dtype=np.float32), 1024, 576)
        rvc.settings.vad = 0
        voiced = rvc.inference((np.random.default_rng(7).standard_normal(4800) * 3000).astype(np.float32), 1024, 576)
        assert silent.shape[0] == voiced.shape[0]

    def test_fade_out_when_speech_stops(self, rvc):
        rvc.settings.vad = 1
        rvc.settings.vadThreshold = 1.0  # noise input must close the gate whatever the VAD scores it
        rvc.inputConditioner.gate_open = True
        rvc.inputConditioner.hangover_left = 0
        rvc.prevGate = 1.0
        out = rvc.inference((np.random.default_rng(6).standard_normal(4800) * 3000).astype(np.float32), 1024, 576)
        assert rvc.pipeline.calls == 1
        assert abs(out[0]) > abs(out[-1])  # ramps down to silence
        assert abs(out[-1]) < 1e-6


class TestDeepFilterNet:
    @pytest.fixture(scope="class")
    def dfn(self):
        try:
            from voice_changer.common.robust.DfnDenoiser import DfnDenoiser

            return DfnDenoiser()
        except (Exception, SystemExit) as e:  # package or model weights not available
            pytest.skip(f"DeepFilterNet3 unavailable: {e}")

    def test_output_length_and_noise_reduction(self, dfn):
        dfn.reset()
        noise = np.random.default_rng(8).standard_normal(2 * SR).astype(np.float32) * 0.05
        out = run_chunks(dfn.process, noise, [1280, 1600, 777])
        assert out.shape == noise.shape
        assert np.std(out[SR // 2 :]) < 0.3 * np.std(noise[SR // 2 :])

    def test_switching_in_after_push_matches_running_all_along(self, dfn):
        x = np.random.default_rng(9).standard_normal(2 * SR).astype(np.float32) * 0.05
        chunks = [x[i : i + 1600] for i in range(0, len(x), 1600)]
        dfn.reset()
        always = [dfn.process(c) for c in chunks]
        dfn.reset()
        for c in chunks[:10]:
            dfn.push(c)
        switched = [dfn.process(c) for c in chunks[10:]]
        np.testing.assert_allclose(switched[0], always[10], atol=1e-4)

"""
LLVC streaming: chunked resampling must match whole-file resampling, and the model's jittery
output length must reach SOLA as a fixed amount per block. Run from the server directory:
    python -m pytest tests -q
"""

import os
import sys

import numpy as np
import resampy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_changer.LLVC.LLVC import LLVC, StreamResampler  # noqa: E402


def test_stream_resampler_matches_offline():
    x = np.sin(2 * np.pi * 220 * np.arange(48000) / 48000).astype(np.float32) * 0.5
    rs = StreamResampler(48000, 16000)
    out = np.concatenate([rs(x[i : i + n]) for i, n in zip(range(0, 48000, 4096), [4096] * 12)])
    ref = resampy.resample(x, 48000, 16000, filter="kaiser_fast")
    # the context delays when samples come out, not where they sit in the stream
    k = len(out) - 200
    np.testing.assert_allclose(out[200:k], ref[200:k], atol=2e-3)


class _UnitInferencer:
    """Emits only whole 208-sample units, like LLVCInferencer, passing audio through."""

    class model:
        dec_chunk_size = 13
        L = 16

    def __init__(self):
        self.buf = np.zeros(0, dtype=np.float32)

    def infer(self, x):
        self.buf = np.concatenate([self.buf, x])
        n = len(self.buf) // 208 * 208
        out, self.buf = self.buf[:n], self.buf[n:]
        return torch.from_numpy(out)


def test_sola_gets_a_fixed_amount_per_block():
    m = LLVC.__new__(LLVC)
    m.processingSampleRate = 16000
    m.inputSampleRate = m.outputSampleRate = 48000
    m.settings = type("S", (), {"crossfade": True, "resample": True})()
    m.inferencer = _UnitInferencer()
    m.prev_audio1 = None
    m._reset_stream()
    x = (np.random.default_rng(0).standard_normal(48000 * 2) * 3000).astype(np.int16)
    lengths = [m.inference(x[i : i + 4096], 1024, 576).shape[0] for i in range(0, len(x) - 4096, 4096)]
    lengths = lengths[2:]  # the crossfade history fills up over the first two blocks
    assert max(lengths) - min(lengths) <= 3  # one 16 kHz sample of rounding at 48 kHz

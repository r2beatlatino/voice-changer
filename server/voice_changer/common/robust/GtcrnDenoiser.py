"""
Streaming speech enhancement with GTCRN (https://github.com/Xiaobin-Rong/gtcrn, MIT).

GTCRN is a ~48k-parameter model trained on DNS3. It runs one STFT frame at a time on
CPU (well under 1 ms per 16 ms hop), so it can sit in front of the embedder and the
pitch extractor without needing the GPU.

Input/output: float32 mono at 16 kHz in [-1, 1]. process() always returns as many
samples as it receives; the output is delayed by LATENCY_SAMPLES (32 ms).
"""

import numpy as np
import onnxruntime


class GtcrnDenoiser:
    SR = 16000
    N_FFT = 512
    HOP = 256
    LATENCY_SAMPLES = 512  # 256 (framing) + 256 (fifo so arbitrary chunk sizes work)

    def __init__(self, model_path: str):
        so = onnxruntime.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        # sqrt of periodic hann: analysis * synthesis = hann, which sums to 1 at 50% overlap.
        self.window = np.sqrt(np.hanning(self.N_FFT + 1)[: self.N_FFT]).astype(np.float32)
        self.reset()

    def reset(self):
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)
        self.in_buf = np.zeros(self.N_FFT - self.HOP, dtype=np.float32)
        self.ola = np.zeros(self.N_FFT, dtype=np.float32)
        self.out_fifo = np.zeros(self.LATENCY_SAMPLES - (self.N_FFT - self.HOP), dtype=np.float32)
        self.dry_fifo = np.zeros(self.LATENCY_SAMPLES, dtype=np.float32)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window)
        mix = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)[None, :, None, :]  # (1, 257, 1, 2)
        enh, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            None,
            {
                "mix": mix,
                "conv_cache": self.conv_cache,
                "tra_cache": self.tra_cache,
                "inter_cache": self.inter_cache,
            },
        )
        enh = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        return (np.fft.irfft(enh, n=self.N_FFT) * self.window).astype(np.float32)

    def process(self, x: np.ndarray, strength: float = 1.0) -> np.ndarray:
        x = x.astype(np.float32)
        n = x.shape[0]
        self.in_buf = np.concatenate([self.in_buf, x])
        emitted = []
        while self.in_buf.shape[0] >= self.N_FFT:
            y = self._process_frame(self.in_buf[: self.N_FFT])
            self.ola += y
            emitted.append(self.ola[: self.HOP].copy())
            self.ola = np.concatenate([self.ola[self.HOP :], np.zeros(self.HOP, dtype=np.float32)])
            self.in_buf = self.in_buf[self.HOP :]
        if len(emitted) > 0:
            self.out_fifo = np.concatenate([self.out_fifo] + emitted)
        wet = self.out_fifo[:n]
        self.out_fifo = self.out_fifo[n:]

        # keep the unprocessed signal aligned with the enhanced one so they can be blended
        self.dry_fifo = np.concatenate([self.dry_fifo, x])
        dry = self.dry_fifo[:n]
        self.dry_fifo = self.dry_fifo[n:]

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength >= 1.0:
            return wet
        return strength * wet + (1.0 - strength) * dry

"""
Stronger speech enhancement with DeepFilterNet3 (https://github.com/Rikorose/DeepFilterNet, MIT/Apache-2.0).

On a real recording with steady rumble it removed about 9 dB more noise than GTCRN while
keeping the voice level. It is heavier (about 9 ms per 80 ms block on an RTX 3090, 13-18 ms
on CPU), so it is an opt-in "strong" mode.

The deepfilternet package has no streaming API, so each call re-runs the model on the last
CONTEXT_S of audio and keeps only the newest samples. LATENCY_SAMPLES of output delay keep
the model's lookahead and the resampler edges out of what is returned.

Input/output: float32 mono at 16 kHz in [-1, 1], same contract as GtcrnDenoiser.
"""

import os

import numpy as np
import torch


class DfnDenoiser:
    SR = 16000
    CONTEXT_S = 0.5
    # 30 ms (2 lookahead hops + window) would do; 32 ms matches GTCRN so switching doesn't shift the signal
    LATENCY_SAMPLES = 512

    def __init__(self):
        from df.enhance import enhance, init_df
        import torchaudio.functional as AF

        self._enhance = enhance
        self._resample = AF.resample
        # a portable install ships the weights and points here, so nothing is downloaded
        model_dir = os.environ.get("VC_DFN_MODEL_DIR") or None
        self.model, self.df_state, _ = init_df(model_base_dir=model_dir, log_level="WARNING", log_file=None)
        self.model_sr = self.df_state.sr()
        self.device = next(self.model.parameters()).device
        self.reset()

    def reset(self):
        self.buf = np.zeros(int(self.CONTEXT_S * self.SR) + self.LATENCY_SAMPLES, dtype=np.float32)
        self.dry_fifo = np.zeros(self.LATENCY_SAMPLES, dtype=np.float32)

    def push(self, x: np.ndarray):
        """Keep the window and delay line current without running the model, so switching to it is seamless."""
        x = x.astype(np.float32)
        keep = int(self.CONTEXT_S * self.SR) + self.LATENCY_SAMPLES
        self.buf = np.concatenate([self.buf, x])[-max(keep, x.shape[0] + self.LATENCY_SAMPLES) :]
        self.dry_fifo = np.concatenate([self.dry_fifo, x])[-self.LATENCY_SAMPLES :]

    def process(self, x: np.ndarray, strength: float = 1.0) -> np.ndarray:
        x = x.astype(np.float32)
        n = x.shape[0]
        if n == 0:
            return x
        keep = int(self.CONTEXT_S * self.SR) + self.LATENCY_SAMPLES
        self.buf = np.concatenate([self.buf, x])[-max(keep, n + self.LATENCY_SAMPLES) :]

        with torch.no_grad():
            audio = torch.from_numpy(self.buf)[None]
            audio = self._resample(audio, self.SR, self.model_sr)
            enh = self._enhance(self.model, self.df_state, audio)
            enh = self._resample(enh.cpu(), self.model_sr, self.SR)[0].numpy()
        enh = enh[: self.buf.shape[0]]
        wet = enh[-(n + self.LATENCY_SAMPLES) : -self.LATENCY_SAMPLES]

        dry_all = np.concatenate([self.dry_fifo, x])
        dry, self.dry_fifo = dry_all[:n], dry_all[n:]
        if strength >= 1.0:
            return wet.astype(np.float32)
        return (strength * wet + (1.0 - strength) * dry).astype(np.float32)

"""
Streaming voice activity detection with Silero VAD v5 (https://github.com/snakers4/silero-vad, MIT).

Feeds 512-sample windows at 16 kHz (32 ms) and keeps the model state between calls.
"""

import numpy as np
import onnxruntime


class SileroVad:
    SR = 16000
    WINDOW = 512
    CONTEXT = 64

    def __init__(self, model_path: str):
        so = onnxruntime.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0

    def process(self, x: np.ndarray) -> float:
        """Returns the highest speech probability among the windows completed by this chunk.
        If the chunk completes no window, the previous probability is returned."""
        self.pending = np.concatenate([self.pending, x.astype(np.float32)])
        probs = []
        while self.pending.shape[0] >= self.WINDOW:
            window = self.pending[: self.WINDOW]
            self.pending = self.pending[self.WINDOW :]
            inp = np.concatenate([self.context, window])[None, :]
            out, self.state = self.session.run(
                None,
                {"input": inp, "state": self.state, "sr": np.array(self.SR, dtype=np.int64)},
            )
            self.context = window[-self.CONTEXT :]
            probs.append(float(out.squeeze()))
        if len(probs) > 0:
            self.last_prob = max(probs)
        return self.last_prob

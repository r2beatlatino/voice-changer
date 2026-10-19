"""
Post-processing for f0 tracks (10 ms frames, Hz, 0 = unvoiced).

Pitch extractors misfire on noisy input: short false "voiced" blips, one-frame dropouts
inside words, and octave jumps. Each of these comes out of RVC as a crackle, a break or
a squeak. This cleans those up without touching real pitch movement.
"""

import numpy as np
from scipy.ndimage import median_filter

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def f0_to_coarse(f0: np.ndarray) -> np.ndarray:
    f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(int)


def _runs(mask: np.ndarray):
    """(start, end) of each run of True in mask, end exclusive."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def stabilize_f0(
    f0: np.ndarray,
    min_voiced_frames: int = 3,
    max_gap_frames: int = 4,
    octave_window: int = 11,
    median_frames: int = 3,
) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float64).copy()
    voiced = f0 > 0
    if not voiced.any():
        return f0

    # 1. drop voiced islands too short to be speech (noise picked up as pitch)
    for s, e in _runs(voiced):
        if e - s < min_voiced_frames:
            f0[s:e] = 0
    voiced = f0 > 0
    if not voiced.any():
        return f0

    # 2. bridge short unvoiced gaps between voiced frames (log-linear interpolation)
    for s, e in _runs(~voiced):
        if s == 0 or e == len(f0) or e - s > max_gap_frames:
            continue
        a, b = np.log(f0[s - 1]), np.log(f0[e])
        f0[s:e] = np.exp(np.linspace(a, b, e - s + 2)[1:-1])
    voiced = f0 > 0

    logf = np.zeros_like(f0)
    logf[voiced] = np.log2(f0[voiced])

    # 3. fold octave errors back towards the local median
    for s, e in _runs(voiced):
        seg = logf[s:e]
        ref = median_filter(seg, size=min(octave_window, len(seg)), mode="nearest")
        diff = seg - ref
        up = (diff > 0.75) & (diff < 1.25)
        down = (diff < -0.75) & (diff > -1.25)
        seg[up] -= 1.0
        seg[down] += 1.0

        # 4. light median smoothing for frame-to-frame jitter
        if median_frames > 1 and len(seg) >= median_frames:
            seg = median_filter(seg, size=median_frames, mode="nearest")
        logf[s:e] = seg

    out = np.zeros_like(f0)
    out[voiced] = np.power(2.0, logf[voiced])
    return out


class PitchFloor:
    """Lifts frames far below the speaker's usual pitch up an octave.

    Dropping the voice at the end of a phrase can halve its pitch for a whole word (on a real
    recording, "seis" at 82 Hz between words at 130-146 Hz). RVC copies pitch exactly, so that
    word came out an octave below the rest, close to another speaker's voice. Normal intonation
    stays well above RATIO and is left alone. It works per frame so it stays consistent across
    streamed blocks; stabilize_f0's octave folding then evens out the edges.
    """

    RATIO = 0.65  # about 7.5 semitones below the usual pitch
    HISTORY = 50  # blocks
    MIN_HISTORY = 10

    def __init__(self):
        self.reset()

    def reset(self):
        self.history: list[float] = []

    def __call__(self, f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0, dtype=np.float64).copy()
        voiced = f0 > 0
        if not voiced.any():
            return f0
        if len(self.history) >= self.MIN_HISTORY:
            limit = self.RATIO * float(np.median(self.history))
            # frames lifted earlier come back in the pitch buffer above the limit, so they are
            # not lifted twice; anything below half the limit is more likely a detector error
            low = voiced & (f0 < limit) & (f0 >= limit / 2)
            f0[low] *= 2.0
        self.history.append(float(np.median(f0[voiced])))
        del self.history[: -self.HISTORY]
        return f0

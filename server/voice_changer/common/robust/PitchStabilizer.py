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

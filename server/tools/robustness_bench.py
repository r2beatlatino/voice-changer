"""
Noise robustness bench for the input side of the RVC pipeline.

Mixes a clean speech recording with noise at several SNRs, runs it through the same
streaming front end the server uses (GTCRN denoiser + Silero VAD gate, chunk by chunk),
and measures what the conversion model would receive:

  si_sdr      quality of the signal fed to the embedder vs the clean speech (dB, higher is better)
  gate_recall share of speech frames the gate lets through (higher is better)
  gate_leak   share of non-speech frames the gate lets through (lower is better)
  f0_gpe      pitch frames more than 50 cents off the clean-speech pitch (lower is better)
  f0_vfa      non-speech frames with a detected pitch, i.e. noise turned into voice (lower is better)
  f0_jump     pitch jumps of more than half an octave between frames, per second; heard as squeaks (lower is better)
  f0_blip     voiced runs shorter than 30 ms, per second; heard as crackles (lower is better)
  rtf         front-end processing time / audio time

Usage (from the server directory):
  python tools/robustness_bench.py --clean my_voice.wav [--noise my_room.wav] [--out bench_out]

Without --noise it uses synthetic fan, keyboard, babble and white noise.
"""

import argparse
import os
import sys
import time

import numpy as np
import resampy
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_changer.common.robust.GtcrnDenoiser import GtcrnDenoiser  # noqa: E402
from voice_changer.common.robust.InputConditioner import GTCRN_MODEL, SILERO_MODEL, InputConditioner  # noqa: E402
from voice_changer.common.robust.PitchStabilizer import _runs, stabilize_f0  # noqa: E402

SR = 16000
HOP = 160  # f0 frame = 10 ms, as in RVC


def load_mono_16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = resampy.resample(x, sr, SR)
    return x.astype(np.float32)


def synthetic_noises(n: int, speech: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = np.arange(n) / SR
    white = rng.standard_normal(n)

    # fan: pink-ish rumble + mains hum
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec /= np.maximum(f, 20) ** 0.5
    spec[f > 4000] *= 0.2
    fan = np.fft.irfft(spec, n) + 0.3 * np.sin(2 * np.pi * 120 * t) * np.std(np.fft.irfft(spec, n))

    # keyboard: short decaying clicks at random times
    keyboard = np.zeros(n)
    click_len = int(0.015 * SR)
    env = np.exp(-np.arange(click_len) / (0.002 * SR))
    for pos in rng.integers(0, n - click_len, size=int(n / SR * 8)):
        keyboard[pos : pos + click_len] += rng.standard_normal(click_len) * env * rng.uniform(0.5, 1.0)

    # babble: several reversed, shifted copies of speech (voice-like but unintelligible)
    babble = np.zeros(n)
    rev = speech[::-1]
    for _ in range(5):
        shift = rng.integers(0, len(rev))
        babble += np.resize(np.roll(rev, shift), n) * rng.uniform(0.6, 1.0)

    return {"fan": fan, "keyboard": keyboard, "babble": babble, "white": white}


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, clean.shape[0])
    speech_power = np.mean(clean[np.abs(clean) > 1e-4] ** 2)
    noise_power = np.mean(noise**2) + 1e-12
    gain = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mix = clean + gain * noise
    peak = np.abs(mix).max()
    return (mix / peak * np.abs(clean).max() if peak > 0.99 else mix).astype(np.float32)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return float(10 * np.log10(np.sum(target**2) / (np.sum((est - target) ** 2) + 1e-12)))


def speech_mask(clean: np.ndarray) -> np.ndarray:
    """Per-10ms-frame speech/non-speech from the clean signal's energy."""
    frames = clean[: len(clean) // HOP * HOP].reshape(-1, HOP)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    return db > (db.max() - 40)


def run_front_end(mix: np.ndarray, denoise: bool, vad: bool, chunk: int, vad_threshold: float = 0.3, vad_hangover_ms: float = 500, strong: bool = False):
    cond = InputConditioner()
    outs, gates = [], []
    t0 = time.perf_counter()
    for i in range(0, len(mix), chunk):
        x = mix[i : i + chunk]
        r = cond.process(x, denoise=denoise, denoise_strength=1.0, vad=vad, vad_threshold=vad_threshold, vad_hangover_ms=vad_hangover_ms, strong=strong)
        outs.append(r.audio)
        gates.append(np.full(len(x), r.gate))
    elapsed = time.perf_counter() - t0
    out = np.concatenate(outs)
    gate = np.concatenate(gates)
    used = cond.strong_denoiser if strong and cond.strong_denoiser is not None else GtcrnDenoiser
    delay = used.LATENCY_SAMPLES if denoise else 0
    # undo the fixed latency so the output lines up with the clean reference
    out = np.concatenate([out[delay:], np.zeros(delay, dtype=np.float32)])
    gate = np.concatenate([gate[delay:], np.zeros(delay)])
    return out, gate, elapsed / (len(mix) / SR)


class F0:
    def __init__(self):
        import torch
        import torchfcpe

        self.torch = torch
        self.model = torchfcpe.spawn_bundled_infer_model("cpu")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            f0 = self.model.infer(self.torch.from_numpy(x).float()[None, :, None], sr=SR, decoder_mode="local_argmax", threshold=0.006)
        return f0.squeeze().cpu().numpy().astype(np.float64)


def f0_errors(f0: np.ndarray, ref: np.ndarray, nonspeech: np.ndarray):
    n = min(len(f0), len(ref), len(nonspeech))
    f0, ref, nonspeech = f0[:n], ref[:n], nonspeech[:n]
    both = (f0 > 0) & (ref > 0)
    cents = 1200 * np.abs(np.log2(np.maximum(f0[both], 1e-3) / np.maximum(ref[both], 1e-3)))
    gpe = float(np.mean(cents > 50)) if both.any() else 0.0
    vfa = float(np.mean(f0[nonspeech] > 0)) if nonspeech.any() else 0.0
    seconds = n / 100
    logf = np.log2(np.where(f0 > 0, f0, np.nan))
    jumps = float(np.nansum(np.abs(np.diff(logf)) > 0.5) / seconds)
    blips = sum(1 for s, e in _runs(f0 > 0) if e - s < 3) / seconds
    return gpe, vfa, jumps, blips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="test.wav")
    parser.add_argument("--noise", default=None, help="noise recording; synthetic noises if omitted")
    parser.add_argument("--snr", default="0,5,10,20")
    parser.add_argument("--chunk-ms", type=float, default=100)
    parser.add_argument("--peak-db", type=float, default=None, help="rescale the clean speech to this peak level (dBFS), e.g. -14 for a quiet mic")
    parser.add_argument("--vad-threshold", type=float, default=0.3)
    parser.add_argument("--vad-hangover-ms", type=float, default=500)
    parser.add_argument("--strong", action="store_true", help="use DeepFilterNet3 instead of GTCRN (needs deepfilternet)")
    parser.add_argument("--no-f0", action="store_true", help="skip pitch metrics (needs torchfcpe)")
    parser.add_argument("--out", default=None, help="write the mixed/processed wavs here")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    clean = load_mono_16k(args.clean)
    if args.peak_db is not None:
        clean = (clean / (np.abs(clean).max() + 1e-9) * 10 ** (args.peak_db / 20)).astype(np.float32)
    # pad with silence so gate leakage is measured on real pauses
    pad = np.zeros(SR, dtype=np.float32)
    clean = np.concatenate([pad, clean, pad])
    noises = {"custom": load_mono_16k(args.noise)} if args.noise else synthetic_noises(len(clean), clean, rng)
    snrs = [float(s) for s in args.snr.split(",")]
    chunk = int(args.chunk_ms * SR / 1000)

    mask = speech_mask(clean)
    sample_mask = np.repeat(mask, HOP)
    sample_mask = np.concatenate([sample_mask, np.zeros(len(clean) - len(sample_mask), dtype=bool)])

    f0 = None if args.no_f0 else F0()
    ref_f0 = f0(clean) if f0 else None
    if f0:
        k = min(len(ref_f0), len(mask))
        ref_f0[k:] = 0
        ref_f0[:k][~mask[:k]] = 0  # pitch only counts where there is speech

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    header = f"{'noise':9} {'snr':>4} {'mode':16} {'si_sdr':>7} {'g_recall':>8} {'g_leak':>7} {'f0_gpe':>7} {'f0_vfa':>7} {'f0_jump':>7} {'f0_blip':>7} {'rtf':>6}"
    print(header)
    print("-" * len(header))
    for name, noise in noises.items():
        for snr in snrs:
            mix = mix_at_snr(clean, noise, snr)
            modes = [("raw", False, False, False), ("denoise", True, False, False), ("denoise+vad", True, True, False), ("denoise+vad+f0fix", True, True, True)]
            for label, dn, vd, fix in modes:
                out, gate, rtf = run_front_end(mix, dn, vd, chunk, args.vad_threshold, args.vad_hangover_ms, args.strong)
                sdr = si_sdr(out * gate, clean)
                g_recall = float(np.mean(gate[sample_mask] > 0))
                g_leak = float(np.mean(gate[~sample_mask] > 0))
                gpe = vfa = jumps = blips = float("nan")
                if f0:
                    est = f0(out)
                    gate_frames = gate[: len(gate) // HOP * HOP].reshape(-1, HOP).max(axis=1)
                    k = min(len(est), len(gate_frames))
                    est[k:] = 0
                    est[:k][gate_frames[:k] == 0] = 0  # gated frames never reach the model
                    if fix:
                        est = stabilize_f0(est)
                    gpe, vfa, jumps, blips = f0_errors(est, ref_f0, ~mask)
                print(f"{name:9} {snr:4.0f} {label:16} {sdr:7.2f} {g_recall:8.2f} {g_leak:7.2f} {gpe:7.3f} {vfa:7.3f} {jumps:7.2f} {blips:7.2f} {rtf:6.3f}")
                if args.out:
                    sf.write(os.path.join(args.out, f"{name}_{snr:.0f}dB_{label}.wav"), out * gate, SR)
            print()


if __name__ == "__main__":
    main()

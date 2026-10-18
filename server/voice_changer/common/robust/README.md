# Noise-robust front end (RVC)

Cleans the microphone signal before RVC turns it into a voice:

```
mic -> GTCRN denoiser -> Silero VAD gate -> ContentVec/HuBERT + pitch extractor -> pitch stabilizer -> RVC
```

Everything here runs on the CPU (about 8% of one core in total), so it does not take
GPU time away from the conversion. It works in both client-audio and server-audio mode;
the browser's Sup1/Sup2 switches only apply to client audio.

| Setting (UI) | Key | Default | What it does |
|---|---|---|---|
| Denoise checkbox | `denoise` | on | GTCRN speech enhancement on the input. Adds 32 ms of latency |
| Denoise slider | `denoiseStrength` | 1.0 | Mix between raw mic (0) and fully denoised (1). Lower it if your voice sounds thin |
| VAD checkbox | `vad` | on | Mutes the output, and skips inference, while you are not talking |
| VAD slider | `vadThreshold` | 0.5 | Speech probability needed to open the gate. Raise it in loud rooms, lower it if word starts get cut |
| — | `vadHangoverMs` | 300 | How long the gate stays open after speech stops |
| F0 Fix | `f0Stabilize` | on | Removes pitch blips, bridges short dropouts and folds octave jumps |

## Models

Both models are bundled in `models/` (MIT licensed, see the LICENSE files):

- `gtcrn_simple.onnx`: streaming GTCRN trained on DNS3, from https://github.com/Xiaobin-Rong/gtcrn (commit 502ebfa)
- `silero_vad.onnx`: Silero VAD v5, from https://github.com/snakers4/silero-vad (commit 1e261b0)

If either fails to load, the server logs a warning and carries on without it.

## Measuring

```
cd server
python tools/robustness_bench.py --clean my_voice.wav            # synthetic fan/keyboard/babble/white noise
python tools/robustness_bench.py --clean my_voice.wav --noise my_room.wav --out bench_out
python -m pytest tests -q
```

The bench measures what the conversion model receives (input quality, gate behaviour and
pitch errors), so it needs no RVC model and runs on CPU.

Limits: babble (other people talking) is speech, so neither the denoiser nor the VAD can
remove it reliably. A strong steady hum near your own pitch can still be picked up as pitch
once the broadband noise around it is removed; the VAD gate covers it while you are silent.

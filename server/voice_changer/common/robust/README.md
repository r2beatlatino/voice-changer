# Noise-robust front end (RVC and LLVC)

Cleans the microphone signal before RVC turns it into a voice:

```
mic -> GTCRN denoiser -> Silero VAD gate -> ContentVec/HuBERT + pitch extractor -> pitch stabilizer -> RVC
```

Everything here runs on the CPU (about 8% of one core in total), so it does not take
GPU time away from the conversion. It works in both client-audio and server-audio mode;
the browser's Sup1/Sup2 switches only apply to client audio.

LLVC uses the same front end and settings, except that `denoiseStrong` defaults to on: the
pretrained LLVC model loses voicing on noisy input, which sounds choppy, and DeepFilterNet3 in
front of it fixed that on a real recording (Whisper word error rate 2.3%). LLVC keeps running
while the VAD gate is closed (it is a streaming model), it just hears silence.

| Setting (UI) | Key | Default | What it does |
|---|---|---|---|
| Denoise checkbox | `denoise` | on | 85 Hz high-pass (removes fan/AC/desk rumble) + GTCRN speech enhancement. Adds 32 ms of latency |
| Strong | `denoiseStrong` | off | Use DeepFilterNet3 instead of GTCRN: about 9 dB more noise removed on a real rumble recording, about 9 ms per block on an RTX 3090 (13-18 ms on CPU). Loads on first use (a short gap in the audio); falls back to GTCRN if `deepfilternet` or its weights are missing |
| Denoise slider | `denoiseStrength` | 1.0 | Mix between raw mic (0) and fully denoised (1). Lower it if your voice sounds thin |
| VAD checkbox | `vad` | on | Mutes the output, and skips inference (RVC), while you are not talking |
| VAD slider | `vadThreshold` | 0.3 | Speech probability needed to open the gate. Raise it in loud rooms, lower it (or turn VAD off) if words get cut |
| — | `vadHangoverMs` | 500 | How long the gate stays open after speech stops |
| F0 Fix | `f0Stabilize` | on | Removes pitch blips, bridges short dropouts and folds octave jumps |
| — | `f0Floor` | on | Lifts frames more than about 7.5 semitones below your usual pitch (the median of the last few seconds) up an octave. Dropping the voice at the end of a phrase otherwise made that word come out an octave low |

Each voice (model slot) remembers its own pitch, index, protect and noise settings in
`stored_setting.json`, so they survive a restart. The "Perfil" menu sets the noise controls in
one click: Casa tranquila (GTCRN, VAD 0.3), Con ruido (Strong, VAD 0.4), Mucho ruido (Strong, VAD 0.5).

The "Asistente" button (next to Perfil) records 10 s of the room and 15 s of the user reading,
and `POST /calibrate` (`Calibration.py`) suggests the settings: Strong when the voice is less than
35 dB above the noise, the highest VAD threshold that still passes 95% of the speech, and a pitch
shift from the user's median pitch to a typical female (260 Hz) or male (125 Hz) voice. It also
flags a quiet or clipping mic, low rumble and speech-like background noise.
`POST /convert_clip` converts the reading block by block so it can be heard before and after.

## Models

Both models are bundled in `models/` (MIT licensed, see the LICENSE files):

- `gtcrn_simple.onnx`: streaming GTCRN trained on DNS3, from https://github.com/Xiaobin-Rong/gtcrn (commit 502ebfa)
- `silero_vad.onnx`: Silero VAD v5, from https://github.com/snakers4/silero-vad (commit 1e261b0)

If either fails to load, the server logs a warning and carries on without it.

Silero's confidence depends on input level, so the VAD hears a copy normalised towards a
0.5 peak (up to +24 dB, 3 s release). The audio sent to the model is not changed.

## Measuring

```
cd server
python tools/robustness_bench.py --clean my_voice.wav            # synthetic fan/keyboard/babble/white noise
python tools/robustness_bench.py --clean my_voice.wav --noise my_room.wav --out bench_out
python tools/robustness_bench.py --clean my_voice.wav --peak-db -26 --vad-threshold 0.3   # quiet mic
python -m pytest tests -q
```

The bench measures what the conversion model receives (input quality, gate behaviour and
pitch errors), so it needs no RVC model and runs on CPU.

Limits: babble (other people talking) is speech, so neither the denoiser nor the VAD can
remove it reliably. A strong steady hum near your own pitch can still be picked up as pitch
once the broadband noise around it is removed; the VAD gate covers it while you are silent.

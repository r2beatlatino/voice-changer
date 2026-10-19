# Portable Windows package

Builds "Cambiador de voz": one folder (and zip) with its own Python runtime, the app, the
DeepFilterNet3 weights and `VoiceChanger.exe`, so it runs on a PC with nothing installed.
The base models (HuBERT, RMVPE, ...) are downloaded on the first run.

Needs, on the build PC: this repo with `server\.venv` (Python 3.10 made from python-build-standalone,
everything in requirements installed), the DeepFilterNet3 weights in its cache (run the server once
with Strong on), and .NET Framework 4 (for `csc.exe`, part of Windows).

```
powershell -ExecutionPolicy Bypass -File packaging\windows\build_portable.ps1
```

Output: `..\build\out\CambiadorDeVoz\` and `CambiadorDeVoz_portable_<commit>.zip` next to the repo.
Only committed files go in (`git archive HEAD`). `-IncludePretrain` also bundles the base models,
which are GPL or non-commercial: only for personal use.

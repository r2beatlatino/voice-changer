# Portable Windows package

Builds "Cambiador de voz": one folder (and zip) with its own Python runtime, the app, the
DeepFilterNet3 weights and `VoiceChanger.exe`, so it runs on a PC with nothing installed.
`VoiceChanger.exe` shows the interface in its own window (WebView2, part of Windows 10/11; the
script downloads the WebView2 SDK from NuGet) and falls back to an Edge window without it.
The base models (HuBERT, RMVPE, ...) are downloaded on the first run.

Needs, on the build PC: this repo with `server\.venv` (Python 3.10 made from python-build-standalone,
everything in requirements installed), the DeepFilterNet3 weights in its cache (run the server once
with Strong on), and .NET Framework 4 (for `csc.exe`, part of Windows).

Run it from Windows PowerShell, not Git Bash (Git Bash's tar can't extract to `D:\...`):

```
powershell -ExecutionPolicy Bypass -File packaging\windows\build_portable.ps1
```

Output: `..\build\out\CambiadorDeVoz\` and `CambiadorDeVoz_portable_<commit>.zip` next to the repo.
Only committed files go in (`git archive HEAD`). `-IncludePretrain` also bundles the base models,
which are GPL or non-commercial: only for personal use.

The folder is about 6 GB (35,000 files) and the zip about 3.5 GB, which is over GitHub's 2 GB
limit for a release file: split it (for example 7-Zip volumes) to publish it there.

To check it, extract the zip into an empty folder and open `VoiceChanger.exe`. The launcher
writes the server log to `datos\registro_servidor.txt`. To drive the window from a script, start
it with `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9333` and connect over the
Chrome DevTools protocol.

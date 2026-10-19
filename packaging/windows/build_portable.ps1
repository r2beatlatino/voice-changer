# Builds the portable "Cambiador de voz" folder and zip.
#   powershell -ExecutionPolicy Bypass -File build_portable.ps1 [-IncludePretrain] [-NoZip]
# Inputs: the repo checkout (git HEAD), its server\.venv (Python 3.10 + everything installed),
# the python-build-standalone interpreter that venv was made from, and the DeepFilterNet3 weights.
param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$Out = "",
    [string]$Name = "CambiadorDeVoz",
    [switch]$IncludePretrain,   # bundle server\pretrain (GPL / non-commercial weights: personal use only)
    [switch]$NoZip,
    # the WebView2 SDK (NuGet) for the program's own window; the runtime itself comes with Windows 10/11
    [string]$WebView2Version = "1.0.2651.64"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
if (-not $Out) { $Out = Join-Path (Split-Path $Repo -Parent) "build\out" }
$tarExe = Join-Path $env:WINDIR "System32\tar.exe"
$dst = Join-Path $Out $Name
$venv = Join-Path $Repo "server\.venv"
$site = Join-Path $venv "Lib\site-packages"

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Robo($from, $to, [string[]]$extra) {
    & robocopy $from $to /E /MT:16 /NFL /NDL /NJH /NJS /NP @extra | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $from -> $to" }
}

Step "Checks"
$home310 = (Get-Content (Join-Path $venv "pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' }) -replace '^home\s*=\s*', ''
$pyHome = (Get-Item $home310).Target; if (-not $pyHome) { $pyHome = $home310 }; if ($pyHome -is [array]) { $pyHome = $pyHome[0] }
if (-not (Test-Path (Join-Path $pyHome "python.exe"))) { throw "python-build-standalone not found at $pyHome" }
$dfn = Join-Path $env:LOCALAPPDATA "DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3"
if (-not (Test-Path (Join-Path $dfn "config.ini"))) { throw "DeepFilterNet3 weights not found at $dfn (run the server once with Strong on)" }
$commit = (& git -C $Repo rev-parse --short HEAD).Trim()
$dirty = (& git -C $Repo status --porcelain --untracked-files=no)
if ($dirty) { Write-Warning "Repo has uncommitted changes; the package uses HEAD ($commit) only" }
Write-Host "python: $pyHome`ncommit: $commit"

Step "Clean $dst"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Force $dst | Out-Null

Step "Runtime (python-build-standalone + site-packages)"
Robo $pyHome (Join-Path $dst "runtime") @()
Robo $site (Join-Path $dst "runtime\Lib\site-packages") @("/XF", "_virtualenv.pth", "_virtualenv.py")
# not needed at run time: C/C++ headers and import libraries (~0.9 GB)
$rsite = Join-Path $dst "runtime\Lib\site-packages"
Remove-Item -Recurse -Force (Join-Path $rsite "torch\include") -ErrorAction SilentlyContinue
Get-ChildItem $rsite -Recurse -Filter *.lib -File | Remove-Item -Force
# app-local Visual C++ runtime: torch/onnxruntime import MSVCP140*.dll and VCOMP140.dll from the system
foreach ($n in "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll", "vcomp140.dll") {
    Copy-Item (Join-Path $env:WINDIR "System32\$n") (Join-Path $dst "runtime\$n") -Force
}

Step "App (git archive ${commit}: server + client/demo/dist)"
$app = Join-Path $dst "app"; New-Item -ItemType Directory -Force $app | Out-Null
$tarFile = Join-Path $Out "app_$commit.tar"
& git -C $Repo archive --format=tar -o $tarFile HEAD server client/demo/dist
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
# Windows' own bsdtar: Git Bash's GNU tar reads "D:\..." as host:path and fails
& $tarExe -xf $tarFile -C $app
if ($LASTEXITCODE -ne 0) { throw "tar extract failed" }
Remove-Item $tarFile
foreach ($f in "server\MMVCServerSIO.py", "client\demo\dist\index.js", "server\voice_changer\common\robust\models\gtcrn_simple.onnx") {
    if (-not (Test-Path (Join-Path $app $f))) { throw "missing after extract: $f" }
}
Set-Content -Path (Join-Path $app "VERSION.txt") -Value "commit $commit`nbuilt $(Get-Date -Format s)" -Encoding UTF8
if ($IncludePretrain) {
    Robo (Join-Path $Repo "server\pretrain") (Join-Path $app "server\pretrain") @()
}

Step "Models (DeepFilterNet3)"
Robo $dfn (Join-Path $dst "modelos\DeepFilterNet3") @("/XF", "*.log")

Step "WebView2 SDK $WebView2Version"
$wv = Join-Path $Out "webview2_$WebView2Version"
if (-not (Test-Path (Join-Path $wv "WebView2Loader.dll"))) {
    $pkg = Join-Path $Out "webview2_$WebView2Version.zip"
    $tmp = Join-Path $Out "webview2_pkg"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/$WebView2Version" -OutFile $pkg -UseBasicParsing
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive $pkg $tmp
    New-Item -ItemType Directory -Force $wv | Out-Null
    foreach ($n in "Microsoft.Web.WebView2.Core.dll", "Microsoft.Web.WebView2.WinForms.dll") {
        $f = Get-ChildItem $tmp -Recurse -Filter $n | Where-Object { $_.FullName -match '\\net4[0-9]*\\' } | Select-Object -First 1
        if (-not $f) { throw "$n (.NET Framework) not found in the WebView2 package" }
        Copy-Item $f.FullName $wv
    }
    Copy-Item (Join-Path $tmp "runtimes\win-x64\native\WebView2Loader.dll") $wv
    Remove-Item -Recurse -Force $tmp, $pkg
}

Step "Launcher"
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$exe = Join-Path $dst "VoiceChanger.exe"
$src = Join-Path $here "launcher\VoiceChanger.cs"
$cscArgs = @("-nologo", "-codepage:65001", "-target:winexe", "-r:System.Windows.Forms.dll", "-r:System.Drawing.dll",
             "-r:$(Join-Path $wv 'Microsoft.Web.WebView2.Core.dll')", "-r:$(Join-Path $wv 'Microsoft.Web.WebView2.WinForms.dll')")
& $csc @cscArgs "-out:$exe" $src
if ($LASTEXITCODE -ne 0) { throw "launcher build failed" }
Copy-Item (Join-Path $wv "*.dll") $dst
# the launcher draws its own icon; build it again with that icon so Explorer and the taskbar show it
$ico = Join-Path $Out "VoiceChanger.ico"
Start-Process -FilePath $exe -ArgumentList "--write-icon", "`"$ico`"" -Wait
if (Test-Path $ico) {
    & $csc @cscArgs "-win32icon:$ico" "-out:$exe" $src
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "launcher build with icon failed; building without it"
        & $csc @cscArgs "-out:$exe" $src
        if ($LASTEXITCODE -ne 0) { throw "launcher build failed" }
    }
} else { Write-Warning "icon not written; the .exe keeps the default icon" }
Copy-Item (Join-Path $here "LEEME.txt") (Join-Path $dst "LEEME.txt")

Step "Licenses"
$lic = Join-Path $dst "LICENCIAS"; New-Item -ItemType Directory -Force $lic | Out-Null
Copy-Item (Join-Path $Repo "LICENSE") (Join-Path $lic "w-okada-voice-changer_LICENSE.txt")
Copy-Item (Join-Path $Repo "LICENSE-NOTICE") (Join-Path $lic "w-okada-voice-changer_LICENSE-NOTICE.txt")
Copy-Item (Join-Path $pyHome "LICENSE.txt") (Join-Path $lic "python_LICENSE.txt")
Get-ChildItem (Join-Path $Repo "server\voice_changer\common\robust\models") -Filter "LICENSE-*" | ForEach-Object { Copy-Item $_.FullName (Join-Path $lic ($_.Name + ".txt")) }
$pk = Join-Path $lic "paquetes-python"; New-Item -ItemType Directory -Force $pk | Out-Null
Get-ChildItem $rsite -Directory -Filter "*.dist-info" | ForEach-Object {
    $files = Get-ChildItem $_.FullName -Recurse -File | Where-Object { $_.Name -match '^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)' }
    if ($files) { $d = Join-Path $pk ($_.Name -replace '\.dist-info$', ''); New-Item -ItemType Directory -Force $d | Out-Null; $files | ForEach-Object { Copy-Item $_.FullName $d -Force } }
}
@"
Componentes incluidos y sus licencias
- w-okada voice-changer: MIT (w-okada-voice-changer_LICENSE.txt, más el aviso LICENSE-NOTICE)
- Python (python-build-standalone): PSF (python_LICENSE.txt)
- Microsoft Edge WebView2 SDK (Microsoft.Web.WebView2.*.dll, WebView2Loader.dll): licencia BSD de Microsoft
- Paquetes de Python (torch, onnxruntime, fairseq, faiss, DeepFilterNet, ...): ver paquetes-python\
- GTCRN y Silero VAD (modelos de limpieza y detección de voz): ver LICENSE-gtcrn / LICENSE-silero-vad
- DeepFilterNet3 (modelo de limpieza fuerte, carpeta modelos\DeepFilterNet3): MIT / Apache-2.0
- Microsoft Visual C++ Runtime (runtime\msvcp140*.dll, vcruntime140*.dll, vcomp140.dll, concrt140.dll):
  archivos redistribuibles de Microsoft Visual C++.

Modelos base que se descargan la primera vez (no incluidos en este paquete salvo que se indique):
HuBERT base, rinna japanese-hubert, ContentVec (ONNX, GPL), RMVPE (ONNX, GPL), CREPE, Whisper tiny,
NSF-HiFiGAN (DiffSinger community vocoder, CC BY-NC-SA 4.0). Cada uno tiene su propia licencia.
"@ | Set-Content -Path (Join-Path $lic "LEEME_LICENCIAS.txt") -Encoding UTF8

Step "Size"
$sz = (Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum
"{0:N2} GB in {1}" -f ($sz / 1GB), $dst

if (-not $NoZip) {
    Step "Zip"
    $zip = Join-Path $Out "${Name}_portable_$commit.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    $t0 = Get-Date
    & $tarExe -a -c -f $zip -C $Out $Name
    if ($LASTEXITCODE -ne 0) { throw "zip failed" }
    "{0:N2} GB zip in {1:N0} s: {2}" -f ((Get-Item $zip).Length / 1GB), ((Get-Date) - $t0).TotalSeconds, $zip
}

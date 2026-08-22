# Install the engine on this machine, so the app has a Local mode.
#
# Why bother when a pod exists: a GPU pod is a poor development loop. It costs
# money while you think, it disappears and takes its 20 GB of weights with it,
# and every edit is a push-pull-restart round trip. Local mode is slower per
# render and immediate for everything else.
#
# What this installs is deliberately the *small* tier -- acestep-v15-turbo
# (0.6B) plus the 0.6B LM -- because that is what fits an 8 GB laptop GPU
# comfortably. The XL checkpoints are 20 GB each and want far more VRAM; those
# stay a cloud-mode thing, and the app says so rather than offering a quality
# tier this machine cannot render.
#
#   pwsh -ExecutionPolicy Bypass -File tools\install-local.ps1
#   ... -Root D:\musicmaker\local     # where it goes (default)
#   ... -SkipModels                   # code and venv only
#
# Resumable: every step checks whether it is already done, so a failed or
# interrupted run can simply be repeated.

[CmdletBinding()]
param(
    [string] $Root = "D:\musicmaker\local",
    [switch] $SkipModels
)

$ErrorActionPreference = "Stop"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Note($text) { Write-Host "    $text" -ForegroundColor DarkGray }
function Good($text) { Write-Host "    $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
Step "Checking what this machine has"

$missing = @()
foreach ($tool in @("uv", "git", "ffmpeg")) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Good "$tool ok"
    } else {
        $missing += $tool
        Warn "$tool MISSING"
    }
}
if ($missing.Count) {
    Write-Host "`nInstall these first:" -ForegroundColor Red
    if ($missing -contains "uv") {
        Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    }
    if ($missing -contains "git")    { Write-Host "  winget install Git.Git" }
    if ($missing -contains "ffmpeg") { Write-Host "  winget install Gyan.FFmpeg" }
    exit 1
}

# The GPU decides whether this is worth doing at all.
$vram = 0
try {
    $line = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) 2>$null
    if ($line) {
        Good "GPU: $line"
        if ($line -match '(\d+)\s*MiB') { $vram = [int]$Matches[1] }
    }
} catch { }

if ($vram -eq 0) {
    Warn "No NVIDIA GPU found. ACE-Step will fall back to CPU, which renders"
    Warn "a few seconds of audio in minutes rather than seconds. Usable for"
    Warn "wiring up the app; not for listening to results."
} elseif ($vram -lt 6000) {
    Warn "$vram MiB of VRAM. Below the ~6 GB the LM + DiT path wants, so the"
    Warn "local engine will run DiT-only and lyrics handling will be weaker."
} else {
    Good "$vram MiB of VRAM -- enough for the turbo tier with the LM."
}

$free = [math]::Round((Get-PSDrive -Name $Root.Substring(0,1)).Free / 1GB)
Note "$free GB free on $($Root.Substring(0,1)):  (about 12 GB is needed)"
if ($free -lt 14) { Write-Host "`nNot enough disk space." -ForegroundColor Red; exit 1 }

New-Item -ItemType Directory -Force -Path $Root | Out-Null
$acestep = Join-Path $Root "ACE-Step-1.5"

# ---------------------------------------------------------------------------
Step "ACE-Step"

if (Test-Path (Join-Path $acestep ".git")) {
    Good "already cloned at $acestep"
} else {
    Note "cloning (shallow -- the history is not wanted)"
    & git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git $acestep
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    Good "cloned"
}

Push-Location $acestep
try {
    if (Test-Path (Join-Path $acestep ".venv\Scripts\python.exe")) {
        Good "virtualenv already built"
    } else {
        Note "uv sync -- this pulls CUDA torch and takes a while (~3 GB)"
        & uv sync
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
        Good "virtualenv built"
    }
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
Step "Models"

$ckpt = Join-Path $acestep "checkpoints"
New-Item -ItemType Directory -Force -Path $ckpt | Out-Null

if ($SkipModels) {
    Warn "skipped (-SkipModels). The engine will download on first render."
} else {
    # The small tier, and only the small tier. Each is fetched into the
    # directory ACE-Step's registry reads, and skipped if already present.
    $wanted = @(
        @{ repo = "ACE-Step/Ace-Step1.5"; sub = ""; note = "turbo DiT, VAE, text encoder (~7 GB)" }
    )
    $py = Join-Path $acestep ".venv\Scripts\python.exe"

    # hf_transfer is the difference between ~18 MB/s and ~0.4 MB/s on an 8 GB
    # download, so it is worth installing -- but not worth failing over.
    Note "ensuring hf_transfer (fast downloads)"
    # No 2>&1 here. In Windows PowerShell, redirecting a native command's stderr
    # wraps every line in an ErrorRecord, which under ErrorActionPreference=Stop
    # turns uv's ordinary progress chatter into a fatal error. The exit code is
    # the thing to check, and even that is advisory: a missing hf_transfer only
    # makes the download slower, and the Python side already copes.
    & uv pip install --python $py hf_transfer | Out-Null
    if ($LASTEXITCODE -ne 0) { Warn "could not install hf_transfer -- downloads will be slower" }

    $script = @'
import os, sys, pathlib
try:
    import hf_transfer                      # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    # Asking for the fast path without the package installed is a hard error
    # inside huggingface_hub, so the flag is only set when it can be honoured.
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    print("  (hf_transfer unavailable -- downloading the slow way)")
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub missing from the ACE-Step venv")

dest = pathlib.Path(sys.argv[1])
# Only the pieces the small tier needs. The base repo also carries the 1.7B LM,
# which is 3.6 GB and not required when the 0.6B one is used.
patterns = ["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*",
            "config.json", "configuration.json", "README.md", ".gitattributes"]
p = snapshot_download(repo_id="ACE-Step/Ace-Step1.5", local_dir=str(dest),
                      allow_patterns=patterns)
size = sum(f.stat().st_size for f in pathlib.Path(p).rglob("*") if f.is_file())
print(f"  core models: {size/1e9:.2f} GB")

lm = dest / "acestep-5Hz-lm-0.6B"
if not lm.exists():
    q = snapshot_download(repo_id="ACE-Step/acestep-5Hz-lm-0.6B", local_dir=str(lm),
                          allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt",
                                          "*.model", "*.pt"])
    size = sum(f.stat().st_size for f in pathlib.Path(q).rglob("*") if f.is_file())
    print(f"  0.6B LM:     {size/1e9:.2f} GB")
else:
    print("  0.6B LM:     already present")
'@
    $tmp = Join-Path $env:TEMP "mm_fetch_models.py"
    Set-Content -Path $tmp -Value $script -Encoding utf8
    & $py $tmp $ckpt
    if ($LASTEXITCODE -ne 0) { throw "model download failed" }
    Good "models in place"
}

# ---------------------------------------------------------------------------
Step "musicmaker server"

$repo   = Split-Path -Parent $PSScriptRoot
$server = Join-Path $repo "server"

# Into ACE-Step's virtualenv, not a second one.
#
# It already carries CUDA torch, torchaudio, soundfile, transformers, fastapi,
# pydantic, numpy and uvicorn -- nearly the whole server requirement -- so a
# separate venv would mean a second 3 GB copy of torch to no purpose. What is
# actually missing is demucs, librosa and mido, which are small once torch is
# present. It also means separation and export work locally rather than failing
# on an import the first time somebody splits a take.
$py = Join-Path $acestep ".venv\Scripts\python.exe"

Note "installing the server requirements"
& uv pip install --python $py -r (Join-Path $server "requirements.txt") | Out-Null
if ($LASTEXITCODE -ne 0) { throw "server requirements failed" }

Note "installing the audio extras (demucs, librosa, mido)"
# Deliberately not the whole of requirements-pod.txt: diffusers, accelerate and
# transformers are ACE-Step's business and already pinned in this venv, and
# re-resolving them here is how a working install gets broken.
& uv pip install --python $py demucs librosa mido | Out-Null
if ($LASTEXITCODE -ne 0) { Warn "audio extras failed -- splitting will not work locally" }
Good "server ready"

# ---------------------------------------------------------------------------
Step "Recording where everything went"

# start-local.ps1 reads this rather than guessing, so moving the install only
# means editing one file.
$conf = @{
    root        = $Root
    acestep_dir = $acestep
    server_venv = Join-Path $acestep ".venv"
    repo        = $repo
    vram_mib    = $vram
} | ConvertTo-Json
# UTF-8 without a BOM. PowerShell 5.1's -Encoding utf8 writes one, and a BOM
# in front of a JSON document is a parse error for anything reading it as plain
# UTF-8 -- which is every other tool that might want to look at this file.
[System.IO.File]::WriteAllText((Join-Path $Root "local.json"), $conf,
                               (New-Object System.Text.UTF8Encoding($false)))
Good (Join-Path $Root "local.json")

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Start it with:  pwsh -File tools\start-local.ps1"
Write-Host "Then in the app, Settings -> Local."

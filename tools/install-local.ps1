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

    $script = @'
import os, sys, pathlib
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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
$venv   = Join-Path $Root "server-venv"

if (Test-Path (Join-Path $venv "Scripts\python.exe")) {
    Good "server virtualenv already built"
} else {
    Note "creating a 3.12 virtualenv for the API server"
    & uv venv --python 3.12 $venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}
Note "installing server requirements"
& uv pip install --python (Join-Path $venv "Scripts\python.exe") `
    -r (Join-Path $server "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "server requirements failed" }
Good "server ready"

# ---------------------------------------------------------------------------
Step "Recording where everything went"

# start-local.ps1 reads this rather than guessing, so moving the install only
# means editing one file.
$conf = @{
    root        = $Root
    acestep_dir = $acestep
    server_venv = $venv
    repo        = $repo
    vram_mib    = $vram
} | ConvertTo-Json
Set-Content -Path (Join-Path $Root "local.json") -Value $conf -Encoding utf8
Good (Join-Path $Root "local.json")

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Start it with:  pwsh -File tools\start-local.ps1"
Write-Host "Then in the app, Settings -> Local."

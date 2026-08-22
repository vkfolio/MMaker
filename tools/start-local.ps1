# Start the local engine: ACE-Step on :8001, the musicmaker API on :8000.
#
# Deliberately the same two processes and the same ports as the pod, because
# the desktop app talks to a URL and nothing else. Local mode is then just a
# different URL, not a different code path -- which is the whole reason this is
# cheap to have.
#
#   pwsh -ExecutionPolicy Bypass -File tools\start-local.ps1
#   ... -Stop        # stop whatever this started
#   ... -Status      # is it up?

[CmdletBinding()]
param(
    [string] $Root = "D:\musicmaker\local",
    [int]    $Port = 8000,
    [switch] $Stop,
    [switch] $Status
)

$ErrorActionPreference = "Stop"
$conf = Join-Path $Root "local.json"
if (-not (Test-Path $conf)) {
    Write-Host "No local install found at $Root." -ForegroundColor Red
    Write-Host "Run:  pwsh -File tools\install-local.ps1"
    exit 1
}
$c = Get-Content $conf -Raw | ConvertFrom-Json
$logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Running($pattern) {
    Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$pattern*" }
}

if ($Status) {
    $ace = Running "acestep-api"
    $api = Running "app.main"
    "ACE-Step  : " + $(if ($ace) { "running (pid $($ace.ProcessId))" } else { "stopped" })
    "musicmaker: " + $(if ($api) { "running (pid $($api.ProcessId))" } else { "stopped" })
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 4
        "health    : $($h.status)"
    } catch { "health    : unreachable" }
    exit 0
}

if ($Stop) {
    foreach ($p in @(Running "acestep-api") + @(Running "app.main")) {
        Write-Host "stopping pid $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "stopped." -ForegroundColor Green
    exit 0
}

# --- ACE-Step ---------------------------------------------------------------
if (Running "acestep-api") {
    Write-Host "ACE-Step already running" -ForegroundColor DarkGray
} else {
    Write-Host "starting ACE-Step on :8001" -ForegroundColor Cyan
    # Only the checkpoint this machine has. Registering a slot for weights that
    # are not here is how a quality tier silently renders as something else --
    # the pod taught that lesson the expensive way.
    $env:ACESTEP_CONFIG_PATH = "acestep-v15-turbo"
    Remove-Item Env:\ACESTEP_CONFIG_PATH2 -ErrorAction SilentlyContinue
    Remove-Item Env:\ACESTEP_CONFIG_PATH3 -ErrorAction SilentlyContinue
    # vLLM has no Windows build; the torch backend is the one that runs here.
    $env:ACESTEP_LM_BACKEND = "pt"
    $env:ACESTEP_LM_MODEL_PATH = "acestep-5Hz-lm-0.6B"

    $exe = Join-Path $c.acestep_dir ".venv\Scripts\acestep-api.exe"
    if (-not (Test-Path $exe)) { throw "acestep-api not found -- rerun install-local.ps1" }
    Start-Process -FilePath $exe `
        -ArgumentList @("--host", "127.0.0.1", "--port", "8001") `
        -WorkingDirectory $c.acestep_dir `
        -RedirectStandardOutput (Join-Path $logs "acestep.log") `
        -RedirectStandardError  (Join-Path $logs "acestep.err.log") `
        -WindowStyle Hidden
}

# --- musicmaker -------------------------------------------------------------
if (Running "app.main") {
    Write-Host "musicmaker already running" -ForegroundColor DarkGray
} else {
    Write-Host "starting musicmaker on :$Port" -ForegroundColor Cyan
    $data = Join-Path $Root "data"
    New-Item -ItemType Directory -Force -Path $data | Out-Null
    $env:ACESTEP_DIR           = $c.acestep_dir
    $env:ACESTEP_URL           = "http://127.0.0.1:8001"
    $env:MUSICMAKER_DATA_DIR   = Join-Path $data "projects"
    $env:MUSICMAKER_MODELS_DIR = Join-Path $data "models"
    # No token locally. It guards a machine that costs money to run; this one is
    # already yours, and a password on localhost is friction without a threat.
    Remove-Item Env:\MUSICMAKER_API_TOKEN -ErrorAction SilentlyContinue

    # The same virtualenv ACE-Step uses: it already has torch, and the
    # server needs torch for separation.
    $py = Join-Path $c.server_venv "Scripts\python.exe"
    Start-Process -FilePath $py `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                        "--port", "$Port") `
        -WorkingDirectory (Join-Path $c.repo "server") `
        -RedirectStandardOutput (Join-Path $logs "musicmaker.log") `
        -RedirectStandardError  (Join-Path $logs "musicmaker.err.log") `
        -WindowStyle Hidden
}

# --- wait for it ------------------------------------------------------------
Write-Host "waiting for the API" -NoNewline
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
        Write-Host "`nready: $($h.status)" -ForegroundColor Green
        Write-Host "Point the app at Local (Settings), or:"
        Write-Host "  build\Release\musicx.exe --connect http://127.0.0.1:$Port"
        exit 0
    } catch { Write-Host "." -NoNewline }
}
Write-Host "`nStill not answering. Logs are in $logs" -ForegroundColor Yellow
exit 1

[CmdletBinding()]
param(
    [string]$PidFile
)

# Resolve PID file path at runtime. Avoid using $PSScriptRoot in param defaults.
$scriptBase = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($PidFile)) {
    $PidFile = Join-Path $scriptBase ".dev-procs.json"
} elseif (-not [System.IO.Path]::IsPathRooted($PidFile)) {
    $PidFile = Join-Path $scriptBase $PidFile
}

# Configure interpreters here (fixed in file, no startup override needed).
$VoiceprintPython = "C:\Users\11979\anaconda3\envs\lc\python.exe"
$XiaozhiAppPython = "C:\Users\11979\anaconda3\envs\xiaozhi\python.exe"
$CodexAppPython = "C:\Users\11979\anaconda3\envs\xiaozhi\python.exe"

$services = @(
    @{
        Name = "voiceprint-api"
        Python = $VoiceprintPython
        WorkDir = "C:\Users\11979\Documents\GitHub\voiceprint-api"
        Script = "start_server.py"
    },
    @{
        Name = "xiaozhi-app"
        Python = $XiaozhiAppPython
        WorkDir = "C:\Users\11979\Documents\GitHub\xiaozhi-esp32-server\main\xiaozhi-server"
        Script = "app.py"
    },
    @{
        Name = "xiaozhi-codex-app"
        Python = $CodexAppPython
        WorkDir = "C:\Users\11979\Documents\GitHub\xiaozhi-esp32-server\main\xiaozhi-server"
        Script = "codex_app.py"
    }
)

if (Test-Path $PidFile) {
    Remove-Item $PidFile -Force -ErrorAction Stop
}

$started = @()
$errors = @()

foreach ($svc in $services) {
    $pythonPath = $svc.Python
    if (-not (Test-Path $pythonPath)) {
        $errors += "[$($svc.Name)] python not found: $pythonPath"
        continue
    }

    $scriptPath = Join-Path $svc.WorkDir $svc.Script
    if (-not (Test-Path $scriptPath)) {
        $errors += "[$($svc.Name)] script not found: $scriptPath"
        continue
    }

    $proc = Start-Process -FilePath $pythonPath -ArgumentList @($svc.Script) -WorkingDirectory $svc.WorkDir -PassThru

    $record = [PSCustomObject]@{
        name = $svc.Name
        pid = $proc.Id
        python = $pythonPath
        workdir = $svc.WorkDir
        script = $svc.Script
        started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }

    $started += $record
    Write-Host "Started $($svc.Name) (PID: $($proc.Id))"
}

if ($started.Count -gt 0) {
    $started | ConvertTo-Json -Depth 3 | Set-Content -Path $PidFile -Encoding UTF8 -ErrorAction Stop
    Write-Host "PID file written: $PidFile"
}

if ($errors.Count -gt 0) {
    Write-Warning "Some services did not start:"
    foreach ($msg in $errors) {
        Write-Warning $msg
    }
}

if ($started.Count -eq 0) {
    throw "No services started. Check python paths and script paths."
}

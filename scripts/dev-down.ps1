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

if (-not (Test-Path $PidFile)) {
    Write-Host "PID file not found: $PidFile"
    exit 0
}

try {
    $records = Get-Content -Path $PidFile -Raw -ErrorAction Stop | ConvertFrom-Json
} catch {
    throw "Failed to parse PID file: $PidFile"
}

if ($records -isnot [System.Array]) {
    $records = @($records)
}

foreach ($r in $records) {
    $procId = [int]$r.pid
    $name = [string]$r.name

    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        Write-Host "Already stopped: $name (PID: $procId)"
        continue
    }

    Stop-Process -Id $procId -Force
    Write-Host "Stopped $name (PID: $procId)"
}

Remove-Item $PidFile -Force -ErrorAction Stop
Write-Host "PID file removed: $PidFile"
